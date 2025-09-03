module Simulation
using ..SimParamModule: SimParams, StateType, Singlet, Triplet, read_simparams
using ..Constants: γe, γ1H, γ14N
using ..SystemModule: System, read_system
using ..MoleculeModule: Molecule, Aiso, read_molecule
using ..Hamiltonian: system_hamiltonian, SchultenWolynes_hamiltonian
using ..Liouvillian: liouvillian, vectorise
using ..SpinOps: Ps
using IniFile
using Base.Threads
using LinearAlgebra
using StaticArrays
using Statistics: mean, std
# using Distributed
using DifferentialEquations: ODEProblem, solve, EnsembleProblem, EnsembleDistributed
using ..Utils: sample_from_sphere, sphere_to_cartesian

function each_process_SW(
    nsteps::Integer,
    ham::SMatrix{4,4,ComplexF64},
    Hsw::SMatrix{4,4,ComplexF64},
    dt::Float64,
    initial_state::StateType,
)::Tuple{Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64}}
    L = liouvillian(ham + Hsw)
    U = exp(L * dt)
    if initial_state == Singlet
        ρ = vectorise(Ps)
    else
        error("Unknown initial state: $initial_state")
    end

    s_i = Vector{Float64}(undef, nsteps)
    tp_i = Vector{Float64}(undef, nsteps)
    tm_i = Vector{Float64}(undef, nsteps)
    t0_i = Vector{Float64}(undef, nsteps)

    for t in 1:nsteps
        tp_i[t] = real(ρ[1])
        t0_i[t] = real(ρ[6])
        s_i[t] = real(ρ[11])
        tm_i[t] = real(ρ[16])
        ρ = U * ρ
    end
    return tp_i, t0_i, s_i, tm_i
end

function SW(input_file::String)::Dict{Float64,Dict{String,Vector{Float64}}}
    cfg = read(Inifile(), input_file)
    sys = read_system(cfg)
    mol1 = read_molecule(cfg, "electron 1")
    mol2 = read_molecule(cfg, "electron 2")
    simparams = read_simparams(cfg)
    return SW(sys, mol1, mol2, simparams)
end

function SW(
    sys::System, mol1::Molecule, mol2::Molecule, simparams::SimParams
)::Dict{Float64,Dict{String,Vector{Float64}}}
    return SW(;
        N_samples=simparams.N_samples,
        simulation_time=simparams.simulation_time,
        dt=simparams.dt,
        initial_state=simparams.initial_state,
        B=simparams.B,
        J=sys.J,
        D=sys.D,
        kS=sys.kS,
        kT=sys.kT,
        a1=Aiso(mol1),
        a2=Aiso(mol2),
        I1=mol1.I,
        I2=mol2.I,
    )
end

function SW(;
    N_samples::Integer,
    simulation_time::Float64,
    dt::Float64,
    initial_state::StateType,
    B::Vector{Float64},
    J::Float64,
    D::Union{Float64,AbstractMatrix{Float64}},
    kS::Float64,
    kT::Float64,
    a1::Vector{Float64},
    a2::Vector{Float64},
    I1::Vector{<:Integer},
    I2::Vector{<:Integer},
)::Dict{Float64,Dict{String,Vector{Float64}}}
    N = N_samples
    time_ns = 0:dt:simulation_time
    dt = dt * abs(γe)
    nsteps = size(time_ns)[1]
    results = Dict{Float64,Dict{String,Vector{Float64}}}()
    for B0 in B
        ham = system_hamiltonian(; B=B0, J=J, D=D, kS=kS, kT=kT)
        Hsw = SchultenWolynes_hamiltonian(a1, a2, I1, I2, N)

        tp = zeros(Float64, nsteps)
        t0 = zeros(Float64, nsteps)
        s = zeros(Float64, nsteps)
        tm = zeros(Float64, nsteps)

        @threads for i in 1:N
            tp_i, t0_i, s_i, tm_i = each_process_SW(nsteps, ham, Hsw[i], dt, initial_state)
            tp .+= tp_i ./ N
            t0 .+= t0_i ./ N
            s .+= s_i ./ N
            tm .+= tm_i ./ N
        end
        results[B0] = Dict("T+" => tp, "T0" => t0, "S" => s, "T-" => tm)
    end

    return results
end

struct SCParams
    A1::Vector{SMatrix{3,3,Float64,9}}
    A2::Vector{SMatrix{3,3,Float64,9}}
    C::SMatrix{3,3,Float64,9}
    ω1::SVector{3,Float64}
    ω2::SVector{3,Float64}
    ω1n::Vector{SVector{3,Float64}}
    ω2n::Vector{SVector{3,Float64}}
    kS::Float64
    kT::Float64
    I1::Vector{Float64}
    I2::Vector{Float64}
end

nspin(p::SCParams) = 2 + length(p.A1) + length(p.A2)
kbar(p::SCParams) = (p.kS + 3*p.kT) / 4.0
delta_k(p::SCParams) = (p.kS - p.kT) / 4.0

# --- Common helpers (moved out of SC) ---
gamma_for_multiplicity(m::Integer)::Float64 = (m == 3) ? γ14N : γ1H
nuclear_omega(B0::Float64, m::Integer)::SVector{3,Float64} = SVector{3,Float64}(
    0.0, 0.0, -B0 * (gamma_for_multiplicity(m) / abs(γe))
)
build_nuclear_omegas(B0::Float64, multiplicities::Vector{<:Integer})::Vector{SVector{3,Float64}} = [
    nuclear_omega(B0, m) for m in multiplicities
]
arrays_to_smatrices(A::Array{Float64,3})::Vector{SMatrix{3,3,Float64,9}} = [
    SMatrix{3,3,Float64,9}(A[i, :, :]) for i in 1:size(A, 1)
]
multiplicities_to_I(m::Vector{<:Integer})::Vector{Float64} = (Float64.(m) .- 1.0) ./ 2.0

crossmat(v::SVector{3,Float64}) = @SMatrix [
    0.0 -v[3] v[2];
    v[3] 0.0 -v[1];
    -v[2] v[1] 0.0
]
function axial(M::SMatrix{3,3,Float64})
    @SVector [M[2, 3]-M[3, 2], M[3, 1]-M[1, 3], M[1, 2]-M[2, 1]]
end

function build_spin_vectors(p::SCParams; seed=nothing)
    Nspin = nspin(p)
    θ, ϕ = if seed === nothing
        sample_from_sphere((Nspin, 1))
    else
        sample_from_sphere((Nspin, 1); seed=seed)
    end
    ux, uy, uz = sphere_to_cartesian(θ, ϕ)
    getunit(i) = SVector{3,Float64}([ux[i]; uy[i]; uz[i]])
    # electron spin S=1/2 => magnitude sqrt(S(S+1)) = sqrt(3)/2
    electron_factor = √3/2
    electron1 = getunit(1) * electron_factor
    electron2 = getunit(2) * electron_factor
    # nuclear spins: magnitude sqrt(I(I+1)) for each nucleus
    @assert length(p.I1) == length(p.A1)
    @assert length(p.I2) == length(p.A2)
    nuclear1 = [getunit(2 + k) * sqrt(p.I1[k]*(p.I1[k] + 1.0)) for k in 1:length(p.A1)]
    nuclear2 = [
        getunit(2 + length(p.A1) + k) * sqrt(p.I2[k]*(p.I2[k] + 1.0)) for
        k in 1:length(p.A2)
    ]
    return electron1, electron2, nuclear1, nuclear2
end

function assemble_initial_state(mode::Symbol, p::SCParams; seed=nothing)
    Nspin = nspin(p)
    S1, S2, I1_list, I2_list = build_spin_vectors(p; seed=seed)
    I1 = reduce(hcat, I1_list)
    I2 = reduce(hcat, I2_list)
    base = reduce(hcat, (S1, S2, I1, I2))
    if mode === :sc1
        return reshape(base, 3*Nspin)
    elseif mode === :sc2 || mode === :sc3
        T12 = S1 * transpose(S2)
        u0 = reshape(reduce(hcat, (base, T12)), 3*Nspin+9)
        if mode === :sc3
            push!(u0, 1.0)
        end
        return u0
    else
        error("Unknown mode: $(mode)")
    end
end

function SC1!(du, u, p::SCParams, t)
    S1 = @view u[1:3]
    S2 = @view u[4:6]
    N1 = length(p.A1)
    N2 = length(p.A2)
    offset = 6
    du[1:3] = cross(
        p.ω1 +
        p.C*S2 +
        sum(p.A1[k]*(u[(offset + 3 * (k - 1) + 1):(offset + 3 * k)]) for k in 1:N1),
        S1,
    )
    offset = 6 + 3*N1
    du[4:6] = cross(
        p.ω2 +
        p.C*S1 +
        sum(p.A2[k]*(u[(offset + 3 * (k - 1) + 1):(offset + 3 * k)]) for k in 1:N2),
        S2,
    )
    offset = 6
    for k in 1:N1
        Ik = @view u[(offset + 3 * (k - 1) + 1):(offset + 3 * k)]
        du[(offset + 3 * (k - 1) + 1):(offset + 3 * k)] = cross(
            p.ω1n[k] + p.A1[k]' * S1, Ik
        )
    end
    offset = 6 + 3*N1
    for k in 1:N2
        Ik = @view u[(offset + 3 * (k - 1) + 1):(offset + 3 * k)]
        du[(offset + 3 * (k - 1) + 1):(offset + 3 * k)] = cross(
            p.ω2n[k] + p.A2[k]' * S2, Ik
        )
    end
end

function SC2_naive!(du, u, p, t)
    """Naive implementation of SC2 is faster when parallelization is not used(?)"""
    S1 = @view u[1:3]
    S2 = @view u[4:6]
    S1x, S1y, S1z = S1
    S2x, S2y, S2z = S2
    A1 = p.A1
    A2 = p.A2
    ω1 = p.ω1
    ω2 = p.ω2
    ω1n = p.ω1n
    ω2n = p.ω2n
    C = p.C
    N1 = length(A1)
    N2 = length(A2)
    Cxx, Cyx, Czx = p.C[:, 1]
    Cxy, Cyy, Czy = p.C[:, 2]
    Cxz, Cyz, Czz = p.C[:, 3]
    #T12 = u[3*(N1+N2+2)+1:3*(N1+N2+2)+9]
    Txx, Tyx, Tzx, Txy, Tyy, Tzy, Txz, Tyz, Tzz = @view u[(3 * (N1 + N2 + 2) + 1):(3 * (N1 + N2 + 2) + 9)]

    # dS1
    bgn = 6
    env1 = ω1 #+ sum(A1[k]'*u[bgn+3*(k-1)+1:bgn+3*k] for k in 1:N1)
    E1x, E1y, E1z = env1

    # ----- S1 |C  = axial(C*T')
    dS1x = (Cyx*Tzx + Cyy*Tzy + Cyz*Tzz) - (Czx*Tyx + Czy*Tyy + Czz*Tyz)
    dS1y = (Czx*Txx + Czy*Txy + Czz*Txz) - (Cxx*Tzx + Cxy*Tzy + Cxz*Tzz)
    dS1z = (Cxx*Tyx + Cxy*Tyy + Cxz*Tyz) - (Cyx*Txx + Cyy*Txy + Cyz*Txz)

    # ----- S2 |C  = - axial(C' * T)
    dS2x = (Cxy*Txz + Cyy*Tyz + Czy*Tzz) - (Cxz*Txy + Cyz*Tyy + Czz*Tzy)
    dS2y = (Cxz*Txx + Cyz*Tyx + Czz*Tzx) - (Cxx*Txz + Cyx*Tyz + Czx*Tzz)
    dS2z = (Cxx*Txy + Cyx*Tyy + Czx*Tzy) - (Cxy*Txx + Cyy*Tyx + Czy*Tzx)
    du[1] = E1y*S1z - E1z*S1y + dS1x
    du[2] = E1z*S1x - E1x*S1z + dS1y
    du[3] = E1x*S1y - E1y*S1x + dS1z

    # keep these: S2 uses ε_{αβγ} C_{βδ} T_{γδ} == axial(C*Tᵀ)
    bgn = 6 + 3*N1
    env2 = ω2 + sum(A2[k]'*u[(bgn + 3 * (k - 1) + 1):(bgn + 3 * k)] for k in 1:N2)
    E2x, E2y, E2z = env2
    du[4] = E2y*S2z - E2z*S2y + dS2x
    du[5] = E2z*S2x - E2x*S2z + dS2y
    du[6] = E2x*S2y - E2y*S2x + dS2z

    # dI1
    bgn = 6
    for k in 1:N1
        du[(bgn + 3 * (k - 1) + 1):(bgn + 3 * k)] .= cross(
            ω1n[k] + A1[k]' * S1, u[(bgn + 3 * (k - 1) + 1):(bgn + 3 * k)]
        )
    end
    # dI2
    bgn = 6 + 3*N1
    for k in 1:N2
        du[(bgn + 3 * (k - 1) + 1):(bgn + 3 * k)] .= cross(
            ω2n[k] + A2[k]' * S2, u[(bgn + 3 * (k - 1) + 1):(bgn + 3 * k)]
        )
    end

    bgn = 6 + 3*N1 + 3*N2

    # dTxx
    du[bgn + 1] = (
        E1y*Tzx - E1z*Tyx + (E2y*Txz - E2z*Txy) - (S1y*Czx - S1z*Cyx) / 4 -
        (S2y*Cxz - S2z*Cxy) / 4
    )

    # dTyx
    du[bgn + 2] = (
        E1z*Txx - E1x*Tzx + (E2y*Tyz - E2z*Tyy) - (S1z*Cxx - S1x*Czx) / 4 -
        (S2y*Cyz - S2z*Cyy) / 4
    )
    # dTzx
    du[bgn + 3] = (
        E1x*Tyx - E1y*Txx + (E2y*Tzz - E2z*Tzy) - (S1x*Cyx - S1y*Cxx) / 4 -
        (S2y*Czz - S2z*Czy) / 4
    )
    # dTxy
    du[bgn + 4] = (
        E1y*Tzy - E1z*Tyy + (E2z*Txx - E2x*Txz) - (S1y*Czy - S1z*Cyy) / 4 -
        (S2z*Cxx - S2x*Cxz) / 4
    )
    # dTyy
    du[bgn + 5] = (
        E1z*Txy - E1x*Tzy + (E2z*Tyx - E2x*Tyz) - (S1z*Cxy - S1x*Czy) / 4 -
        (S2z*Cyx - S2x*Cyz) / 4
    )
    # dTzy
    du[bgn + 6] = (
        E1x*Tyy - E1y*Txy + (E2z*Tzx - E2x*Tzz) - (S1x*Cyy - S1y*Cxy) / 4 -
        (S2z*Czx - S2x*Czz) / 4
    )
    # dTxz
    du[bgn + 7] = (
        E1y*Tzz - E1z*Tyz + (E2x*Txy - E2y*Txx) - (S1y*Czz - S1z*Cyz) / 4 -
        (S2x*Cxy - S2y*Cxx) / 4
    )
    # dTyz
    du[bgn + 8] = (
        E1z*Txz - E1x*Tzz + (E2x*Tyy - E2y*Tyx) - (S1z*Cxz - S1x*Czz) / 4 -
        (S2x*Cyy - S2y*Cyx) / 4
    )
    # dTzz
    du[bgn + 9] = (
        E1x*Tyz - E1y*Txz + (E2x*Tzy - E2y*Tzx) - (S1x*Cyz - S1y*Cxz) / 4 -
        (S2x*Czy - S2y*Czx) / 4
    )
end

function SC2!(du, u, p::SCParams, t)
    S1 = SVector{3,Float64}(u[1:3])
    S2 = SVector{3,Float64}(u[4:6])
    N1 = length(p.A1)
    N2 = length(p.A2)
    offset = 6
    ω1′ =
        p.ω1 + sum(p.A1[k]'*(u[(offset + 3 * (k - 1) + 1):(offset + 3 * k)]) for k in 1:N1)
    for k in 1:N1
        Ik = @view u[(offset + 3 * (k - 1) + 1):(offset + 3 * k)]
        du[(offset + 3 * (k - 1) + 1):(offset + 3 * k)] .= cross(
            p.ω1n[k] + p.A1[k]' * S1, Ik
        )
    end
    offset = 6 + 3*N1
    ω2′ =
        p.ω2 + sum(p.A2[k]'*(u[(offset + 3 * (k - 1) + 1):(offset + 3 * k)]) for k in 1:N2)
    for k in 1:N2
        Ik = @view u[(offset + 3 * (k - 1) + 1):(offset + 3 * k)]
        du[(offset + 3 * (k - 1) + 1):(offset + 3 * k)] .= cross(
            p.ω2n[k] + p.A2[k]' * S2, Ik
        )
    end
    offset = 6 + 3*N1 + 3*N2
    T = @SMatrix [
        u[offset + 1] u[offset + 4] u[offset + 7];
        u[offset + 2] u[offset + 5] u[offset + 8];
        u[offset + 3] u[offset + 6] u[offset + 9]
    ]
    dS1 = cross(ω1′, S1) + axial(p.C*T')
    dS2 = cross(ω2′, S2) + axial(p.C' * T)
    dT = (
        crossmat(ω1′)*T - T*crossmat(ω2′) - crossmat(S1)*p.C/4 -
        p.C*transpose(crossmat(S2))/4
    )
    du[1:3] .= dS1
    du[4:6] .= dS2
    du[(offset + 1):(offset + 9)] .= vec(dT)
    return nothing
end

function SC3!(du, u, p::SCParams, t)
    S1 = SVector{3,Float64}(u[1:3])
    S2 = SVector{3,Float64}(u[4:6])
    n = u[end]
    N1 = length(p.A1)
    N2 = length(p.A2)
    offset = 6
    ω1′ = p.ω1 + sum(p.A1[k]'*u[(offset + 3 * (k - 1) + 1):(offset + 3 * k)] for k in 1:N1)
    factor1 = √3/2/norm(S1)
    for k in 1:N1
        Ik = @view u[(offset + 3 * (k - 1) + 1):(offset + 3 * k)]
        du[(offset + 3 * (k - 1) + 1):(offset + 3 * k)] .=
            cross(p.ω1n[k] + p.A1[k]' * S1, Ik) * factor1
    end
    offset = 6 + 3*N1
    ω2′ = p.ω2 + sum(p.A2[k]'*u[(offset + 3 * (k - 1) + 1):(offset + 3 * k)] for k in 1:N2)
    factor2 = √3/2/norm(S2)
    for k in 1:N2
        Ik = @view u[(offset + 3 * (k - 1) + 1):(offset + 3 * k)]
        du[(offset + 3 * (k - 1) + 1):(offset + 3 * k)] .=
            cross(p.ω2n[k] + p.A2[k]' * S2, Ik) * factor2
    end
    offset = 6 + 3*N1 + 3*N2
    T = @SMatrix [
        u[offset + 1] u[offset + 4] u[offset + 7];
        u[offset + 2] u[offset + 5] u[offset + 8];
        u[offset + 3] u[offset + 6] u[offset + 9]
    ]
    trT = tr(T)
    dS1 = cross(ω1′, S1) + axial(p.C*T') - kbar(p)*S1 + delta_k(p)*S2
    dS2 = cross(ω2′, S2) + axial(p.C' * T) - kbar(p)*S2 + delta_k(p)*S1
    dT = (
        crossmat(ω1′)*T - T*crossmat(ω2′) - crossmat(S1)*p.C/4 -
        p.C*transpose(crossmat(S2))/4 - kbar(p)*T +
        delta_k(p)*T' +
        delta_k(p)*(n/4-trT)*I(3)
    )
    du[1:3] .= dS1
    du[4:6] .= dS2
    du[(offset + 1):(offset + 9)] .= vec(dT)
    du[end] = -kbar(p)*n + 4*delta_k(p)*trT
    return nothing
end

function prob_func1(prob, i, repeat)
    prob.u0 .= assemble_initial_state(:sc1, prob.p)
    return prob
end
function prob_func2(prob, i, repeat)
    prob.u0 .= assemble_initial_state(:sc2, prob.p)
    return prob
end
function prob_func3(prob, i, repeat)
    prob.u0 .= assemble_initial_state(:sc3, prob.p)
    return prob
end

function output_func1(sol, i)
    t = sol.t
    N = length(t)
    Ps = ones(Float64, N) * 1/4
    Pt0 = ones(Float64, N) * 1/4
    Ptp = ones(Float64, N) * 1/4
    Ptm = ones(Float64, N) * 1/4
    S0 = sol.u[1]
    S01x, S01y, S01z, S02x, S02y, S02z = S0[1:6]
    for k in 1:N
        St = sol.u[k]
        St1x, St1y, St1z, St2x, St2y, St2z = St[1:6]
        Ps[k] += 4.0*(St1x*St2x+St1y*St2y+St1z*St2z)*(S01x*S02x+S01y*S02y+S01z*S02z)
        Pt0[k] -= 4.0*(St1x*St2x+St1y*St2y-St1z*St2z)*(S01x*S02x+S01y*S02y+S01z*S02z)
        Ptp[k] -=
            (4.0*St1z*St2z + 2.0*St1z + 2.0*St2z) * (S01x*S02x + S01y*S02y + S01z*S02z)
        Ptm[k] -=
            (4.0*St1z*St2z - 2.0*St1z - 2.0*St2z) * (S01x*S02x + S01y*S02y + S01z*S02z)
    end
    if i % 10_000 == 0
        println("Completed $(i) trajectories")
    end
    ((t, Ptp, Pt0, Ps, Ptm), false)
end

function output_func2(sol, i)
    t = sol.t
    N = length(t)
    Ps = Array{Float64,1}(undef, N)
    Pt0 = Array{Float64,1}(undef, N)
    Ptp = Array{Float64,1}(undef, N)
    Ptm = Array{Float64,1}(undef, N)
    u = sol.u[1]
    N1 = length(sol.prob.p.A1)
    N2 = length(sol.prob.p.A2)
    T12 = u[(3 * (N1 + N2 + 2) + 1):(3 * (N1 + N2 + 2) + 9)]
    Txx0, Tyx0, Tzx0, Txy0, Tyy0, Tzy0, Txz0, Tyz0, Tzz0 = T12
    tr0 = Txx0 + Tyy0 + Tzz0
    for k in 1:N
        u = sol.u[k]
        St1x, St1y, St1z, St2x, St2y, St2z = u[1:6]
        T12 = u[(3 * (N1 + N2 + 2) + 1):(3 * (N1 + N2 + 2) + 9)]
        Txx, Tyx, Tzx, Txy, Tyy, Tzy, Txz, Tyz, Tzz = T12
        trk = Txx + Tyy + Tzz
        Ps[k] = (1/4 - trk) * (1/4 - tr0) * 4
        Pt0[k] = (1/4 + Txx + Tyy - Tzz) * (1/4 - tr0) * 4
        Ptp[k] = (1/4 + 1/2*St1z + 1/2*St2z + Tzz) * (1/4 - tr0) * 4
        Ptm[k] = (1/4 - 1/2*St1z - 1/2*St2z + Tzz) * (1/4 - tr0) * 4
    end
    if i % 10_000 == 0
        println("Completed $(i) trajectories")
    end
    ((t, Ptp, Pt0, Ps, Ptm), false)
end

function output_func3(sol, i)
    t = sol.t
    N = length(t)
    Ps = Array{Float64,1}(undef, N)
    Pt0 = Array{Float64,1}(undef, N)
    Ptp = Array{Float64,1}(undef, N)
    Ptm = Array{Float64,1}(undef, N)
    u = sol.u[1]
    N1 = length(sol.prob.p.A1)
    N2 = length(sol.prob.p.A2)
    T12 = u[(3 * (N1 + N2 + 2) + 1):(3 * (N1 + N2 + 2) + 9)]
    Txx0, Tyx0, Tzx0, Txy0, Tyy0, Tzy0, Txz0, Tyz0, Tzz0 = T12
    p0 = u[end]
    tr0 = Txx0 + Tyy0 + Tzz0
    for k in 1:N
        u = sol.u[k]
        p = u[end]
        St1x, St1y, St1z, St2x, St2y, St2z = u[1:6]
        T12 = u[(3 * (N1 + N2 + 2) + 1):(3 * (N1 + N2 + 2) + 9)]
        Txx, Tyx, Tzx, Txy, Tyy, Tzy, Txz, Tyz, Tzz = T12
        trk = Txx + Tyy + Tzz
        Ps[k] = (p/4 - trk) * (p0/4 - tr0) * 4
        Pt0[k] = (p/4 + Txx + Tyy - Tzz) * (p0/4 - tr0) * 4
        Ptp[k] = (p/4 + 1/2*St1z + 1/2*St2z + Tzz) * (p0/4 - tr0) * 4
        Ptm[k] = (p/4 - 1/2*St1z - 1/2*St2z + Tzz) * (p0/4 - tr0) * 4
    end
    if i % 10_000 == 0
        println("Completed $(i) trajectories")
    end
    ((t, Ptp, Pt0, Ps, Ptm), false)
end

function average_ensemble(data)
    N = length(data)
    t = data.u[1][1]
    colmat(i) = hcat((d[i] for d in data.u)...)
    Ptp = mean(colmat(2); dims=2)[:]
    Pt0 = mean(colmat(3); dims=2)[:]
    Ps = mean(colmat(4); dims=2)[:]
    Ptm = mean(colmat(5); dims=2)[:]
    return t, Ptp, Pt0, Ps, Ptm
end

function C_from(J::Float64, D::Union{Float64,AbstractMatrix{Float64}})::SMatrix{3,3,Float64}
    if typeof(D) == Float64
        @assert D ≤ 0 "D must be non-positive under point-dipole approximation"
        Dtensor = diagm(0 => [2D/3, 2D/3, -4D/3])
    else
        @assert size(D) == (3, 3) "D must be a 3×3 matrix"
        @assert D == D' "D must be symmetric"
        Dtensor = D
    end
    Cmat = Dtensor + 2.0 * J * I(3)
    C = SMatrix{3,3,Float64}(-Cmat)
    return C
end

function SC(input_file::String)::Dict{Float64,Dict{String,Vector{Float64}}}
    cfg = read(Inifile(), input_file)
    sys = read_system(cfg)
    mol1 = read_molecule(cfg, "electron 1")
    mol2 = read_molecule(cfg, "electron 2")
    simparams = read_simparams(cfg)
    return SC(sys, mol1, mol2, simparams)
end

function SC(
    sys::System, mol1::Molecule, mol2::Molecule, simparams::SimParams
)::Dict{Float64,Dict{String,Vector{Float64}}}
    return SC(;
        N_samples=simparams.N_samples,
        simulation_time=simparams.simulation_time,
        dt=simparams.dt,
        initial_state=simparams.initial_state,
        B=simparams.B,
        J=sys.J,
        D=sys.D,
        kS=sys.kS,
        kT=sys.kT,
        a1=mol1.A,
        a2=mol2.A,
        I1=mol1.I,
        I2=mol2.I,
    )
end

function SC(;
    N_samples::Integer,
    simulation_time::Float64,
    dt::Float64,
    initial_state::StateType,
    B::Vector{Float64},
    J::Float64,
    D::Union{Float64,AbstractMatrix{Float64}},
    kS::Float64,
    kT::Float64,
    a1::Array{Float64,3},
    a2::Array{Float64,3},
    I1::Vector{<:Integer},
    I2::Vector{<:Integer},
)::Dict{Float64,Dict{String,Vector{Float64}}}
    N = N_samples
    time_ns = 0:dt:simulation_time
    dt = dt * abs(γe)
    nsteps = size(time_ns)[1]
    C = C_from(J, D)
    solver = (C ≈ zeros(3, 3) && kS == kT) ? :SC1 : (kS == kT ? :SC2 : :SC3)
    results = Dict{Float64,Dict{String,Vector{Float64}}}()
    #addprocs(8)
    #println("Using $(nprocs()) processes")
    for B0 in B
        ω1 = [0.0, 0.0, B0]
        ω2 = [0.0, 0.0, B0]
        # Nuclear Zeeman frequencies (rescaled by |γe|): -B0 * γ_n / |γe|
        ω1n = build_nuclear_omegas(B0, I1)
        ω2n = build_nuclear_omegas(B0, I2)
        A1 = arrays_to_smatrices(a1)
        A2 = arrays_to_smatrices(a2)
        # Convert nuclear spin multiplicities m to spin quantum numbers I = (m-1)/2
        Ivals1 = multiplicities_to_I(I1)
        Ivals2 = multiplicities_to_I(I2)

        p = SCParams(
            A1,
            A2,
            C,
            SVector{3,Float64}(ω1...),
            SVector{3,Float64}(ω2...),
            ω1n,
            ω2n,
            kS,
            kT,
            Ivals1,
            Ivals2,
        )

        if solver == :SC1
            println("Using SC1 solver (6 variables for electrons)")
            u0 = assemble_initial_state(:sc1, p; seed=123)
            prob = ODEProblem(SC1!, u0, (0.0, simulation_time * abs(γe)), p)
            eprob = EnsembleProblem(prob; output_func=output_func1, prob_func=prob_func1)
        elseif solver == :SC2
            println("Using SC2 solver (15 variables for electrons)")
            u0 = assemble_initial_state(:sc2, p; seed=123)
            # prob = ODEProblem(SC2!, u0, (0.0, simulation_time * abs(γe)), p)
            prob = ODEProblem(SC2_naive!, u0, (0.0, simulation_time * abs(γe)), p)
            eprob = EnsembleProblem(prob; output_func=output_func2, prob_func=prob_func2)
        else
            println("Using SC3 solver (16 variables for electrons)")
            u0 = assemble_initial_state(:sc3, p; seed=123)
            prob = ODEProblem(SC3!, u0, (0.0, simulation_time * abs(γe)), p)
            eprob = EnsembleProblem(prob; output_func=output_func3, prob_func=prob_func3)
        end

        # data = solve(eprob, EnsembleDistributed(); dt=dt, saveat=0.0:dt:simulation_time, trajectories=N)
        data = solve(eprob; dt=dt, saveat=0.0:dt:simulation_time, trajectories=N)
        t, Ptp, Pt0, Ps, Ptm = average_ensemble(data)
        results[B0] = Dict("T+" => Ptp, "T0" => Pt0, "S" => Ps, "T-" => Ptm)
    end

    return results
end

export SW, SC

end # module
