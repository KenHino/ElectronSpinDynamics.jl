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
using DifferentialEquations: ODEProblem, solve, EnsembleProblem, EnsembleDistributed
using ..Utils: sample_from_sphere, sphere_to_cartesian
using HDF5
using Dates
using Printf

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
        a1=mol1.A,
        a2=mol2.A,
        mult1=mol1.mult,
        mult2=mol2.mult,
        output_folder=simparams.output_folder,
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
    a1::Union{Vector{Float64},Array{Float64,3}},
    a2::Union{Vector{Float64},Array{Float64,3}},
    mult1::Vector{<:Integer},
    mult2::Vector{<:Integer},
    output_folder::String=nothing,
    Nbatch::Int=10_000,
)::Dict{Float64,Dict{String,Vector{Float64}}}
    N = N_samples
    time_ns = 0:dt:simulation_time
    dt = dt * abs(γe)
    nsteps = size(time_ns)[1]
    results = Dict{Float64,Dict{String,Vector{Float64}}}()
    for B0 in B
        ham = system_hamiltonian(; B=B0, J=J, D=D, kS=kS, kT=kT)
        Hsw = SchultenWolynes_hamiltonian(a1, a2, mult1, mult2, N)

        if Nbatch >= N
            tp = Array{Float64,2}(undef, nsteps, N)
            t0 = Array{Float64,2}(undef, nsteps, N)
            s = Array{Float64,2}(undef, nsteps, N)
            tm = Array{Float64,2}(undef, nsteps, N)

            @threads for i in 1:N
                tp_i, t0_i, s_i, tm_i = each_process_SW(
                    nsteps, ham, Hsw[i], dt, initial_state
                )
                tp[:, i] = tp_i
                t0[:, i] = t0_i
                s[:, i] = s_i
                tm[:, i] = tm_i
            end
            μ_tp, se_tp = mean_std_se(tp)
            μ_t0, se_t0 = mean_std_se(t0)
            μ_s, se_s = mean_std_se(s)
            μ_tm, se_tm = mean_std_se(tm)
            results[B0] = Dict(
                "T+" => μ_tp,
                "T0" => μ_t0,
                "S" => μ_s,
                "T-" => μ_tm,
                "se_T+" => se_tp,
                "se_T0" => se_t0,
                "se_S" => se_s,
                "se_T-" => se_tm,
                "time_ns" => collect(time_ns),
            )
            continue
        end

        remaining = N

        rs_tp = RunningStats(nsteps)
        rs_t0 = RunningStats(nsteps)
        rs_s = RunningStats(nsteps)
        rs_tm = RunningStats(nsteps)

        while remaining > 0
            b = min(Nbatch, remaining)
            tp = Array{Float64,2}(undef, nsteps, b)
            t0 = Array{Float64,2}(undef, nsteps, b)
            s = Array{Float64,2}(undef, nsteps, b)
            tm = Array{Float64,2}(undef, nsteps, b)

            @threads for i in 1:b
                tp_i, t0_i, s_i, tm_i = each_process_SW(
                    nsteps, ham, Hsw[i], dt, initial_state
                )
                tp[:, i] = tp_i
                t0[:, i] = t0_i
                s[:, i] = s_i
                tm[:, i] = tm_i
            end
            μ_tp_b, se_tp_b = mean_std_se(tp)
            μ_t0_b, se_t0_b = mean_std_se(t0)
            μ_s_b, se_s_b = mean_std_se(s)
            μ_tm_b, se_tm_b = mean_std_se(tm)

            update!(rs_tp, b, μ_tp_b, batch_M2_from_se(se_tp_b, b))
            update!(rs_t0, b, μ_t0_b, batch_M2_from_se(se_t0_b, b))
            update!(rs_s, b, μ_s_b, batch_M2_from_se(se_s_b, b))
            update!(rs_tm, b, μ_tm_b, batch_M2_from_se(se_tm_b, b))

            remaining -= b
            z = 1.959963984540054 # quantile(Normal(), 0.975)
            mean_se_s = mean(se(rs_s.M2, rs_s.count); dims=1)[1] * z
            println(
                "$(Dates.format(Dates.now(), "HH:MM:SS")) Completed $(N - remaining) trajectories ($(Printf.@sprintf("%.3f", (N - remaining) / N * 100))%): ΔPs(95%) = $(Printf.@sprintf("%.3e", mean_se_s))",
            )
        end

        μ_tp, se_tp = finalize(rs_tp)
        μ_t0, se_t0 = finalize(rs_t0)
        μ_s, se_s = finalize(rs_s)
        μ_tm, se_tm = finalize(rs_tm)

        results[B0] = Dict(
            "T+" => μ_tp,
            "T0" => μ_t0,
            "S" => μ_s,
            "T-" => μ_tm,
            "se_T+" => se_tp,
            "se_T0" => se_t0,
            "se_S" => se_s,
            "se_T-" => se_tm,
            "time_ns" => collect(time_ns),
        )
    end

    if output_folder !== nothing
        output_folder = joinpath(output_folder, "SW")
        save_results(results, output_folder)
    end

    return results
end

function mean_std_se(M)
    N = size(M, 2)
    μ = vec(mean(M; dims=2))
    # corrected=true => divide by N-1 => sample std
    s = vec(std(M; corrected=true, dims=2))
    se = s ./ sqrt(N) # standard error
    μ, se
end

function save_results(
    results::Dict{Float64,Dict{String,Vector{Float64}}}, output_folder::String
)
    mkpath(output_folder)
    h5open(joinpath(output_folder, "results.h5"), "w") do file
        # Save magnetic field values list for reference
        B_values = sort(collect(keys(results)))
        write(file, "B", B_values)

        # Save data for each B as a separate group
        for B0 in B_values
            gname = "B=$(B0)"
            g = create_group(file, gname)
            data_for_B = results[B0]
            for (key, vec) in data_for_B
                write(g, key, vec)
            end
        end
    end
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
    qn1::Vector{Float64}
    qn2::Vector{Float64}
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
    @assert length(p.qn1) == length(p.A1)
    @assert length(p.qn2) == length(p.A2)
    norm1 = sqrt.(p.qn1 .* (p.qn1 .+ 1.0))
    norm2 = sqrt.(p.qn2 .* (p.qn2 .+ 1.0))
    nuclear1 = [getunit(2 + k) * norm1[k] for k in 1:length(p.A1)]
    nuclear2 = [getunit(2 + length(p.A1) + k) * norm2[k] for k in 1:length(p.A2)]
    return electron1, electron2, nuclear1, nuclear2
end

function assemble_initial_state(mode::Symbol, p::SCParams; seed=nothing)
    Nspin = nspin(p)
    S1, S2, I1_vec, I2_vec = build_spin_vectors(p; seed=seed)
    I1 = reduce(hcat, I1_vec)
    I2 = reduce(hcat, I2_vec)
    base = reduce(hcat, (S1, S2, I1, I2))
    if mode === :SC1
        return reshape(base, 3*Nspin)
    elseif mode === :SC2 || mode === :SC3
        T12 = S1 * transpose(S2)
        u0 = reshape(reduce(hcat, (base, T12)), 3*Nspin+9)
        if mode === :SC3
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
    offset1 = 6
    offset2 = 6 + 3*N1
    offset3 = 6 + 3*N1 + 3*N2
    I1 = @view u[(offset1 + 1):(offset1 + 3 * N1)]
    I2 = @view u[(offset2 + 1):(offset2 + 3 * N2)]
    dI1 = @view du[(offset1 + 1):(offset1 + 3 * N1)]
    dI2 = @view du[(offset2 + 1):(offset2 + 3 * N2)]
    ω1′ = p.ω1 + p.C*S2 + sum(p.A1[k]*I1[(3k - 2):3k] for k in 1:N1)
    ω2′ = p.ω2 + p.C*S1 + sum(p.A2[k]*I2[(3k - 2):3k] for k in 1:N2)
    du[1:3] .= cross(ω1′, S1)
    du[4:6] .= cross(ω2′, S2)
    for k in 1:N1
        dI1[(3k - 2):3k] .= cross(p.ω1n[k] + p.A1[k]' * S1, I1[(3k - 2):3k])
    end
    for k in 1:N2
        dI2[(3k - 2):3k] .= cross(p.ω2n[k] + p.A2[k]' * S2, I2[(3k - 2):3k])
    end
    return nothing
end

function SC2_naive!(du, u, p, t)
    """Naive implementation of SC2 is faster when parallelization is used(?)"""
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
    offset1 = 6
    offset2 = 6 + 3*N1
    offset3 = 6 + 3*N1 + 3*N2
    Cxx, Cyx, Czx = p.C[:, 1]
    Cxy, Cyy, Czy = p.C[:, 2]
    Cxz, Cyz, Czz = p.C[:, 3]
    Txx, Tyx, Tzx, Txy, Tyy, Tzy, Txz, Tyz, Tzz = @view u[(offset3 + 1):(offset3 + 9)]
    dTxx, dTyx, dTzx, dTxy, dTyy, dTzy, dTxz, dTyz, dTzz = @view du[(offset3 + 1):(offset3 + 9)]
    dI1 = @view du[(offset1 + 1):(offset1 + 3 * N1)]
    dI2 = @view du[(offset2 + 1):(offset2 + 3 * N2)]
    I1 = @view u[(offset1 + 1):(offset1 + 3 * N1)]
    I2 = @view u[(offset2 + 1):(offset2 + 3 * N2)]

    # dS1

    # ω1 x S1 + axial(C*T')
    dS1x = (Cyx*Tzx + Cyy*Tzy + Cyz*Tzz) - (Czx*Tyx + Czy*Tyy + Czz*Tyz)
    dS1y = (Czx*Txx + Czy*Txy + Czz*Txz) - (Cxx*Tzx + Cxy*Tzy + Cxz*Tzz)
    dS1z = (Cxx*Tyx + Cxy*Tyy + Cxz*Tyz) - (Cyx*Txx + Cyy*Txy + Cyz*Txz)
    ω1x, ω1y, ω1z = ω1 + sum(A1[k]*I1[(3k - 2):3k] for k in 1:N1)
    du[1] = ω1y*S1z - ω1z*S1y + dS1x
    du[2] = ω1z*S1x - ω1x*S1z + dS1y
    du[3] = ω1x*S1y - ω1y*S1x + dS1z

    # ω2 x S2 + axial(C' * T)
    dS2x = (Cxy*Txz + Cyy*Tyz + Czy*Tzz) - (Cxz*Txy + Cyz*Tyy + Czz*Tzy)
    dS2y = (Cxz*Txx + Cyz*Tyx + Czz*Tzx) - (Cxx*Txz + Cyx*Tyz + Czx*Tzz)
    dS2z = (Cxx*Txy + Cyx*Tyy + Czx*Tzy) - (Cxy*Txx + Cyy*Tyx + Czy*Tzx)
    ω2x, ω2y, ω2z = ω2 + sum(A2[k]*I2[(3k - 2):3k] for k in 1:N2)
    du[4] = ω2y*S2z - ω2z*S2y + dS2x
    du[5] = ω2z*S2x - ω2x*S2z + dS2y
    du[6] = ω2x*S2y - ω2y*S2x + dS2z

    # dI1
    for k in 1:N1
        dI1[(3k - 2):3k] .= cross(ω1n[k] + A1[k]' * S1, I1[(3k - 2):3k])
    end
    # dI2
    for k in 1:N2
        dI2[(3k - 2):3k] .= cross(ω2n[k] + A2[k]' * S2, I2[(3k - 2):3k])
    end

    # dTxx
    dTxx = (
        ω1y*Tzx - ω1z*Tyx + (ω2y*Txz - ω2z*Txy) - (S1y*Czx - S1z*Cyx) / 4 -
        (S2y*Cxz - S2z*Cxy) / 4
    )

    # dTyx
    dTyx = (
        ω1z*Txx - ω1x*Tzx + (ω2y*Tyz - ω2z*Tyy) - (S1z*Cxx - S1x*Czx) / 4 -
        (S2y*Cyz - S2z*Cyy) / 4
    )
    # dTzx
    dTzx = (
        ω1x*Tyx - ω1y*Txx + (ω2y*Tzz - ω2z*Tzy) - (S1x*Cyx - S1y*Cxx) / 4 -
        (S2y*Czz - S2z*Czy) / 4
    )
    # dTxy
    dTxy = (
        ω1y*Tzy - ω1z*Tyy + (ω2z*Txx - ω2x*Txz) - (S1y*Czy - S1z*Cyy) / 4 -
        (S2z*Cxx - S2x*Cxz) / 4
    )
    # dTyy
    dTyy = (
        ω1z*Txy - ω1x*Tzy + (ω2z*Tyx - ω2x*Tyz) - (S1z*Cxy - S1x*Czy) / 4 -
        (S2z*Cyx - S2x*Cyz) / 4
    )
    # dTzy
    dTzy = (
        ω1x*Tyy - ω1y*Txy + (ω2z*Tzx - ω2x*Tzz) - (S1x*Cyy - S1y*Cxy) / 4 -
        (S2z*Czx - S2x*Czz) / 4
    )
    # dTxz
    dTxz = (
        ω1y*Tzz - ω1z*Tyz + (ω2x*Txy - ω2y*Txx) - (S1y*Czz - S1z*Cyz) / 4 -
        (S2x*Cxy - S2y*Cxx) / 4
    )
    # dTyz
    dTyz = (
        ω1z*Txz - ω1x*Tzz + (ω2x*Tyy - ω2y*Tyx) - (S1z*Cxz - S1x*Czz) / 4 -
        (S2x*Cyy - S2y*Cyx) / 4
    )
    # dTzz
    dTzz = (
        ω1x*Tyz - ω1y*Txz + (ω2x*Tzy - ω2y*Tzx) - (S1x*Cyz - S1y*Cxz) / 4 -
        (S2x*Czy - S2y*Czx) / 4
    )
end

function SC2!(du, u, p::SCParams, t)
    N1 = length(p.A1)
    N2 = length(p.A2)
    offset1 = 6
    offset2 = 6 + 3*N1
    offset3 = 6 + 3*N1 + 3*N2
    S1 = SVector{3,Float64}(u[1:3])
    S2 = SVector{3,Float64}(u[4:6])
    I1 = @view u[(offset1 + 1):(offset1 + 3 * N1)]
    I2 = @view u[(offset2 + 1):(offset2 + 3 * N2)]
    dI1 = @view du[(offset1 + 1):(offset1 + 3 * N1)]
    dI2 = @view du[(offset2 + 1):(offset2 + 3 * N2)]
    ω1′ = p.ω1 + sum(p.A1[k]*I1[(3k - 2):3k] for k in 1:N1)
    for k in 1:N1
        dI1[(3k - 2):3k] .= cross(p.ω1n[k] + p.A1[k]' * S1, I1[(3k - 2):3k])
    end
    ω2′ = p.ω2 + sum(p.A2[k]*I2[(3k - 2):3k] for k in 1:N2)
    for k in 1:N2
        dI2[(3k - 2):3k] .= cross(p.ω2n[k] + p.A2[k]' * S2, I2[(3k - 2):3k])
    end
    T = @SMatrix [
        u[offset3 + 1] u[offset3 + 4] u[offset3 + 7];
        u[offset3 + 2] u[offset3 + 5] u[offset3 + 8];
        u[offset3 + 3] u[offset3 + 6] u[offset3 + 9]
    ]
    dS1 = cross(ω1′, S1) + axial(p.C*T')
    dS2 = cross(ω2′, S2) + axial(p.C'*T)
    dT = (
        crossmat(ω1′)*T - T*crossmat(ω2′) - crossmat(S1)*p.C/4 -
        p.C*transpose(crossmat(S2))/4
    )
    du[1:3] .= dS1
    du[4:6] .= dS2
    du[(offset3 + 1):(offset3 + 9)] .= vec(dT)
    return nothing
end

function SC3!(du, u, p::SCParams, t)
    S1 = SVector{3,Float64}(u[1:3])
    S2 = SVector{3,Float64}(u[4:6])
    n = u[end]
    N1 = length(p.A1)
    N2 = length(p.A2)
    offset1 = 6
    offset2 = 6 + 3*N1
    offset3 = 6 + 3*N1 + 3*N2
    I1 = @view u[(offset1 + 1):(offset1 + 3 * N1)]
    I2 = @view u[(offset2 + 1):(offset2 + 3 * N2)]
    dI1 = @view du[(offset1 + 1):(offset1 + 3 * N1)]
    dI2 = @view du[(offset2 + 1):(offset2 + 3 * N2)]
    ω1′ = p.ω1 + sum(p.A1[k]*I1[(3k - 2):3k] for k in 1:N1)
    factor1 = √3/2/norm(S1)
    for k in 1:N1
        dI1[(3k - 2):3k] .= cross(p.ω1n[k] + p.A1[k]' * S1, I1[(3k - 2):3k]) * factor1
    end
    ω2′ = p.ω2 + sum(p.A2[k]*I2[(3k - 2):3k] for k in 1:N2)
    factor2 = √3/2/norm(S2)
    for k in 1:N2
        dI2[(3k - 2):3k] .= cross(p.ω2n[k] + p.A2[k]' * S2, I2[(3k - 2):3k]) * factor2
    end
    T = @SMatrix [
        u[offset3 + 1] u[offset3 + 4] u[offset3 + 7];
        u[offset3 + 2] u[offset3 + 5] u[offset3 + 8];
        u[offset3 + 3] u[offset3 + 6] u[offset3 + 9]
    ]
    trT = tr(T)
    dS1 = cross(ω1′, S1) + axial(p.C*T') - kbar(p)*S1 + delta_k(p)*S2
    dS2 = cross(ω2′, S2) + axial(p.C'*T) - kbar(p)*S2 + delta_k(p)*S1
    dT = (
        crossmat(ω1′)*T - T*crossmat(ω2′) - crossmat(S1)*p.C/4 -
        p.C*transpose(crossmat(S2))/4 - kbar(p)*T +
        delta_k(p)*T' +
        delta_k(p)*(n/4-trT)*I(3)
    )
    du[1:3] .= dS1
    du[4:6] .= dS2
    du[(offset3 + 1):(offset3 + 9)] .= vec(dT)
    du[end] = -kbar(p)*n + 4*delta_k(p)*trT
    return nothing
end

function prob_func1(prob, i, repeat)
    prob.u0 .= assemble_initial_state(:SC1, prob.p)
    return prob
end
function prob_func2(prob, i, repeat)
    prob.u0 .= assemble_initial_state(:SC2, prob.p)
    return prob
end
function prob_func3(prob, i, repeat)
    prob.u0 .= assemble_initial_state(:SC3, prob.p)
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
        Ps[k] += 4.0*(St1x*St2x + St1y*St2y + St1z*St2z)*(S01x*S02x + S01y*S02y + S01z*S02z)
        Pt0[k] -=
            4.0*(St1x*St2x + St1y*St2y - St1z*St2z)*(S01x*S02x + S01y*S02y + S01z*S02z)
        Ptp[k] -=
            (4.0*St1z*St2z + 2.0*St1z + 2.0*St2z) * (S01x*S02x + S01y*S02y + S01z*S02z)
        Ptm[k] -=
            (4.0*St1z*St2z - 2.0*St1z - 2.0*St2z) * (S01x*S02x + S01y*S02y + S01z*S02z)
    end
    (
        (
            t,
            Ptp .* exp.(-sol.prob.p.kT*t),
            Pt0 .* exp.(-sol.prob.p.kT*t),
            Ps .* exp.(-sol.prob.p.kS*t),
            Ptm .* exp.(-sol.prob.p.kT*t),
        ),
        false,
    )
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
    (
        (
            t,
            Ptp .* exp.(-sol.prob.p.kT*t),
            Pt0 .* exp.(-sol.prob.p.kT*t),
            Ps .* exp.(-sol.prob.p.kS*t),
            Ptm .* exp.(-sol.prob.p.kT*t),
        ),
        false,
    )
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
    ((t, Ptp, Pt0, Ps, Ptm), false)
end

function average_ensemble(data)
    N = length(data)
    t = data.u[1][1]
    colmat(i) = hcat((d[i] for d in data.u)...)

    μ_tp, se_tp = mean_std_se(colmat(2))
    μ_t0, se_t0 = mean_std_se(colmat(3))
    μ_s, se_s = mean_std_se(colmat(4))
    μ_tm, se_tm = mean_std_se(colmat(5))
    return t, μ_tp, μ_t0, μ_s, μ_tm, se_tp, se_t0, se_s, se_tm
end

function C_from(J::Float64, D::Float64)::SMatrix{3,3,Float64}
    @assert D ≤ 0 "D must be non-positive under point-dipole approximation"
    Dtensor = diagm(0 => [2D/3, 2D/3, -4D/3])
    Cmat = Dtensor + 2.0 * J * I(3)
    C = SMatrix{3,3,Float64}(-Cmat)
    return C
end

function C_from(J::Float64, D::AbstractMatrix{Float64})::SMatrix{3,3,Float64}
    @assert size(D) == (3, 3) "D must be a 3×3 matrix"
    @assert D == D' "D must be symmetric"
    Dtensor = D
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
        mult1=mol1.mult,
        mult2=mol2.mult,
        output_folder=simparams.output_folder,
    )
end

mutable struct RunningStats
    count::Integer         # N = length(x): Number of samples
    mean::Vector{Float64}  # μ = ∑x_i / N: Mean of the samples
    M2::Vector{Float64}    # M2 = ∑(x_i - μ)²: Sum of squared deviations from the mean
end

RunningStats(nsteps::Integer) = RunningStats(0, zeros(nsteps), zeros(nsteps))

function update!(
    rs::RunningStats, b::Integer, bmean::AbstractVector{<:Real}, bM2::AbstractVector{<:Real}
)
    """
    Δ = μ_b - μ_N
    μnew = (N * μ_N + b * μ_b) / (N + b)
         = μ_N + Δ * (b / (N + b))
         = μ_B - Δ * (N / (N + b))
    M2new = ∑ (x_i - μnew)² + ∑ (x_j - μnew)²
          = ∑ (x_i - μ_N - Δ * (b / (N + b)))²
          + ∑ (x_j - μ_B + Δ * (N / (N + b)))²
          = M2_N + M2_B + Δ² * (N * b / (N + b))
    """

    new_count = rs.count + b
    Δ = bmean .- rs.mean
    rs.mean .+= Δ .* (b / new_count)
    rs.M2 .+= bM2 .+ (Δ .^ 2) .* (rs.count * b / new_count)
    rs.count = new_count
    return rs
end

function finalize(rs::RunningStats)
    n = rs.count
    if n <= 1
        return (copy(rs.mean), fill(NaN, length(rs.mean)))  # SE undefined for n<=1
    else
        s = sqrt.(rs.M2 ./ (n - 1))  # sample std
        se = s ./ sqrt(n)
        return (copy(rs.mean), se)
    end
end

function batch_M2_from_se(se::AbstractVector{<:Real}, b::Integer)
    b > 1 ? (se .^ 2) .* (b * (b - 1)) : zeros(length(se))
end  # since se = s/√b, M2 = s²*(b-1)

function se(M2::AbstractVector{<:Real}, n::Integer)
    std = sqrt.(M2 ./ (n - 1))
    return std ./ sqrt(n)
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
    mult1::Vector{<:Integer},
    mult2::Vector{<:Integer},
    output_folder::Union{String,Nothing}=nothing,
    Nbatch::Int=10000,
)::Dict{Float64,Dict{String,Vector{Float64}}}
    N = N_samples
    time_ns = 0:dt:simulation_time
    dt = dt * abs(γe)
    T = simulation_time * abs(γe)
    nsteps = length(time_ns)
    C = C_from(J, D)
    solver = (C ≈ zeros(3, 3) && kS == kT) ? :SC1 : (kS == kT ? :SC2 : :SC3)
    results = Dict{Float64,Dict{String,Vector{Float64}}}()

    for B0 in B
        ω1 = [0.0, 0.0, B0];
        ω2 = [0.0, 0.0, B0]
        ω1n = build_nuclear_omegas(B0, mult1)
        ω2n = build_nuclear_omegas(B0, mult2)
        A1 = arrays_to_smatrices(a1)
        A2 = arrays_to_smatrices(a2)
        qn1 = multiplicities_to_I(mult1)
        qn2 = multiplicities_to_I(mult2)

        p = SCParams(
            A1,
            A2,
            C,
            SVector{3,Float64}(ω1...),
            SVector{3,Float64}(ω2...),
            ω1n,
            ω2n,
            kS * 1e-03 / abs(γe),
            kT * 1e-03 / abs(γe),
            qn1,
            qn2,
        )

        u0 = assemble_initial_state(solver, p)
        if solver == :SC1
            println("Using SC1 solver (6 variables for electrons)")
            prob = ODEProblem(SC1!, u0, (0.0, T), p)
            eprob = EnsembleProblem(prob; output_func=output_func1, prob_func=prob_func1)
        elseif solver == :SC2
            println("Using SC2 solver (15 variables for electrons)")
            # prob = ODEProblem(SC2_naive!, u0, (0.0, T), p)
            prob = ODEProblem(SC2!, u0, (0.0, T), p)
            eprob = EnsembleProblem(prob; output_func=output_func2, prob_func=prob_func2)
        else
            println("Using SC3 solver (16 variables for electrons)")
            prob = ODEProblem(SC3!, u0, (0.0, T), p)
            eprob = EnsembleProblem(prob; output_func=output_func3, prob_func=prob_func3)
        end

        # --- single-shot (default) keeps old behavior
        if Nbatch >= N
            data = solve(eprob; dt=dt, saveat=0.0:dt:T, trajectories=N)
            t, μ_tp, μ_t0, μ_s, μ_tm, se_tp, se_t0, se_s, se_tm = average_ensemble(data)
            results[B0] = Dict(
                "T+" => μ_tp,
                "T0" => μ_t0,
                "S" => μ_s,
                "T-" => μ_tm,
                "se_T+" => se_tp,
                "se_T0" => se_t0,
                "se_S" => se_s,
                "se_T-" => se_tm,
                "time_ns" => t / abs(γe),
            )
            continue
        end

        # --- streamed/batched evaluation
        rs_tp = RunningStats(nsteps)
        rs_t0 = RunningStats(nsteps)
        rs_s = RunningStats(nsteps)
        rs_tm = RunningStats(nsteps)

        remaining = N
        first = true
        time_out = nothing

        while remaining > 0
            b = min(Nbatch, remaining)
            data = solve(eprob; dt=dt, saveat=0.0:dt:T, trajectories=b)
            t, μ_tp_b, μ_t0_b, μ_s_b, μ_tm_b, se_tp_b, se_t0_b, se_s_b, se_tm_b = average_ensemble(
                data
            )

            if first
                time_out = t / abs(γe)
                first = false
            end

            update!(rs_tp, b, μ_tp_b, batch_M2_from_se(se_tp_b, b))
            update!(rs_t0, b, μ_t0_b, batch_M2_from_se(se_t0_b, b))
            update!(rs_s, b, μ_s_b, batch_M2_from_se(se_s_b, b))
            update!(rs_tm, b, μ_tm_b, batch_M2_from_se(se_tm_b, b))

            remaining -= b
            z = 1.959963984540054 # quantile(Normal(), 0.975)
            mean_se_s = mean(se(rs_s.M2, rs_s.count); dims=1)[1] * z
            println(
                "$(Dates.format(Dates.now(), "HH:MM:SS")) Completed $(N - remaining) trajectories ($(Printf.@sprintf("%.3f", (N - remaining) / N * 100))%): ΔPs(95%) = $(Printf.@sprintf("%.3e", mean_se_s))",
            )
        end

        μ_tp, se_tp = finalize(rs_tp)
        μ_t0, se_t0 = finalize(rs_t0)
        μ_s, se_s = finalize(rs_s)
        μ_tm, se_tm = finalize(rs_tm)

        results[B0] = Dict(
            "T+" => μ_tp,
            "T0" => μ_t0,
            "S" => μ_s,
            "T-" => μ_tm,
            "se_T+" => se_tp,
            "se_T0" => se_t0,
            "se_S" => se_s,
            "se_T-" => se_tm,
            "time_ns" => time_out,
        )
    end

    if output_folder !== nothing
        output_folder = joinpath(output_folder, "SC")
        save_results(results, output_folder)
    end
    return results
end

export SW, SC

end # module
