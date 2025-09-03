using LinearAlgebra
using DifferentialEquations: ODEProblem, solve, EnsembleProblem, EnsembleSerial
using StaticArrays
using Plots
using ElectronSpinDynamics:
    sample_from_sphere,
    sphere_to_cartesian,
    dipolar_hamiltonian,
    liouvillian,
    vectorise,
    zeeman_hamiltonian
using Statistics
using Distributions

struct SpinParams
    A1::Vector{SMatrix{3,3,Float64,9}}
    A2::Vector{SMatrix{3,3,Float64,9}}
    C::SMatrix{3,3,Float64,9}
    ω1::SVector{3,Float64}
    ω2::SVector{3,Float64}
    ω1n::Vector{SVector{3,Float64}}
    ω2n::Vector{SVector{3,Float64}}
    kS::Float64
    kT::Float64
end
nspin(p::SpinParams) = 2 + length(p.A1) + length(p.A2)
kbar(p::SpinParams) = (p.kS + 3*p.kT) / 4.0
delta_k(p::SpinParams) = (p.kS - p.kT) / 4.0

A1 = [
    SMatrix{3,3}([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]) * 1.0,
    SMatrix{3,3}([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]) * 1.0,
]
A2 = [
    SMatrix{3,3}([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]) * 1.0,
    SMatrix{3,3}([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]) * 1.0,
]
C = SMatrix{3,3}([1.0 1.0 0.1; 1.0 0.0 0.0; 0.1 0.0 0.0]) * 4.0
B = SVector{3}([0.0; 0.0; 1.0]) * 2.0
ω1 = SVector{3}([0.4; 0.0; 0.0]) * 2.0 # asymmetric
ω2 = B
ω1n = [B .* 1e-03, B .* 1e-03]
ω2n = [B .* 1e-03, B .* 1e-03]
kS = 1.0
kT = 1.0
params = SpinParams(A1, A2, C, ω1, ω2, ω1n, ω2n, kS, kT)
@assert delta_k(params) ≈ 0.0
@assert length(A1) == length(ω1n)
@assert length(A2) == length(ω2n)

function build_spin_vectors(p::SpinParams; seed=nothing)
    Nspin = nspin(p)
    θ, ϕ = if seed === nothing
        sample_from_sphere((Nspin, 1))
    else
        sample_from_sphere((Nspin, 1); seed=seed)
    end
    ux, uy, uz = sphere_to_cartesian(θ, ϕ)
    getvec(i) = SVector{3,Float64}([ux[i]; uy[i]; uz[i]]) * √3/2
    electron1 = getvec(1)
    electron2 = getvec(2)
    nuclear1 = [getvec(2 + k) for k in 1:length(p.A1)]
    nuclear2 = [getvec(2 + length(p.A1) + k) for k in 1:length(p.A2)]
    return electron1, electron2, nuclear1, nuclear2
end

function assemble_initial_state(mode::Symbol, p::SpinParams; seed=nothing)
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

# Cross matrix and axial (vee) map (single definition)
crossmat(v::SVector{3,Float64}) = @SMatrix [
    0.0 -v[3] v[2];
    v[3] 0.0 -v[1];
    -v[2] v[1] 0.0
]
function axial(M::SMatrix{3,3,Float64})
    @SVector [M[2, 3]-M[3, 2], M[3, 1]-M[1, 3], M[1, 2]-M[2, 1]]
end

u0 = assemble_initial_state(:sc1, params; seed=123)

function prob_func1(prob, i, repeat)
    prob.u0 .= assemble_initial_state(:sc1, prob.p)
    return prob
end

function SC1!(du, u, p, t)
    """one dimensional u"""
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

dt = 0.05
saveat = 0.0:dt:2.0
tspan = (0.0, last(saveat))
prob = ODEProblem(SC1!, u0, tspan, params)
sol = solve(prob; dt=dt, saveat=saveat)

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

    (
        (t, Ptp .* exp.(-kT*t), Pt0 .* exp.(-kT*t), Ps .* exp.(-kS*t), Ptm .* exp.(-kT*t)),
        false,
    )
end
eprob1 = EnsembleProblem(prob; output_func=output_func1, prob_func=prob_func1)

data1 = solve(eprob1; dt=dt, saveat=saveat, trajectories=10_000)

# 95% CI for the mean (Normal)
z = 1.959963984540054  # quantile(Normal(), 0.975)
function show(data)
    N = length(data)
    t = data[1][1]                 # all replicates share the same x
    # Helper: stack replicates column-wise => size = length(x) × N
    colmat(i) = hcat((d[i] for d in data)...)

    Ptp = colmat(2)
    Pt0 = colmat(3)
    Ps = colmat(4)
    Ptm = colmat(5)
    Tr = Ptp + Pt0 + Ps + Ptm

    # helper: empirical 95% band + mean (no parametric assumption)
    # mean & sample std (Bessel) per x, then SE = s/sqrt(N)
    mean_std_se(M) = begin
        μ = vec(mean(M, dims=2))
        s = vec(std(M, dims=2))          # corrected=true by default -> sample std
        se = s ./ sqrt(N)
        μ, se
    end

    μ_tp, se_tp = mean_std_se(Ptp)
    μ_t0, se_t0 = mean_std_se(Pt0)
    μ_s, se_s = mean_std_se(Ps)
    μ_tm, se_tm = mean_std_se(Ptm)
    μ_tr, se_tr = mean_std_se(Tr)

    plt = plot(t, μ_tp; ribbon=z .* se_tp, label="T+ (95% CI of mean)", lw=3)
    plot!(plt, t, μ_t0; ribbon=z .* se_t0, label="T0 (95% CI of mean)", lw=3)
    plot!(plt, t, μ_s; ribbon=z .* se_s, label="S (95% CI of mean)", lw=3)
    plot!(plt, t, μ_tm; ribbon=z .* se_tm, label="T- (95% CI of mean)", lw=3)
    plot!(plt, t, μ_tr; ribbon=z .* se_tr, label="Tr (95% CI of mean)", lw=3)
    plt, μ_s, se_s
end

plt, μ_s_sc1, se_s_sc1 = show(data1)
#display(plt)

u0 = assemble_initial_state(:sc2, params; seed=123)

# Cross matrix and axial (vee) map

function SC2!(du, u, p, t)
    S1 = SVector{3,Float64}(u[1:3])
    S2 = SVector{3,Float64}(u[4:6])
    # unpack T (column-major)
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
    # Note:
    # If Heisenberg EOM is based on pauli matirces σ rather than S
    # (final results would be identical),
    # axial(C*T') -> axial(C*T') / 2
    # axial(C' * T) -> axial(C' * T) / 2
    # crossmat(S1)*C/4 -> crossmat(S1)*C / 2
    # C*transpose(crossmat(S2))/4 -> C*transpose(crossmat(S2)) / 2
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
prob2 = ODEProblem(SC2!, u0, tspan, params)

function SC3!(du, u, p, t)
    S1 = SVector{3,Float64}(u[1:3])
    S2 = SVector{3,Float64}(u[4:6])
    n = u[end] # norm
    # unpack T (column-major)
    N1 = length(p.A1)
    N2 = length(p.A2)

    offset = 6
    ω1′ = p.ω1 + sum(p.A1[k]'*u[(offset + 3 * (k - 1) + 1):(offset + 3 * k)] for k in 1:N1)
    factor1 = √3/2/norm(S1)
    for k in 1:N1
        Ik = @view u[(offset + 3 * (k - 1) + 1):(offset + 3 * k)]
        du[(offset + 3 * (k - 1) + 1):(offset + 3 * k)] .= (
            cross(p.ω1n[k] + p.A1[k]' * S1, Ik) * factor1
        )
    end

    offset = 6 + 3*N1
    ω2′ = p.ω2 + sum(p.A2[k]'*u[(offset + 3 * (k - 1) + 1):(offset + 3 * k)] for k in 1:N2)
    factor2 = √3/2/norm(S2)
    for k in 1:N2
        Ik = @view u[(offset + 3 * (k - 1) + 1):(offset + 3 * k)]
        du[(offset + 3 * (k - 1) + 1):(offset + 3 * k)] .= (
            cross(p.ω2n[k] + p.A2[k]' * S2, Ik) * factor2
        )
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

u0 = assemble_initial_state(:sc3, params; seed=123)
prob3 = ODEProblem(SC3!, u0, tspan, params)

function prob_func2(prob, i, repeat)
    prob.u0 .= assemble_initial_state(:sc2, prob.p)
    return prob
end

function prob_func3(prob, i, repeat)
    prob.u0 .= assemble_initial_state(:sc3, prob.p)
    return prob
end

function output_func2(sol, i)
    t = sol.t
    N = length(t)
    Ps = Array{Float64,1}(undef, N)
    Pt0 = Array{Float64,1}(undef, N)
    Ptp = Array{Float64,1}(undef, N)
    Ptm = Array{Float64,1}(undef, N)

    u = sol.u[1]
    S01x, S01y, S01z, S02x, S02y, S02z = u[1:6]
    N1 = length(prob.p.A1)
    N2 = length(prob.p.A2)
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
            Ptp .* exp.(-prob.p.kT*t),
            Pt0 .* exp.(-prob.p.kT*t),
            Ps .* exp.(-prob.p.kS*t),
            Ptm .* exp.(-prob.p.kT*t),
        ),
        false,
    )
end
eprob2 = EnsembleProblem(prob2; output_func=output_func2, prob_func=prob_func2)

function output_func3(sol, i)
    t = sol.t
    N = length(t)
    Ps = Array{Float64,1}(undef, N)
    Pt0 = Array{Float64,1}(undef, N)
    Ptp = Array{Float64,1}(undef, N)
    Ptm = Array{Float64,1}(undef, N)

    u = sol.u[1]
    S01x, S01y, S01z, S02x, S02y, S02z = u[1:6]
    N1 = length(prob.p.A1)
    N2 = length(prob.p.A2)
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
eprob3 = EnsembleProblem(prob3; output_func=output_func3, prob_func=prob_func3)

@time data2 = solve(eprob2, dt=dt, saveat=saveat, trajectories=10_000)

@time data3 = solve(eprob3, dt=dt, saveat=saveat, trajectories=10_000)

plt, μ_s_sc2, se_s_sc2 = show(data2)
#display(plt)
plt, μ_s_sc3, se_s_sc3 = show(data3)
#display(plt)

ham = -dipolar_hamiltonian(params.C) + zeeman_hamiltonian(params.ω1, params.ω2)
@show ishermitian(ham)

L = liouvillian(ham)

U = exp(L*dt)
ρ = vectorise(diagm(0=>[0.0, 0.0, 1.0, 0.0]))
t = sol.t
s = []
tp = []
t0 = []
tm = []
for k in t
    push!(tp, real(ρ[1]))
    push!(t0, real(ρ[6]))
    push!(s, real(ρ[11]))
    push!(tm, real(ρ[16]))
    ρ .= U * ρ
end
plt = plot(t, tp; label="T+", lw=3)
plot!(plt, t, t0; label="T0", lw=3)
plot!(plt, t, s; label="S", lw=3)
plot!(plt, t, tm; label="T-", lw=3)
#display(plt)

plt = plot(t, s; label="QM", lw=3)
plot!(plt, t, μ_s_sc1; ribbon=z .* se_s_sc1, label="SC1 (95% CI of mean)", lw=3)
plot!(plt, t, μ_s_sc2; ribbon=z .* se_s_sc2, label="SC2 (95% CI of mean)", lw=3)
plot!(plt, t, μ_s_sc3; ribbon=z .* se_s_sc3, label="SC3 (95% CI of mean)", lw=3)
display(plt)
