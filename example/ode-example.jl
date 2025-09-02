using LinearAlgebra
using DifferentialEquations: ODEProblem, solve, EnsembleProblem, EnsembleSummary
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

A1 = [
    SMatrix{3,3}([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]) * 0.0,
    SMatrix{3,3}([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]) * 0.0,
]
A2 = [
    SMatrix{3,3}([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]) * 0.0,
    SMatrix{3,3}([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]) * 0.0,
]
C = SMatrix{3,3}([1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]) * 3.0
B = SVector{3}([0.0; 0.0; 1.0]) * 2.0
ω1 = SVector{3}([0.4; 0.0; 0.0]) * 1.0 # asymmetric
ω2 = B
ω1n = [B .* 1e-03, B .* 1e-03]
ω2n = [B .* 1e-03, B .* 1e-03]

@assert length(A1) == length(ω1n)
@assert length(A2) == length(ω2n)
Nspin = 2 + length(A1) + length(A2)

θ, ϕ = sample_from_sphere((Nspin, 1); seed=123)
ux, uy, uz = sphere_to_cartesian(θ, ϕ)
S1 = [ux[1]; uy[1]; uz[1]] * √3/2
S2 = [ux[2]; uy[2]; uz[2]] * √3/2
I1 = reduce(hcat, [[ux[3]; uy[3]; uz[3]] * √3/2, [ux[3]; uy[3]; uz[3]] * √3/2])
I2 = reduce(hcat, [[ux[4]; uy[4]; uz[4]] * √3/2, [ux[3]; uy[3]; uz[3]] * √3/2])
@assert length(A1) == size(I1)[2]
@assert length(A2) == size(I2)[2]
_u0 = reduce(hcat, [S1, S2, I1, I2])
u0 = reshape(_u0, 3*Nspin)
#u0 = reshape(u0, (3, Nspin))
@show u0[1:3]
@show _u0

function prob_func(prob, i, repeat)
    θ, ϕ = sample_from_sphere((Nspin, 1))
    ux, uy, uz = sphere_to_cartesian(θ, ϕ)
    S1 = [ux[1]; uy[1]; uz[1]] * √3/2
    S2 = [ux[2]; uy[2]; uz[2]] * √3/2
    _I1 = []
    _I2 = []
    for k in 3:(length(A1) + 2)
        push!(_I1, [ux[k]; uy[k]; uz[k]] * √3/2)
    end
    for k in (length(A1) + 3):(length(A1) + length(A2) + 2)
        push!(_I2, [ux[k]; uy[k]; uz[k]] * √3/2)
    end
    I1 = reduce(hcat, _I1)
    I2 = reduce(hcat, _I2)
    newu0 = reduce(hcat, (S1, S2, I1, I2))
    newu0 = reshape(newu0, 3*Nspin)
    prob.u0 .= newu0
    return prob# return a fresh problem
end

function SC1!(du, u, p, t)
    """one dimensional u"""
    S1 = u[1:3]
    S2 = u[4:6]
    N1 = length(A1)
    N2 = length(A2)
    bgn = 6
    du[1:3] = cross(
        ω1 + C*S2 + sum(A1[k]*u[(bgn + 3 * (k - 1) + 1):(bgn + 3 * k)] for k in 1:N1), S1
    )
    bgn = 6 + 3*N1
    du[4:6] = cross(
        ω2 + C*S1 + sum(A2[k]*u[(bgn + 3 * (k - 1) + 1):(bgn + 3 * k)] for k in 1:N2), S2
    )
    bgn = 6
    for k in 1:N1
        du[(bgn + 3 * (k - 1) + 1):(bgn + 3 * k)] = cross(
            ω1n[k] + A1[k]' * S1, u[(bgn + 3 * (k - 1) + 1):(bgn + 3 * k)]
        )
    end
    bgn = 6 + 3*N1
    for k in 1:N2
        du[(bgn + 3 * (k - 1) + 1):(bgn + 3 * k)] = cross(
            ω2n[k] + A2[k]' * S2, u[(bgn + 3 * (k - 1) + 1):(bgn + 3 * k)]
        )
    end
end

dt = 0.025
saveat = 0.0:dt:2.0
tspan = (0.0, last(saveat))
prob = ODEProblem(SC1!, u0, tspan)

sol = solve(prob; dt=dt, saveat=saveat)

function output_func(sol, i)
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

    ((t, Ptp, Pt0, Ps, Ptm), false)
end
eprob = EnsembleProblem(prob; output_func=output_func, prob_func=prob_func)

data = solve(eprob; dt=dt, saveat=saveat, trajectories=10_000)

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

μ_s_sc1 = μ_s
se_s_sc1 = se_s

# 95% CI for the mean (Normal)
z = 1.959963984540054  # quantile(Normal(), 0.975)

plt = plot(t, μ_tp; ribbon=z .* se_tp, label="T+ (95% CI of mean)", lw=3)
plot!(plt, t, μ_t0; ribbon=z .* se_t0, label="T0 (95% CI of mean)", lw=3)
plot!(plt, t, μ_s; ribbon=z .* se_s, label="S (95% CI of mean)", lw=3)
plot!(plt, t, μ_tm; ribbon=z .* se_tm, label="T- (95% CI of mean)", lw=3)
plot!(plt, t, μ_tr; ribbon=z .* se_tr, label="Tr (95% CI of mean)", lw=3)
display(plt)

θ, ϕ = sample_from_sphere((Nspin, 1); seed=123)
ux, uy, uz = sphere_to_cartesian(θ, ϕ)
S1 = [ux[1]; uy[1]; uz[1]] * √3/2
S2 = [ux[2]; uy[2]; uz[2]] * √3/2
I1 = reduce(hcat, [[ux[3]; uy[3]; uz[3]] * √3/2, [ux[3]; uy[3]; uz[3]] * √3/2])
I2 = reduce(hcat, [[ux[4]; uy[4]; uz[4]] * √3/2, [ux[3]; uy[3]; uz[3]] * √3/2])
@assert length(A1) == size(I1)[2]
@assert length(A2) == size(I2)[2]
_u0 = reduce(hcat, [S1, S2, I1, I2])
@show _u0
@show transpose(S2)
T12 = S1 * transpose(S2)
@assert reshape(T12, 9)[1:2] ≈ 3/4 * [ux[1]*ux[2], uy[1]*ux[2]]
@show T12
_u0 = reduce(hcat, [_u0, T12])
@show _u0
u0 = reshape(_u0, 3*Nspin+9)

#u0 = reshape(u0, (3, Nspin))
#@show u0[1:3]
#@show _u0

function SC2!(du, u, p, t)
    """one dimensional u"""
    S1 = u[1:3]
    S2 = u[4:6]
    S1x, S1y, S1z = S1
    S2x, S2y, S2z = S2
    N1 = length(A1)
    N2 = length(A2)
    T12 = u[(3 * (N1 + N2 + 2) + 1):(3 * (N1 + N2 + 2) + 9)]
    Txx, Tyx, Tzx, Txy, Tyy, Tzy, Txz, Tyz, Tzz = T12

    Cxx, Cyx, Czx = C[:, 1]
    Cxy, Cyy, Czy = C[:, 2]
    Cxz, Cyz, Czz = C[:, 3]

    # dS1
    bgn = 6
    env1 = ω1 + sum(A1[k]'*u[(bgn + 3 * (k - 1) + 1):(bgn + 3 * k)] for k in 1:N1)
    E1x, E1y, E1z = env1
    # du[1:3] = cross(env1, S1)

    # --- S1 needs ε_{αβγ} C_{βδ} T_{δγ}  == axial(C*T)
    res1 = (Cyx*Tzx + Cyy*Tzy + Cyz*Tzz - Czx*Tyx - Czy*Tyy - Czz*Tyz)
    res2 = (Czx*Txx + Czy*Txy + Czz*Txz - Cxx*Tzx - Cxy*Tzy - Cxz*Tzz)
    res3 = (Cxx*Tyx + Cxy*Tyy + Cxz*Tyz - Cyx*Txx - Cyy*Txy - Cyz*Txz)
    #du[1] = E1y*S1z - E1z*S1y + res1
    #du[2] = E1z*S1x - E1x*S1z + res2
    #du[3] = E1x*S1y - E1y*S1x + res3

    # ----- S1 |C  = axial(C*T')
    dS1x = (Cyx*Tzx + Cyy*Tzy + Cyz*Tzz) - (Czx*Tyx + Czy*Tyy + Czz*Tyz)
    dS1y = (Czx*Txx + Czy*Txy + Czz*Txz) - (Cxx*Tzx + Cxy*Tzy + Cxz*Tzz)
    dS1z = (Cxx*Tyx + Cxy*Tyy + Cxz*Tyz) - (Cyx*Txx + Cyy*Txy + Cyz*Txz)

    # ----- S2 |C  = - axial(C' * T)
    dS2x = -((Cxy*Txz + Cyy*Tyz + Czy*Tzz) - (Cxz*Txy + Cyz*Tyy + Czz*Tzy))
    dS2y = -((Cxz*Txx + Cyz*Tyx + Czz*Tzx) - (Cxx*Txz + Cyx*Tyz + Czx*Tzz))
    dS2z = -((Cxx*Txy + Cyx*Tyy + Czx*Tzy) - (Cxy*Txx + Cyy*Tyx + Czy*Tzx))
    du[1] = E1y*S1z - E1z*S1y + dS1x
    du[2] = E1z*S1x - E1x*S1z + dS1x
    du[3] = E1x*S1y - E1y*S1x + dS1z

    # keep these: S2 uses ε_{αβγ} C_{βδ} T_{γδ} == axial(C*Tᵀ)
    bgn = 6 + 3*N1
    env2 = ω2 + sum(A2[k]'*u[(bgn + 3 * (k - 1) + 1):(bgn + 3 * k)] for k in 1:N2)
    E2x, E2y, E2z = env2
    # du[4:6] = cross(env2, S2)
    #du[4] = E2y*S2z - E2z*S2y + dS2x
    #du[5] = E2z*S2x - E2x*S2z + dS2y
    #du[6] = E2x*S2y - E2y*S2x + dS2z
    du[4] = E2y*S2z - E2z*S2y - res1
    du[5] = E2z*S2x - E2x*S2z - res2
    du[6] = E2x*S2y - E2y*S2x - res3

    # dI1
    bgn = 6
    for k in 1:N1
        du[(bgn + 3 * (k - 1) + 1):(bgn + 3 * k)] = cross(
            ω1n[k] + A1[k]' * S1, u[(bgn + 3 * (k - 1) + 1):(bgn + 3 * k)]
        )
    end
    # dI2
    bgn = 6 + 3*N1
    for k in 1:N2
        du[(bgn + 3 * (k - 1) + 1):(bgn + 3 * k)] = cross(
            ω2n[k] + A2[k]' * S2, u[(bgn + 3 * (k - 1) + 1):(bgn + 3 * k)]
        )
    end

    bgn = 6 + 3*N1 + 3*N2

    # dTxx
    du[bgn + 1] = E1y*Tzx - E1z*Tyx + (+ E2y*Txz - E2z*Txy)
    - 0.25*(S1y*Czx - S1z*Cyx) - 0.25*(S2y*Cxz - S2z*Cxy)

    # dTyx
    du[bgn + 2] = E1z*Txx - E1x*Tzx + (+ E2y*Tyz - E2z*Tyy)
    - 0.25*(S1z*Cxx - S1x*Czx) - 0.25*(S2y*Cyz - S2z*Cyy)

    # dTzx
    du[bgn + 3] = E1x*Tyx - E1y*Txx + (+ E2y*Tzz - E2z*Tzy)
    - 0.25*(S1x*Cyx - S1y*Cxx) - 0.25*(S2y*Czz - S2z*Czy)

    # dTxy
    du[bgn + 4] = E1y*Tzy - E1z*Tyy + (+ E2z*Txx - E2x*Txz)
    - 0.25*(S1y*Czy - S1z*Cyy) - 0.25*(S2z*Cxx - S2x*Cxz)

    # dTyy
    du[bgn + 5] = E1z*Txy - E1x*Tzy + (+ E2z*Tyx - E2x*Tyz)
    - 0.25*(S1z*Cxy - S1x*Czy) - 0.25*(S2z*Cyx - S2x*Cyz)

    # dTzy
    du[bgn + 6] = E1x*Tyy - E1y*Txy + (+ E2z*Tzx - E2x*Tzz)
    - 0.25*(S1x*Cyy - S1y*Cxy) - 0.25*(S2z*Czx - S2x*Czz)

    # dTxz
    du[bgn + 7] = E1y*Tzz - E1z*Tyz + (+ E2x*Txy - E2y*Txx)
    - 0.25*(S1y*Czz - S1z*Cyz) - 0.25*(S2x*Cxy - S2y*Cxx)

    # dTyz
    du[bgn + 8] = E1z*Txz - E1x*Tzz + (+ E2x*Tyy - E2y*Tyx)
    - 0.25*(S1z*Cxz - S1x*Czz) - 0.25*(S2x*Cyy - S2y*Cyx)

    # dTzz
    du[bgn + 9] = E1x*Tyz - E1y*Txz + (+ E2x*Tzy - E2y*Tzx)
    - 0.25*(S1x*Cyz - S1y*Cxz) - 0.25*(S2x*Czy - S2y*Czx)
end

prob = ODEProblem(SC2!, u0, tspan)

function prob_func(prob, i, repeat)
    θ, ϕ = sample_from_sphere((Nspin, 1))
    ux, uy, uz = sphere_to_cartesian(θ, ϕ)
    S1 = [ux[1]; uy[1]; uz[1]] * √3/2
    S2 = [ux[2]; uy[2]; uz[2]] * √3/2
    _I1 = []
    _I2 = []
    for k in 3:(length(A1) + 2)
        push!(_I1, [ux[k]; uy[k]; uz[k]] * √3/2)
    end
    for k in (length(A1) + 3):(length(A1) + length(A2) + 2)
        push!(_I2, [ux[k]; uy[k]; uz[k]] * √3/2)
    end
    I1 = reduce(hcat, _I1)
    I2 = reduce(hcat, _I2)
    _u0 = reduce(hcat, (S1, S2, I1, I2))
    T12 = S1 * transpose(S2)
    _u0 = reduce(hcat, [_u0, T12])
    newu0 = reshape(_u0, 3*Nspin+9)
    prob.u0 .= newu0
    return prob# return a fresh problem
end

function output_func(sol, i)
    t = sol.t
    N = length(t)
    Ps = Array{Float64,1}(undef, N)
    Pt0 = Array{Float64,1}(undef, N)
    Ptp = Array{Float64,1}(undef, N)
    Ptm = Array{Float64,1}(undef, N)

    u = sol.u[1]
    S01x, S01y, S01z, S02x, S02y, S02z = u[1:6]
    N1 = length(A1)
    N2 = length(A2)
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
    ((t, Ptp, Pt0, Ps, Ptm), false)
end
eprob = EnsembleProblem(prob; output_func=output_func, prob_func=prob_func)

data = solve(eprob; dt=dt, saveat=saveat, trajectories=10_000)

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

μ_s_sc2 = μ_s
se_s_sc2 = se_s

# 95% CI for the mean (Normal)
z = 1.959963984540054  # quantile(Normal(), 0.975)

plt = plot(t, μ_tp; ribbon=z .* se_tp, label="T+ (95% CI of mean)", lw=3)
plot!(plt, t, μ_t0; ribbon=z .* se_t0, label="T0 (95% CI of mean)", lw=3)
plot!(plt, t, μ_s; ribbon=z .* se_s, label="S (95% CI of mean)", lw=3)
plot!(plt, t, μ_tm; ribbon=z .* se_tm, label="T- (95% CI of mean)", lw=3)
plot!(plt, t, μ_tr; ribbon=z .* se_tr, label="Tr (95% CI of mean)", lw=3)
display(plt)

ham = -dipolar_hamiltonian(C) + zeeman_hamiltonian(ω1, ω2)
@show ishermitian(ham)

L = liouvillian(ham)

U = exp(L*dt)
ρ = vectorise(diagm(0=>[0.0, 0.0, 1.0, 0.0]))

s = []
tp = []
t0 = []
tm = []
for k in t
    push!(tp, real(ρ[1]))
    push!(t0, real(ρ[6]))
    push!(s, real(ρ[11]))
    push!(tm, real(ρ[16]))
    ρ = U * ρ
end
plt = plot(t, tp; label="T+", lw=3)
plot!(plt, t, t0; label="T0", lw=3)
plot!(plt, t, s; label="S", lw=3)
plot!(plt, t, tm; label="T-", lw=3)
display(plt)

plt = plot(t, s; label="QM", lw=3)
plot!(plt, t, μ_s_sc1; ribbon=z .* se_s_sc1, label="SC1 (95% CI of mean)", lw=3)
plot!(plt, t, μ_s_sc2; ribbon=z .* se_s_sc2, label="SC2 (95% CI of mean)", lw=3)
display(plt)
