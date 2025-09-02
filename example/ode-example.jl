using LinearAlgebra
using DifferentialEquations: ODEProblem, solve, EnsembleProblem, EnsembleSummary
using StaticArrays
using Plots
using Base.Threads
using ElectronSpinDynamics: sample_from_sphere, sphere_to_cartesian

A1 = [SMatrix{3,3}([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0])]
A2 = [SMatrix{3,3}([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0])]
C = SMatrix{3,3}([0.4 0.0 0.0; 0.0 0.4 0.0; 0.0 0.0 0.8])
ω1 = SVector{3}([0.0; 0.0; 1.0])
ω2 = SVector{3}([0.0; 0.0; 1.0])

θ, ϕ = sample_from_sphere((4, 1))
ux, uy, uz = sphere_to_cartesian(θ, ϕ)
S1 = [ux[1]; uy[1]; uz[1]] * √3/2
S2 = [ux[2]; uy[2]; uz[2]] * √3/2
I1 = [ux[3]; uy[3]; uz[3]] * √3/2
I2 = [ux[4]; uy[4]; uz[4]] * √3/2
u0 = reduce(hcat, [S1, S2, I1, I2])

function prob_func(prob, i, repeat)
  θ, ϕ = sample_from_sphere((4, 1))
  ux, uy, uz = sphere_to_cartesian(θ, ϕ)
  S1 = [ux[1]; uy[1]; uz[1]] * √3/2
  S2 = [ux[2]; uy[2]; uz[2]] * √3/2
  I1 = [ux[3]; uy[3]; uz[3]] * √3/2
  I2 = [ux[4]; uy[4]; uz[4]] * √3/2
  newu0 = reduce(hcat, (S1, S2, I1, I2))   # no @., no broadcasting
  prob.u0 .= newu0
  return prob# return a fresh problem
end

function SC1!(du, u, p, t)
  S1 = u[:, 1]
  S2 = u[:, 2]
  N1 = length(A1)
  N2 = length(A2)
  du[:, 1] = cross(ω1 + C*S2 + sum(A1[k]*u[:, 2 + k] for k in 1:N1), S1)
  du[:, 2] = cross(ω2 + C*S1 + sum(A2[k]*u[:, 2 + N1 + k] for k in 1:N2), S2)
  for k in 1:N1
    du[:, 2 + k] = cross(A1[k]' * S1, u[:, 2 + k])
  end
  for k in 1:N2
    du[:, 2 + k + N1] = cross(A2[k]' * S2, u[:, 2 + k + N1])
  end
end

dt = 0.05
saveat = 0.0:dt:1.0
tspan = (0.0, last(saveat))

prob = ODEProblem(SC1!, u0, tspan)

sol = solve(prob; dt=dt, saveat=saveat)

plot(sol.t, [sol.u[i][1, 1] for i in 1:size(sol.t)[1]]; label="S1x", marker=:o)
plot!(sol.t, [sol.u[i][2, 1] for i in 1:size(sol.t)[1]]; label="S1y", marker=:o)
plot!(sol.t, [sol.u[i][3, 1] for i in 1:size(sol.t)[1]]; label="S1z", marker=:o)
plot!(sol.t, [sol.u[i][1, 2] for i in 1:size(sol.t)[1]]; label="S2x", marker=:x)
plot!(sol.t, [sol.u[i][2, 2] for i in 1:size(sol.t)[1]]; label="S2y", marker=:x)
plot!(sol.t, [sol.u[i][3, 2] for i in 1:size(sol.t)[1]]; label="S2z", marker=:x)

output_func(sol, i) = (sol, false)
function output_func(sol, i)
  t = sol.t
  N = length(t)
  corr_S = Array{Float64,4}(undef, N, 2, 3, 3)
  corr_T0 = Array{Float64,4}(undef, N, 2, 3, 3)
  S0 = sol.u[1]
  for k in 1:N
    Sk = sol.u[k]
    for i in 1:2
      S0i = S0[:, i]
      Ski = Sk[:, i]
      for α in 1:3
        Skiα = Ski[α]
        sign = α ≤ 2 && i == 1 ? +1 : -1
        for β in 1:3
          S0iβ = S0i[β]
          corr_S[k, i, α, β] = Skiα * S0iβ
          corr_T0[k, i, α, β] = Skiα * S0iβ * (-sign)
        end
      end
    end
  end
  Ps = 1 / 4 .+ sum(prod(corr_S; dims=2) .* 4.0; dims=(3, 4))[:, 1, 1, 1]
  Pt0 = 1 / 4 .+ sum(prod(corr_T0; dims=2) .* 4.0; dims=(3, 4))[:, 1, 1, 1]

  Ptp = ones(Float64, N) * 1/4
  Ptm = ones(Float64, N) * 1/4

  for k in 1:N
    Sk = sol.u[k]
    S01 = S0[:, 1]
    S02 = S0[:, 2]
    Sk1 = Sk[:, 1]
    Sk2 = Sk[:, 2]

    S01x = S01[1]
    S02x = S02[1]
    S01y = S01[2]
    S02y = S02[2]
    S01z = S01[3]
    S02z = S02[3]

    Sk1z = Sk1[3]
    Sk2z = Sk2[3]

    Ptp[k] -= (4.0*Sk1z*Sk2z + 2.0*Sk1z + 2.0*Sk2z) * (S01x*S02x + S01y*S02y + S01z*S02z)
    Ptm[k] -= (4.0*Sk1z*Sk2z - 2.0*Sk1z - 2.0*Sk2z) * (S01x*S02x + S01y*S02y + S01z*S02z)
  end

  ((t, Ptp, Pt0, Ps, Ptm), false)
end
eprob = EnsembleProblem(prob; output_func=output_func, prob_func=prob_func)

data = solve(eprob; dt=dt, saveat=saveat, trajectories=100_000)
@show data[1][1]
@show data[1][2]

plot(data[1][1], data[1][2])
for i in 2:10
  plot!(data[i][1], data[i][2])
end
plot!()

N = length(data)
Ptp_avg = sum([data[k][2] for k in 1:N]; dims=1) / length(data)
Pt0_avg = sum([data[k][3] for k in 1:N]; dims=1) / length(data)
Ps_avg = sum([data[k][4] for k in 1:N]; dims=1) / length(data)
Ptm_avg = sum([data[k][5] for k in 1:N]; dims=1) / length(data)
@show Ps_avg
plot(data[1][1], Ptp_avg; label="T+")
plot!(data[1][1], Pt0_avg; label="T0")
plot!(data[1][1], Ps_avg; label="S")
plot!(data[1][1], Ptm_avg; label="T-")
plot!(data[1][1], Ptp_avg + Pt0_avg + Ps_avg + Ptm_avg; label="Tr")
