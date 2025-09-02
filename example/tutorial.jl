using Base.Threads
using LinearAlgebra, IniFile, DifferentialEquations, Plots, LaTeXStrings
using ElectronSpinDynamics
using StaticArrays

cfg = read(Inifile(), "input.ini")   # Dict{String,Dict}
# 1. Pick a section

mol1 = read_molecule(cfg, "electron 1")
mol2 = read_molecule(cfg, "electron 2")
sys = read_system(cfg)
simparams = read_simparams(cfg)

@show mol1
@show mol2
@show sys
@show simparams

N = simparams.N_samples
time_ns = 0:simparams.dt:simparams.simulation_time
dt = simparams.dt / abs(γe)
nsteps = size(time_ns)[1]
B0 = simparams.B[2] # 0.05 mT

function each_process(
  nsteps::Integer, ham::SMatrix{4,4,ComplexF64}, Hsw::SMatrix{4,4,ComplexF64}, dt::Float64
)::Tuple{Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64}}
  L = liouvillian(ham + Hsw)
  U = exp(L * dt)
  ρ = vectorise(Ps)

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

function SW_main(N)
  dt = 1.0 * abs(γe)
  s = Vector{Float64}(undef, nsteps)
  tp = Vector{Float64}(undef, nsteps)
  tm = Vector{Float64}(undef, nsteps)
  t0 = Vector{Float64}(undef, nsteps)
  ham = system_hamiltonian(sys, B0)
  Hsw = SchultenWolynes_hamiltonian(mol1, mol2, N)
  for i in 1:N
    tp_i, t0_i, s_i, tm_i = each_process(nsteps, ham, Hsw[i], dt)
    s .+= s_i
    tp .+= tp_i
    tm .+= tm_i
    t0 .+= t0_i
  end
  tp ./ N, t0 ./ N, s ./ N, tm ./ N
end

tp, t0, s, tm = SW_main(N)

plot(time_ns, s; xlabel="time / ns", ylabel="P", label=L"$S$", lw=2)
plot!(time_ns, tp; label=L"$T_+$", lw=2)
plot!(time_ns, tm; label=L"$T_0$", lw=2)
plot!(time_ns, t0; label=L"$T_-$", lw=2)

@time begin
  SW_main(N)
end

function SW_thread(N)
  dt = 1.0 * abs(γe)
  ham = system_hamiltonian(sys, B0)
  Hsw = SchultenWolynes_hamiltonian(mol1, mol2, N)

  tp = Vector{Float64}(undef, nsteps)
  t0 = Vector{Float64}(undef, nsteps)
  s = Vector{Float64}(undef, nsteps)
  tm = Vector{Float64}(undef, nsteps)

  @threads for i in 1:N
    tp_i, t0_i, s_i, tm_i = each_process(nsteps, ham, Hsw[i], dt)
    tp .+= tp_i ./ N
    t0 .+= t0_i ./ N
    s .+= s_i ./ N
    tm .+= tm_i ./ N
  end

  return tp, t0, s, tm
end

# usage:
tp, t0, s, tm = SW_thread(N)

plot(time_ns, s; xlabel="time / ns", ylabel="P", label=L"$S$", lw=2)
plot!(time_ns, tp; label=L"$T_+$", lw=2)
plot!(time_ns, tm; label=L"$T_0$", lw=2)
plot!(time_ns, t0; label=L"$T_-$", lw=2)

@time begin
  SW_thread(N)
end
