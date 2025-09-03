using LinearAlgebra, IniFile, DifferentialEquations, Plots, LaTeXStrings
using ElectronSpinDynamics

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

time_ns = 0:simparams.dt:simparams.simulation_time

@time results = SC(sys, mol1, mol2, simparams)
B0 = 0.05
tp = results[B0]["T+"]
t0 = results[B0]["T0"]
s = results[B0]["S"]
tm = results[B0]["T-"]

plt = plot(time_ns, tp; label=L"$T_+$", lw=2)
plot!(plt, time_ns, s; xlabel="time / ns", ylabel="P", label=L"$S$", lw=2)
plot!(plt, time_ns, t0; label=L"$T_0$", lw=2)
plot!(plt, time_ns, tm; label=L"$T_-$", lw=2)
plot!(plt, time_ns, s+tp+tm+t0; label="trace", lw=2, ylims=(0.15, 0.35))
display(plt)

results = SW(sys, mol1, mol2, simparams)
B0 = 0.05
tp = results[B0]["T+"]
t0 = results[B0]["T0"]
s = results[B0]["S"]
tm = results[B0]["T-"]

plot(time_ns, tp; label=L"$T_+$", lw=2)
plot!(time_ns, s; xlabel="time / ns", ylabel="P", label=L"$S$", lw=2)
plot!(time_ns, t0; label=L"$T_0$", lw=2)
plot!(time_ns, tm; label=L"$T_-$", lw=2)
plot!(time_ns, s+tp+tm+t0; label="trace", lw=2, ylims=(0.15, 0.35))
