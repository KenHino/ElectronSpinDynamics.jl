using LinearAlgebra, IniFile, DifferentialEquations, Plots, LaTeXStrings
using ElectronSpinDynamics

cfg = read(Inifile(), "input.ini")   # Dict{String,Dict}
# cfg = read(Inifile(), "input-wo-nuclei.ini")   # Dict{String,Dict}
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

results = SC(sys, mol1, mol2, simparams)
B0 = 0.05
tp = results[B0]["T+"]
t0 = results[B0]["T0"]
s = results[B0]["S"]
tm = results[B0]["T-"]
@show s

plt = plot(time_ns, tp; label=L"$T_+$", lw=2)
plot!(plt, time_ns, s; xlabel="time / ns", ylabel="P", label=L"$S$", lw=2)
plot!(plt, time_ns, t0; label=L"$T_0$", lw=2)
plot!(plt, time_ns, tm; label=L"$T_-$", lw=2)
plot!(plt, time_ns, s+tp+tm+t0; label="trace", lw=2)
ylims!(plt, 0, 0.5)
display(plt)

results = SW(sys, mol1, mol2, simparams)
B0 = 0.05
tp = results[B0]["T+"]
t0 = results[B0]["T0"]
s = results[B0]["S"]
tm = results[B0]["T-"]

plt = plot(time_ns, tp; label=L"$T_+$", lw=2)
plot!(plt, time_ns, s; xlabel="time / ns", ylabel="P", label=L"$S$", lw=2)
plot!(plt, time_ns, t0; label=L"$T_0$", lw=2)
plot!(plt, time_ns, tm; label=L"$T_-$", lw=2)
ylims!(plt, 0, 0.5)
plot!(plt, time_ns, s+tp+tm+t0; label="trace", lw=2)
display(plt)

results = read_results("out/SC")
se_tp = results[0.05]["se_T+"] # standard error std(M) / √N
se_t0 = results[0.05]["se_T0"] # standard error std(M) / √N
se_s = results[0.05]["se_S"] # standard error std(M) / √N
se_tm = results[0.05]["se_T-"] # standard error std(M) / √N
μ_tp = results[0.05]["T+"] # mean of M
μ_t0 = results[0.05]["T0"] # mean of M
μ_s = results[0.05]["S"] # mean of M
μ_tm = results[0.05]["T-"] # mean of M
z = 1.959963984540054  # quantile(Normal(), 0.975)
time_ns = results[0.05]["time_ns"]

plt = plot(time_ns, μ_tp; ribbon=(z*se_tp, z*se_tp), label=L"$T_+$", lw=2)
plot!(plt, time_ns, μ_t0; ribbon=(z*se_t0, z*se_t0), label=L"$T_0$", lw=2)
plot!(plt, time_ns, μ_s; ribbon=(z*se_s, z*se_s), label=L"$S$", lw=2)
plot!(plt, time_ns, μ_tm; ribbon=(z*se_tm, z*se_tm), label=L"$T_-$", lw=2)
plot!(plt, time_ns, μ_tp+μ_t0+μ_s+μ_tm; label="trace", lw=2)
ylims!(plt, 0, 0.5)
display(plt)
savefig(plt, "out/SC_error.png")
