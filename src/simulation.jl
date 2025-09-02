module Simulation
using ..SimParamModule: SimParams, StateType, Singlet, Triplet, read_simparams
using ..Constants: γe
using ..SystemModule: System, read_system
using ..MoleculeModule: Molecule, Aiso, read_molecule
using ..Hamiltonian: system_hamiltonian, SchultenWolynes_hamiltonian
using ..Liouvillian: liouvillian, vectorise
using ..SpinOps: Ps
using IniFile
using Base.Threads
using LinearAlgebra
using StaticArrays

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

function each_process_SC(
    nsteps::Integer,
    a1::Vector{Float64},
    a2::Vector{Float64},
    I1::Vector{<:Integer},
    I2::Vector{<:Integer},
    k::Float64,
    ω::Vector{Float64},
    dt::Float64,
    initial_state::StateType,
)::Tuple{Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64}}
    """
    Ref:
        - Manolopoulos and Hore (2013).
           - Improvement starting from the SW.
           - Assuming D=J=0 and kS=kT=k.
        - Lewis, Manolopoulos and Hore (2014).
           - Introduced correlation tensor T_12 to support kS != kT.
        - Fay, Lindoy, Manolopoulos and Hore (2020).
           - Introduced two methods to enable D != 0 and J != 0.
              - Method 1: Based on the straightforward precision EOM.
              - Method 2: Based on the T_12. Exact when the absence of nuclei. (probably better)
    Assuming
    - Dipolar and exchange interactions are neglected
    - kS == kT == k

    If nuclear spins are not moving, the results are the same as the SW.
    """
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
        a1=Aiso(mol1),
        a2=Aiso(mol2),
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
            tp_i, t0_i, s_i, tm_i = each_process_SC(nsteps, ham, Hsw[i], dt, initial_state)
            tp .+= tp_i ./ N
            t0 .+= t0_i ./ N
            s .+= s_i ./ N
            tm .+= tm_i ./ N
        end
        results[B0] = Dict("T+" => tp, "T0" => t0, "S" => s, "T-" => tm)
    end

    return results
end

export SW, SC

end # module
