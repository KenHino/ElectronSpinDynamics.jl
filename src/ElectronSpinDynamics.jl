module ElectronSpinDynamics

# Write your package code here.
include("constants.jl")
using .Constants      # brings ℏ, μB, … into scope

export ℏ, μ0, γe, γ1H, γ14N, g_electron

include("utils.jl")
using .Utils

export sample_from_sphere, sphere_to_cartesian, read_results

include("spinops.jl")
using .SpinOps

export σx,
    σy,
    σz,
    Sx,
    Sy,
    Sz,
    Sp,
    Sm,
    ST,
    ST_basis,
    Sx1,
    Sx2,
    Sy1,
    Sy2,
    Sz1,
    Sz2,
    S1S2,
    Ps,
    Pt,
    Pt0,
    Ptp,
    Ptm

include("molecule.jl")
using .MoleculeModule

export Molecule, read_molecule, Aiso, is_isotropic

include("system.jl")
using .SystemModule
export System, read_system

include("simparam.jl")
using .SimParamModule
export SimParams, read_simparams, StateType, Singlet, Triplet

include("hamiltonian.jl")
using .Hamiltonian

export system_hamiltonian,
    zeeman_hamiltonian,
    exchange_hamiltonian,
    dipolar_hamiltonian,
    haberkorn_hamiltonian,
    SchultenWolynes_hamiltonian

include("liouvillian.jl")
using .Liouvillian
export liouvillian, vectorise, linearise, normalise, trace

include("simulation.jl")
using .Simulation
export SW, SC

end
