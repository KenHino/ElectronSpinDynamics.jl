module ElectronSpinDynamics

# Write your package code here.
include("constants.jl")
using .Constants      # brings ℏ, μB, … into scope

export ℏ, μ0, γe, γ1H, γ14N, g_electron

include("utils.jl")
using .Utils


include("molecule.jl")

export Molecule, read_molecule

include("system.jl")

export System, read_system

include("simparam.jl")

export SimParams, read_simparams

end
