module MoleculeModule

using IniFile
using ..Utils
using LinearAlgebra: tr, diagm

"""Typed container for one electron subsystem."""

struct Molecule{T <: Integer}
    """
    g::Float64                 # Landé g-factor (current implementation assumes g = 2.002_319_304_362_56 in any case)
    I::Vector{T}               # nuclear spins (any integer type)
    A::Array{Float64, 3}       # coupling tensors  (size    N × 3 × 3)
    mults::Vector{Int}         # multiplicities
    """
    g::Float64                 # Landé g-factor
    I::Vector{T}               # nuclear spins
    A::Array{Float64, 3}       # coupling tensors  (size    N × 3 × 3)
end

function Aiso(A::Array{Float64, 3})::Vector{Float64}
    return [tr(A[k, :, :]) / 3 for k in 1:size(A, 1)]
end

function Aiso(A::Array{Float64, 2})::Float64
    return tr(A) / 3.0
end

function Aiso(molecule::Molecule)::Vector{Float64}
    return Aiso(molecule.A)
end

function Aiso(molecule::Molecule, k::Int)::Float64
    return Aiso(molecule.A[k, :, :])
end

function is_isotropic(A::Array{Float64, 2})::Bool
    Aiso_val = Aiso(A)
    return A ≈ diagm(0 => [Aiso_val, Aiso_val, Aiso_val])
end

function is_isotropic(A::Array{Float64, 3})::Bool
    return all(is_isotropic(A[k, :, :]) for k in 1:size(A, 1))
end

function is_isotropic(molecule::Molecule)::Bool
    return is_isotropic(molecule.A)
end

# ---------------------------------------------------------------------
# main API
# ---------------------------------------------------------------------
function read_molecule(cfg::IniFile.Inifile, section::AbstractString)::Molecule
    # ---- scalars & short vectors ------------------------------------
    g  = parse(Float64, get(cfg, section, "g", "NaN"))
    NI = vecparse(UInt, get(cfg, section, "N_I", ""))
    @assert allequal(NI, 1)
    I = vecparse(Int, get(cfg, section, "I", ""))  # Use Int as default integer type

    N = length(NI)                                 # number of nuclei
    A = Array{Float64, 3}(undef, N, 3, 3)

    for k in 1:N
        key = "A$(k)"
        if haskey(cfg, section, key)
            A[k, :, :] = mat3(get(cfg, section, key, ""))
        else
            A[k, :, :] .= 0.0                      # absent tensor → zeros
        end
    end

    return Molecule(g, I, A)
end

export Molecule, read_molecule, Aiso, is_isotropic

end # module
