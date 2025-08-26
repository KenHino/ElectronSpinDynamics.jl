module MoleculeModule

using IniFile
using ..Utils

"""Typed container for one electron subsystem."""

struct Molecule
    g::Float64                 # Landé g-factor
    I::Vector{UInt8}             # nuclear spins
    A::Array{Float64, 3} # coupling tensors  (size    N × 3 × 3)
end

# ---------------------------------------------------------------------
# main API
# ---------------------------------------------------------------------
function read_molecule(cfg::IniFile.Inifile, section::AbstractString)
    # ---- scalars & short vectors ------------------------------------
    g  = parse(Float64, get(cfg, section, "g", "NaN"))
    NI = vecparse(UInt, get(cfg, section, "N_I", ""))
    @assert allequal(NI, 1)
    I = vecparse(UInt8, get(cfg, section, "I", ""))

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

export Molecule, read_molecule

end # module
