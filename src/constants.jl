# constants.jl
# This submodule collects physical constants commonly used in spin‑dynamics
# simulations. Values are CODATA 2022 unless otherwise noted.

# We employs
# time: ns
# energy: mT (x γe)

module Constants

# Fundamental constants (SI units)
const ℏ::Float64 = 1.054_571_817e-34      # Reduced Planck constant [J·s]
const μ0::Float64 = 4π * 1e-7               # Vacuum permeability     [N·A⁻²]
const μB::Float64 = 9.274_010_0657e-24       # Bohr magneton           [J·T⁻¹]

# Gyromagnetic ratio
const γe::Float64 = -176.085_962_784 * 1e-03     # ns-1 mT-1
const γ1H::Float64 = 0.267_522_187_08 * 1e-03   # ns-1 mT-1
const γ14N::Float64 = 0.019_329_779_2 * 1e-03    # ns-1 mT-1

# g‑factors
const g_electron::Float64 = 2.002_319_304_362_56

@assert γe / 1e-9 / 1e-3 ≈ - g_electron * μB / ℏ "γe is not equal to -g_electron * μB / ℏ"

export ℏ, μ0, μB, γe, γ1H, γ14N, g_electron

end # module
