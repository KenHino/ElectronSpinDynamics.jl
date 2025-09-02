# utils.jl ---------------------------------------------------------------
# Lightweight helpers shared across the package.
# Placed in a tiny sub‑module so it is included exactly **once**,
# avoiding duplicate definitions during precompilation.
# --------------------------------------------------------------------

module Utils

using StaticArrays
using Distributions: Uniform

export vecparse, mat3, mat3static, clean!, clean, sample_from_sphere, sphere_to_cartesian

"""
    vecparse(T, s::AbstractString) → Vector{T}

Split the whitespace‑delimited string `s` and `parse` each token as the
requested numeric type `T` (e.g. `Float64`, `Int`).

```julia
vecparse(Int, "1 2 3")  # == [1, 2, 3]
```
"""
vecparse(::Type{T}, s::AbstractString) where {T} = parse.(T, split(strip(s)))

"""
    mat3(s::AbstractString) → Matrix{Float64}

Interpret the nine whitespace‑separated numbers in `s` as *row‑major* data
for a 3 × 3 tensor and return a regular `Matrix{Float64}`.

The transpose at the end converts from the row‑major order used in most
text files to Julia/Fortran column‑major storage.
"""
mat3(s::AbstractString)::Matrix{Float64} = reshape(vecparse(Float64, s), 3, 3)

"""
    mat3static(s::AbstractString) → SMatrix{3,3,Float64}

Same as [`mat3`](@ref) but returns a stack‑allocated `SMatrix` from
StaticArrays.jl for maximum performance in tight inner loops.
"""
mat3static(s::AbstractString)::SMatrix{3,3,Float64} = SMatrix{3,3}(mat3(s))

"""
    clean!(M::AbstractMatrix)

Set all elements of `M` to zero if they are less than 1e-15.
"""
function clean!(M::AbstractMatrix)
  for i in axes(M, 1), j in axes(M, 2)
    if abs(M[i, j]) < 1e-15
      M[i, j] = 0.0
    elseif abs(M[i, j] - 0.5) < 1e-15
      M[i, j] = 0.5
    elseif abs(M[i, j] + 0.5) < 1e-15
      M[i, j] = -0.5
    elseif abs(M[i, j] - √2/4) < 1e-15
      M[i, j] = √2/4
    elseif abs(M[i, j] + √2/4) < 1e-15
      M[i, j] = -√2/4
    elseif abs(M[i, j] - √2/4im) < 1e-15
      M[i, j] = √2/4im
    elseif abs(M[i, j] + √2/4im) < 1e-15
      M[i, j] = -√2/4im
    end
  end
end

function clean(M::SMatrix{N,N,T})::SMatrix{N,N,T} where {N,T}
  M_mut = Matrix(M)
  clean!(M_mut)
  return SMatrix{N,N,T}(M_mut)
end

function sample_from_sphere(shape)::Tuple{Array{Float64,2},Array{Float64,2}}
  """
  Since the volume element of a sphere is dΩ = sin(θ) dθ dϕ,
  uniformly sampled points on a sphere are given by
  θ = acos(2x - 1), ϕ = 2πy, where x and y are uniformly sampled from [0, 1].
  """
  ϕ = rand(Uniform(0, 2π), shape)
  θ = acos.(rand(Uniform(-1, 1), shape))
  @assert all(0 .≤ θ .≤ π) "θ must be in [0, π]"
  @assert all(0 .≤ ϕ .≤ 2π) "ϕ must be in [0, 2π]"
  return θ, ϕ
end

function sphere_to_cartesian(
  θ::Array{Float64,T}, ϕ::Array{Float64,T}
)::Tuple{Array{Float64,T},Array{Float64,T},Array{Float64,T}} where {T}
  x = sin.(θ) .* cos.(ϕ)
  y = sin.(θ) .* sin.(ϕ)
  z = cos.(θ)
  return x, y, z
end

end # module
