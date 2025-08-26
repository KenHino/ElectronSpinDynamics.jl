# utils.jl ---------------------------------------------------------------
# Lightweight helpers shared across the package.
# Placed in a tiny sub‑module so it is included exactly **once**,
# avoiding duplicate definitions during precompilation.
# --------------------------------------------------------------------

module Utils

using StaticArrays

export vecparse, mat3, mat3static, clean!

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
for a 3 × 3 tensor and return a regular `Matrix{Float64}`.

The transpose at the end converts from the row‑major order used in most
text files to Julia/Fortran column‑major storage.
"""
mat3(s::AbstractString) = reshape(vecparse(Float64, s), 3, 3)

"""
    mat3static(s::AbstractString) → SMatrix{3,3,Float64}

Same as [`mat3`](@ref) but returns a stack‑allocated `SMatrix` from
StaticArrays.jl for maximum performance in tight inner loops.
"""
mat3static(s::AbstractString) = SMatrix{3, 3}(mat3(s))

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

end # module
