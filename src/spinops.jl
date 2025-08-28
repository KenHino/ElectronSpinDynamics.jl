"""
Spin operators module
"""
module SpinOps

using ..Utils: clean!, clean
using LinearAlgebra: I, kron, diagm
using StaticArrays: SMatrix

σx = SMatrix{2, 2}([0.0 1.0; 1.0 0.0])
σy = SMatrix{2, 2}([0.0 -im; im 0.0])
σz = SMatrix{2, 2}([1.0 0.0; 0.0 -1.0])

Sx = σx / 2.0
Sy = σy / 2.0
Sz = σz / 2.0

Sp = SMatrix{2, 2}([0.0 1.0; 0.0 0.0])
Sm = SMatrix{2, 2}([0.0 0.0; 1.0 0.0])

ST = SMatrix{4, 4}([
    1  0    0    0;
     0  1/√2 1/√2 0;
     0 -1/√2 1/√2 0;
     0  0    0    1
])

function ST_basis(M::AbstractMatrix)::AbstractMatrix
    @assert size(M) == (4, 4) "Matrix must be 4×4"
    result = ST * M * transpose(ST)
    if result isa SMatrix
        return clean(result)
    else
        clean!(result)
        return result
    end
end

Sx1 = SMatrix{4, 4}(ST_basis(kron(Sx, I(2))))
Sx2 = SMatrix{4, 4}(ST_basis(kron(I(2), Sx)))
Sy1 = SMatrix{4, 4}(ST_basis(kron(Sy, I(2))))
Sy2 = SMatrix{4, 4}(ST_basis(kron(I(2), Sy)))
Sz1 = SMatrix{4, 4}(ST_basis(kron(Sz, I(2))))
Sz2 = SMatrix{4, 4}(ST_basis(kron(I(2), Sz)))

Ps = SMatrix{4, 4}(diagm(0 => [0.0, 0.0, 1.0, 0.0])) # projection operator for singlet state
Pt = SMatrix{4, 4}(diagm(0 => [1/3, 1/3, 0.0, 1/3])) # projection operator for triplet state
Pt0 = SMatrix{4, 4}(diagm(0 => [0.0, 1.0, 0.0, 0.0])) # projection operator for T0 state
Ptp = SMatrix{4, 4}(diagm(0 => [1.0, 0.0, 0.0, 0.0])) # projection operator for T+ state
Ptm = SMatrix{4, 4}(diagm(0 => [0.0, 0.0, 0.0, 1.0])) # projection operator for T- state

# same as ST_basis(kron(Sx, Sx) + kron(Sy, Sy) + kron(Sz, Sz))
# Since Ps = 1/4 - S1S2, we have
S1S2 = SMatrix{4, 4}(diagm(0=>[1/4, 1/4, -3/4, 1/4]))

@assert S1S2 ≈ ST_basis(kron(Sx, Sx) .+ kron(Sy, Sy) .+ kron(Sz, Sz))  # Assertion commented out due to numerical precision issues

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

end # module
