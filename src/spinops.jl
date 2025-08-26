"""
Spin operators module
"""
module SpinOps

using ..Utils: clean!
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

function ST_basis(M::AbstractMatrix)
    @assert size(M) == (4, 4) "Matrix must be 4×4"
    M = ST * M * transpose(ST)
    clean!(M)
    return M
end

Sx1 = ST_basis(kron(Sx, I(2)))
Sx2 = ST_basis(kron(I(2), Sx))
Sy1 = ST_basis(kron(Sy, I(2)))
Sy2 = ST_basis(kron(I(2), Sy))
Sz1 = ST_basis(kron(Sz, I(2)))
Sz2 = ST_basis(kron(I(2), Sz))

Ps = diagm(0 => [0.0, 0.0, 1.0, 0.0]) # projection operator for singlet state
Pt = diagm(0 => [1/3, 1/3, 0.0, 1/3]) # projection operator for triplet state
Pt0 = diagm(0 => [0.0, 1.0, 0.0, 0.0]) # projection operator for T0 state
Ptp = diagm(0 => [1.0, 0.0, 0.0, 0.0]) # projection operator for T+ state
Ptm = diagm(0 => [0.0, 0.0, 0.0, 1.0]) # projection operator for T- state

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
    Ps,
    Pt,
    Pt0,
    Ptp,
    Ptm

end # module
