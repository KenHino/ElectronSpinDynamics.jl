using ElectronSpinDynamics
using Test
using LinearAlgebra
using StaticArrays
using IniFile

# Import modules for testing
import ElectronSpinDynamics.Constants
import ElectronSpinDynamics.SpinOps

@testset "ElectronSpinDynamics.jl" begin
    @testset "Constants" begin
        # Test physical constants are available and positive
        @test ElectronSpinDynamics.ℏ > 0
        @test ElectronSpinDynamics.μ0 > 0
        @test ElectronSpinDynamics.g_electron ≈ 2.0 atol=0.1

        # Test gyromagnetic ratios are defined
        @test abs(ElectronSpinDynamics.γe) > 0
        @test ElectronSpinDynamics.γ1H > 0
        @test ElectronSpinDynamics.γ14N > 0
    end

    @testset "Spin Operators" begin
        # Test Pauli matrices
        σx = SpinOps.σx
        σy = SpinOps.σy
        σz = SpinOps.σz

        # Check dimensions
        @test size(σx) == (2, 2)
        @test size(σy) == (2, 2)
        @test size(σz) == (2, 2)

        # Check trace-free property
        @test tr(σx) ≈ 0 atol=1e-15
        @test tr(σy) ≈ 0 atol=1e-15
        @test tr(σz) ≈ 0 atol=1e-15

        # Check anticommutation relations {σᵢ, σⱼ} = 2δᵢⱼI
        I2 = Matrix{ComplexF64}(I, 2, 2)
        @test σx * σx ≈ I2
        @test σy * σy ≈ I2
        @test σz * σz ≈ I2
        @test σx * σy + σy * σx ≈ zeros(ComplexF64, 2, 2) atol=1e-15
        @test σy * σz + σz * σy ≈ zeros(ComplexF64, 2, 2) atol=1e-15
        @test σz * σx + σx * σz ≈ zeros(ComplexF64, 2, 2) atol=1e-15

        # Test spin operators (S = σ/2)
        Sx = SpinOps.Sx
        Sy = SpinOps.Sy
        Sz = SpinOps.Sz

        @test Sx ≈ σx / 2
        @test Sy ≈ σy / 2
        @test Sz ≈ σz / 2

        # Test ladder operators
        Sp = SpinOps.Sp
        Sm = SpinOps.Sm

        @test size(Sp) == (2, 2)
        @test size(Sm) == (2, 2)
        @test Sp ≈ (σx + im * σy) / 2 atol=1e-15
        @test Sm ≈ (σx - im * σy) / 2 atol=1e-15
    end

    @testset "Singlet-Triplet Basis" begin
        # Test ST transformation matrix
        ST = SpinOps.ST
        @test size(ST) == (4, 4)

        # Test that ST is unitary (ST * ST' = I)
        @test ST * transpose(ST) ≈ I atol=1e-15

        # Test ST_basis function
        test_matrix = Matrix{Float64}(I, 4, 4)
        transformed = SpinOps.ST_basis(test_matrix)
        @test size(transformed) == (4, 4)
    end

    @testset "Module Structure" begin
        # Test that modules are properly loaded
        @test isdefined(ElectronSpinDynamics, :Constants)
        @test isdefined(ElectronSpinDynamics, :SpinOps)

        # Test that constants are exported
        @test isdefined(ElectronSpinDynamics, :ℏ)
        @test isdefined(ElectronSpinDynamics, :μ0)
        @test isdefined(ElectronSpinDynamics, :g_electron)
    end

    @testset "Parser" begin
        cfg = read(Inifile(), "input.ini")   # Dict{String,Dict}
        # 1. Pick a section
        mol1 = read_molecule(cfg, "electron 1")
        mol2 = read_molecule(cfg, "electron 2")
        sys = read_system(cfg)
        simparams = read_simparams(cfg)

        H = system_hamiltonian(sys, 1.0, 0.0, 0.0)
        # H should be Hermitian + skew-Hermitian (Haberkorn term)
        # Therefore, H=H' except for diagonal elements
        @test H - diagm(0 => diag(H)) ≈ H' - diagm(0 => diag(H')) atol=1e-15

        L = liouvillian(H)
        @test L isa Matrix{ComplexF64}
    end
end
