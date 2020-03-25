module TestInverse
using LinearAlgebra
using LazyInverse: inverse, Inverse, pseudoinverse, PseudoInverse
using Test

@testset "inverse" begin
    A = randn(3, 3)
    A = A'A
    Inv = inverse(A)
    @test A*Inv ≈ I(3)

    # determinant
    @test det(Inv) ≈ 1/det(A)
    @test logdet(Inv) ≈ -logdet(A)
    @test all(logabsdet(Inv) .≈ (-1, 1) .* logabsdet(A))
end

@testset "pseudoinverse" begin
    A = randn(3, 2)
    LInv = pseudoinverse(A)
    @test LInv*A ≈ I(2)
    @test Matrix(LInv)*A ≈ I(2)

    A = randn(2, 3)
    RInv = pseudoinverse(A, Val(:R))
    @test A*RInv ≈ I(2)
    @test A*Matrix(RInv) ≈ I(2)
end

end # TestInverse
