module TestInverse
using LinearAlgebra
using LazyInverse: inverse, Inverse, pseudoinverse, PseudoInverse
using Test

@testset "inverse" begin
    n = 3
    A = randn(n, n)
    A = A'A
    Inv = inverse(A)
    @test A*Inv ≈ I(n)

    # determinant
    @test det(Inv) ≈ 1/det(A)
    @test logdet(Inv) ≈ -logdet(A)
    @test all(logabsdet(Inv) .≈ (-1, 1) .* logabsdet(A))

    # factorize
    @test factorize(Inv) ≡ Inv # no-op
    @test isposdef(Inv)

    # inv
    @test inv(Inv) isa AbstractMatrix
    @test inv(Inv) ≈ A
    D = Diagonal(randn(n))
    @test inv(Inverse(D)) isa AbstractMatrix
    @test inv(Inverse(D)) ≈ D
    @test inv(Inverse(Inv)) isa AbstractMatrix
    @test inv(Inverse(Inv)) ≈ inv(A)

    @test AbstractMatrix(Inv) ≈ inv(A)
    @test AbstractMatrix(Inverse(D)) isa Diagonal
    @test Matrix(Inverse(D)) isa Matrix

    x = randn((1, 1))
    @test inverse(x) isa Real
    @test inverse(x) ≈ inv(x)[1]
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

    # factorize
    @test factorize(LInv) ≡ LInv
end

@testset "cholesky" begin
    n = 1024
    x = randn(n)
    y = randn(n)
    A = randn(n, n)
    A = A'A
    C = cholesky(A)
    D = Inverse(C)

    @test dot(x, D, x) ≈ dot(x, C\x)
    @test dot(x, D, y) ≈ dot(x, C\y)

    # @time dot(x, D, x)
    # @time dot(x, C\x)
    # @time dot(x, D, y)
    # @time dot(x, C\y)

    # X = randn(n, n)
    # Y = randn(n, n)
    # @test *(X, D, X) ≈ *(X, C\X)
    # @test *(X, D, Y) ≈ *(X, C\Y)

    # @time *(X, D, X)
    # @time *(X, C\X)
    # @time *(X, D, Y)
    # @time *(X, C\Y)
end

end # TestInverse
