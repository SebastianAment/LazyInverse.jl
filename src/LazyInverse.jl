module LazyInverse
using LinearAlgebra
const AbstractMatOrFac{T} = Union{AbstractMatrix{T}, Factorization{T}}
# TODO: export inverse, pinverse, pseudoinverse?
# TODO: extend Zygote's logdet adjoint with a lazy inverse ...
# TODO: create custom printing, so that it doesn't fail in the repl

# converts multiplication into a backsolve and vice versa
# applications:
# - WoodburyIdentity
# -  x' A^{-1} y = *(x, Inverse(A), y) can be 2x for cholesky
# - Zygote logdet adjoint
# - Laplace's approximation
############################ Lazy Inverse Matrix ###############################
struct Inverse{T, M} <: Factorization{T} # <: AbstractMatrix{T} ?
    parent::M
    function Inverse(A::M) where {T, M<:AbstractMatOrFac{T}}
        size(A, 1) == size(A, 2) ? new{T, M}(A) : error("Input of size $(size(A)) not square")
    end
end
Base.size(L::Inverse) = size(L.parent)
Base.size(L::Inverse, dim::Integer) = size(L.parent, dim)

# Base.show(io::IO, Inv::Inverse) = (println(io, "Inverse of "); show(io, Inv.parent))
function inverse end # smart pseudo-constructor
inverse(Inv::Inverse) = Inv.parent
LinearAlgebra.inv(Inv::Inverse) = inverse(Inv)
inverse(x::Union{Number, Diagonal, UniformScaling}) = inv(x)
inverse(A::AbstractMatOrFac) = Inverse(A)
Base.AbstractMatrix(Inv::Inverse) = AbstractMatrix(inv(Inv.parent))
Base.Matrix(Inv::Inverse) = Matrix(inv(Inv.parent))

# factorize the underlying matrix
import LinearAlgebra: factorize, det, logdet, logabsdet, dot
# factorize is used to compute a type which makes it easy to apply the inverse
# therefore, it should be a no-op on Inverse
factorize(Inv::Inverse) = Inv
det(Inv::Inverse) = 1/det(Inv.parent)
logdet(Inv::Inverse) = -logdet(Inv.parent)
function logabsdet(Inv::Inverse)
    l, s = logabsdet(Inv.parent)
    (-l, s)
end

# TODO: allows for stochastic approximation:
# A Probing Method for Cοmputing the Diagonal of the Matrix Inverse
import LinearAlgebra: diag
diag(Inv::Inverse) = diag(Matrix(Inv))
diag(Inv::Inverse{<:Any, <:Factorization}) = diag(inv(Inv.parent))

# TODO: specialize
# mul!(Y, A, B, α, β)
# ldiv!
# rdiv!
import LinearAlgebra: adjoint, transpose, ishermitian, issymmetric
adjoint(Inv::Inverse) = Inverse(adjoint(Inv.parent))
tranpose(Inv::Inverse) = Inverse(tranpose(Inv.parent))
ishermitian(Inv::Inverse) = ishermitian(Inv.parent)
issymmetric(Inv::Inverse) = issymmetric(Inv.parent)
symmetric(Inv::Inverse) = Inverse(Symmetric(Inv.parent))

# TODO: should override factorizations' get factors method instead
import LinearAlgebra: UpperTriangular, LowerTriangular
UpperTriangular(U::Inverse{T, <:UpperTriangular}) where {T} = U
LowerTriangular(L::Inverse{T, <:LowerTriangular}) where {T} = L

# TODO: have to check if these are correct of uplo = L
# inverse(C::Cholesky) = Cholesky(inverse(C.U), C.uplo, C.info)
# inverse(C::CholeskyPivoted) = CholeskyPivoted(inverse(C.U), C.uplo, C.piv, C.rank, C.tol, C.info)

# const Chol = Union{Cholesky, CholeskyPivoted}
# # this should be faster if C is low rank
# *(C::Chol, B::AbstractVector) = C.L * (C.U * B)
# *(C::Chol, B::AbstractMatrix) = C.L * (C.U * B)
# *(B::AbstractVector, C::Chol) = (B * C.L) * C.U
# *(B::AbstractMatrix, C::Chol) = (B * C.L) * C.U

# this implements the right pseudoinverse
# is defined if A has linearly independent columns
# ⁻¹, ⁺ syntax
struct PseudoInverse{T, M<:AbstractMatOrFac{T}} <: Factorization{T}
    parent::M
end
const AbstractInverse{T} = Union{Inverse{T}, PseudoInverse{T}}

Base.size(P::PseudoInverse) = size(P.parent')
Base.size(P::PseudoInverse, k::Integer) = size(P.parent', k::Integer)

function Base.Matrix(P::PseudoInverse)
    A = Matrix(P.parent)
    inverse(A'A)*A' # left pseudo inverse #P.parent'inverse(P.parent*P.parent') # right pseudo inverse
end
# Base.Matrix(P::PseudoInverse) = AbstractMatrix(P)
Base.Matrix(A::Adjoint{<:Number, <:PseudoInverse}) = Matrix(A.parent)'
LinearAlgebra.factorize(P::PseudoInverse) = P # same reasoning as for Inverse

# smart constructor
# TODO: have to figure out how to make right inverse work correctly
# calls regular inverse if matrix is square
function pseudoinverse end
const pinverse = pseudoinverse
function pseudoinverse(A::AbstractMatOrFac, side::Union{Val{:L}, Val{:R}} = Val(:L))
    if size(A, 1) == size(A, 2)
        return inverse(A)
    elseif side isa Val{:L}
        size(A, 1) > size(A, 2) || error("A does not have linearly independent columns")
        return PseudoInverse(A) # left pinv
    else
        size(A, 2) > size(A, 1) || error("A does not have linearly independent rows")
        return PseudoInverse(A')' # right pinv
    end
end
# pseudoinverse(A, side::Val{:R}) = pseudoinverse(A')' # cleaner but more confusing error

pseudoinverse(A::Union{Number, Diagonal, UniformScaling}) = inv(A)
pseudoinverse(P::AbstractInverse) = P.parent

LinearAlgebra.adjoint(P::PseudoInverse) = Adjoint(P)
# LinearAlgebra.transpose(P::PseudoInverse) = Transpose(P)
import LinearAlgebra: *, /, \
*(L::AbstractInverse, B::AbstractVector) = L.parent \ B
*(L::AbstractInverse, B::AbstractMatrix) = L.parent \ B
*(L::AbstractInverse, B::Factorization) = L.parent \ B

# since left pseudoinverse behaves differently for right multiplication
*(B::AbstractVector, P::PseudoInverse) = B * Matrix(P) #(A = L.parent; (B * inverse(A'A)) * A')
*(B::AbstractMatrix, P::PseudoInverse) = B * Matrix(P) #(A = L.parent; (B * inverse(A'A)) * A')
*(B::Factorization, P::PseudoInverse) = B * Matrix(P)

*(B::AbstractVector, L::Inverse) = B / L.parent
*(B::AbstractMatrix, L::Inverse) = B / L.parent
*(B::Factorization, L::Inverse) = B / L.parent

\(L::AbstractInverse, B::AbstractVector) = L.parent * B
\(L::AbstractInverse, B::AbstractMatrix) = L.parent * B
\(L::AbstractInverse, B::Factorization) = L.parent * B

/(B::AbstractVector, L::AbstractInverse) = B * L.parent
/(B::AbstractMatrix, L::AbstractInverse) = B * L.parent
/(B::Factorization, L::AbstractInverse) = B * L.parent

# Adjoints of pseudo-inverses
*(B::AbstractVector, L::Adjoint{<:Real, <:PseudoInverse}) = (L'*B')'
*(B::AbstractMatrix, L::Adjoint{<:Real, <:PseudoInverse}) = (L'*B')'
*(B::Factorization, L::Adjoint{<:Real, <:PseudoInverse}) = (L'*B')'

############################ ternary dot products ##############################
dot(x::AbstractVecOrMat, A::Inverse, y::AbstractVecOrMat) = dot(x, A*y)

function dot(x::AbstractVector, A::Inverse{<:Any, <:Cholesky}, y::AbstractVector)
	if x ≡ y
		L = A.parent.U' # since getting L causes allocations
		Ly = L\y
		dot(Ly, Ly)
	else # if x != y, this is more efficient because it avoid a temporary
		C = A.parent
		dot(x, C\y)
	end
end

# TODO: could have non-allocating mul! for this
# advantage seems to be less pronounced than for dot
# function *(x, A::Inverse{<:Any, <:Cholesky}, y)
# 	# println("hi")
# 	if x ≡ y'
# 		L = A.parent.U'
# 		Ly = L\y
# 		*(Ly', Ly)
# 	else
# 		C = A.parent
# 		*(x, C\y)
# 	end
# end

end # LazyInverse

# import Base: +, -
# +(A::AbstractInverse, B::AbstractMatrix) = AbstractMatrix(A) + B
# +(A::AbstractMatrix, B::AbstractInverse) = A + AbstractMatrix(B)
# -(A::AbstractInverse, B::AbstractMatrix) = AbstractMatrix(A) - B
# -(A::AbstractMatrix, B::AbstractInverse) = A - AbstractMatrix(B)
# +(A::AbstractInverse, B::AbstractInverse) = AbstractMatrix(A) + AbstractMatrix(B)
# -(A::AbstractInverse, B::AbstractInverse) = AbstractMatrix(A) - AbstractMatrix(B)

# Iterative method to compute pseudo inverse:
# https://en.wikipedia.org/wiki/Moore–Penrose_inverse#Updating_the_pseudoinverse
# function iterativepinv(A)
#     # B # has to be chosen so that AB = BA'
#     for i in 1:iter
#         B .= 2B - B*A*B
#     end
#     return B
# end
