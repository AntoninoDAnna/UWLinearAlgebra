function symmetric_from_upper(d, n)
    D = zeros(eltype(d), n, n)
    k = 1
    for j in 1:n
        for i in 1:j
            D[i,j] = d[k]
            D[j,i] = d[k]
            k += 1
        end
    end
    return D
end

function triangular_from_upper(d, n)
    D = zeros(eltype(d), n, n)
    k = 1
    for j in 1:n
        for i in 1:j
            D[i,j] = d[k]
            k += 1
        end
    end
    return LinearAlgebra.UpperTriangular(D)
end

@inline packed_index(i, j) = j * (j - 1) ÷ 2 + i

function cholesky_F!(f, x, d, n)
    k = 1
    @inbounds for j in 1:n
        for i in 1:j
            s = zero(eltype(x))
            for r in 1:i
                s += x[packed_index(r, i)] *
                     x[packed_index(r, j)]
            end
            f[k] = d[k] - s
            k += 1
        end
    end

    return nothing
end

@doc raw"""
     uwcholesky(M::AbstractMatrix{uwreal})

Computes the Cholesky decomposition of a Symmetric matrix.
`M` must built in such a way that `M[i,j] === M[j,i]` for all `(i,j)`, that is the underlying data must be the same!
It return a `LinearAlgebra.Cholesky` containing the lower triangular matrix `L` and the upper triangular matrix `U` such that
```math
        M ≈ L L^T ≈ U^T U
```

The code uses `LinearAlgebra.cholesky` on the central values of `M`, and then propagates the error on `U`. Using the function

```math
        F(M) = M - U^T(M) U(M) = 0
```
Calling `LinearAlgebra.cholesky` with a `Matrix{uwreal}` redirects to `uweigen`

"""
function uwcholesky(M::AbstractMatrix{uwreal};kw...)
    ndata = size(M,1)
    C = LinearAlgebra.cholesky(value.(M);kw...)
    n = size(M, 1)
    idx = [(i,j) for j in 1:n for i in 1:j]
    D = [M[i,j] for (i,j) in idx]
    U = [C.U[i,j] for (i,j) in idx]
    F!(y, x, d) = cholesky_F!(y, x, d, n)
    res = propagate_error!(F!,U,D)
    return LinearAlgebra.Cholesky(triangular_from_upper(res,n))
end

LinearAlgebra.cholesky(M::AbstractMatrix{uwreal};kw...) =
    uwcholesky(M;kw...)

LinearAlgebra.symmetric_type(::Type{uwreal}) = uwreal
LinearAlgebra.symmetric(uw::uwreal,::Symbol) = uw
export uwcholesky
