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
function uwcholesky(M::AbstractMatrix{uwreal})
    ndata = size(M,1)
    C = LinearAlgebra.cholesky(value.(M))
    n = size(M, 1)
    idx = [(i,j) for j in 1:n for i in 1:j]
    D = [M[i,j] for (i,j) in idx]
    U = [C.U[i,j] for (i,j) in idx]
    function F(x,d, n=n, idx = idx)
        D = symmetric_from_upper(d,n)
        U = triangular_from_upper(x,n)
        res = D - U'*U
        return [res[i,j] for (i,j) in idx]
    end
    res = triangular_from_upper(propagate_error(F,U,D),3)
    return LinearAlgebra.Cholesky(res)
end

LinearAlgebra.cholesky(M::AbstractMatrix{uwreal}) =
    uwcholesky(M)

LinearAlgebra.symmetric_type(::Type{uwreal}) = uwreal
LinearAlgebra.symmetric(uw::uwreal,::Symbol) = uw
export uwcholesky
