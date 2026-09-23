@doc raw"""
        uweigen(M::AbstractMatrix{uwreal})

Compute eigenvalues and eigenvector of `M` and return them in a `LinearAlgebra.Eigen` structure. Eigenvalues and eigenvector are computed using `value.(M)` and by calling `LineraAlgebra`'s routines. The error are then propagated

In particular, the error on the eigenvalues are propagated by calling `ADerrors.root_error` with the employing the function
```math
       det(M-λI) = 0
```
and propagating the error from `M`.
The error on the eigenvector are propagated in a similar manner as for `ADerros.root_error`, by employing the fuction
```math
        (M-λ_k I)v_k =0
```
and the condition $|v_k|^2 = 1$.  to propagate the error from `M` and `λ`

Calling `LinearAlgebre.eigen` with an `AbstractMatrix{uwreal}` redirect to `uweigen`
"""
function uweigen(M::AbstractMatrix{uwreal})
    n = size(M,1)
    E = LinearAlgebra.eigen(value.(M))
    id = UWLinearAlgebra.identity(Float64, M)
    f(λ,d) = [LinearAlgebra.det(reshape(d,n,n)-_λ*id) for _λ in λ]
    λ = propagate_error(f,E.values,reshape(M,:))
    function g(v,d)
        res = similar(v)
        n = length(v)
        aux = reshape(d,n,n)*v
        res[1:n-1] = aux[1:n-1,:]
        res[n]= sum(_v^2 for _v in v) -1
        return res
    end
    vuw = similar(E.vectors,uwreal)
    for I = 1:n
        Cuw =  M .- λ[I].*id
        vuw[:,I] .= propagate_error(g,E.vectors[:,I],reshape(Cuw,:))
    end
    return LinearAlgebra.Eigen(λ,vuw)
end



@doc raw"""
        eigen(M::AbstractMatrix{uwreal})

Compute eigenvalues and eigenvector of `M` and return them in a `LinearAlgebra.Eigen` structure. Eigenvalues and eigenvector are computed using `value.(M)` and by calling `LineraAlgebra`'s routines. The error are then propagated

In particular, the error on the eigenvalues are propagated by calling `ADerrors.root_error` with the employing the function
```math
       det(M-λI) = 0
```
and propagating the error from `M`.
The error on the eigenvector are propagated in a similar manner as for `ADerros.root_error`, by employing the fuction
```math
        (M-λ_k I)v_k =0
```
and the condition $|v_k|^2 = 1$.  to propagate the error from `M` and `λ`

"""
LinearAlgebra.eigen(M::AbstractMatrix{uwreal}) = uweigen(M)
