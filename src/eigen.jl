using CatViews
@doc raw"""
        uweigen(M::AbstractMatrix{uwreal};kw...)

Compute the eigenvalue decomposition of `M`, propagating the uncertainty. It first computes eigenvalues and eigenvectors by calling  `LinearAlgebra.eigen(ADerrors.value.(M);kw...)`. Then it propagates the uncertainties.

## Error Propagation
The errors on the eigenvalues are propagated by calling `UWLinearAlgebra.propagate_error` with the function

```math
       det(M-λI) = 0
```

The error on the eigenvector are propagated in a similarly way, using the fuction
```math
        (M-λ_k I)v_k =0
```
and the condition $|v_k|^2 = 1$  to propagate the error from `M` and `λ`.

Calling `LinearAlgebre.eigen` with an `AbstractMatrix{uwreal}` redirect to `uweigen`

"""
function uweigen(M::AbstractMatrix{uwreal};kw...)
    n = size(M,1)
    E = LinearAlgebra.eigen(value.(M);kw...)
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
    uweigvals(M::AbstractMatrix{uwreal};kw...)

Computes the eigenvalues of `M` and propagates the uncertainty. It first computes the eigenvalues by calling `LinearAlgebra.eigvals(value.(M);kw...)`. Then it propagates the uncertainties.

## Error Propagation
The errors on the eigenvalues are propagated by calling `UWLinearAlgebra.propagate_error` with the function

```math
       det(M-λI) = 0
```

Calling `LinearAlgebra.eigval` with an `Abstractmatrix{uwreal}` redirects to `uweigvals`.
"""
function uweigvals(M::AbstractMatrix{uwreal};kw...)
    E = LinearAlgebra.eigvals(value.(M);kw...)
    id = UWLinearAlgebra.identity(Float64, M)
    f(λ,d) = [LinearAlgebra.det(reshape(d,size(M))-_λ*id) for _λ in λ]
    λ = propagate_error(f,E,reshape(M,:))
    return λ
end


@doc raw"""
        uweigen(A::AbstractMatrix{uwreal},B::AbstractMatrix{uwreal};kw...)

Compute the generalized eigenvalue decomposition of `A` and `B` and propagates the uncertainties. It first computes the generalized eigenvalues and eigenvectors by calling `LinearAlgebra.eigen(ADerrors.value.(A),ADerrors.value.(B);kw...)`. Then it propagates the uncertainties.

##  Error propagation
The uncertainties are propagated first on the eigenvalues by calling `UWLinearAlgebra.propagate_error` with the function
```math
       det(A-λB) = 0
```
then on the eigenvectors, using  the fuction
```math
        (A-λ_k B)v_k =0
```
and the condition $|v_k|^2 = 1$  to propagate the uncertanties from `M` and `λ`

Calling `LinearAlgebra.eigen` with two `AbstractMatrix{uwreal}` redirect to `uweigen`
"""
function uweigen(A::AbstractMatrix{uwreal},B::AbstractMatrix{uwreal};kw...)
    n = size(A,1)
    E = LinearAlgebra.eigen(value.(A),value.(B);kw...)
    function f(λ::AbstractVector,d::AbstractVector; n=n)
        n2 = n*n
        A = reshape(@view(d[1:n2]), n,n)
        B = reshape(@view(d[n2+1:end]),n,n)
        return [LinearAlgebra.det(A-_λ*B) for _λ in λ]
    end
    D = CatView(reshape(A,:),reshape(B,:))
    λ = propagate_error(f,E.values,D)
    function g(v::AbstractVector,d::AbstractVector,n=n)
        res = similar(v)
        aux = reshape(d,n,n)*v
        res[1:n-1] = aux[1:n-1,:]
        res[n]= sum(_v^2 for _v in v) -1
        return res
    end
    vuw = similar(E.vectors,uwreal)
    for I = 1:n
        Cuw =  A .- λ[I].*B
        vuw[:,I] .= propagate_error(g,E.vectors[:,I],reshape(Cuw,:))
    end
    return LinearAlgebra.GeneralizedEigen(λ,vuw)
end

@doc raw"""
    uweigvals(A::AbstractMatrix{uwreal},B::AbstractMatrix{uwreal};kw...)

Computes the generalized eigenvalues of `A` and `B`, and propagates the uncertainty. It first computes the eigenvalues by calling `LinearAlgebra.eigvals(value.(M);kw...)`. Then it propagates the uncertainties.

## Error Propagation
The errors on the eigenvalues are propagated by calling `UWLinearAlgebra.propagate_error` with the function

```math
       det(A-λB) = 0
```

Calling `LinearAlgebra.eigval` with two `Abstractmatrix{uwreal}` redirects to `uweigvals`.
"""
function uweigvals(A::AbstractMatrix{uwreal},B::AbstractMatrix{uwreal};kw...)
    E = LinearAlgebra.eigvals(value.(A),value.(B);kw...)
    function f(λ::AbstractVector,d::AbstractVector; n=size(A,1))
        n2 = n*n
        A = reshape(@view(d[1:n2]), n,n)
        B = reshape(@view(d[n2+1:end]),n,n)
        return [LinearAlgebra.det(A-_λ*B) for _λ in λ]
    end
    D = CatView(reshape(A,:),reshape(B,:))
    return propagate_error(f,E.values,D)
end




@doc raw"""
        eigen(M::AbstractMatrix{uwreal};kw...)

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
LinearAlgebra.eigen(M::AbstractMatrix{uwreal};kw...) = uweigen(M;kw...)

@doc raw"""
    eigvals(M::AbstractMatrix{uwreal}; kw...)

Computes the eigenvalues of `M` and propagate the uncertainty`
"""
LinearAlgebra.eigvals(M::AbstractMatrix{uwreal};kw...) = uweigvals(M;kw...)
