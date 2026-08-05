"""
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
and propagating the error from `M` and `λ`

Calling `LinearAlgebrea.eigen` with an `AbstractMatrix{uwreal}` redirect to `uweigen`
"""
function uweigen(M::AbstractMatrix{uwreal})
    val = value.(M)
    E = LinearAlgebra.eigen(val)
    id = UWLinearAlgebra.identity(Float64, M)
    f(λ,d) = LinearAlgebra.det(reshape(d,n,n)-λ*id)
    λ = similar(E.values,uwreal)
    vuw = similar(E.vectors,uwreal)
    n = size(M,1)
    for (I,_l) = enumerate(E.values)
        λ[I] = root_error(f,E.values[I],reshape(M,:))
        Cuw =  M .- id .*λ[I]
        C= value.(Cuw)
        Cinv = LinearAlgebra.pinv(C)
        v = E.vectors[:,I]
        @inbounds for i in 1:n
            der = zeros(size(C))
            @inbounds for j in 1:n, k in 1:n
                der[j,k] = -Cinv[i,j]*v[k]
            end
            vuw[i,I] = addobs(reshape(Cuw,length(Cuw)),
                              reshape(der,length(der)),v[i])
        end
    end
    return LinearAlgebra.Eigen(λ,vuw)
end

LinearAlgebra.eigen(M::AbstractMatrix{uwreal}) = uweigen(M)
