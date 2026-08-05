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
