function get_E(i,j; n)
    res = zeros(n,n)
    res[i,j]=1;
    res[j,i]=1;
    res
end

function phi!(res::Matrix{T},M::Matrix{T},
              n::Int64 =size(M,1)) where T
    res .= zero(T)
    @inbounds for i in 1:n
        res[i,i] = 0.5*M[i,i]
        for j in 1:i-1
            res[i,j] = M[i,j]
        end
    end
    return res
end

lowertriangular(M) = [M[i,j] for i in axes(M,1) for j in 1:i]


function uwcholesky(M::AbstractMatrix{uwreal})
    val = value.(M)
    ndata = size(M,1)
    C = LinearAlgebra.cholesky(val)
    L = C.L
    res = similar(L, uwreal)
    aux1 = zeros(ndata,ndata)
    Jall = Vector{Matrix{Float64}}()
    for i in 1:ndata, j in 1:i
        Eij = get_E(i,j,n=ndata)
        J = L \ Eij / L'
        phi!(aux1,J)
        J  = L*aux1
        push!(Jall,copy(J))
    end
    D = lowertriangular(M)
    for i in 1:ndata, j in 1:i
        der = [_j[i,j] for _j in Jall]
        res[i,j] = addobs(D,der,L[i,j])
    end
    return LinearAlgebra.Cholesky(res)
end

LinearAlgebra.cholesky(M::AbstractMatrix{uwreal}) =
    uwcholesky(M)


export uwcholesky


# (d11m11 + d12m21 +d13m31) (d11m12 + d12m22 + d13m32) (d11m13 + d12m23 + d13m33)
# (d21m11 + d22m21 +d23m31) (d21m12 + d22m22 + d23m32) (d21m13 + d22m23 + d23m33)
# (d31m11 + d32m21 +d33m31) (d31m12 + d32m22 + d33m32) (d31m13 + d32m23 + d33m33)
# (d11+d21+d31) * m11
