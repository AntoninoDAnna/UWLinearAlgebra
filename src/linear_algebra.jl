Base.conj(uw::uw_wrap) = uw
Base.conj(uw::uwreal) = unwrap(Base.conj(uw_wrap(uw)))
Base.adjoint(uw::uw_wrap) = Base.conj(uw)
Base.adjoint(uw::uwreal) = unwrap(Base.conj(uw_wrap(uw)))
Base.transpose(uw::uwreal) = unwrap(Base.transpose(uw_wrap(uw)))
Base.abs(uw::uw_wrap) = sqrt(uw^2)
Base.abs2(uw::uw_wrap) = uw^2
Base.inv(uw::uw_wrap) = 1/uw
Base.real(::Type{uw_wrap}) = uw_wrap
Base.real(uw::uw_wrap) = uw
Base.copysign(x::uw_wrap,y::uw_wrap) = x.uw.mean*y.uw.mean<0 ? -x : x

function LinearAlgebra.norm(uw::uw_wrap;p::Real=2)
    res = abs(uw)
    if p==0
        iszero(uw) && return zero(res)
        !isnan(res.mean) && return oneunit(res)
        return res
    else
        return res
    end
end

function norm2(itr::AbstractArray{uw_wrap})
    s = sum(abs2(i) for i in itr)
    return sqrt(s)
end

function LinearAlgebra.norm(itr::AbstractArray{uw_wrap})
    isempty(itr) && return zero(uw_wrap)
    LinearAlgebra.norm_recursive_check(itr)
    norm2(itr)
end

# overloading LinearAlgebra methods
LinearAlgebra.choltype(::AbstractMatrix{uw_wrap}) = uw_wrap
LinearAlgebra.choltype(::Type{uw_wrap}) = uw_wrap

function identity(::Type{T}, n::Integer) where T
    res = zeros(T,n,n)
    _one = one(T)
    @inbounds for i in axes(res,1)
        res[i,i] = _one
    end
    return res
end
identity(::Type{T},M::AbstractMatrix) where T = identity(T,size(M,1))
identity(M::AbstractMatrix{T}) where T = identity(T,size(M,1))

LinearAlgebra.qr(M::AbstractMatrix{uwreal}) =
    unwrap(LinearAlgebra.qr(uw_wrap.(M)))
