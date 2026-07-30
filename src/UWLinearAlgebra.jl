module UWLinearAlgebra
using ADerrors
import LinearAlgebra as LA

Base.zero(::Type{uwreal}) = uwreal(0.0)


function iszero(a::uwreal)
    a.mean == 0.0 || return false
    uwerr(a)
    a.err == 0.0
end

function Base.:(==)(a::uwreal,b::uwreal)
    a.mean == b.mean || return false
    c = a-b;
    uwerr(c)
    c.err == 0.0
end

Base.conj(uw::uwreal) = uw
Base.adjoint(uw::uwreal) = Base.conj(uw)
Base.abs(uw::uwreal) = sqrt(uw^2)
Base.abs2(uw::uwreal) = uw^2
Base.inv(uw::uwreal) = 1/uw
Base.real(::Type{uwreal}) = uwreal
Base.real(uw::uwreal) = uw
ADerrors.uwreal(uw::uwreal) = uw
Base.:^(x::uwreal, y::Integer) = Base.invoke(^, Tuple{uwreal,Number}, x, y)

Base.isless(a::uwreal,b::uwreal) = Base.isless(a.mean,b.mean)

function Base.copysign(x::uwreal,y::uwreal)
    z = copysign(x.mean,y.mean)
    z == x.mean && return copy(x)
    return -x
end

function LA.norm(uw::uwreal;p::Real=2)
    res = abs(uw)
    if p==0
        iszero(uw) && return zero(res)
        !isnan(res.mean) && return oneunit(res)
        return res
    else
        return res
    end
end

function norm2(itr::AbstractArray{uwreal})
    s = sum(abs2(i) for i in itr)
    return sqrt(s)
end

function LA.norm(itr::AbstractArray{uwreal})
    isempty(itr) && return zero(uwreal)
    LA.norm_recursive_check(itr)
    norm2(itr)
end

# overloading LinearAlgebra methods
LA.eigtype(::AbstractMatrix{uwreal}) = uwreal
LA.eigtype(::Type{uwreal}) = uwreal
LA.choltype(::AbstractMatrix{uwreal}) = uwreal
LA.choltype(::Type{uwreal}) = uwreal





end # module UWLinearAlgebra
