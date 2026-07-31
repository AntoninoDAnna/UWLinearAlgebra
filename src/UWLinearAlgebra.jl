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
Base.copysign(x::uwreal,y::uwreal) = x.mean*y.mean<0 ? -x : x

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
# LA.eigtype(::AbstractMatrix{uwreal}) = uwreal
# LA.eigtype(::Type{uwreal}) = uwreal
LA.choltype(::AbstractMatrix{uwreal}) = uwreal
LA.choltype(::Type{uwreal}) = uwreal
#LA.dot()

function uwidentity(n::Int64)
    M =zeros(uwreal,n,n)
    _one = one(uwreal)
    @inbounds for i in axes(M,1)
        M[i,i] = _one
    end
    return M
end

get_tol(tol::Real,M::AbstractMatrix{uwreal}) = tol
get_tol(::Nothing,M::AbstractMatrix{uwreal}) =
    eps(minimum(abs(M[i,i].mean) for i in axes(M,1)))

function uweigen(M::AbstractMatrix{uwreal};
                 tol::Union{Real,Nothing} = nothing)
    n = size(M,1)
    v = uwidentity(n)
    m = LA.copy_similar(M,uwreal)
    temp = zeros(uwreal,n,n)
    old = zeros(uwreal,n,n)
    done = false
    while !done
        old .= m
        Q = LA.qr(m)
        LA.mul!(m,Q.R,Q.Q)
        _tol = get_tol(tol,m)
        done = sum(abs(_mold.mean-_m.mean) for (_mold,_m) in zip(old,m)) < _tol
        old .=v
        LA.mul!(temp,v, Q.Q)
        v.=temp
    end
    return LA.Eigen(LA.diag(m),v)
end
end # module UWLinearAlgebra
