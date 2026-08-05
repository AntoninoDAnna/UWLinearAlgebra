mutable struct uw_wrap <:Number
    uw::uwreal
end

Base.zero(::Type{uw_wrap}) = uw_wrap(uwreal(0.0))
Base.zero(::Type{uwreal}) = uwreal(0.0)
Base.zero(::uw_wrap) = uw_wrap(uwreal(0.0))
Base.one(::Type{uw_wrap}) = uw_wrap(one(uwreal))
Base.one(::uw_wrap) = uw_wrap(one(uwreal))
Base.length(::uw_wrap) = 1
Base.length(::uwreal) = 1
Base.iterate(uw::uwreal,state=1) = state>1 ? nothing : (uw,state+1)
Base.iterate(uw::uw_wrap,state=1) = state>1 ? nothing : (uw,state+1)

function Base.iszero(a::uw_wrap)
    a.uw.mean == 0.0 || return false
    uwerr(a.uw)
    a.uw.err <= eps()
end

Base.show(io::IO,a::uw_wrap) = Base.show(io,a.uw)

Base.convert(::Type{uwreal},a::uw_wrap) =  a.uw
Base.convert(::Type{uw_wrap},a::uwreal) =  uw_wrap(a.uw)

for op in (:sin, :cos, :tan, :log, :exp, :sqrt, :sind, :cosd, :tand, :sinpi, :cospi, :sinh, :cosh, :tanh, :asin, :acos, :atan, :asind, :acosd, :atand, :sec, :csc, :cot, :secd, :cscd, :cotd, :asec, :acsc, :acot, :asecd, :acscd, :acotd, :sech, :csch, :coth, :asinh, :acosh, :atanh, :asech, :acsch, :acoth, :sinc, :cosc, :deg2rad, :rad2deg, :log2, :log10, :log1p, :exp2, :exp10, :expm1, :-,:+)
    @eval Base.$op(a::uw_wrap) = uw_wrap(Base.$op(a.uw))
end

for op in (:+, :-, :*, :/, :^, :atan, :hypot)
    @eval Base.$op(a::uw_wrap, b::uw_wrap) =uw_wrap(Base.$op(a.uw,b.uw))
    @eval Base.$op(a::uw_wrap, b::uwreal) =uw_wrap(Base.$op(a.uw,b))
    @eval Base.$op(a::uwreal, b::uw_wrap) =uw_wrap(Base.$op(a,b.uw))
    @eval Base.$op(a::uw_wrap, b::Number) = uw_wrap(Base.$op(a.uw,b))
    @eval Base.$op(b::Number, a::uw_wrap) = uw_wrap(Base.$op(b,a.uw))
end

for op in (:<, :>, :≤, :≥, :≠)
    @eval Base.$op(a::uw_wrap, b::uw_wrap) = Base.$op(a.uw, b.uw)
    @eval Base.$op(a::uwreal, b::uw_wrap) = Base.$op(a, b.uw)
    @eval Base.$op(a::uw_wrap, b::uwreal) = Base.$op(a.uw, b)
    @eval Base.$op(a::uw_wrap, b::Number) = Base.$op(a.uw, b)
    @eval Base.$op(a::Number, b::uw_wrap) = Base.$op(a, b.uw)
end

Base.:(==)(a::uw_wrap, b::uw_wrap) = iszero(a-b)
Base.:^(x::uw_wrap, y::Integer) = uw_wrap(x.uw^y)

uw_wrap(a::uw_wrap) = a

unwrap(a::uw_wrap) = a.uw
unwrap(a::AbstractArray{T}) where T = unwrap.(a)

function unwrap(a::T) where T
    S = T.name.wrapper
    fields = Vector{Any}(undef, length(fieldnames(T)))
    if length(fields) == 0
        return a
    end
    for (idx,f) in enumerate(fieldnames(T))
        fields[idx] = unwrap(getfield(a,f))
    end
    return S(fields...)
end
