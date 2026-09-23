@doc raw"""
     propagate_error(fnew::Function, x0::AbstractVector{Float64}, data::AbstractVector{uwreal})

Propagates the error of `data` on `x0` assuming `fnew(x0,data)= 0`.
The function `fnew` return a vector of the same length as `x0`.

```@example
using ADerrors, UWLinearAlgebra, LinearAlgebra

A =  [1.0 2.0; 3.0 4.0];
Auw = [uwreal(randn(1000).+a, "test") for a in A]
Auw = [a+(_a-a.mean) for (a,_a) in zip(Auw,A)] ## ensure the mean values of A match with Auw

E = eigvals(A)
f(λ,d) = [det(reshape(d,2,2).-_λ.*[1 0; 0 1]) for _λ in λ]
f(λ,d::Matrix) = [det(d.-_λ.*[1 0; 0 1]) for _λ in λ]

λ = propagate_error(f,E,reshape(Auw,:))
uwerr.(λ)

#Check
λtrue = let
    tr = Auw[1,1]+Auw[2,2]
    det = Auw[1,1]*Auw[2,2] - Auw[1,2]*Auw[2,1]
    [0.5*(tr - sqrt(tr^2-4*det)),0.5*(tr + sqrt(tr^2-4*det))]
end

z = λ .- λtrue
uwerr.(z)
print("Better be zero (with zero error): ")
details.(z)
```

"""
function propagate_error(fnew::Function, x0::AbstractVector{Float64}, data::AbstractVector{uwreal})
    nx = length(x0)
    nd = length(data)
    xv = [x0;value.(data)]
    f(x) = fnew( @view(x[1:nx]), @view(x[nx+1:end]))
    J = ForwardDiff.jacobian(f,xv)
    Jx = view(J,:,1:nx)
    Jd = view(J,:,nx+1:nx+nd)
    der = -Jx \ Jd
    return addobs(data,der,x0)
end

function propagate_error!(fnew!::Function, x0::AbstractVector{Float64}, data::AbstractVector{uwreal}, nf::Int64=length(x0))
    nx = length(x0)
    nd = length(data)
    xv = [x0;value.(data)]
    y = Vector{Float64}(undef, nf)
    J = Matrix{Float64}(undef, nf, nx + nd)
    f!(y,x) = fnew!(y, @view(x[1:nx]), @view(x[nx+1:end]))
    ForwardDiff.jacobian!(J, f!, y, xv,cfg)
    Jx = @view J[:,1:nx]
    Jd = @view J[:,nx+1:end]
    der = -Jx \ Jd
    return addobs(data,der,x0)
end


export propagate_error
