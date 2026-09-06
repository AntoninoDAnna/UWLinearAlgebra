using Revise, ADerrors, UWLinearAlgebra, LinearAlgebra

M = randn(2,2) |> x-> 0.5*(x+x')
Muw = [uwreal(randn(1000).*0.01.+m,"test") for m in M];
M = value.(Muw)
Euw = UWLinearAlgebra.uweigen(Muw)
E = LinearAlgebra.eigen(M)

##analythic results
function eigenvalues(M::Matrix)
    s = size(M,1);
    if s==2
        tr = M[1,1] + M[2,2]
        det = M[1,1]*M[2,2] - M[2,1]*M[1,2]
        D = sqrt(tr^2 - 4det)
        return [0.5*(tr-D), 0.5*(tr+D)]
    else
        error("unsupported size")
    end
end

λ1 = eigenvalues(M)
test = λ1 .- value.(Euw.values)

λ2 = eigenvalues(Muw)
test = λ2 .- Euw.values

uwerr.(test); test
