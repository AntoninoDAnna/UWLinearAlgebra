using Revise
using ADerrors
using LinearAlgebra
using Test
using UWLinearAlgebra
using BenchmarkTools

function exact_cholesky(M::Matrix{T}) where T
    res = LowerTriangular(zeros(T,size(M)))
    res[1,1] = sqrt(M[1,1])
    res[2,1] = M[1,2]/res[1,1]
    res[3,1] = M[1,3]/res[1,1]
    res[2,2] = sqrt(M[2,2] - res[2,1]^2)
    res[3,2] = (M[2,3] - res[2,1]*res[3,1])/res[2,2]
    res[3,3] = sqrt(M[3,3] -res[3,1]^2 - res[3,2]^2)
    Cholesky(res)
end

@testset "uwcholesky" begin
    A = [1.0 0.3 0.4; 0.3 2.0 0.1; 0.4 0.1 4.0 ]
    Auw = [uwreal(randn(10000).+a,"test") for a in A]
    Auw = [a+(_a-a.mean) for (a,_a) in zip(Auw,A)]
    for i in 1:3, j in 1:2
        Auw[i,j] = Auw[j,i]
    end
    Cuw = uwcholesky(Auw)
    Cex = exact_cholesky(Auw)
    diff = Cuw.L .- Cex.L;
    uwerr.(diff);
    uwerr.(Cuw.L);
    uwerr.(Cex.L);
    @test value.(Cuw.L) ≈ value.(Cex.L)
    @test ADerrors.err.(Cuw.L) ≈ ADerrors.err.(Cex.L)
    @test ADerrors.err.(diff) ≈ zeros(3,3) atol=eps()
end
