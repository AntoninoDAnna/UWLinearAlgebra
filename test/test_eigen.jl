using ADerrors
using LinearAlgebra
using Test
using UWLinearAlgebra

function exact_eigen(M)
    tr = M[1,1]+M[2,2]
    det = M[1,1]*M[2,2] - M[1,2]*M[2,1]
    values = [0.5*(tr - sqrt(tr^2-4*det)),
              0.5*(tr + sqrt(tr^2-4*det))]
    vectors = similar(M)
    vectors[1,1] = one(eltype(M))
    vectors[2,1] = (values[1]-M[1,1])/M[1,2]
    vectors[1,2] = one(eltype(M))
    vectors[2,2] = (values[2]-M[1,1])/M[1,2]
    vectors[:,1] ./=sqrt(sum(vectors[:,1].^2))
    vectors[:,2] ./=sqrt(sum(vectors[:,2].^2))
    return Eigen(values,vectors)
end

@testset "uweigen" begin
    A = [1.0 2.0; 3.0 4.0]
    Auw = [uwreal(randn(10000).+a,"test") for a in A]
    Auw = [a+(_a-a.mean) for (a,_a) in zip(Auw,A)]
    E=eigen(A)
    Euw = eigen(Auw)
    Euw_ex = exact_eigen(Auw)
    p = sortperm(Euw.values,by=value)
    p_ex = sortperm(Euw.values,by=value)
    diff = Euw.values[p].-Euw_ex.values[p_ex]
    uwerr.(diff)
    @test value.(diff) ≈ zeros(2) atol=eps()
    @test ADerrors.err.(diff) ≈ zeros(2) atol=eps()
    diff = Euw.vectors[:,p].^2 - Euw_ex.vectors[:,p_ex].^2
    uwerr.(diff)
    @test value.(diff) ≈ zeros(2,2) atol=eps()
    @test ADerrors.err.(diff) ≈ zeros(2,2) atol=eps()
end

@testset "uweigen" begin
    A = [1.0 2.0; 3.0 4.0]
    id = [1.0 0.0; 0.0 1.0]
    Auw = [uwreal(randn(10000).+a,"test") for a in A]
    Auw = [a+(_a-a.mean) for (a,_a) in zip(Auw,A)]
    iduw = [uwreal(randn(10000).+a,"test") for a in id]
    iduw = [a+(_a-a.mean) for (a,_a) in zip(iduw,id)]
    E=eigen(A,id)
    Euw = eigen(Auw,)
    Euw_ex = exact_eigen(Auw)
    p = sortperm(Euw.values,by=value)
    p_ex = sortperm(Euw.values,by=value)
    diff = Euw.values[p].-Euw_ex.values[p_ex]
    uwerr.(diff)
    @test value.(diff) ≈ zeros(2) atol=eps()
    @test ADerrors.err.(diff) ≈ zeros(2) atol=eps()
    diff = Euw.vectors[:,p].^2 - Euw_ex.vectors[:,p_ex].^2
    uwerr.(diff)
    @test value.(diff) ≈ zeros(2,2) atol=eps()
    @test ADerrors.err.(diff) ≈ zeros(2,2) atol=eps()
end
