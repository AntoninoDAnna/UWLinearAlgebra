using Revise, ADerrors, UWLinearAlgebra, LinearAlgebra

M = [0.1 0.03; 0.03 0.1]
LinearAlgebra.cholesky(M)
LinearAlgebra.eigen(M)

Muw = [uwreal([m,m*0.01],"test2") for m in M];
LinearAlgebra.cholesky(Muw)
LinearAlgebra.qr(Muw)
#LinearAlgebra.eigen(Muw)
