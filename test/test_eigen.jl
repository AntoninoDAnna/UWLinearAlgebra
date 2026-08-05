using Revise, ADerrors, UWLinearAlgebra, LinearAlgebra

M = rand(5,5) |> x->0.5*(x+x')
Q = LinearAlgebra.qr(M)
E = LinearAlgebra.eigen(M)

Muw = [uwreal([m,m*0.01],"test2") for m in M];
Quw = LinearAlgebra.qr(Muw)
Euw = UWLinearAlgebra.uweigen(Muw)

Mpd = Muw*Muw'

C = LinearAlgebra.cholesky(Mpd)
