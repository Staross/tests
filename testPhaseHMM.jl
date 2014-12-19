reload("formulas.jl")
using ASCIIPlots

function Pem(th,A,data_th)

    P = normpdf(waveth(th)*A, data_th, 0.01)
end


function Pt_th(th,thp)
    
    P=0
    for k=-1:1
        P += normpdf(thp, 2*k*pi + th  + dt*(2*pi/24),  sqrt(dt)* 0.05)
    end
    return P
end

function Pt_A(A,Ap)

    P = normpdf(Ap,  A ,  sqrt(dt)* 0.01)
end


function normpdf(x,mu,sigma)

    return exp(-0.5*( ((x-mu)/sigma)).^2.0 ) ./ (sqrt(2.0*pi)*sigma)
end

function waveth(th)

    return (0.5 + 0.5*cos(th)).^1.2
end

##

th = linspace(0,2*pi,61);
th = th[1:end-1]
A = linspace(0.0,1.5,40);

dt = 0.5

d = linspace(-0.1,1.1,80);

tr =  f(th,thp,A,Ap) ~ Pt_th(th,thp)Pt_A(A,Ap) ;
em =  f(th,A,d) ~ Pem(th,A,d)

h = jHMM.HMM()
h = setStates(h,(:th,th),(:A,A))
h = setTransitions(h, tr )

h = setObservationSpace(h,(:d,d))
h = setEmission(h, em )

Nt = 150;

t = linspace(0,dt*Nt,Nt)

realPhase = 2*pi*t/24 + 0.1*cos(2*pi/15*t)
realA = linspace(1.0,0.2,Nt)
d1 =  realA.*waveth( realPhase ) + 0.01*randn(size(t));
d1 = mapData(d1,d)

h = setObservations(h,(:d,d1))

show(h)

m = jHMM.Model( tr )

exf = buildForward(h)
exb = buildBackward(h)

@time forward(h)
backward(h)
posterior(h)

s = maxPosterior(h)
th_ = s[:,1]

imagesc( h.trMatrices[1]  )

R = abs( mean( exp(im*(th_-realPhase))))

println(R)


