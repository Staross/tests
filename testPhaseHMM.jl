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
h = set_states(h,(:th,th),(:A,A))
h = set_transitions(h, tr )

h = set_observation_space(h,(:d,d))
h = set_emission(h, em )

Nt = 15;

t = linspace(0,dt*Nt,Nt)

realPhase = 2*pi*t/24 + 0.1*cos(2*pi/15*t)
realA = linspace(1.0,0.2,Nt)
d1 =  realA.*waveth( realPhase ) + 0.01*randn(size(t));
d1 = map_data(d1,d)

h = set_observations(h,(:d,d1))

show(h)

m = jHMM.Model( tr )

exf = build_forward(h)
exb = build_backward(h)

@time forward(h)
@time backward(h)
posterior(h)

s = max_posterior(h)
th_ = s[:,1]

imagesc( h.trMatrices[1]  )

R = abs( mean( exp(im*(th_-realPhase))))

println(R)

## coing toss example
if false
    
p = 1/30;
trMatrix = [1-p p; p 1-p];

function P_tr(state,statep)
    return trMatrix[state,statep]
end

bias = 0.25;
emMatrix = [0.5 0.5; 0.5-bias 0.5+bias];

function P_em(state, coin)
    return emMatrix[state,coin]
end

states = [1.0; 2.0]; #fair, unfair
coins = [1.0; 2.0]; #head, tail

h = jHMM.HMM()
h = set_states(h,(:state,states))
h = set_transitions(h, f(state,statep) ~ P_tr(state,statep)  )

h = set_observation_space(h,(:coin,coins))
h = set_emission(h, f(state,coin) ~ P_em(state,coin) )

data = (rand(30) .< 0.5 ) + 1.0;
data = [data; (rand(30) .< 0.5+bias ) + 1.0];
data = [data; (rand(30) .< 0.5 ) + 1.0];

h = set_observations(h,(:coin,data))

exf = build_forward(h)
exb = build_backward(h)

@time forward(h)
@time backward(h)
posterior(h)

s = max_posterior(h)

end


