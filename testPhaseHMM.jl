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

Nt = 150;

t = linspace(0,dt*Nt,Nt)

realPhase = 2*pi*t/24 + 0.1*cos(2*pi/15*t)
realA = linspace(1.0,0.2,Nt)
d1 =  realA.*waveth( realPhase ) + 0.01*randn(size(t));
d1 = obs2ind(d1,d)

h = set_observations(h,(:d,d1))

show(h)

m = jHMM.Model( tr )

exf = build_forward(h)
exb = build_backward(h)
exjoint = build_joint_of_hidden_states(h)

@time forward(h)
@time backward(h)
@time joint_of_hidden_states(h,5)

posterior(h)

s = max_posterior(h)
th_ = s[1]

imagesc( h.trMatrices[1]  )

R = abs( mean( exp(im*(th_-realPhase))))

println(R)

## coing toss example
if true
    
states = ["fair"; "unfair"]; 
coins = ["head", "tail"]; 

p = 1/30;
trMatrix = [1-p p; p 1-p];

function P_tr(state,statep)
    return trMatrix[state2ind(state,states),state2ind(statep,states)]
end

bias = 0.4;
emMatrix = [0.5 0.5; 0.5-bias 0.5+bias];

function P_em(state, coin)
    return emMatrix[state2ind(state,states),obs2ind(coin,coins)]
end

h = jHMM.HMM()
h = set_states(h,(:state,states))
h = set_transitions(h, f(state,statep) ~ P_tr(state,statep)  )

h = set_observation_space(h,(:coin,coins))
h = set_emission(h, f(state,coin) ~ P_em(state,coin) )

#flipping coins
cflip = (rand(30) .< 0.5 ) ;
cflip = [cflip; (rand(30) .< 0.5+bias ) ];
cflip = [cflip; (rand(30) .< 0.5 ) ];

obs = fill("head",length(cflip));
obs[cflip] = "tail";

h = set_observations(h,(:coin,obs))

exf = build_forward(h)
exb = build_backward(h)
exjoint = build_joint_of_hidden_states(h)

@time forward(h)
@time backward(h)
@time joint_of_hidden_states(h,5)
posterior(h)

s = max_posterior(h)

scatterplot(state2ind(s[1],states))

end


