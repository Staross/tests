reload("formulas.jl")
using ASCIIPlots


function Pt_th(th,thp)

    P=0
    for k=-1:1
        P += normpdf(thp, 2*k*pi + th  + dt*(2*pi/24),  sqrt(dt)* 0.05)
    end
    return P
end

Pt_A(A,Ap) = normpdf(Ap,  A ,  sqrt(dt)* 0.01)
normpdf(x,mu,sigma) = exp(-0.5*( ((x-mu)/sigma)).^2.0 ) ./ (sqrt(2.0*pi)*sigma)
waveform(th) = (0.5 + 0.5*cos(th)).^1.2
Pem(th,A,data_th) = normpdf(waveform(th)*A, data_th, 0.01)

## phase and amplitude

th = linspace(0,2*pi,61);
th = th[1:end-1]
A = linspace(0.0,1.5,40);

dt = 0.5

obs = linspace(-0.1,1.1,80);

tr =  f(th,thp,A,Ap) ~ Pt_th(th,thp)Pt_A(A,Ap) ;
em =  f(th,A,obs) ~ Pem(th,A,obs)

h = jHMM.HMM()
h = set_states(h,(:th,th),(:A,A))
h = set_transitions(h, tr )

h = set_observation_space(h,(:obs,obs))
h = set_emission(h, em )

Nt = 100;

t = linspace(0,dt*Nt,Nt)

realPhase = 2*pi*t/24 + 0.1*cos(2*pi/15*t)
realA = linspace(1.0,0.3,Nt)
data =  realA.*waveform( realPhase ) + 0.05*randn(size(t));

h = set_observations(h,(:obs,data))

show(h)

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

## phase, amplitude and background


th = linspace(0,2*pi,51);
th = th[1:end-1]

obs = linspace(-0.1,1.5,80);

Pem(th,A,B,data_th) = normpdf(waveform(th)*A + B, data_th, 0.01)

em =  f(th,A,B,obs) ~ Pem(th,A,B,obs)
tr =  f(th,thp,A,Ap,B,Bp) ~ Pt_th(th,thp)Pt_A(A,Ap)Pt_A(B,Bp);

h = jHMM.HMM()
h = set_states(h,(:th,th),(:A,A),(:B,A))
h = set_transitions(h, tr )

h = set_observation_space(h,(:obs,obs))
h = set_emission(h, em )

Nt = 2;

t = linspace(0,dt*Nt,Nt)

realPhase = 2*pi*t/24 + 0.1*cos(2*pi/15*t)
realA = linspace(1.0,0.3,Nt)
realB = linspace(0.0,0.5,Nt)
data =  realA.*waveform( realPhase ) + realB + 0.05*randn(size(t));

h = set_observations(h,(:obs,data))

show(h)

exf = build_forward(h)
exb = build_backward(h)
exjoint = build_joint_of_hidden_states(h)

@time forward(h)
@time backward(h)
#@time joint_of_hidden_states(h,1)

posterior(h)

s = max_posterior(h)
th_ = s[1]

scatterplot(h.observations[1])

R = abs( mean( exp(im*(th_-realPhase))))

println(R)

## coing toss example
if false
    
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


