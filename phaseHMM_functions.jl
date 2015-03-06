using Distributions
normpdf(x,mu,sigma) = pdf(Normal(mu,sigma),x)

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