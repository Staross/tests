    
A = rand(50,40,30);
B = rand(50,50);
C = rand(40,40);
D = rand(30,30);
E = rand(50,40,30);
alpha = zeros(size(A))

writecsv("A.csv",A)
writecsv("B.csv",B)
writecsv("C.csv",C)
writecsv("D.csv",D)
writecsv("E.csv",E)

    
function testSum!(A::Array{Float64,3},B::Array{Float64,2},C::Array{Float64,2},
                  D::Array{Float64,2},E::Array{Float64,3},alpha::Array{Float64,3})
    
    tmp_1 = zero(Float64)
    tmp_2 = zero(Float64)
    tmp_3 = zero(Float64)
        
    @inbounds begin
    for x_3p = 1:30
        for x_2p = 1:40
            for x_1p = 1:50

                tmp_3 = zero(tmp_3)             
                for x_3 = 1:30 

                    tmp_2 = zero(tmp_2) 
                    for x_2 = 1:40 

                        tmp_1 = zero(tmp_1) 
                        @simd for x_1 = 1:50 
                             tmp_1 += A[x_1,x_2,x_3] * B[x_1,x_1p]                                                        
                        end
                        tmp_2 += tmp_1 * C[x_2,x_2p]
                    end
                    tmp_3 += tmp_2 * D[x_3,x_3p]
                end

                alpha[x_1p,x_2p,x_3p] =  E[x_1p,x_2p,x_3p] * tmp_3
            end
        end
    end
    end

end

function testSum2!(A::Array{Float64,3},B::Array{Float64,2},C::Array{Float64,2},
                  D::Array{Float64,2},E::Array{Float64,3},alpha::Array{Float64,3})
    
    tmp_1 = 0.0
        
    @inbounds begin
    for x_3p = 1:30*30*40*40
        for x_2p = 1:50*50

            tmp_1 += 1.0                                                       

        end
    end
    end

end

if false
Nt = 2000;
A = ones(50,40,40);
B = ones(50,Nt)/2;
C = ones(40,Nt)/3;
D = ones(40,Nt)/4;
E = ones(Nt,Nt)/5; 
idx = int([Nt:-1:1]);
end

#


function naive_times(A::Array{Float64,2},B::Array{Float64,2})

    C = zeros(size(A,1),size(B,2))
    @assert size(A,2) == size(B,1)
    
    @inbounds for i=1:size(C,1)
        for j=1:size(C,2)
            
            s = 0.0
            @simd for k=1:size(A,2)
                s += A[i,k]*B[k,j]
            end
            C[i,j] = s
        end
    end
    return C
end

A = rand(1,1500); B = rand(1500,2000); C = rand(2000,1);
 
function testSum3(A::Array{Float64,2},B::Array{Float64,2},C::Array{Float64,2})
 
    s = 0.0
            
    tmp_j = 0.0
    @inbounds for j=1:size(B,2)
        tmp_i = 0.0
        @simd for i=1:size(B,1)
            tmp_i += B[i,j] * A[i]            
        end
        tmp_j += tmp_i * C[j]
    end

    s = tmp_j

    return s
end


function testSum3v(A::Array{Float64,2},B::Array{Float64,2},C::Array{Float64,2})

    return  A*B*C
end

function testSum3v_naive(A::Array{Float64,2},B::Array{Float64,2},C::Array{Float64,2})

    return  naive_times(naive_times(A,B),C)
end

@assert mean(abs( testSum3(A,B,C) - testSum3v(A,B,C) ) ) < 1e-8

@time testSum3(A,B,C);
@time testSum3v(A,B,C);

#

A = rand(100,1500); B = rand(1500,2000); C = rand(2000,1);
At = A';
 
function testSum4(A::Array{Float64,2},B::Array{Float64,2},C::Array{Float64,2})
 
    s = zeros(size(A,2))
    
    @inbounds for k=1:size(A,2)
    
        tmp_j = 0.0
        for j=1:size(B,2)
            tmp_i = 0.0        
            @simd for i=1:size(B,1)
                tmp_i += B[i,j] * A[i,k]
            end
            tmp_j += tmp_i * C[j]
        end
        s[k] = tmp_j 
    
    end
    return s
end

@assert mean(abs( testSum4(At,B,C) - testSum3v(A,B,C) ) ) < 1e-8

@time testSum4(At,B,C);
@time testSum3v(A,B,C);

t1=[];
t2=[];

B = rand(800,600); C = rand(600,1);
for N = [1:20:500]
    println(N)
    A = rand(N,800); 
    At = A';

    tt = Base.time();
    testSum4(At,B,C);
    tt2 = Base.time();
    t1 = [t1; tt2-tt];

    tt = Base.time();
    testSum3v_naive(A,B,C);
    tt2 = Base.time();
    t2 = [t2; tt2-tt];

end

N = [1:20:500]


