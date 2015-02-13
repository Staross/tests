using Base.Cartesian
using Base.Cartesian.inlineanonymous


function get_output_size(sA::Dims,sB::Dims,dimsA::Array{Int64,1}, dimsB::Array{Int64,1})

    if length(sB) > length(sA) 
        (sA,sB) = (sB,sA) 
        (dimsA,dimsB) = (dimsB,dimsA) 
    end

    sC = ()
    for i=1:length(sA)
        if i != dimsA[2]
           sC = tuple(sC...,sA[i]) 
        else
           sC = tuple(sC...,sB[dimsB[2]]) 
        end
    end
    
    return sC
end


@assert get_output_size((30,40),(40,35),[1; 2],[1; 2]) == (30,35)

@assert get_output_size((30,40,60),(40,35,60),[1; 2],[1; 2]) == (30,35,60)
@assert get_output_size((60,40,20),(60,20,30),[2; 3],[2; 3]) == (60,40,30)
@assert get_output_size((60,40,20),(20,60,30),[2; 3],[1; 3]) == (60,40,30)

@assert get_output_size((5,30,40,60),(5,40,35,60),[2; 3],[2; 3]) == (5,30,35,60)
@assert get_output_size((30,5,40,60),(5,40,35,60),[1; 3],[2; 3]) == (30,5,35,60)

@assert get_output_size((30,35),(20,40,30),[1; 2],[2; 3]) == (20,40,35)
@assert get_output_size((20,40,30),(30,35),[2; 3],[1; 2]) == (20,40,35)

@assert get_output_size((30,35),(20,40,30,10),[1; 2],[2; 3]) == (20,40,35,10)
@assert get_output_size((30,35),(20,40,10,30),[1; 2],[2; 4]) == (20,40,10,35)

@assert get_output_size((6,30,10,3,4),(10,20,6,3,4),[2; 3],[1; 2]) == (6,30,20,3,4)

function get_loop_indices(sA::Dims,sB::Dims,sC::Dims,dimsA::Array{Int64,1}, dimsB::Array{Int64,1})

    switched = false
    if length(sB) > length(sA) 
        switched = true
        (sA,sB) = (sB,sA) 
        (dimsA,dimsB) = (dimsB,dimsA) 
    end

    idx = ()
    dims = setdiff(1:length(sA),dimsA)
    
    for i=1:length(dims)
        
        N = sA[dims[i]]
        a = dims[i]
        b = 0
        c = a
        
        if length(sB) >= dims[i] & sB[dims[i]] == N
            
            b = dims[i]            
        else
            
            k = find([sB[i] == N for i=1:length(sB)])                
            k = setdiff(k,dimsB)
            
            if !isempty(k)
                b = k[1]
            end
        end

        idx = switched ? tuple(idx...,N,b,a,c) : tuple(idx...,N,a,b,c)
    end
    
    if length(sA) - length(intersect(idx,0)) != length(sB) 
        error("Dimension mismatch:  $(sC) = $(sA) * $(sB), looping over $(idx[1:4:end])")
    end
        
    
    return idx
end

sA = (30,40,60); sB = (40,35,60); dimsA = [1; 2]; dimsB = [1; 2];
@assert get_loop_indices(sA,sB, get_output_size(sA,sB,dimsA,dimsB) ,dimsA,dimsB) == (60,3,3,3)

sA = (30,40); sB = (40,35); dimsA = [1; 2]; dimsB = [1; 2];
@assert get_loop_indices(sA,sB, get_output_size(sA,sB,dimsA,dimsB) ,dimsA,dimsB) == ()

sA = (30,35); sB = (20,40,30); dimsA = [1; 2]; dimsB = [2; 3];
@assert get_loop_indices(sA,sB, get_output_size(sA,sB,dimsA,dimsB) ,dimsA,dimsB) == (20,0,1,1)

sA = (5,30,40,60); sB = (5,40,35,60); dimsA = [2; 3]; dimsB = [2; 3];
@assert get_loop_indices(sA,sB, get_output_size(sA,sB,dimsA,dimsB) ,dimsA,dimsB) == (5,1,1,1,60,4,4,4)

sA = (6,30,10,3,4); sB = (10,20,6,3,4); dimsA = [2; 3]; dimsB = [1; 2];
@assert get_loop_indices(sA,sB, get_output_size(sA,sB,dimsA,dimsB) ,dimsA,dimsB) == (6,1,3,1,3,4,4,4,4,5,5,5)
@assert get_output_size(sA,sB,[2; 3],[1; 2]) == (6,30,20,3,4)

#sA = (6,30,10,3,4); sB = (10,20,25,3,4); dimsA = [1; 3]; dimsB = [1; 2];


function multiprod{N,K,T}(A::Array{T,N},B::Array{T,K},dimsA::Array{Int64,1}, dimsB::Array{Int64,1})

    #todo: check that dimsA and dimsB are in order, otherwise need to transpose and rearange
    
    sC = get_output_size(size(A),size(B),dimsA, dimsB)
    
    loop_idx = get_loop_indices(size(A),size(B),sC,dimsA,dimsB)
        
    C = zeros(Float64, sC...)
    
    loop_mult!(A,B,C,loop_idx...)
             
    return C
end


function multiprod!{N,K,T}(A::Array{T,N},B::Array{T,K},C, dimsA::Array{Int64,1}, dimsB::Array{Int64,1})

    #todo: check that dimsA and dimsB are in order, otherwise need to transpose and rearange
    
    sC = get_output_size(size(A),size(B),dimsA, dimsB)
    
    loop_idx = get_loop_indices(size(A),size(B),sC,dimsA,dimsB)
        
    loop_mult!(A,B,C,loop_idx...)
             
    return C
end

getref(j) = (:)
getref(j,K1,k1) = (j == K1 ? k1 : getref(j))
getref(j,K1,K2,k1,k2) =  j == K1 ? k1 : getref(j,K2,k2)
getref(j,K1,K2,K3,k1,k2,k3) =  j == K1 ? k1 : getref(j,K2,K3,k2,k3)


function loop_mult!(A,B,C) #2D

    A_mul_B!(slice(C,:,:),slice(A,:,:),slice(B,:,:))
    
end


function build_loop_mult_functions()

   N = 3

   args = Array(Any,0)
   for i=1:N
       push!(args, inlineanonymous(:N,i) )
       for j=1:3
           push!(args, inlineanonymous(:i,j+(i-1)*3) )       
       end
   end   
   args = Expr(args...)
   #args = Expr(:tuple,:A,:B,:C,args...)
   #fundef = Expr(:call, :(loop_mult!), args)

   getref_args = Array(Any,3,2*N)
   
   for i=1:N
       for j=1:3
            getref_args[j,i] = inlineanonymous(:i, j+(i-1)*3 )  
       end        
       for j=1:3
            getref_args[j,N+i] = inlineanonymous(:k, i )  
       end 
   end
   
   getrefA = Expr(:call, getref, Expr(:tuple,:j,getref_args[1,:]...) )
   getrefB = Expr(:call, getref, Expr(:tuple,:j,getref_args[2,:]...) )
   getrefC = Expr(:call, getref, Expr(:tuple,:j,getref_args[3,:]...) )
   
   loop_body = quote
   
        refA = ntuple(length(size(A)),j -> $(getrefA) )
        refB = ntuple(length(size(B)),j -> $(getrefB)  )
        refC = ntuple(length(size(C)),j -> $(getrefC)  )
                
        A_mul_B!(slice(C,refC...), slice(A,refA...), slice(B,refB...))        
   end

   for i=1:N
       
       iter_var = inlineanonymous(:k, N-i+1 )
       range = inlineanonymous(:N, N-i+1 )
       loop_body =  quote
           for $iter_var=1:$range
              $loop_body
           end
       end
    end
   
    fun = quote
    function loop_mult!($(args...))   
            $loop_body            
        end
    end
   
   show(fun)
   eval(fun)
   
      
end

build_loop_mult_functions()

function loop_mult!(A,B,C,N1,i1,i2,i3) #3D

    for k1=1:N1
             
        refA = ntuple(length(size(A)),j -> getref(j,i1,k1) )
        refB = ntuple(length(size(B)),j -> getref(j,i2,k1) )
        refC = ntuple(length(size(C)),j -> getref(j,i3,k1) )
                
        A_mul_B!(slice(C,refC...), slice(A,refA...), slice(B,refB...))
    end
    
end

function loop_mult!(A,B,C,N1,i1,i2,i3,N2,i4,i5,i6)

    for k1=1:N1
        for k2=1:N2

            refA = ntuple(length(size(A)),j -> getref(j,i1,i4,k1,k2) )
            refB = ntuple(length(size(B)),j -> getref(j,i2,i5,k1,k2) )
            refC = ntuple(length(size(C)),j -> getref(j,i3,i6,k1,k2) )

            A_mul_B!(slice(C,refC...), slice(A,refA...), slice(B,refB...))
        
        end
    end
end

function loop_mult!(A,B,C,N1,i1,i2,i3,N2,i4,i5,i6,N3,i7,i8,i9)

    for k1=1:N1
        for k2=1:N2
            for k3=1:N3

                refA = ntuple(length(size(A)),j -> getref(j,i1,i4,i7,k1,k2,k3) )
                refB = ntuple(length(size(B)),j -> getref(j,i2,i5,i8,k1,k2,k3) )
                refC = ntuple(length(size(C)),j -> getref(j,i3,i6,i9,k1,k2,k3) )
             
                A_mul_B!(slice(C,refC...), slice(A,refA...), slice(B,refB...))
            
            end        
        end
    end
end

#just for testing
function multiprod2{N,K,P,T}(A::Array{T,N},B::Array{T,K},C::Array{T,P})

    for j=1:size(C,4)
        for i=1:size(C,3)            
            A_mul_B!(slice(C,:,:,i,j),slice(A,:,:,i,j),slice(B,:,:,i,j))
        end
    end    
        
    return C
end

function multiprod1{N,K,P,T}(A::Array{T,N},B::Array{T,K},C::Array{T,P})

    for i=1:size(C,3)                 
        A_mul_B!(slice(C,:,:,i),slice(A,:,:,i),slice(B,:,:,i))
    end    
        
    return C
end

A = rand(30,40); B = rand(40,30);
@assert multiprod(A,B,[1; 2],[1; 2]) == A*B

A = rand(30,40,60); B = rand(40,30,60);
C =  multiprod(A,B,[1; 2],[1; 2]) 
@assert multiprod(A,B,[1; 2],[1; 2]) == multiprod1(A,B,C)
@time multiprod(A,B,[1; 2],[1; 2])


C = zeros(300,20,60,3);
A = rand(300,400,60,3); B = rand(400,20,60,3);

multiprod2(A,B,C)

@assert multiprod!(A,B,C,[1; 2],[1; 2]) == multiprod2(A,B,C)
@assert multiprod(A,B,[1; 2],[1; 2]) == multiprod2(A,B,C)

gc()
@time multiprod!(A,B,C,[1; 2],[1; 2]) 
@time multiprod2(A,B,C)



A = rand(30,10,6,3,4); B = rand(10,20,6,3,4);
C = zeros(30,20,6,3,4);

multiprod!(A,B,C,[1; 2],[1; 2]) 
@time multiprod!(A,B,C,[1; 2],[1; 2]) 
@assert multiprod!(A,B,C,[1; 2],[1; 2]) == multiprod(A,B,[1; 2],[1; 2]) 


A = rand(6,30,10,3,4); B = rand(10,20,6,3,4);
multiprod(A,B,[2; 3],[1; 2]) 

A = rand(6,30,10,3,4); B = rand(10,20,30,3,4);
multiprod(A,B,[1; 3],[1; 2]) 

A = rand(6,30,10,3,4); B = rand(10,20,30,3,4);
multiprod(A,B,[1; 3],[1; 2]) 

