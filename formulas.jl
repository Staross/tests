using DataFrames
using Cartesian
using Cartesian.inlineanonymous

module jHMM

    export getModel

    using DataFrames
    
    type HMM
        v::Array{Symbol,1} #symbols of states variables
        X::Array{Array{Float64,1},1} #discrete values of states variables
        ndim::Integer #dimension of state space
        trFormula::Formula
        trMatrices::Array{Any,1}
        trDimensions::Array{Any,1}
        vO::Array{Symbol,1} #symbols of observation variables
        O::Array{Array{Float64,1},1} #discrete values of observation variables
        ndimO::Integer
        emFormula::Formula
        emMatrices::Array{Any,1}
        emDimensions::Array{Any,1}
        observations::Array{Any,1}
        forward
        backward
        posterior
        
        HMM() = new( Array(Symbol,0),
                     Array(Array{Float64,1},0),
                     0,
                     Formula(:tr,:f),
                     Array(Any,0),
                     Array(Any,0),
                     Array(Symbol,0),
                     Array(Any,0),
                     0,
                     Formula(:em,:f),
                     Array(Any,0),
                     Array(Any,0),
                     Array(Any,0),
                     nothing,
                     nothing,
                     nothing
                     )
    end
    
    
    type Model
        v::Array{Symbol,1}     
        f::Array{Array{Any,1},1}                               
    end
        
    function Model(f::Formula) 
        v,f = getModel(f)
        Model(v,f)
    end
    
    function getModel(f::Formula)

        args = f.lhs.args
        variables = Array(Symbol,length(args)-1)

        for i = 1:length(args) 
            if typeof(args[i]) != Symbol
                error("Right hand side should have one expression")
            end 
            if i > 1
                variables[i-1] = args[i]
            end
        end
        
        #variables = unique(variables)
        fs = parseRhs(f.rhs.args)

        return variables, fs    
    end

    function parseRhs(args::Array{Any,1})

        f = Array(Array{Any,1},0)
        parseRhs!(args,f)
        return f
    end

    function parseRhs!(args::Array{Any,1},f::Array{Array{Any,1},1})

       T = getTypes(args)

       if any( T .!= Symbol) 
           idx = find( T .== Expr )
           for i = 1:length(idx)
               parseRhs!(args[idx[i]].args,f)
           end       
       else
           push!(f,args)
       end

    end
    
    function getTypes(x::Array{Any,1})

        t = Array(DataType,length(x))
        for i=1:length(x)
           t[i] = typeof(x[i])
        end
        return t
    end
        
end

using jHMM
import Base.show


function packVariables(x...)
    X = Array(Array{Float64,1},length(x))
    for i=1:length(x)
       X[i] = x[i]; 
    end
    return X
end

function packTuples(x...)
    X = Array(Array{Float64,1},length(x))
    v = Array(Symbol,length(x))
    for i=1:length(x)
       v[i] = x[i][1];
       X[i] = x[i][2]; 
    end
    return v,X
end

function setStates(h::jHMM.HMM,x...)

    v,X = packTuples(x...)
    h.X = X
    h.ndim = length(h.X)
    h.v = v
    return h
end

function show(io::IO,h::jHMM.HMM)
    
    @printf io "%dD-HMM" h.ndim
    for i=1:h.ndim
       print("\t$(h.v[i]): [" * string(minimum(h.X[i])) * ",.. ," * string(maximum(h.X[i]))"], N_$i=" * string(length(h.X[i])) )
    end
    
    println("\nTransitions:")
    show(h.trFormula)
    
    s = string("\nMatrices: ")
    for i=1:length(h.trMatrices)
        s = s * "M_$i" * string(size(h.trMatrices[i])) * "\t"
    end
    println(s)
    
    print("\nEmission: ")        
    @printf io "%dD-observation space\n" h.ndimO
    for i=1:h.ndimO
       println("$(h.vO[i]): [" * string(minimum(h.O[i])) * " ,.. ," * string(maximum(h.O[i]))"], N_$i=" * string(length(h.O[i])) )
    end
    show(h.emFormula)
    s = string("\nMatrices: ")
    for i=1:length(h.observations)
        s = s * "M_$i" * string(size(h.emMatrices[i])) * "\t"
    end
    println(s)
    if length(h.observations) > 0 
        println("Number of observations=" * string(length(h.observations[1])) * "*"  * string(length(h.observations)) )
    end
    
    
end

function setTransitions(h::jHMM.HMM,f::Formula)

    h.trFormula = f
    m  = jHMM.Model(f)
    
    M,D = getTransitionMatrix(m,h)
    h.trMatrices = M
    h.trDimensions = D
    
    #call the function
    for i=1:length(m.f)
        fname = symbol( string("fill_transition_matrix_",i) )        
        eval(Expr(:call, fname,:h))
    end
    
    return h
end

function removePrime(s::Symbol)

    s = string(s)    
    if s[end] == 'p' 
       if length(s)>1
            s = s[1:end-1]
       else
           error("variable cannot be named p")
       end
    end
    return symbol(s)
end

function removePrime(v::Array{Symbol,1})

    out = Array(Symbol,0)
    for i=1:length(v)
        push!(out, removePrime(v[i]) )
    end
    return unique(out)
end

function addPrime(s::Symbol)
    return symbol( string(s) * "p")
end

function addPrime(v::Array{Symbol,1})

    out = Array(Symbol,0)
    for i=1:length(v)
        push!(out, addPrime(v[i]) )
    end
    return out
end

function isPrime(s::Symbol)
    s = string(s)   
    return s[end] == 'p' 
end

#compute dependence matrix
function getDependenceMatrices(m::jHMM.Model)

    fs = m.f
    v =  m.v
    ndim = length((v))
    
    dep = zeros(length(fs),ndim)
    v_order = zeros(length(fs),ndim)
    for i=1:length(fs)
        for j=1:ndim        
            if any(fs[i].== v[j])
                dep[i,j] = true   
                v_order[i,j] = find( fs[i] .== v[j])[1]-1
            end        
        end
    end

    return dep,v_order
end

function getTransitionMatrix(m::jHMM.Model,h::jHMM.HMM)
    
    fs = m.f
    v = m.v
    dep,v_order = getDependenceMatrices(m)
    
    uniquev = removePrime(v)
    
    if !isempty(symdiff(uniquev,h.v))        
        error("The definition of the transitions is inconsistent with the states variables")        
    end
    
    M = Array(Any,length(fs))
    D = Array(Any,length(fs)) #dimensions of the transition matrices, trDimensions
    for i=1:length(fs)
        
        f = fs[i][2:end]
        N = zeros(Int64, length(f) )   
        d = zeros(Int64, length(f) )   
        
        for k=1:length( f )
            s = removePrime( f[k] ) 
            j = find( s .== uniquev)[1]
            N[k] = length(h.X[j])
            d[k] = j
        end
        
        M[i] = eval( Expr(:call,:zeros,Expr(:tuple, N...)) )
        D[i] = d
    end
        
    #now build the functions
    for i=1:length(fs)
        d = find( dep[i,:] )
        
        args = Array(Expr,length(d))
        idx = Array(Symbol,length(d))
        for j=1:length(d)
            s = :(X[$(D[i][j])])
            itervar = inlineanonymous(:i, v[d[j]]  )
            args[ v_order[i,d[j]] ] = :( $s[ $itervar ] ) #x[i_x]
            
            idx[ v_order[i,d[j]] ] = itervar 
        end

        rhs = Expr(:call,fs[i][1],args...) #f1(X[1][i_x])                
        lhs = Expr(:ref,:(M[$i]),idx...)   #M[1][i_x,i_xp]
    
        ex = quote
            $lhs = $rhs
        end
        
        s = fs[i][2:end]        
        for j=1:length(s)

            itervar = inlineanonymous(:i, s[j]  )

            ex = quote
                for $itervar=1:length(X[$(D[i][j])])
                   $ex
                end
            end
        end
                
        fname = symbol( string("fill_transition_matrix_",i) )
        ex = quote
        function $fname(h::jHMM.HMM)
            M = h.trMatrices
            X = h.X
            $ex            
            h.trMatrices = M                                    
        end
        end
        #build the function
        eval(ex)
        
    end
                
    return M,D
end


function setObservationSpace(h::jHMM.HMM,x...)

    vO,O = packTuples(x...)    
    h.O = O
    h.ndimO = length(h.O)
    h.vO = vO;
    return h
end

function setEmission(h::jHMM.HMM,f::Formula)

    h.emFormula = f
    m  = jHMM.Model(f)
    
    M,D = getEmissionMatrix(m,h)
    
    h.emMatrices = M
    h.emDimensions = D
        
    #call the functions
    for i=1:length(m.f)
        fname = symbol( string("fill_emission_matrix_",i) )        
        eval(Expr(:call, fname,:h))
    end
            
    return h
end

function getEmissionMatrix(m::jHMM.Model,h::jHMM.HMM)
    
    fs = m.f
    v = m.v
            
    M = Array(Any,length(fs))
    D = Array(Any,length(fs)) #dimensions of the emission matrices
    X_OR_O = Array(Any,length(fs)) #state or observation space
    for i=1:length(fs)
        
        f = fs[i][2:end]
        N = zeros(Int64, length(f) )   
        d = zeros(Int64, length(f) )
        x_or_O = zeros(Bool, length(f) )
        
        for k=1:length(f)
            
            s =  f[k]             
            x_or_O[k] = length( intersect(h.v,[s]) ) > 0
            
            if x_or_O[k]    
                j = find( s .== h.v)[1]
                N[k] = length(h.X[j])                
            else
                j = find( s .== h.vO)[1]
                N[k] = length(h.O[j])                
            end
            
            d[k] = j
        end
        
        M[i] = eval( Expr(:call,:zeros,Expr(:tuple, N...)) )
        D[i] = d
        X_OR_O[i] = x_or_O
    end
    
    #now build the functions
    for i=1:length(fs)
        
        f = fs[i][2:end]
                
        args = Array(Expr,length(f))
        idx = Array(Symbol,length(f))
        for j=1:length(f)
            
            k = D[i][j]                     
            s = X_OR_O[i][j]  ? :(X[$(k)]) : :(O[$(k)])
            itervar = inlineanonymous(:i, f[j]  )
            args[ j ] = :( $s[ $itervar ] ) #x[i_x]            
            idx[ j ] = itervar 
        end

        rhs = Expr(:call,fs[i][1],args...) #f1(X[1][i_x])                
        lhs = Expr(:ref,:(M[$i]),idx...)   #M[1][i_x,i_xp]
    
        ex = quote
            $lhs = $rhs
        end
                   
        for j=1:length(f)

            k = D[i][j]
            r = X_OR_O[i][j] ? :(X[$(k)]) : :(O[$(k)])
            
            itervar = inlineanonymous(:i, f[j]  )

            ex = quote
                for $itervar=1:length($r)
                   $ex
                end
            end
        end
                
        fname = symbol( string("fill_emission_matrix_",i) )
        ex = quote
        function $fname(h::jHMM.HMM)
            M = h.emMatrices
            X = h.X
            O = h.O
            $ex            
            h.emMatrices = M
        end
        end
        #finally build the function
        #show(ex)
        eval(ex)
        
    end
    
    return M,D
end

function setObservations(h::jHMM.HMM,x...)

    #TODO: reorder and check if dimensions match, discretize if needed
    vd,d = packTuples(x...)    
    h.observations = d
    
    return h
end

function buildForward_(h::jHMM.HMM)

    vs,fs = getModel(h.trFormula)
    
    Nt = length(h.observations[1])
    N = [length(h.X[i]) for i=1:length(h.X)]
    alpha = eval( Expr(:call,:ones,Expr(:tuple, [Nt; N]...)) )#TODO: initial condition
    
    ref = Array(Symbol,1+length(N)); ref[1] = :t    
    ref[2:end] = :(:)
    
    ex = Expr(:ref,:alpha, ref... )   #alpha(t,:,:)
    ex = :(alpha_t = squeeze($ex,1))
        
    #determine the order in which we need to sum 
    dims = h.trDimensions
    
    dep = zeros(Bool,length(dims),length(dims))
    for i=1:length(dims)
        for j=1:length(dims)
            dep[i,j] = !isempty( intersect(dims[i],dims[j]) )
        end
    end
    
    val,vect = eig(dep)
    val = abs(val) .> 1e-10
    vect = abs(vect) .> 1e-10
    
    vect = vect[:,val]
    v_summed = Array(Symbol,0)
    
    for b=1:size(vect,2) #for each block
    
        idx = sort( vec(sum(vect,1)),rev=true )
    
        idx = find(vect[:,idx[b]])
        
        f = fs[idx]

        vars = Array(Symbol,0)
        for i=1:length(idx)
            v = h.v[ h.trDimensions[idx[i]] ]
            for j=1:length(v)
                push!( vars, v[j] )
            end
        end    
        vars = unique(vars)    
        v_summed = union(vars,v_summed)
        missingvars = setdiff(h.v, v_summed )#variables we haven't summed over yet

        #build temporary vector
        N = zeros(Int64, length(missingvars) )  
        for i=1:length(missingvars)
            k = find(h.v .== missingvars[i])[1]
            N[i] = length(h.X[k])
        end

        rhs = length(N)>0 ? Expr(:call,:zeros,Expr(:tuple, N...)) : 0.0
        lhs = inlineanonymous(:tmp,b)
        ex = :( $lhs = $rhs)
        
        show(ex)
        
        #generate internal part of the loop
        for d = 1:length(vars)
            
        end
    
    end
   
end

function cleanExpr(ex::Expr)

    ex2 = deepcopy(ex)
    ex2.args = cleanExpr(ex2.args)
    return ex2
end

#this shit doesn't work
function cleanExpr(args::Array{Any,1})
    
    nargs = Array(Any,0)
    for i=1:length(args)
        
       if typeof( args[i] ) == Expr
           if args[i].head != :line
               if args[i].head == :block
                   for j=1:length(args[i].args)
                       
                       if typeof(args[i].args[j])== Expr && args[i].args[j].head == :line                           
                          continue 
                       end
                       
                       if typeof(args[i].args[j]) != LineNumberNode
                           push!(nargs, cleanExpr(args[i].args[j]))
                       end                                              
                   end
               else
                   
                  if typeof(args[i]) !=  LineNumberNode 
                        push!(nargs, cleanExpr(args[i]))
                  end                                     
               end
           end        
       else
           push!(nargs, args[i])
       end
    end
    return nargs
end

function getMainLoopForwardOpt(h::jHMM.HMM,fs,N)

    #get level of functions in the loops
    level = zeros(length(fs))
    for i=1:length(fs)       
        for j=1:length(h.v)
            if sum( h.v[j] .== fs[i][2:end] ) > 0
               level[i] = j
               break
            end
        end        
    end
        
    #build inner part of the main loop: tmp_1 += (alpha_t[x,y] * tr_1[x,xp]) * tr_2[x,y,yp]
    lhs = Expr(:ref,:alpha_t,h.v...)
        
    for i=1:length(h.trMatrices)       
       if level[i] == 1
           M = inlineanonymous(:tr,i) 
           ex = Expr(:ref,M,fs[i][2:end]...)           
           lhs = Expr(:call,:*,lhs,ex)
       end
    end

    ex = quote        
        @inbounds tmp_1 += $lhs
    end
    
    rhs = Expr(:tmp)
    #loops over state variables    
    for i=1:length(h.v)
        
        lhs = inlineanonymous(:tmp,i)        
        ex2 = Expr(:block)
        if i>1 
           rhs = inlineanonymous(:tmp,i-1)
           
           idx = find( level .== i)
           
           for j = 1:length(idx) 
               M = inlineanonymous(:tr,idx[j])   

               ex2 = Expr(:ref,M,fs[idx[j]][2:end]...)           
               rhs = Expr(:call,:*,rhs,ex2)

           end
        end
        
        itervar = h.v[i]
        tmp_p1 = inlineanonymous(:tmp,i+1)   
        if i == 1
            ex = quote
                $(lhs) = 0.0
                @simd for $itervar=1:$(N[i])
                   $ex 
                end
            end
        else
            ex = quote
                $(lhs) = 0.0
                for $itervar=1:$(N[i])
                   $ex                    
                   $(lhs) += $(rhs)
                end                
            end        
        end
        
    end
    
    #show(ex)
    return ex    
end

function getMainLoopForward(h::jHMM.HMM,fs,N)

        
    #build inner part of the main loop: tmp += (alpha_t[x,y] * tr_1[x,xp]) * tr_2[x,y,yp]
    lhs = Expr(:ref,:alpha_t,h.v...)
        
    for i=1:length(h.trMatrices)              
           M = inlineanonymous(:tr,i) 
           ex = Expr(:ref,M,fs[i][2:end]...)           
           lhs = Expr(:call,:*,lhs,ex)       
    end
    
    tmp_ex = inlineanonymous(:tmp,length(N))
    ex = quote        
        @inbounds $tmp_ex += $lhs
    end
        
    #loops over state variables    
    for i=1:length(h.v)
                
        itervar = h.v[i]
        if i == 1
            ex = quote
                @simd for $itervar=1:$(N[i])
                   $ex 
                end
            end
        else 
            ex = quote
                for $itervar=1:$(N[i])
                   $ex                    
                end
            end        
        end
        
        end
    
    ex = Expr(:block,:($tmp_ex = 0.0),ex)
    
    return ex    
end


function buildForward(h::jHMM.HMM)

    vs,fs = getModel(h.trFormula)
    vse,fse = getModel(h.emFormula)
        
    N = [length(h.X[i]) for i=1:length(h.X)]
    
    ex = getMainLoopForwardOpt(h,fs,N)
    
    #build part with emission: alpha[tp1,xp,yp] = tmp * em_1[xp,o1[tp1]]
    ref = addPrime(h.v)
    ref = [:(tp1); ref]

    rhs = Expr(:ref,:alpha,ref...)   
        
    #emission term
    em = inlineanonymous(:(tmp),length(N))
    for i=1:length(h.emMatrices)
        
       M = inlineanonymous(:em,i) 
       
       f = fse[i][2:end]
       args = Array(Any,length(f))
       for j=1:length(f)
            x_or_O = sum( h.v .== f[j] ) >0
            args[j] = x_or_O ? addPrime(f[j]) : Expr(:ref,f[j],:tp1)
       end
           
       em = Expr(:call,:*,em, Expr(:ref,M,args...) )
    end
         
    ex = quote         
        $ex        
        @inbounds $rhs = $em 
    end
    
    #loops over state variables prime    
    for i=1:length(h.v)

        itervar = addPrime(h.v[i])
        ex = quote
            for $itervar=1:$(N[i])
               $ex 
            end
        end
    end 
        
    ref = Array(Symbol,1+length(N)); ref[1] = :t    
    ref[2:end] = :(:)
    
    alpha_t = Expr(:ref,:alpha, ref... )   #alpha(t,:,:)
    ex_a = quote        
        scale[t] = sum($alpha_t)
        $alpha_t = $alpha_t /  scale[t]
        alpha_t = squeeze($alpha_t,1)::Array{Float64,$(length(h.v))}
        tp1 = t+1
    end
            
    exs = Array(Expr,0)
    
    push!(exs,ex_a)
    push!(exs,ex)
    ex = Expr(:block,exs...)
    
    a_dec = Expr(:call,:ones,Expr(:tuple, [:Nt; N]...)) #TODO: initial condition
    scale_dec =  Expr(:call,:ones,:Nt) 

    exs = Array(Expr,0)
    #unpack transition matrices
    for i=1:length(h.trMatrices)
       M = inlineanonymous(:tr,i) 
       push!(exs, :($M = h.trMatrices[$i]::Array{Float64,$(ndims(h.trMatrices[i]))} ) )#these annotations are critical for performance
    end
    #unpack emission matrices
    for i=1:length(h.emMatrices)
       M = inlineanonymous(:em,i) 
       push!(exs, :($M = h.emMatrices[$i]::Array{Float64,$(ndims(h.emMatrices[i]))} ) )
    end  
    #unpack observations
    for i=1:length(h.vO)
       push!(exs, :($(h.vO[i]) = h.observations[$i]::Array{Float64,$(ndims(h.observations[i]))} ) )
    end
    
    matricesDef = Expr(:block,exs...)
    
    #finish
    ex = quote
        function forward(h::jHMM.HMM)
    
            Nt = length(h.observations[1])::Int64
            alpha = $a_dec
            scale = $scale_dec
            $matricesDef
            for t=1:Nt-1
                #println(t)
                $ex
            end
            
            t = Nt 
            scale[t] = sum($alpha_t)
            $alpha_t = $alpha_t /  scale[t]
            L = sum(log(scale))
            h.forward = alpha

            return alpha, L, scale  
        end        
    end
        
    #show(ex)
    eval(ex)
    return ex
            
end

function buildBackward(h::jHMM.HMM)

    vs,fs = getModel(h.trFormula)
    vse,fse = getModel(h.emFormula)
        
    N = [length(h.X[i]) for i=1:length(h.X)]
    
    #build inner part of the main loop: tmp += (beta_t[x,y] * tr_1[xp,x]) * tr_2[xp,yp,y] em_1[x,o1[tp1]]
    lhs = Expr(:ref,:beta_t,h.v...)
        
    for i=1:length(h.trMatrices)
        
       M = inlineanonymous(:tr,i) 
              
       v = fs[i][2:end]
       #transform variables
       for j=1:length(v)
          v[j] = isPrime(v[j]) ? removePrime(v[j]) : addPrime(v[j])
       end
       
       ex = Expr(:ref,M,v...)           
       lhs = Expr(:call,:*,lhs,ex)
    end
    
    #emission term
    for i=1:length(h.emMatrices)
        
       M = inlineanonymous(:em,i) 
       
       f = fse[i][2:end]
       args = Array(Any,length(f))
       for j=1:length(f)
            x_or_O = sum( h.v .== f[j] ) >0
            args[j] = x_or_O ? f[j] : Expr(:ref,f[j],:tp1)
       end
           
       lhs = Expr(:call,:*,lhs, Expr(:ref,M,args...) )
    end


    ex = quote        
        @inbounds tmp += $lhs
    end
    
    #loops over state variables    
    for i=1:length(h.v)
        
        itervar = h.v[i]
        if i == 1
            ex = quote
                @simd for $itervar=1:$(N[i])
                   $ex 
                end
            end
        else
            ex = quote
                for $itervar=1:$(N[i])
                   $ex 
                end
            end        
        end
    end 
    
    #build part : beta[t,xp,yp] = tmp 
    ref = addPrime(h.v)
    ref = [:t; ref]

    rhs = Expr(:ref,:beta,ref...)   
                   
    ex = quote 
        tmp = 0.0
        $ex        
        @inbounds $rhs = tmp 
    end
    
    #loops over state variables prime    
    for i=1:length(h.v)

        itervar = addPrime(h.v[i])
        ex = quote
            for $itervar=1:$(N[i])
               $ex 
            end
        end
    end 
        
    ref = Array(Symbol,1+length(N)); ref[1] = :(tp1)    
    ref[2:end] = :(:)
    
    beta_t = Expr(:ref,:beta, ref... )   #beta(t+1,:,:)
    ex_a = quote        
        tp1 = t+1
        scale[tp1] = sum($beta_t)
        $beta_t = $beta_t /  scale[tp1]
        beta_t = squeeze($beta_t,1)::Array{Float64,$(length(h.v))}        
    end
            
    exs = Array(Expr,0)
    
    push!(exs,ex_a)
    push!(exs,ex)
    ex = Expr(:block,exs...)
    
    a_dec = Expr(:call,:ones,Expr(:tuple, [:Nt; N]...)) #TODO: initial condition
    scale_dec =  Expr(:call,:ones,:Nt) 

    exs = Array(Expr,0)
    #unpack transition matrices
    for i=1:length(h.trMatrices)
       M = inlineanonymous(:tr,i) 
       push!(exs, :($M = h.trMatrices[$i]::Array{Float64,$(ndims(h.trMatrices[i]))} ) )#these annotations are critical for performance
    end
    #unpack emission matrices
    for i=1:length(h.emMatrices)
       M = inlineanonymous(:em,i) 
       push!(exs, :($M = h.emMatrices[$i]::Array{Float64,$(ndims(h.emMatrices[i]))} ) )
    end  
    #unpack observations
    for i=1:length(h.vO)
       push!(exs, :($(h.vO[i]) = h.observations[$i]::Array{Float64,$(ndims(h.observations[i]))} ) )
    end
    
    matricesDef = Expr(:block,exs...)
    
    #finish
    ex = quote
        function backward(h::jHMM.HMM)
    
            Nt = length(h.observations[1])::Int64
            beta = $a_dec
            scale = $scale_dec
            $matricesDef
            for t=(Nt-1):-1:1
                #println(t)
                $ex
            end
            
            tp1 = 1 
            scale[tp1] = sum($beta_t)
            $beta_t = $beta_t /  scale[tp1]
            L = sum(log(scale))
            h.backward = beta

            return beta, L        
        end        
    end
        
    #show(ex)
    eval(ex)
    return ex
            
end


function posterior(h::jHMM.HMM)

    p =  h.forward .* h.backward
    
    for t=1:size(p,1)
       p[t,:] = p[t,:]/sum(p[t,:]) 
    end
    
    h.posterior = p
    
    return p
end

function maxPosterior(h::jHMM.HMM)
    
    p = h.posterior
    Nt = size(p,1)
    s = zeros(Nt,length(h.X))
    
    for t=1:Nt
        
        tmp = p[t,:] 
        idx = ind2sub(size(p)[2:end], indmax(tmp))
        
        for i=1:length(idx)
           s[t,i] = h.X[i][idx[i]]
        end
                
    end
        
    return s
end

function mapData(d,x)

    ind = zeros(size(d))
    for i=1:length(d)        
        ind[i] = indmin( abs( d[i] - x ) )
    end
    return ind
        
end

x = linspace(-2,2,60);
y = linspace(-2,2,40);
z = linspace(-2,2,20);

o1 = linspace(0,6,100);
o2 = linspace(0,6,100);

f1(x,xp) = exp(-(x+xp)^2) ;
f2(x,y,yp) = exp(-(x+y*yp)^2);
f3(z,zp) = exp(-z^2)*exp(-zp^2);

em1(x,o1) = exp(-(x+o1)^2)
em2(y,z,o1) = exp(-(y+z+o1)^2)

tr =  f(x,xp,y,yp,z,zp) ~ f1(x,xp)f2(x,y,yp)f3(z,zp) ;

#tr =  f(x,xp,y,yp) ~ f1(x,xp)f2(x,y,yp);

em =  f(x,y,o1) ~ em2(x,y,o1);

h = jHMM.HMM()
h = setStates(h,(:x,x),(:y,y),(:z,z))
h = setTransitions(h, tr )

h = setObservationSpace(h,(:o1,o1))
h = setEmission(h, em )

Nt = 2;

d1 = ceil( length(o1)*rand(Nt) ); d2 = ceil( length(o2)*rand(Nt) )

h = setObservations(h,(:o1,d1))

show(h)

m = jHMM.Model( tr )

exf = buildForward(h)
exb = buildBackward(h)

#@time alpha, L = forward(h)


#v,fs = getModel( f(x,y,z) ~ f1(x)f2(y)f3(z) )
#v,fs = getModel( f(x,y,z) ~ f1(x,y,z) )

#v =  m.v
#N = int( [length(eval(v[i])) for i=1:length(v)] )
#ndim = length(N);

#F = eval( Expr(:call,:zeros,Expr(:tuple, N...)) )


#dep,v_order = getDependenceMatrices(m)


if(false)
m = jHMM.Model( f(x,y) ~ f1(x)f2(x,y) )
fs = m.f
v = m.v
dep,v_order = getDependenceMatrices(m)

#compute sum
function _getSum(fs::Array{Array{Any,1},1},dep,v)
	
    exs = Array(Expr,length(fs))
    sum_term = Array(Symbol,length(fs))
    for i=1:length(fs)
                
        d = find( dep[i,:] )
        sum_term[i] = inlineanonymous(:s,i)

        args = Array(Expr,length(d))
        for j=1:length(d)            
            itervar = inlineanonymous(:i,v[ d[j] ])
            args[ v_order[i,d[j]] ] = :( $(v[d[j]])[ $itervar ] ) #x[i_x]
        end

        ex = Expr(:call,fs[i][1],args...) #f1(x[i_x])
        ex = quote
            @inbounds $(sum_term[i]) += $ex #s_1 += f1(x[i_x])
        end

        for j=1:length(d)

            itervar = inlineanonymous(:i,v[ d[j] ])

            ex = quote
                for $itervar=1:$(N[d[j]])
                   $ex 
                end
            end
        end            

        println("")

        ex = quote
            $(sum_term[i]) = 0.0
            $ex         
        end

        exs[i] = ex
    end

    ex = Expr(:block,exs...)

    finalSum = Expr(:call,:*,sum_term...)
    ex = quote
        $ex
        s = $finalSum
    end
  
    return ex
end


body = _getSum(fs,dep,v)
ex = quote
    function getSum()
        s = 0.0
        $body
        return s
    end
end

#show(ex)
#eval(ex)

function getSumNaive(x,y,z,f1,f2)

    s=0.0
    for i=1:length(x)
        for j=1:length(y)
            for k=1:length(z)
                s += f1(x[i])*f2(y[j],z[k])
            end
        end
    end
    return s
end

##
#s = getSum()
#s2 = getSumNaive(x,y,z,f1,f2)

#@time getSum()
#@time getSumNaive(x,y,z,f1,f2)

end




