using DataFrames
using ASCIIPlots
using TensorOperations

module jHMM

    export get_model

    using DataFrames

    type HMM
        v::Array{Symbol,1} #symbols of states variables
        X::Array{Any,1} #discrete values of states variables
        ndim::Integer #dimension of state space
        trFormula::Formula
        trMatrices::Array{Any,1}
        trDimensions::Array{Any,1}
        vO::Array{Symbol,1} #symbols of observation variables
        O::Array{Any,1} #discrete values of observation variables
        ndimO::Integer
        emFormula::Formula
        emMatrices::Array{Any,1}
        emDimensions::Array{Any,1}
        observations::Array{Any,1}
        forward
        backward
        posterior
        h2::jHMM.HMM

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
        f::Array{Array{Symbol,1},1}
    end

    function Model(f::Formula)
        v,f = get_model(f)
        Model(v,f)
    end

    function get_model(f::Formula)

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
        fs = parse_rhs(f.rhs.args)

        return variables, fs
    end

    function parse_rhs(args::Array{Any,1})

        f = Array(Array{Any,1},0)
        parse_rhs!(args,f)
        return f
    end

    function parse_rhs!(args::Array{Any,1},f::Array{Array{Any,1},1})

       T = get_types(args)

       if any( T .!= Symbol)
           idx = find( T .== Expr )
           for i = 1:length(idx)
               parse_rhs!(args[idx[i]].args,f)
           end
       else
           push!(f,args)
       end

    end

    function get_types(x::Array{Any,1})

        t = Array(DataType,length(x))
        for i=1:length(x)
           t[i] = typeof(x[i])
        end
        return t
    end

end

using jHMM
import Base.show
import Base.zero

#From cartesian.jl: Given :i and 3, this generates :i_3
inlineanonymous(base::Symbol, ext) = symbol(string(base)*"_"*string(ext))

function pack_variables(x...)
    X = Array(Array{Any,1},length(x))
    for i=1:length(x)
       X[i] = x[i];
    end
    return X
end

function pack_tuples(t::Type,x...)

    X = Array(Any,0);    
    v = Array(Symbol,length(x))
    for i=1:length(x)
       v[i] = x[i][1];
       push!(X,x[i][2])
    end
    return v,X
end

function set_states(h::jHMM.HMM,x...)

    v,X = pack_tuples(Any,x...)
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
    for i=1:length(h.emMatrices)
        s = s * "M_$i" * string(size(h.emMatrices[i])) * "\t"
    end
    println(s)
    if length(h.observations) > 0
        println("Number of observations=" * string(length(h.observations[1])) * "*"  * string(length(h.observations)) )
    end


end

obs2ind{T<:Union(Float64,String)}(d::Array{T,1},ospace::Array{T,1}) = [obs2ind(d[i],ospace) for i=1:length(d)]
obs2ind{T<:Number}(d::T,ospace::Array{T,1}) = float(indmin( abs( d - ospace ) ))

function obs2ind{T<:String}(d::T,ospace::Array{T,1}) 

    for i=1:length(ospace)
        if ospace[i] == d
            return float(i)
        end
    end
    error("data point was not found in observation space")
end

ind2obs{T<:Any}(ind::Array{Float64,1},ospace::Array{T,1}) = ospace[ind]

function state2ind{T}(state::T,states::Array{T,1})
    
    for i=1:length(states)
        if states[i] == state
            return i
        end
    end
    error("state was not found.")
end

state2ind{T}(v::Array{T,1},states::Array{T,1}) = [state2ind(v[i],states) for i=1:length(v)]

function remove_prime(s::Symbol)

    s = string(s)
    if length(s)>1 && s[end-1:end] == "_p"
       if length(s)>2
            s = s[1:end-2]
       else
           error("variable cannot be named _p")
       end
    end
    return symbol(s)
end

function remove_prime(v::Array{Symbol,1})

    out = Array(Symbol,0)
    for i=1:length(v)
        push!(out, remove_prime(v[i]) )
    end
    return unique(out)
end

function add_prime(s::Symbol)
    return symbol( string(s) * "_p")
end

function add_prime(v::Array{Symbol,1})

    out = Array(Symbol,0)
    for i=1:length(v)
        push!(out, add_prime(v[i]) )
    end
    return out
end

function is_prime(s::Symbol)
    s = string(s)
    return s[end] == "_p"
end

#compute dependence matrix
function get_dependence_matrices(m::jHMM.Model)

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

#return initialized matrices, and build functions to fill them
function get_transition_matrix(m::jHMM.Model,h::jHMM.HMM)

    fs = m.f
    v = m.v
    dep,v_order = get_dependence_matrices(m)

    uniquev = remove_prime(v)

    if !isempty(symdiff(uniquev,h.v))
        error("The definition of the transitions is inconsistent with the states variables")
    end

    trMatrices = Array(Any,length(fs))
    trDimensions = Array(Any,length(fs)) #dimensions of the transition matrices
    
    for i=1:length(fs)

        f = fs[i][2:end]
        N = zeros(Int64, length(f) )
        d = zeros(Int64, length(f) )

        for k=1:length( f )
            s = remove_prime( f[k] )
            j = find( s .== uniquev)[1]
            N[k] = length(h.X[j])
            d[k] = j
        end

        trMatrices[i] = eval( Expr(:call,:zeros,Expr(:tuple, N...)) )
        trDimensions[i] = d
    end

    #now build the functions
    for i=1:length(fs)
        d = find( dep[i,:] )

        args = Array(Expr,length(d))
        idx = Array(Symbol,length(d))
        for j=1:length(d)
            s = :(X[$(trDimensions[i][j])])
            itervar = inlineanonymous(:i, v[d[j]]  )
            args[ v_order[i,d[j]] ] = :( $s[ $itervar ] ) #x[i_x]

            idx[ v_order[i,d[j]] ] = itervar
        end

        rhs = Expr(:call,fs[i][1],args...) #f1(X[1][i_x])
        lhs = Expr(:ref,:(trMatrices[$i]),idx...)   #M[1][i_x,i_xp]

        ex = quote
            $lhs = $rhs
        end

        s = fs[i][2:end]
        for j=1:length(s)

            itervar = inlineanonymous(:i, s[j]  )

            ex = quote
                for $itervar=1:length(X[$(trDimensions[i][j])])
                   $ex
                end
            end
        end

        fname = symbol( string("fill_transition_matrix_",i) )
        ex = quote
            function $fname(h::jHMM.HMM)
                trMatrices = h.trMatrices
                X = h.X
                $ex
                h.trMatrices = trMatrices
            end
        end
        #build the function
        #show(ex)
        eval(ex)

    end

    return trMatrices,trDimensions
end

function get_independent_variables(h::jHMM.HMM)

    fs = jHMM.Model(h.trFormula).f
    v =  h.v
    ndim = h.ndim

    dep = zeros(ndim,ndim)
    
    for j=1:ndim
        for i=1:ndim
            for k=1:length(fs)        
                if any( remove_prime(fs[k][2:end]).== v[j] ) && any( remove_prime(fs[k][2:end]).== v[i] )
                    dep[i,j] = true                    
                end
            end
        end
    end

    (eigval,eigvect) = eig(dep)
    
    dims = find( abs(eigval) .> 1e-12 )
    ndimm_eff = length(dims)
    
    v_eff = Array(Array{Symbol,1},0)
        
    for i=1:ndimm_eff        
        idx = abs(eigvect[:,dims[i]]) .> 1e-12        
        push!(v_eff, v[idx])
    end
    
    #need to reorder
    order = zeros(length(v_eff))
    for k=1:length(fs) 
        for i=1:length(v_eff) 
            if isempty( symdiff( remove_prime(fs[k][2:end]), v_eff[i] ) ) 
                order[i] = k
            end
        end
    end
        
    dims_conc = abs(eigvect) .> 0
    dims_conc = dims_conc[:,dims]
    dims_conc = dims_conc[:,order]
    
    return (v_eff[order], dims_conc)
end

function set_transitions(h::jHMM.HMM,f::Formula)

    h.trFormula = f
    trModel  = jHMM.Model(f)

    trMatrices, trDimensions = get_transition_matrix(trModel,h)
    h.trMatrices = trMatrices
    h.trDimensions = trDimensions

    #call the functions
    for i=1:length(trModel.f)
        fname = symbol( string("fill_transition_matrix_",i) )
        eval(Expr(:call, fname, h))
    end

    return h
end

function set_observation_space(h::jHMM.HMM,x...)

    vO,O = pack_tuples(Any,x...)
    h.O = O
    h.ndimO = length(h.O)
    h.vO = vO;
    return h
end

function set_emission(h::jHMM.HMM,f::Formula)

    h.emFormula = f
    emModel  = jHMM.Model(f)

    emMatrices,emDimensions = get_emissionMatrix(emModel,h)

    h.emMatrices = emMatrices
    h.emDimensions = emDimensions

    #call the functions
    for i=1:length(emModel.f)
        fname = symbol( string("fill_emission_matrix_",i) )
        eval(Expr(:call, fname, h))
    end

    return h
end

#return initialized matrices, and build functions to fill them
function get_emissionMatrix(m::jHMM.Model,h::jHMM.HMM)

    fs = m.f
    v = m.v
        
    emMatrices = Array(Any,length(fs))
    emDimensions = Array(Any,length(fs)) #dimensions of the emission matrices
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

        emMatrices[i] = eval( Expr(:call,:zeros,Expr(:tuple, N...)) )
        emDimensions[i] = d
        X_OR_O[i] = x_or_O
    end

    #now build the functions
    for i=1:length(fs)

        f = fs[i][2:end]

        args = Array(Expr,length(f))
        idx = Array(Symbol,length(f))
        for j=1:length(f)

            k = emDimensions[i][j]
            s = X_OR_O[i][j]  ? :(X[$(k)]) : :(O[$(k)])
            itervar = inlineanonymous(:i, f[j]  )
            args[ j ] = :( $s[ $itervar ] ) #x[i_x]
            idx[ j ] = itervar
        end

        rhs = Expr(:call,fs[i][1],args...) #f1(X[1][i_x])
        lhs = Expr(:ref,:(emMatrices[$i]),idx...)   #emMatrices[1][i_x,i_xp]

        ex = quote
            $lhs = $rhs
        end

        for j=1:length(f)

            k = emDimensions[i][j]
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
                emMatrices = h.emMatrices
                X = h.X
                O = h.O
                $ex
                h.emMatrices = emMatrices
            end
        end
        #finally build the function
        #show(ex)
        eval(ex)
    end

    return emMatrices,emDimensions
end

function set_observations(h::jHMM.HMM,x...)

    #TODO: reorder and check if dimensions match, discretize if needed
    vd, d = pack_tuples(Any,x...)
    
    for i=1:length(d)
        indv = find( h.vO .== vd[i] )[1]
        d[i] = obs2ind(d[i],h.O[indv])
    end
    
    h.observations = d

    return h
end


function build_next_hmm(h::jHMM.HMM)

    (v,dims_conc) = get_independent_variables(h)
    
    s      = fill(:x,length(v));    
    states = Array(Any,0)
    
    N =  [length(h.X[j])::Integer for j=1:length(h.X)]
    
    for i=1:length(v)
       
        idx = findin(h.v,v[i])        
        s[i] = symbol(h.v[idx]...)
            
        Xi = Array(eltype(h.X[1]),0)
        for j=1:length(idx)
            Xi =  [Xi; h.X[idx[j]]]
        end       
                        
        push!(states,tuple(s[i], vec(1:prod(N[idx]))) )
    end
    
    h2 = jHMM.HMM()
    h2 = set_states(h2,states...)
    
    ## transitions
    fs = Array(Expr,0)            
    for i=1:length(v)
        
        idx = findin(h.v,v[i])        
                        
        lhs = Expr(:call, inlineanonymous(:tr_h2,i), tuple(s[i],add_prime(s[i]))...) 
                
        rhs = Expr(:call,:reshape, Expr(:ref,:(h.trMatrices),i), prod(N[idx]), prod(N[idx]))    
        rhs = Expr(:ref,rhs, s[i], add_prime(s[i])) 
                            
        eval( :( $lhs = $rhs) )        
        push!(fs, :( $lhs  ) )
    end
    
    #now build formula
    args = Array(Symbol,0)
    for i=1:length(s)               
       push!(args,s[i],add_prime(s[i]))         
    end
    
    lhs = Expr(:call,:f,args...)
    rhs = Expr(:call,*,fs...)

    tr = Formula(lhs,rhs)    
    h2 = set_transitions(h2, tr )
    
    ## observations
    
    tmp = Array(Any,0)
    for i=1:length(h.O)
        push!(tmp,vec(1:length(h.O[i])))
    end
    h2.O = tmp
    h2.ndimO = copy(h.ndimO)
    h2.vO = copy(h.vO)
                    
    ## emission
    eFs = jHMM.Model(h.emFormula).f
    
    
    ##I need to know what variables the original emission uses
    h_em_v = Array(Symbol,0)
    for i=1:length(eFs)
       push!(h_em_v,eFs[i][2:end]...)
    end
    h_em_v = unique(h_em_v)
    
    ##remove observation ones
    h_em_v_without_o = Array(Symbol,0)
    for i=1:length(h_em_v)
        if all( h.vO .!= h_em_v[i] )
            push!(h_em_v_without_o, h_em_v[i])
        end
    end
    
    ## build ind2sub calls
    
    idx = zeros(Integer,length(h_em_v_without_o)) #find where emission variable map into contracted variables
    for i=1:length(idx)
        for j=1:length(v)
           if any(v[j] .==  h_em_v_without_o[i])
               idx[i] = j
           end
        end
    end
    
    idx = unique(idx)
    
    ind2sub_calls = Array(Expr,0)
    for i in idx
        rhs = Expr(:tuple,v[i]...)            
        sizes =  Expr(:tuple,N[findin(h.v,v[i])]...)
        
        ind2sub_call = Expr(:call, :ind2sub, sizes, h2.v[i])        
        push!(ind2sub_calls, :($rhs = $ind2sub_call) )            
    end
    
    ind2sub_calls = Expr(:block,ind2sub_calls...)
   
    ##now I need to transform these variables into the new ones (e.g. x -> xz)
    for i=1:length(h_em_v)
       for j=1:length(v)
          if any( h_em_v[i] .== v[j] )
            h_em_v[i] = h2.v[j]
          end
       end        
    end
    
    h_em_v = unique(h_em_v)
    
    ## build ref to emission matrices
    em_mat = Array(Expr,0)
    for i=1:length(h.emMatrices)
        push!(em_mat, Expr(:ref, h.emMatrices[i], eFs[i][2:end]...) ) #note, 
        #push!(em_mat, Expr(:ref, Expr(:ref,:(h.emMatrices),i), eFs[i][2:end]...) ) #note,  will get h in the global space
    end
    em_mat = Expr(:call,*, em_mat...)
    
    #now I can build the new emission function!
    args = tuple(h_em_v...)
    fdef = inlineanonymous(:em_h2,1)
        
    fdef = Expr(:call,fdef,args...)

    ex =  quote 
        $fdef = begin
            $ind2sub_calls
            return $em_mat
        end
    end
    
    #show(ex)
    eval(ex)
        
    #let's build the formula for the emission
    lhs = Expr(:call,:f,args...)
    rhs = fdef

    emFormula = Formula(lhs,rhs)
           
    h2 = set_emission(h2, emFormula )
    
    h2.observations = copy(h.observations)
        
    return h2
    
end


function normalize!(A)

       c = 1/sum(A)
       for i=1:prod(size(A))
           A[i] =  A[i]*c            
       end
       return c
end
    

function forward(h::jHMM.HMM)

    #initialize everything
    N = [length(h.X[i])::Integer for i=1:length(h.X)]
        
    alpha = ones(N...)
    out = similar(alpha)

    if length(h.observations)==0
        error("No observations. Use set_observations to provide observations to the HMM.")
    end
    
    Nt =  length(h.observations[1]) 
    
    h.forward = ones([N;Nt]...)
    scale = zeros(Nt)
    
    tr  = h.trMatrices
    em  = h.emMatrices[1]
    obs = h.observations
    
    c = normalize!(alpha)
    scale[1] = c
        
    
    for t=2:Nt

        ref_em = tuple([fill(Colon(),ndims(em)-1); t]...)
                
        forward_iter!(alpha,out,tr...,slice(em,ref_em...))                
        c = normalize!(out)
        
        ref = tuple([fill(Colon(),length(N)); t]...)
                
        copy!(slice(h.forward,ref...),out)
        scale[t] = c
        
        copy!(alpha,out)

    end       
    
end

#1D
function forward_iter!{T}(alpha::Array{T,1}, out::Array{T,1}, Tx::Array{T,2}, em)
            
    A_mul_B!(out, Tx, alpha)         
    broadcast!(*,alpha,out,em)   
    copy!(out,alpha)   
end

#2D
function forward_iter!{T}(alpha::Array{T,2}, out::Array{T,2}, Tx::Array{T,2}, Ty::Array{T,2}, em)
        
    A_mul_B!(out, alpha, Ty)
    A_mul_B!(alpha, Tx, out)     
    
    broadcast!(*,out,alpha,em)        
end

#3D
function forward_iter!(alpha::Array{Float64,3}, out::Array{Float64,3}, Tx::Array{Float64,2}, Ty::Array{Float64,2}, Tz::Array{Float64,2}, em)

   TensorOperations.tensorcontract!(1.0,alpha,[:x; :y; :z]  ,'N',Tz   ,[:z; :zp]     ,'N',0.0, out,   [:x; :y; :zp]);
   TensorOperations.tensorcontract!(1.0,out,  [:x; :y; :zp] ,'N',Ty   ,[:y; :yp]     ,'N',0.0, alpha, [:x; :yp; :zp]);
   TensorOperations.tensorcontract!(1.0,Tx,   [:xp; :x]     ,'N',alpha,[:x; :yp; :zp],'N',0.0, out,   [:xp; :yp; :zp]);

   broadcast!(*,alpha,out,em)   
   copy!(out,alpha)  
   
end

# some tests    

reload("phaseHMM_functions.jl")

th = linspace(0,2*pi,25); th = th[1:end-1]
A = linspace(0.0,1.5,10);
B = linspace(-1.0,1.0,10);

dt = 0.5

obs = linspace(-0.1,1.1,80);



## 3D sep

if 1 == 1

h = jHMM.HMM()
h = set_states(h,(:x,x),(:y,y),(:z,z))

tr =  f(th,thp,A,Ap) ~ Pt_th(th,thp)Pt_A(A,Ap) ;
em =  f(th,A,obs) ~ Pem(th,A,obs)

h = set_transitions(h, tr )

h = set_observation_space(h,(:o1,o1))
h = set_observations(h,(:o1,o1))
h = set_emission(h, em )

h2 = build_next_hmm(h)

for i=1:3
    @assert norm(h.trMatrices[i] - h2.trMatrices[i]) == 0
end

M = zeros(size(h.emMatrices[1]));
for i=1:size(M,2)
   M[:,i,:] =  h.emMatrices[2];
end

@assert sum(abs((h.emMatrices[1].*M - h2.emMatrices[1]))) == 0

forward(h2)
@time forward(h2)

end



