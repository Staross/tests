  function myForward(h::jHMM.HMM) # /Users/bieler/Desktop/julia/tests/formulas.jl, line 773:
        Nt = length(h.observations[1])::Int64 # line 774:
        alpha = ones((50,40,40,Nt))::Array{Float64,4} # line 775:
        scale = ones(Nt) # line 776:
        begin 
            tr_1 = h.trMatrices[1]::Array{Float64,2}
            tr_2 = h.trMatrices[2]::Array{Float64,2}
            tr_3 = h.trMatrices[3]::Array{Float64,2}
            em_1 = h.emMatrices[1]::Array{Float64,4}
            obs = h.observations[1]::Array{Float64,1}
        end # line 777:
        for t = 1:Nt - 1 # line 779:
            begin 
                begin  # /Users/bieler/Desktop/julia/tests/formulas.jl, line 688:
                    scale[t] = sum(alpha[:,:,:,t]) # line 689:
                    alpha[:,:,:,t] = alpha[:,:,:,t] / scale[t] # line 690:
                    alpha_t = squeeze(alpha[:,:,:,t],4)::Array{Float64,3} # line 691:
                    tp1 = t + 1
                end
                begin  # /Users/bieler/Desktop/julia/tests/formulas.jl, line 857:
                    for Bp = 1:40 # line 858:
                        begin  # /Users/bieler/Desktop/julia/tests/formulas.jl, line 857:
                            for Ap = 1:40 # line 858:
                                begin  # /Users/bieler/Desktop/julia/tests/formulas.jl, line 857:
                                
                                    tmp_1 = zero(Float64)
                                    tmp_2 = zero(Float64)
                                    tmp_3 = zero(Float64)
                                    tmp_4 = zero(Float64)
                                
                                    for thp = 1:50 # line 858:
                                        tmp_3 = 0.0 # line 557:
                                        tmp_4 = rand() # line 557:
                                                                                                                                
                                        for B = 1:40 # line 558:

                                            tmp_2 = 0.0 # line 557:
                                            for A = 1:40 # line 558:

                                                tmp_1 = 0.0 # line 550:
                                                @simd for th = 1:50 # line 551:

                                                    @inbounds tmp_1 += alpha_t[th,A,B] * tr_1[th,thp]                                                        
                                                end

                                                @inbounds tmp_2 += tmp_1 * tr_2[A,Ap]
                                            end
                                            
                                            @inbounds tmp_3 += tmp_2 * tr_3[B,Bp]
                                        end

                                        @inbounds alpha[thp,Ap,Bp,tp1] =  em_1[thp,Ap,Bp,obs[tp1]] * tmp_3                    
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end # line 782:
        t = Nt # line 783:
        scale[t] = sum(alpha[:,:,:,t]) # line 784:
        alpha[:,:,:,t] = alpha[:,:,:,t] / scale[t] # line 785:
        L = sum(log(scale)) # line 786:
        h.forward = alpha # line 788:
        return (alpha,L,scale)
    end