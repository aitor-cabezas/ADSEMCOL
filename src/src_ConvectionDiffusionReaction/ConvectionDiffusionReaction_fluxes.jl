#-------------------------------------------------------------------------------
#AUXILIARY FUNCTIONS FOR DOMAIN INTEGRALS:
#Update flux terms with contribution of hyperbolic terms. 
#This function receives several matrices: u1,...,uN,p. Then, updates flux_qp,
#which is a nVars x 2 matrix of Matrix{Float64} with the corresponding values of 
#f_{Ij}, I=1,..,,nVars, j=1:2.
#The function also updates dflux_du_qp, which is a nVars x 2 x nVars array with the values of
#df_{Ij}/du_J.

function OregonatorFlux!(model::Oregonator,du::Matrix{Matrix{Float64}},flux::Matrix{Matrix{Float64}},dflux_dgradu::Array{Matrix{Float64},4},ComputeJ::Bool)

    nSpecies        =   model.nSpecies
    D               =   [model.Du,model.Dv,model.Dw] #Diffusion Coefficients
    epsilonv        =   [model.epsilon,1.0,model.epsilonp]  

    ## Diffusion Flux

    for alpha = 1:nSpecies, i= 1:2

        @tturbo @. flux[alpha,i]   +=   -(D[alpha]/epsilonv[alpha])*du[alpha,i]

    end
    
    if ComputeJ
            
            @tturbo @. dflux_dgradu[1,1,1,1]   +=   -D[1]/epsilonv[1]
            @tturbo @. dflux_dgradu[1,2,1,2]   +=   -D[1]/epsilonv[1]
            @tturbo @. dflux_dgradu[3,1,3,1]   +=   -D[3]/epsilonv[3]
            @tturbo @. dflux_dgradu[3,2,3,2]   +=   -D[3]/epsilonv[3]
        
    end
    
    return

end


function source!(model::Oregonator, u::Vector{Matrix{Float64}}, Q::Vector{Matrix{Float64}}, dQ_du::Matrix{Matrix{Float64}},ComputeJ::Bool)
    
    nSpecies        =   model.nSpecies
    D               =   [model.Du,model.Dv,model.Dw] #Diffusion Coefficients
    epsilonv        =   [model.epsilon,1.0,model.epsilonp]
    q               =   model.q
    f               =   model.f
    phi             =   model.phi
    @tturbo @. Q[1] +=   (1/epsilonv[1])*(u[1]*(1-u[1])-u[3]*(u[1]-q))
    Q[2]            +=   u[1]-u[2]
    @tturbo @. Q[3] +=   (1/epsilonv[3])*(phi+f*u[2]-u[3]*(u[1]+q))
    
    if ComputeJ
       
        @tturbo @. dQ_du[1,1]   +=   (1/epsilonv[1])*(1-2*u[1]-u[3])
        @tturbo @. dQ_du[1,3]   +=   (1/epsilonv[1])*(q-u[1])
        @tturbo @. dQ_du[2,1]   +=   1.0
        @tturbo @. dQ_du[2,2]   +=   -1.0
        @tturbo @. dQ_du[3,1]   +=   (-1/epsilonv[3])*u[3]
        @tturbo @. dQ_du[3,2]   +=   f/epsilonv[3]
        @tturbo @. dQ_du[3,3]   +=   -1/epsilonv[3]*(u[1]+q)
       
        # Deltat_CFL_reac Computation
       
        dQ_du_inf = zeros(size(u[1]))
       
        for i=1:nSpecies

            sum_i = zeros(size(u[1]))

            for j=1:nSpecies

               @tturbo @. sum_i += abs(dQ_du[i,j])

            end

            @tturbo @. dQ_du_inf = max(dQ_du_inf,sum_i)

        end
    
        Deltat_CFL_reac =  @tturbo @. 1/dQ_du_inf
       
    else 
       
        Deltat_CFL_reac = NaN
        
    end
    
    return Deltat_CFL_reac
    
end

function NonlinearFlux!(model::NCD,u::Vector{Matrix{Float64}},du::Matrix{Matrix{Float64}},flux::Matrix{Matrix{Float64}},dflux_du::Array{Matrix{Float64},3},dflux_dgradu::Array{Matrix{Float64},4},a::Vector{Matrix{Float64}},da_du::Vector{Matrix{Float64}},DT::Vector{Matrix{Float64}},dDT_du::Vector{Matrix{Float64}},ComputeJ::Bool)

    nSpecies = 1

    ## Diffusion Flux

    for alpha = 1:nSpecies, i= 1:2

        @tturbo @. flux[alpha,i]   +=   -DT[1]*du[alpha,i]

    end
    
    ## Convection Flux
    
    for alpha = 1:nSpecies, i= 1:2

        @tturbo @. flux[alpha,i]   +=   a[i]*u[alpha]

    end
    
    if ComputeJ
       
       #dfconv/du
       
       @tturbo @. dflux_du[1,1,1]   +=   a[1] + da_du[1]*u[1]
       @tturbo @. dflux_du[1,2,1]   +=   a[2] + da_du[2]*u[1]
       
       #dfdiff/du
       
       @tturbo @. dflux_du[1,1,1]   +=   -du[1,1]*dDT_du[1]
       @tturbo @. dflux_du[1,2,1]   +=   -du[1,2]*dDT_du[1]
       
       #dfdiff/dfdiffdgradu
       
       @tturbo @. dflux_dgradu[1,1,1,1]   += -DT[1]
       @tturbo @. dflux_dgradu[1,2,1,2]   += -DT[1]
        
    end
    
end

function DiffusiveFlux!(model::NCD,DT::Vector{Matrix{Float64}},dDT_du::Vector{Matrix{Float64}},u::Vector{Matrix{Float64}},du::Matrix{Matrix{Float64}},ComputeJ::Bool,flux::Matrix{Matrix{Float64}},dflux_du::Array{Matrix{Float64},3},dflux_dgradu::Array{Matrix{Float64},4})

    nSpecies = 1


    for alpha = 1:nSpecies, i= 1:2

        @tturbo @. flux[alpha,i]   +=   -DT[1]*du[alpha,i]

    end

    if ComputeJ

        #dfdiff/du

        @tturbo @. dflux_du[1,1,1]   +=   -du[1,1]*dDT_du[1]
        @tturbo @. dflux_du[1,2,1]   +=   -du[1,2]*dDT_du[1]

        #dfdiff/dfdiffdgradu

        @tturbo @. dflux_dgradu[1,1,1,1]   += -DT[1]
        @tturbo @. dflux_dgradu[1,2,1,2]   += -DT[1]

    end



    return

end

#Add monolithic diffusion:
#   f_Ij = -tau d_j u_I

function epsilonFlux!(model::Oregonator, tau::MFloat, duB::Matrix{MFloat}, 
    ComputeJ::Bool, 
    fB::Matrix{MFloat}, dfB_dduB::Array{MFloat,4};
    IIv::Vector{Int}=Vector{Int}(1:model.nVars)) where MFloat<:Matrix{Float64}
   
    for II=IIv, jj=1:2
        @avxt @. fB[II,jj]              -= tau*duB[II,jj]
    end
    
    if ComputeJ
        for II=IIv, jj=1:2
            @avxt @. dfB_dduB[II,jj,II,jj]  -= tau
        end
    end
    
    return
    
end

function SSDiffusiveFlux!(model::NCD, DTSS::Vector{MFloat},
                        du::Matrix{MFloat},
                        ComputeJ::Bool, flux::Matrix{MFloat}, dflux_du::Array{MFloat,3},
                        dflux_dgradu::Array{MFloat,4}) where MFloat<:Matrix{Float64}

     flux[1,1]  .-= DTSS[1].*du[1,1]
     flux[1,2]  .-= DTSS[1].*du[1,2]
    
    if ComputeJ #&& false
        dflux_dgradu[1,1,1,1]  .-= DTSS[1]
        dflux_dgradu[1,2,1,2]  .-= DTSS[1]
    end

    return

end
