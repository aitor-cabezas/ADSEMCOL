#-------------------------------------------------------------------------------
#AUXILIARY FUNCTIONS FOR DOMAIN INTEGRALS:
#Update flux terms with contribution of hyperbolic terms. 
#This function receives several matrices: u1,...,uN,p. Then, updates flux_qp,
#which is a nVars x 2 matrix of Matrix{Float64} with the corresponding values of 
#f_{Ij}, I=1,..,,nVars, j=1:2.
#The function also updates dflux_du_qp, which is a nVars x 2 x nVars array with the values of
#df_{Ij}/du_J.


#Add monolithic diffusion:
#   f_Ij = -tau d_j u_I

function epsilonFlux!(model::GasModel, tau::MFloat, duB::Matrix{MFloat}, 
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



function OregonatorFlux!(model::Oregonator,du::Matrix{Matrix{Float64}},flux::Matrix{Matrix{Float64}},dflux_dgradu::Array{Matrix{Float64},4},ComputeJ::Bool)

    nSpecies        =   model.nSpecies
    D               =   [model.Du,model.Dv,model.Dw] #Diffusion Coefficients
    epsilonv        =   [model.epsilon,1.0,model.epsilonp]  

    ## Diffusion Flux

    for alpha = 1:nSpecies, i= 1:2

        flux[alpha,i]   +=   -(D[alpha]/epsilonv[alpha])*du[alpha,i]

    end
    
    if ComputeJ
            
            dflux_dgradu[1,1,1,1]   +=   -D[1]/epsilonv[1]
            dflux_dgradu[1,2,1,2]   +=   -D[1]/epsilonv[1]
            dflux_dgradu[3,1,3,1]   +=   -D[3]/epsilonv[3]
            dflux_dgradu[3,2,3,2]   +=   -D[3]/epsilonv[3]
        
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
    Q[1]            +=   (1/epsilonv[1])*(u[1]*(1-u[1])-u[3]*(u[1]-q))
    Q[2]            +=   u[1]-u[2]
    @tturbo @. Q[3] +=   (1/epsilonv[3])*(phi+f*u[2]-u[3]*(u[1]+q))
    
    if ComputeJ
       
       dQ_du[1,1]   +=   (1/epsilonv[1])*(1-2*u[1]-u[3])
       dQ_du[1,2]   +=   0.0
       dQ_du[1,3]   +=   (1/epsilonv[1])*(q-u[1])
       dQ_du[2,1]   +=   1.0
       dQ_du[2,2]   +=   -1.0
       dQ_du[2,3]   +=   0.0
       dQ_du[3,1]   +=   (-1/epsilonv[3])*u[3]
       dQ_du[3,2]   +=   f/epsilonv[3]
       dQ_du[3,3]   +=   -1/epsilonv[3]*(u[1]+q)
       
    end
    
end
