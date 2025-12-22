include("../src_LIRKHyp/LIRKHyp.jl")

#------------------------------------------------------------------
#STRUCTURES WITH CONSTITUTIVE MODELS AND BOUNDARY CONDITIONS:

abstract type OregonatorModel <: ConstModels end

Base.@kwdef mutable struct Oregonator <: OregonatorModel

    #Model's characteristic fields:
    f               ::Float64           = 1.8
    epsilon         ::Float64           = 1/8
    epsilonp        ::Float64           = 1/720
    q               ::Float64           = 0.002
    Du              ::Float64           = 1.0
    Dv              ::Float64           = 0.0
    Dw              ::Float64           = 1.12
    phi             ::Float64           = 0.0025
    nSpecies        ::Int64             = 3
    CSS             ::Float64           = 0.1   #Subgrid stabilization
    CW              ::Float64           = 50.0  #Boundary penalty (50.0-200.0 for IIPG)

    #Mandatory fields:
    nVars           ::Int               = 3

end

abstract type ConvectionDiffusionModel <: ConstModels end


mutable struct NCD <: ConvectionDiffusionModel

    #Model's characteristic fields. These functions receive (t,[x1,x2],[u]) and return
    a               ::FWt21                     #Returns velocity [a1, a2]
    epsilon         ::FWt21                     #Returns viscosity coefficient [epsilon]
    Q               ::FWt21                     #Returns source [Q]
    A               ::Float64           
    B               ::Float64          
    DT0             ::Float64           
    
    #Functions to compute the jacobians:
    da_du           ::FWt21
    depsilon_du     ::FWt21
    dQ_du           ::FWt21
    
    #Stabilization variables:
    CSS             ::Float64   #Subgrid stabilization
    CW              ::Float64   #Boundary penalty (50.0-200.0 for IIPG)
    
    #Mandatory fields:
    nVars           ::Int                 
    
    NCD()           = new()
    
end

Base.@kwdef mutable struct NonlinearDiffusion <: ConvectionDiffusionModel

    #Model's characteristic fields:
    A               ::Float64           = 1.0
    B               ::Float64           = 1.0
    DT0             ::Float64           = 1.0
    delta           ::Float64           = 1e-6
    nSpecies        ::Int64             = 1
    CSS             ::Float64           = 0.1   #Subgrid stabilization
    CW              ::Float64           = 50.0  #Boundary penalty (50.0-200.0 for IIPG)

    #Mandatory fields:
    nVars           ::Int               = 1
    
    #Dependent variables. NOTE: DepVars contains variables to be evaluated when
    #Jacobian is not necessary. DepVarsJ contains variables to be evaluated when
    #Jacobian is to be computed. Variables in DepVars and DepVarsJ must be sorted in 
    #the same way.
    
    DepVarsJ         ::Vector{String}    = ["vx", "vy", "DT"]

end

#-------------------------------------------------------------------------------
# Boundary Conditions

mutable struct Dirichlet <: BoundConds
    uDir            ::FWt11     #must return Dirichlet condition at the boundary [u]
end

mutable struct Neumann <: BoundConds
    q               ::FWt11     #must return diffusive flux [q=-epsilon*du/dn]
end

#--------------------------------------------------------------------------------
#Auxiliary Functions



#Return index corresponding to dependent variable "var":
function DepVarIndex(model::ConvectionDiffusionModel, var::String)
    return findfirst(model.DepVarsJ.==var)
end

function DepVars(model::NonlinearDiffusion, t::Float64, x::Vector{Matrix{Float64}},
                 u::Vector{Matrix{Float64}}, vout::Vector{String})
        
        nSpecies    =   model.nSpecies
        A           =   model.A
        B           =   model.B
        DT0         =   model.DT0
            
        DT          =   @tturbo @. DT0 + A*(u[alpha]) + B*(u[alpha]*u[alpha])
        vx          =   @tturbo @. (8*pi/25)*sin((pi*x[1])/25)*sin((pi*x[2])/25)
        vy          =   @tturbo @. (8*pi/25)*cos((pi*x[1])/25)*cos((pi*x[2])/25)

        nout        = length(vout)
        xout        = Vector{Matrix{Float64}}(undef, nout)
        
        for ivar in eachindex(vout)
            
            vble    = vout[ivar]
            
            if vble=="DT"
                
                    xout[ivar]      = DT
                    
            elseif vble=="vx"
                
                    xout[ivar]      = vx
                    
            elseif vble=="vy"
                
                    xout[ivar]      = vy

            else
                
                error("Variable $(vble) not supported")
                
            end  
        
        end
        
        return xout
            
            
end




#-------------------------------------------------------------------------------
#MANDATORY FUNCTIONS:

include("../src_ConvectionDiffusionReaction/ConvectionDiffusionReaction_fluxes.jl")
include("../src_ConvectionDiffusionReaction/ConvectionDiffusionReaction_BC.jl")

#Compute normalization factors from solution. Mass matrix has already been computed.

function nFactsCompute!(solver::SolverData{<:OregonatorModel})

    #Normalization factors:
    solver.nFacts        .= 1.0

    return

end

function nFactsCompute!(solver::SolverData{<:ConvectionDiffusionModel})

    #Normalization factors:
    solver.nFacts        .= 1.0

    return

end

#Function to evaluate flux and source terms at quadrature nodes:

function FluxSource!(model::Oregonator, _qp::TrIntVars, ComputeJ::Bool)

    t               = _qp.t
    x               = _qp.x
    u               = _qp.u
    du              = _qp.gradu
    duB             = _qp.graduB
    
    #Terms due to diffusion flux
    OregonatorFlux!(model, du, _qp.f, _qp.df_dgradu, ComputeJ)
    
    #Subgrid stabilization - monolithic diffusion:
    lambda          = 0.0
    #     h_Elems         = _hElems(_qp.Integ2D.mesh)
    A_Elems         = areas(_qp.Integ2D.mesh)
    h_Elems         = @tturbo @. sqrt(A_Elems)
    hp              = h_Elems./_qp.FesOrder * ones(1, _qp.nqp)
    tau             = @mlv model.CSS*lambda*hp
    epsilonFlux!(model, tau, duB, ComputeJ, _qp.fB, _qp.dfB_dgraduB)

    #Source terms:
    
    Deltat_CFL_reac = source!(model, u, _qp.Q, _qp.dQ_du, ComputeJ)

    #CFL number:
    hp_min              = _hmin(_qp.Integ2D.mesh)./_qp.FesOrder * ones(1, _qp.nqp)
    D_max               = max(model.Du,model.Dw)
    Deltat_CFL_D        = @. $minimum(hp_min^2/D_max)
    Deltat_CFL_reac     = minimum(Deltat_CFL_reac)
    _qp.Deltat_CFL      = min(Inf, Deltat_CFL_D,Deltat_CFL_reac)

    return

end



function FluxSource!(model::NonlinearDiffusion, _qp::TrIntVars, ComputeJ::Bool)

    t               = _qp.t
    x               = _qp.x
    u               = _qp.u
    du              = _qp.gradu
    duB             = _qp.graduB

    #Terms due to convection-diffusion flux
    NonlinearDiffusionFlux!(model, u, du, _qp.f, _qp.df_du, _qp.df_dgradu, ComputeJ)

    #Subgrid stabilization - monolithic diffusion:
    lambda          = 0.0
    #     h_Elems         = _hElems(_qp.Integ2D.mesh)
    A_Elems         = areas(_qp.Integ2D.mesh)
    h_Elems         = @tturbo @. sqrt(A_Elems)
    hp              = h_Elems./_qp.FesOrder * ones(1, _qp.nqp)
    tau             = @mlv model.CSS*lambda*hp
    epsilonFlux!(model, tau, duB, ComputeJ, _qp.fB, _qp.dfB_dgraduB)

    #Source terms:

    source!(model, u, _qp.Q, _qp.dQ_du,_qp.dQ_dgradu, ComputeJ)

    #     #CFL number:
    #     hp_min              = _hmin(_qp.Integ2D.mesh)./_qp.FesOrder * ones(1, _qp.nqp)
    #     D_max               = @mlv max(epsilon, nu, beta, kappa_rho_cv)
    #     Deltat_CFL_lambda   = @. $minimum(hp_min/lambda)
    #     Deltat_CFL_D        = @. $minimum(hp_min^2/D_max)
    #     _qp.Deltat_CFL      = min(Deltat_CFL_lambda, Deltat_CFL_D)

    return

end


