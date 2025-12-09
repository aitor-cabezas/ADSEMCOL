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

mutable struct Neumann <: BoundConds
    q               ::FWt11     #must return diffusive flux [q=-epsilon*du/dn]
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
    
    source!(model, u, _qp.Q, _qp.dQ_du, ComputeJ)

#     #CFL number:
#     hp_min              = _hmin(_qp.Integ2D.mesh)./_qp.FesOrder * ones(1, _qp.nqp)
#     D_max               = @mlv max(epsilon, nu, beta, kappa_rho_cv)
#     Deltat_CFL_lambda   = @. $minimum(hp_min/lambda)
#     Deltat_CFL_D        = @. $minimum(hp_min^2/D_max)
#     _qp.Deltat_CFL      = min(Deltat_CFL_lambda, Deltat_CFL_D)

    return

end
