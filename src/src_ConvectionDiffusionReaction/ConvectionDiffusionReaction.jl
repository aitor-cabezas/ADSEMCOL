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

    #Model's characteristic fields (nonlinear diffusion).
    a               ::FWt21                       #Returns velocity [vx, vy]
    da_du           ::FWt21                       #Function to compute the jacobians
    DT              ::FWt21                       #Returns thermal diffusion DT(u)
    dDT_du          ::FWt21                       #Function to compute the jacobians
    Q               ::FWt11                       #Returns source term
    dQ_du           ::FWt11                       #Function to compute the jacobians
    
    #Stabilization variables:
    CSS             ::Float64   #Subgrid stabilization
    CW              ::Float64   #Boundary penalty (50.0-200.0 for IIPG)
    
    #Mandatory fields:
    nVars           ::Int                 
    
    NCD()           = new()
    
end

function NCD(a::FWt21,da_du::FWt21,DT::FWt21,dDT_du::FWt21,Q::FWt11,dQ_du::FWt11)

    NCDS              = NCD()
    NCDS.a            = a
    NCDS.da_du        = da_du
    NCDS.DT           = DT
    NCDS.dDT_du       = dDT_du
    NCDS.Q            = Q
    NCDS.dQ_du        = dQ_du
    NCDS.CSS          = 0.1
    NCDS.CW           = 50.0
    NCDS.nVars        = 1

    return NCDS

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

include("../src_ConvectionDiffusionReaction/ConvectionDiffusionReaction_fluxes.jl")
include("../src_ConvectionDiffusionReaction/ConvectionDiffusionReaction_BC.jl")

function DepVars(model::NCD, t::Float64, x::Vector{Matrix{Float64}}, u::Vector{Matrix{Float64}},vout::Vector{String})

    nout        = length(vout)
    xout        = Vector{Vector{Matrix{Float64}}}(undef,nout)
    for ivar in eachindex(vout)
        vble    = vout[ivar]
        if vble=="u"
            xout[ivar]  = [ copy(u[1]) ]
            elseif vble=="lambda_max"
            a           = model.a(t,x,u)
            da_du       = model.da_du(t,x,u)
            #ahat_i = d(a_i u)/du = da_i/du * u + a_i
            lambda      = @tturbo @. sqrt( (da_du[1]*u[1]+a[1])^2 + (da_du[2]*u[1]+a[2])^2 )
            xout[ivar]  = [ lambda ]
        else
            error("Variable $(vble) not supported")
        end
    end

    return xout

end





#-------------------------------------------------------------------------------
#MANDATORY FUNCTIONS:

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

#Function to evaluate flux and source terms at quadrature nodes:
function FluxSource!(model::NCD, _qp::TrIntVars, ComputeJ::Bool)

    t               = _qp.t
    x               = _qp.x
    u               = _qp.u
    du              = _qp.gradu
    duB             = _qp.graduB
    metric          = _qp.Integ2D.mesh.metric

    #Natural viscosity:
    a                       = model.a(t,x,u)
    da_du                   = Vector{Matrix{Float64}}(undef,2)
    lambda_max              = DepVars(model,t,x,u,["lambda_max"])[1][1]
    DT                      = model.DT(t,x,u)
    dDT_du                  = Vector{Matrix{Float64}}() 
    if ComputeJ
        da_du               = model.da_du(t,x,u)
        dDT_du              = model.dDT_du(t,x,u)
    end

    #Nonlinear convective and diffusive fluxes:

    NonlinearDiffusionFlux!(model, u, du, _qp.f, _qp.df_du, _qp.df_dgradu,a,DT,dDT_du,ComputeJ)

    #Evaluate subgrid stabilization flux:
    A_Elems             = areas(_qp.Integ2D.mesh)
    h_Elems             = @tturbo @. sqrt(A_Elems)
    hp                  = h_Elems./_qp.FesOrder * ones(1, _qp.nqp)
    DTSS                = [@tturbo @. model.CSS*lambda_max*hp]
    if ComputeJ
        dDT_du        = model.dDT_du(t,x,u)
        #         @avxt @. depsilon_du        = model.CSS*hp * (a[1]*da_du[1]+a[2]*da_du[2])/anorm
    end
    
    SSDiffusiveFlux!(model, DTSS, dDT_du, u, duB, ComputeJ,_qp.fB, _qp.dfB_du, _qp.dfB_dgraduB)

    #Evaluate source terms:
    _qp.Q[1]            .= model.Q(t,x)[1]
    if ComputeJ
        _qp.dQ_du[1]    .= model.dQ_du(t,x)[1]
    end

    #Deltat imposed by CFL=1 (do not use @avxt, it does not work well with $ symbol)
    hp_min              = _hmin(_qp.Integ2D.mesh)./_qp.FesOrder .* ones(1, _qp.nqp)
    Deltat_CFL_a        = minimum(hp_min ./ lambda_max)
    Deltat_CFL_DT       = minimum((hp_min.^2) ./DT[1])
    _qp.Deltat_CFL      = min(Deltat_CFL_a, Deltat_CFL_DT)

    return

end

