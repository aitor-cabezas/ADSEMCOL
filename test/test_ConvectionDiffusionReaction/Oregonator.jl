include("test_ConvectionDiffusionReaction.jl")

function Oregonator(;hp::Float64=1.0, FesOrder::Int=5, tf::Float64=1.0, TMSName::String= "RoW",RKMethod::String="Ascher3", RoWMethod::String="ROS34PRW",PlotVars::Vector{String}=String[], PlotCode::Vector{String}=fill("nodes", length(PlotVars)),SaveFig::Bool=false, wFig::Float64=9.50, hFig::Float64=6.50, mFig::Int=max(1,length(PlotCode)), nFig::Int=Int(ceil(length(PlotCode)/mFig)), Nt_SaveFig::Int=typemax(Int), cmap::String="jet",SC::Int=0, CSS::Float64=0.1, CDC::Float64=5.0, CFLa::Float64=1.0, phi::Float64=0.0025,epsilon::Float64= 1/8,epsilonp::Float64= 1/720,q               ::Float64 = 0.002,f::Float64=1.8,A::Float64=0.5,sigma::Float64=5.0,  Deltat0::Float64=1e-4,AMA_MaxIter::Int=200,TolS::Float64=1e-5,TolT::Float64=1e-3,AMA_SizeOrder::Int=FesOrder,AMA_AnisoOrder::Int=2,AMA_ProjN::Int=1,AMA_ProjOrder::Int=0,SpaceAdapt::Bool=true, TimeAdapt::Bool=true,SaveRes::Bool=false, Nt_SaveRes::Int=typemax(Int), Deltat_SaveRes::Float64=0.01)
    
    
    #---------------------------------------------------------------------
    #PRE-PROCESS STAGE:
    
    #Define Oregonator Model and parameters of the model:
    model                   = Oregonator()
    model.CSS               = CSS
    model.f                 = f
    model.phi               = phi
    model.q                 = q
    model.epsilon           = epsilon
    model.epsilonp          = epsilonp
    
    
    #Mesh:
    MeshFile                = "$(@__DIR__)/../../temp/Oregonator_SC$(SC).geo"
    NX                      = Int(ceil(24.0/(hp*FesOrder)))
    NY                      = Int(ceil(14.0/(hp*FesOrder)))
    x1                      = -7.0
    x2                      = 17.0
    y1                      = -7.0
    y2                      = 7.0
    TrMesh_Rectangle_Create!(MeshFile, x1, x2, NX, y1, y2, NY)
    
    #Load LIRKHyp solver structure with default data. Modify the default data if necessary:
    solver                  = LIRKHyp_Start(model)
    solver.ProblemName      = "Oregonator"
    solver.SC               = SC
#     solver.MeshFile         = "$(MeshUbi)SmoothVortex/MeshCase$(MeshCase).geo"
    solver.MeshFile         = MeshFile
    solver.nBounds          = 4
    solver.FesOrder         = FesOrder
    solver.TMSName          = TMSName
    solver.RKMethod         = RKMethod
    solver.RoWMethod        = RoWMethod
    solver.Deltat0          = Deltat0
    solver.tf               = tf
    solver.AMA_MaxIter      = AMA_MaxIter
    solver.AMA_SizeOrder    = AMA_SizeOrder
    solver.AMA_AnisoOrder   = AMA_AnisoOrder
    solver.AMA_ProjN        = AMA_ProjN
    solver.AMA_ProjOrder    = AMA_ProjOrder
    solver.TolS_max         = TolS
    solver.TolS_min         = 0.01*TolS
    solver.TolT             = TolT
    solver.SpaceAdapt       = SpaceAdapt
    solver.TimeAdapt        = TimeAdapt
    
    #Set initial condition:
    
    function u0_Oregonator(x::Vector{Matrix{Float64}})
        
        xc                  =   (x2-x1)/2
        yc                  =   (y2-y1)/2
        @tturbo @.  rxy     =   sqrt((x[1]-xc)^2 + (x[2]-yc)^2)
        
        #Anderson's method to obtain the equilibrium solution u*
        
        function J0(u0::Float64) #Jacobian of the residual function
            u0v           = fill(u0,length(x[1]))
            diag_value    = 1 - 2*u0 + (((f*q-phi-2*f*u0)*(u0+q)-(q*phi+f*q*u0-phi*u0-f*u0*u0))/
                            ((u0+q)*(u0+q)))
                            
            diagJ0 = spdiagm(0 => fill(diag_value, length(x[1])))
            return diagJ0,u0v
        end
        
        J0_diag,u0v   =   J0(0.1)
        
        function gfun!(u::Vector{Float64},g::Vector{Float64}) #Preconditioned residual computation
            
            @tturbo @. R   = u*(1-u) - ((phi+f*u)/(u+q))*(u-q) #Residual
            g              .= J0_diag.\R  #Preconditioned Residual
            return 1 #flag
            
        end
        
        ueq, ch     = Anderson(FW_NLS((u,g)->gfun!(u,g)), u0v, memory=50, AbsTolG=1e-12, MaxIter=iter, Display="notify") #FW_NLS is a wrapper that takes the function (y,g)->gfun!(y,g) and adapts the interface that Anderson's method needs for working as a NonLinear Solver. In this way, Anderson doesn't need how you compute the residual exactly, only recieves and object(FW_NLS) that responds with whatever it needs.  
        #Display = "none", "iter", "final", "notify"
        if ch.flag<=0 #ch=convergence history
            @warn "Nonlinear solver did not converge"
        end
        
        @tturbo @. u_in  +=  ueq + A*exp(-(rxy*rxy)/(2*sigma*sigma))
        @tturbo @. v_in  +=  u_in
        @tturbo @. w_in  +=  (phi+f*v_in)/(u_in+q)
        
        return [u_in,v_in,w_in]
        
    end
    
    
    solver.u0fun        = FW11((x) -> u0_Oregonator(x)) 
    
    
    #-----------------------------------------------------------------------------
    #INITIAL CONDITION:
    
    #Compute initial condition:
    ConvFlag            = LIRKHyp_InitialCondition!(solver)
    CheckJacobian(solver, Plot_dQ_du=true, Plot_df_dgradu=true)
#     BC_CheckJacobian(solver, 4, Plot_df_du=true, Plot_df_dgradu=true)
    return
    
    
    
    
    
end
