include("test_ConvectionDiffusionReaction.jl")

function NonlinearDiffusion_test(;hp::Float64=1.0, FesOrder::Int64=5, tf::Float64=1.0, TMSName::String= "RoW",RKMethod::String="Ascher3", RoWMethod::String="ROS34PRW",  CSS::Float64=0.1, CDC::Float64=5.0, CFLa::Float64=1.0,CW::Float64=50.0,SC::Int64=0,
A::Float64=0.05, B::Float64= 0.001, Dt0::Float64= 0.01,omegat::Float64=1.0, Deltat0::Float64=1e-4,AMA_MaxIter::Int=200,TolS::Float64=1e-5,TolT::Float64=1e-3,AMA_SizeOrder::Int=FesOrder,AMA_AnisoOrder::Int=2,AMA_ProjN::Int=1,AMA_ProjOrder::Int=0,SpaceAdapt::Bool=true, TimeAdapt::Bool=true)



    #---------------------------------------------------------------------
    #PRE-PROCESS STAGE:

    #Define NonlinearDiffusion Model and parameters of the model:
    model                   = NCD()
    model.CSS               = CSS
    model.CW                = CW


    #Mesh:
    MeshFile                = "$(@__DIR__)/../../temp/NonlinearDiffusion$(SC).geo"
    NX                      = Int(ceil(50.0/(hp*FesOrder)))
    NY                      = Int(ceil(50.0/(hp*FesOrder)))
    x1                      = 0.0
    x2                      = 50.0
    y1                      = 0.0
    y2                      = 50.0
    TrMesh_Rectangle_Create!(MeshFile, x1, x2, NX, y1, y2, NY)

    #Load LIRKHyp solver structure with default data. Modify the default data if necessary:
    solver                  = LIRKHyp_Start(model)
    solver.ProblemName      = "NonlinearDiffusion"
    solver.SC               = SC
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


    function a(x::Vector{Matrix{Float64}},t::Float64,u::Vector{Matrix{Float64}})

        vx          =   @. (8*pi/25)*sin((pi*x[1])/25)*sin((pi*x[2])/25)
        vy          =   @. (8*pi/25)*cos((pi*x[1])/25)*cos((pi*x[2])/25)

        return [vx,vy]

    end

    function da_du(x::Vector{Matrix{Float64}},t::Float64,u::Vector{Matrix{Float64}})

        return [zeros(size(u[1])),zeros(size(u[1]))]

    end


    function dvdxv(x::Vector{Matrix{Float64}})

        dvdx          =   @. ((8*pi/25)^2)*cos((pi*x[1])/25)*sin((pi*x[2])/25)
        dvdy          =   @. -((8*pi/25)^2)*cos((pi*x[1])/25)*sin((pi*x[2])/25)

        return [dvdx,dvdy]

    end

    function Tfun(t::Float64)

        T = sin(omegat*t)

        return T

    end
    
    function dTdtfun(t::Float64)

        dTdt = omegat*cos(omegat*t)

        return dTdt

    end

    function Xfun(x::Vector{Matrix{Float64}})

        sigma  =  y2/6
        EXPON  =  @. ((x[1]-x2/2)^2 + (x[2]-y2/2)^2)/(sigma*sigma)
        X      =  @. x[1]*(x2-x[1])*x[2]*(y2-x[2])*exp(-EXPON)
        return X

    end
    
    function dXdxvfun(x::Vector{Matrix{Float64}})
        
        sigma  =  y2/6
        EXPON  =  @.  ((x[1]-x2/2)^2 + (x[2]-y2/2)^2)/(sigma*sigma)
        dXdx   =  @.  x[2]*(y2-x[2])*exp(-EXPON)*
                              ((x2-2*x[1])-(2/sigma^2)*(x2*x[1]^2-x[1]^3-x[1]*x2^2/2+x2/2*x[1]^2))
        dXdy   =  @.  x[1]*(x2-x[1])*exp(-EXPON)*
                              ((y2-2*x[2])-(2/sigma^2)*(y2*x[2]^2-x[2]^3-x[2]*y2^2/2+y2/2*x[2]^2))

        
        return [dXdx,dXdy]
        
    end
    
    function d2Xdxv2fun(x::Vector{Matrix{Float64}})
        
        sigma  =  y2/6
        EXPON  =  @.((x[1]-x2/2)^2 + (x[2]-y2/2)^2)/(sigma*sigma)
        Px     =  @.(x2-2*x[1])-(2/sigma^2)*(x2*x[1]^2-x[1]^3-x[1]*x2^2/2+x2/2*x[1]^2)
        Pprimx =  @. -2/sigma^2*(2*x[1]*x2 - 3*x[1]^2 -1/2*x2^2 + 2*x[1]*x2/2) - 2
        d2Xdx2 =  @. x[1]*(x2-x[1])*exp(-EXPON)*((-2/sigma^2)*(x[1]-x2/2)*Px+Pprimx)
        Py     =  @.(y2-2*x[2])-(2/sigma^2)*(y2*x[2]^2-x[2]^3-x[2]*y2^2/2+y2/2*x[2]^2)
        Pprimy =  @. -2/sigma^2*(2*x[2]*y2 - 3*x[2]^2 -1/2*y2^2 + 2*x[2]*y2/2) - 2
        d2Xdy2 =  @. x[2]*(y2-x[2])*exp(-EXPON)*((-2/sigma^2)*(x[2]-y2/2)*Py+Pprimy)
        
        
        return [d2Xdx2,d2Xdy2]
        
    end

    function H(x::Vector{Matrix{Float64}}, t::Float64)

        Xaux   =   Xfun(x)
        Taux   =   Tfun(t)

        Hfun   =   @. Xaux*Taux

        return [Hfun]

    end

    function H0(x::Vector{Matrix{Float64}})

        T0  = Tfun(0.0)
        X0  = Xfun(x)

        return [@. X0*T0]

    end

    function DT(x::Vector{Matrix{Float64}}, t::Float64, u::Vector{Matrix{Float64}})

        H0x = H0(x)[1]
        DTfun = @. DT0 + A*(u[1] - H0x) + B*(u[1] - H0x)^2
        return [DTfun]

    end

    function dDT_du(x::Vector{Matrix{Float64}}, t::Float64, u::Vector{Matrix{Float64}})

        H0x = H0(x)[1]
        dDT_dufun = @. A + 2*B*(u[1] - H0x)
        return [dDT_dufun]

    end

    function d2DT_du2(x::Vector{Matrix{Float64}}, t::Float64, u::Vector{Matrix{Float64}})

        return [fill(2*B, size(u[1]))]

    end

    function Q(x::Vector{Matrix{Float64}},t::Float64,u::Vector{Matrix{Float64}})

        X   = Xfun(x)
        T   = Tfun(t)
        dT  = dTdtfun(t)

        a1, a2      = a(x,t,u)
        da1, da2    = dvdxv(x)
        dX1, dX2    = dXdxvfun(x)
        d2X1, d2X2  = d2Xdxv2fun(x)

        DTm   = DT(x,t,u)[1]
        dDTm  = dDT_du(x,t,u)[1]

        dHdt     = @. X * dT
        dconvdxi = @. T*X*(da1 + da2) + a1*T*dX1 + a2*T*dX2
        ddiffdxi = @. DTm*T*(d2X1 + d2X2) + dDTm*T^2*(dX1^2 + dX2^2)

        return [@. dHdt + dconvdxi - ddiffdxi]

    end

    function dQ_du(x::Vector{Matrix{Float64}},t::Float64, u::Vector{Matrix{Float64}})

        T           = Tfun(t)
        dDTm        = dDT_du(x,t,u)[1]
        d2DTm       = d2DT_du2(x,t,u)[1]
        d2X1, d2X2  = d2Xdxv2fun(x)
        dX1, dX2    = dXdxvfun(x)

        dQdu = @. -dDTm*T*(d2X1 + d2X2) - T^2*(dX1^2 + dX2^2)*d2DTm
        return [dQdu]

    end


    ProblemData     = NCD(  FWt21((x,t,u)->a(x,t,u)),
                          FWt21((x,t,u)->da_du(x,t,u)),
                          FWt21((x,t,u)->DT(x,t,u)),
                          FWt21((x,t,u)->dDT_du(x,t,u)),
                          FWt21((x,t,u)->Q(x,t,u)),
                          FWt21((x,t,u)->dQ_du(x,t,u)) )

    #Set initial condition:

    solver.u0fun        = FW1((x) -> H0(x))
    
    # Set boundary conditions:
    
    function uDir(t::Float64, x::Vector{Matrix{Float64}})
        
        return H(x,t)
        
    end

    BC_Dirichlet        = Dirichlet(FWt11((t,x)->uDir(t,x)))
    solver.BC           = [BCW(BC_Dirichlet), BCW(BC_Dirichlet), BCW(BC_Dirichlet), BCW(BC_Dirichlet)]

    #-----------------------------------------------------------------------------
    #INITIAL CONDITION:

    #Compute initial condition:
    ConvFlag            = LIRKHyp_InitialCondition!(solver)
        CheckJacobian(solver, Plot_dQ_du=true, Plot_df_dgradu=true)
        BC_CheckJacobian(solver, 4, Plot_df_du=true, Plot_df_dgradu=true)
        return
    
    
end
    
    
    
