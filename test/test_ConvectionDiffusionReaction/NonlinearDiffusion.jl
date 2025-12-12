include("test_ConvectionDiffusionReaction.jl")

function NonlinearDiffusion_test(;hp::Float64=1.0, FesOrder::Int64=5, tf::Float64=1.0, TMSName::String= "RoW",RKMethod::String="Ascher3", RoWMethod::String="ROS34PRW",  CSS::Float64=0.1, CDC::Float64=5.0, CFLa::Float64=1.0, A::Float64=1.0, B::Float64= 1.0, Dt0::Float64= 1.0, delta::Float64=1e-6,xc::Float64=50.0,yc::Float64=50.0,Rc::Float64=15.0,omegat::Float64=1.0, Deltat0::Float64=1e-4,AMA_MaxIter::Int=200,TolS::Float64=1e-5,TolT::Float64=1e-3,AMA_SizeOrder::Int=FesOrder,AMA_AnisoOrder::Int=2,AMA_ProjN::Int=1,AMA_ProjOrder::Int=0,SpaceAdapt::Bool=true, TimeAdapt::Bool=true)

    #---------------------------------------------------------------------
    #PRE-PROCESS STAGE:

    #Define Oregonator Model and parameters of the model:
    model                   = NonlinearDiffusion()
    model.CSS               = CSS
    model.A                 = A
    model.B                 = B
    model.DT0               = DT0
    model.delta             = delta


    #Mesh:
    MeshFile                = "$(@__DIR__)/../../temp/Oregonator_SC$(SC).geo"
    NX                      = Int(ceil(100.0/(hp*FesOrder)))
    NY                      = Int(ceil(100.0/(hp*FesOrder)))
    x1                      = 25.0
    x2                      = 75.0
    y1                      = 12.5
    y2                      = 87.5
    TrMesh_Rectangle_Create!(MeshFile, x1, x2, NX, y1, y2, NY)

    #Load LIRKHyp solver structure with default data. Modify the default data if necessary:
    solver                  = LIRKHyp_Start(model)
    solver.ProblemName      = "ConvectionDiffusion"
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


    function Xfun(x::Vector{Matrix{Float64}})

        rxy   = @tturbo @. sqrt((x[1]-xc)*(x[1]-xc) + (x[2]-yc)*(x[2]-yc)+delta)
        X     = similar(rxy)

        @tturbo for i = eachindex(rxy)
            if rxy[i] < Rc
                X[i] = 1 - rxy[i]/Rc
            else
                X[i] = 0.0
            end
        end

            return X

    end

    function Tfun(t::Float64)

        T = exp(omegat*t)

        return T

    end

    function Htheor(X::function,T::function,x::Vector{Matrix{Float64}},t::Float64)

        H   =   @tturbo @. X(x)*T(t)

        return H

    end


    #Set initial condition:

    solver.u0fun        = FW11((x) -> Xfun(x))
    
    # Set boundary conditions:
    
    function uDir(t::Float64, x::Vector{Matrix{Float64}})
        
        u_Dir   = @tturbo @. 0.0*x[1]
        
        return u_Dir 
        
    end

    BC_Dirichlet        = Dirichlet(FWt11((t,x)->uDir(t,x)))
    solver.BC           = [BCW(BC_Dirichlet), BCW(BC_Dirichlet ), BCW(BC_Dirichlet), BCW(BC_Dirichlet)]
    
    
    
    
    
    
