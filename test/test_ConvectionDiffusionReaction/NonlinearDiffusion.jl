include("test_ConvectionDiffusionReaction.jl")

function NonlinearDiffusion_test(;hp::Float64=0.01, FesOrder::Int64=5, tf::Float64=2.5, TMSName::String= "RoW",RKMethod::String="Ascher3", RoWMethod::String="ROS34PRW",  CSS::Float64=0.1, CDC::Float64=5.0, CFLa::Float64=1.0,CW::Float64=50.0,SC::Int64=0,
A::Float64=0.0, B::Float64= 0.001, DT0::Float64= 0.0005,omegat::Float64=1.0, Lx::Float64 = 1.0, Ly::Float64=1.0,
PlotFig::Bool=true, Deltat_SaveFig::Float64=0.01, SaveFig::Bool=false, Nt_SaveFig::Int=typemax(Int),
SaveRes::Bool=false, Nt_SaveRes::Int=typemax(Int), Deltat_SaveRes::Float64=0.01,
Deltat0::Float64=1e-4,AMA_MaxIter::Int=200,TolS::Float64=1e-5,TolT::Float64=1e-3,AMA_SizeOrder::Int=FesOrder,AMA_AnisoOrder::Int=2,AMA_ProjN::Int=1,AMA_ProjOrder::Int=0,SpaceAdapt::Bool=true, TimeAdapt::Bool=true)

    #---------------------------------------------------------------------
    #PROBLEM DATA:
    
    function a(t::Float64,x::Vector{Matrix{Float64}},u::Vector{Matrix{Float64}})

        vx          =   @. (0.5/(Lx/2))*sin((pi*x[1])/(Lx/2))*sin((pi*x[2])/(Lx/2))
        vy          =   @. (0.5/(Ly/2))*cos((pi*x[1])/(Ly/2))*cos((pi*x[2])/(Ly/2))

        return [vx,vy]

    end

    function da_du(t::Float64,x::Vector{Matrix{Float64}},u::Vector{Matrix{Float64}})

        return [zeros(size(u[1])),zeros(size(u[1]))]

    end


    function dvdxv(x::Vector{Matrix{Float64}})

        dvdx          =   @. 8*(pi/(Lx/2))^2*cos((pi*x[1])/(Lx/2))*sin((pi*x[2])/(Lx/2))
        dvdy          =   @. -8*(pi/(Ly/2))^2*cos((pi*x[1])/(Ly/2))*sin((pi*x[2])/(Ly/2))

        return [dvdx,dvdy]

    end

    function Tfun(t::Float64)

        T = 2.0 + sin(omegat*t)

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
        EXPON  =  @.((x[1]-x2/2).^2 + (x[2]-y2/2).^2)/(sigma*sigma)
        Px     =  @.(x2-2*x[1])-(2/sigma^2)*(x2*x[1].^2-x[1].^3-x[1]*x2^2/2+x2/2*x[1].^2)
        Pprimx =  @. -2/sigma^2*(2*x[1]*x2 - 3*x[1].^2 -1/2*x2^2 + 2*x[1]*x2/2) - 2
        d2Xdx2 =  @. x[1]*(x2-x[1])*exp(-EXPON)*((-2/sigma^2)*(x[1]-x2/2)*Px+Pprimx)
        Py     =  @.(y2-2*x[2])-(2/sigma^2)*(y2*x[2].^2-x[2].^3-x[2]*y2^2/2+y2/2*x[2].^2)
        Pprimy =  @. -2/sigma^2*(2*x[2]*y2 - 3*x[2].^2 -1/2*y2^2 + 2*x[2]*y2/2) - 2
        d2Xdy2 =  @. x[2]*(y2-x[2])*exp(-EXPON)*((-2/sigma^2)*(x[2]-y2/2)*Py+Pprimy)
        
        
        return [d2Xdx2,d2Xdy2]
        
    end

    function H(t::Float64, x::Vector{Matrix{Float64}})

        Xaux   =   Xfun(x)
        Taux   =   Tfun(t)

        Hfun   =   @. Xaux*Taux

        return [Hfun]

    end

    function H0(x::Vector{Matrix{Float64}})

        T0  = Tfun(0.0)
        X0  = Xfun(x)

        return [@.X0*T0]

    end

    function DT(t::Float64,x::Vector{Matrix{Float64}},u::Vector{Matrix{Float64}})

        H0x = H0(x)[1]
        DTfun = @. DT0 + B*(u[1] - H0x)^2
        return [DTfun]

    end

    function dDT_du(t::Float64,x::Vector{Matrix{Float64}},u::Vector{Matrix{Float64}})

        H0x = H0(x)[1]
        dDT_dufun = @. 2*B*(u[1] - H0x)
        return [dDT_dufun]

    end

    function d2DT_du2(t::Float64,x::Vector{Matrix{Float64}},u::Vector{Matrix{Float64}})

        return [fill(2*B, size(u[1]))]

    end

    function Q(t::Float64,x::Vector{Matrix{Float64}},u::Vector{Matrix{Float64}})

        X   = Xfun(x)
        T   = Tfun(t)
        dT  = dTdtfun(t)

        a1, a2      = a(t,x,u)
        da1, da2    = dvdxv(x)
        dX1, dX2    = dXdxvfun(x)
        d2X1, d2X2  = d2Xdxv2fun(x)

        DTm   = DT(t,x,u)[1]
        dDTm  = dDT_du(t,x,u)[1]

        dHdt     = @. X * dT
        dconvdxi = @. T*X*(da1 + da2) + a1*T*dX1 + a2*T*dX2
        ddiffdxi = @. DTm*T*(d2X1 + d2X2) + dDTm*T^2*(dX1^2 + dX2^2)

        return [@. 0*dHdt + 0*dconvdxi - 0*ddiffdxi]

    end

    function dQ_du(t::Float64,x::Vector{Matrix{Float64}},u::Vector{Matrix{Float64}})

        T           = Tfun(t)
        dDTm        = dDT_du(t,x,u)[1]
        d2DTm       = d2DT_du2(t,x,u)[1]
        d2X1, d2X2  = d2Xdxv2fun(x)
        dX1, dX2    = dXdxvfun(x)

        dQdu = @. 0*(-dDTm*T*(d2X1 + d2X2) - T^2*(dX1^2 + dX2^2)*d2DTm)
        return [dQdu]

    end
    
    # Boundary conditions:
    
    function uDir(t::Float64, x::Vector{Matrix{Float64}})
        
        return H(t,x)
        
    end

    BC_Dirichlet        = Dirichlet(FWt11((t,x)->uDir(t,x)))


    #Structure with convection--NonlinearDiffusion problem data:
    
    ProblemData     = NCD(  FWt21((t,x,u)->a(t,x,u)),
                          FWt21((t,x,u)->da_du(t,x,u)),
                          FWt21((t,x,u)->DT(t,x,u)),
                          FWt21((t,x,u)->dDT_du(t,x,u)),
                          FWt21((t,x,u)->Q(t,x,u)),
                            FWt21((t,x,u)->dQ_du(t,x,u)) )
    
    #Mesh:
    MeshFile                = "$(@__DIR__)/../../temp/NonlinearDiffusion$(SC).geo"
    NX                      = Int(ceil(1.0/(hp*FesOrder)))
    NY                      = Int(ceil(1.0/(hp*FesOrder)))
    x1                      = 0.0
    x2                      = Lx
    y1                      = 0.0
    y2                      = Ly
    TrMesh_Rectangle_Create!(MeshFile, x1, x2, NX, y1, y2, NY)

    #Load LIRKHyp solver structure with default data. Modify the default data if necessary:
    solver                  = LIRKHyp_Start(ProblemData)
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
    
    # Set Boundary Conditions
    
    solver.BC           = [BCW(BC_Dirichlet), BCW(BC_Dirichlet), BCW(BC_Dirichlet), BCW(BC_Dirichlet)]
    
    #Set initial condition:

    solver.u0fun        = FW11((x) -> H0(x))

    #Compute initial condition:
    ConvFlag            = LIRKHyp_InitialCondition!(solver)
   #= 
    CheckJacobian(solver, Plot_dQ_du=true, Plot_df_dgradu=true,Plot_df_du=true,Plot_dQ_dgradu=true)
    BC_CheckJacobian(solver, 4, Plot_df_du=true, Plot_df_dgradu=true)
    return=#

    #Change TolT:
    if TolT==0.0
        TolT            = 0.01*solver.etaS
        solver.TolT     = TolT
    end
    
    #Compute Lq error:
    errLq,              = LqError(solver, FW11((x) -> H(solver.t, x)), q=2.0)
    hmean               = 2.0*sqrt(solver.Omega/solver.mesh.nElems/TrElem_Area)
    errL2L2             = errLq
    etaL2L2             = solver.etaS
    
    println("hmean=", sprintf1("%.2e", hmean), ", e_L2L2=", sprintf1("%.2e", errL2L2))
    
    #Function to plot solution:
    figv                = Vector{Figure}(undef,2)
    if PlotFig
        for ii=1:length(figv)
            figv[ii]    = figure()
        end
    end
    
    t_lastFig           = 0.0
    ct_SaveFig          = 0
    nb_SaveFig          = 0

    function PlotSol()

        ct_SaveFig      += 1

        if PlotFig && ( solver.t-t_lastFig>=Deltat_SaveFig ||
            ct_SaveFig==Nt_SaveFig || solver.t==tf || solver.t==0.0 )

            figure(figv[1].number)
            PyPlot.cla()
            PlotContour(solver.u[1], solver.fes)
            PlotMesh!(solver.mesh, color="w")
            title(latexstring("H", "; t^n=", sprintf1("%.2e", solver.t)),
            fontsize=10)
            if SaveFig
                savefig("$(VideosUbi)NonlinearDiffusion_Mesh$(SC)_$(nb_SaveFig).png", dpi=400, pad_inches=0)
            end

            figure(figv[2].number)
            PyPlot.cla()
            semilogy(solver.tv, solver.etaSv, ".-b")
            semilogy(solver.tv, solver.etaTv, ".-g")
            semilogy(solver.tv, solver.etaAv, ".-r")
            if true
                validv  = solver.validv .== 1
                semilogy(solver.tv[validv], solver.etaSv[validv], "sb")
                semilogy(solver.tv[validv], solver.etaTv[validv], "sg")
                semilogy(solver.tv[validv], solver.etaAv[validv], "sr")
            end
            legend(["space", "time", "algebraic"])
            xlabel(L"t")
            if SaveFig && solver.t==tf
                savefig("$(VideosUbi)$(SC)_$(nb_SaveFig).png", dpi=400, pad_inches=0)
            end

            t_lastFig           += Deltat_SaveFig
            ct_SaveFig          = 0
            nb_SaveFig          += 1

        end
        
        return

    end

    PlotSol()

 #-----------------------------------------------------------------------------
    #MARCH IN TIME:
    
    N=0
    
    while solver.t<tf
        N+=1
        ConvFlag            = LIRKHyp_Step!(solver)
#         if N ==20
#             CheckJacobian(solver, Plot_dQ_du=true, Plot_df_dgradu=true,Plot_df_du=true,Plot_dQ_dgradu=true)
#             BC_CheckJacobian(solver, 4, Plot_df_du=true, Plot_df_dgradu=true)
#             N=0
#         end
        if ConvFlag<=0
            break
        end
        PlotSol()
        
        #Compute Lq error:
        errLq,              = LqError(solver, FW11((x) -> H(solver.t, x)), q=2.0)
        hmean               = 2.0*sqrt(solver.Omega/solver.mesh.nElems/TrElem_Area)
        errL2L2             = errL2L2_(solver.t-solver.Deltat, errL2L2, solver.t, errLq)
        etaL2L2             = errL2L2_(solver.t-solver.Deltat, etaL2L2, solver.t, solver.etaS+solver.etaT)
        
        println("hmean=", sprintf1("%.2e", hmean), 
                ", Deltat_mean=", sprintf1("%.2e", solver.t/solver.Nt), 
                ", e_L2L2=", sprintf1("%.2e", errL2L2), 
                ", etaL2L2=", sprintf1("%.2e", etaL2L2))
        
#         SaveSol()
        
    end


    #Save results:
        if SaveRes
            save("$(ResUbi)LIRKHyp_SC$(SC)_1000.jld2", "StudyCase", "NonlinearDiffusion",
                "ConvFlag", ConvFlag, "solver", save(solver) )
        end
    
    
end
    
    
    
