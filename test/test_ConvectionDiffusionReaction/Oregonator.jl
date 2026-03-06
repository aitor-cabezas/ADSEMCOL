include("test_ConvectionDiffusionReaction.jl")

function Oregonator_test(;hp::Float64=1.0, FesOrder::Int64=5, tf::Float64=1.0, TMSName::String= "RoW",RKMethod::String="Ascher3", RoWMethod::String="ROS34PRW",PlotVars::Vector{String}= ["u","v","w"], PlotCode::Vector{String}=fill("nodes", length(PlotVars)), PlotFig::Bool=true, Deltat_SaveFig::Float64=0.01, SaveFig::Bool=false, wFig::Float64=9.50, hFig::Float64=6.50, mFig::Int=max(1,length(PlotCode)), nFig::Int=Int(ceil(length(PlotCode)/mFig)), Nt_SaveFig::Int=typemax(Int), cmap::String="jet",SC::Int=0, CSS::Float64=0.1, CDC::Float64=5.0, CFLa::Float64=1.0, phi::Float64=0.0025,epsilon::Float64= 1/8,epsilonp::Float64= 1/720,Du::Float64=1.0,Dw::Float64=1.12,q::Float64 = 0.002,f::Float64=1.8,A::Float64=1e-3,sigma::Float64=5.0,  Deltat0::Float64=1e-4,AMA_MaxIter::Int=200,TolS::Float64=1e-5,TolT::Float64=1e-3,AMA_SizeOrder::Int=FesOrder,AMA_AnisoOrder::Int=2,AMA_ProjN::Int=1,AMA_ProjOrder::Int=0,SpaceAdapt::Bool=true, TimeAdapt::Bool=true,SaveRes::Bool=false, Nt_SaveRes::Int=typemax(Int), Deltat_SaveRes::Float64=0.01)
    
    
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
    model.Du                = Du
    model.Dw                = Dw
    
    
    #Mesh:
    MeshFile                = "$(@__DIR__)/../../temp/Oregonator_SC$(SC).geo"
    NX                      = Int(ceil(2*3.1416/(hp*FesOrder)))
    NY                      = Int(ceil(2*3.1416/(hp*FesOrder)))
    x1                      = -3.1416
    x2                      = 3.1416
    y1                      = -3.1416
    y2                      = 3.1416
    TrMesh_Rectangle_Create!(MeshFile, x1, x2, NX, y1, y2, NY)
    
    #Load LIRKHyp solver structure with default data. Modify the default data if necessary:
    solver                  = LIRKHyp_Start(model)
    solver.ProblemName      = "Oregonator"
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
    
    #Boundary conditions (q_alpha = 0 for the whole boundary):
    function q_alpha(t::Float64,x::Vector{Matrix{Float64}})
        
        q   = [zero.(x[1]), zero.(x[1]), zero.(x[1])]
        
        return q
        
    end
    
    BC_Neumann        = Neumann(FWt11((t,x)->q_alpha(t,x)))
    
    #Set initial condition:
    
    function u0_Oregonator(x::Vector{Matrix{Float64}})
        
#         xc                  =   (x2-x1)/2
#         yc                  =   (y2-y1)/2
        xr                  =   @view x[1][:]
        yr                  =   @view x[2][:]
#         rxy                 =   similar(xr) 
#         @tturbo @. rxy      =   sqrt((xr-xc)^2 + (yr-yc)^2)
        
        #Anderson's method to obtain the equilibrium solution u*
        
        function J0(u0::Float64) #Jacobian of the residual function
            u0v           = fill(u0,length(x[1]))
            diag_value    = 1 - 2*u0 + (((f*q-phi-2*f*u0)*(u0+q)-(q*phi+f*q*u0-phi*u0-f*u0*u0))/
                            ((u0+q)*(u0+q)))
            return diag_value,u0v
        end
        
        diagJ0_value,u0v   =   J0(0.1)
        
        function gfun!(u::Vector{Float64},g::Vector{Float64}) #Preconditioned residual computation
            R              = zeros(length(u0v))
            @tturbo @. R   = u*(1-u) - ((phi+f*u)/(u+q))*(u-q) #Residual
            g              .= diagJ0_value.\R  #Preconditioned Residual
            return 1 #flag
            
        end
        
        ueq, ch     = Anderson(FW_NLS((u,g)->gfun!(u,g)), u0v, memory=50, AbsTolG=1e-12, MaxIter=100, Display="notify") #FW_NLS is a wrapper that takes the function (y,g)->gfun!(y,g) and adapts the interface that Anderson's method needs for working as a NonLinear Solver. In this way, Anderson doesn't need how you compute the residual exactly, only recieves and object(FW_NLS) that responds with whatever it needs.  
        #Display = "none", "iter", "final", "notify"
        if ch.flag<=0 #ch=convergence history
            @warn "Nonlinear solver did not converge"
        end
        u_in             =   zeros(length(x[1]))
        v_in             =   zeros(length(x[1]))
        w_in             =   zeros(length(x[1]))
        @tturbo @. u_in  +=  ueq + A*sin(xr)*sin(yr)
#         @tturbo @. u_in  +=  2*A + A*sin(xr)*sin(yr)
        @tturbo @. v_in  +=  u_in
        @tturbo @. w_in  +=  (phi+f*v_in)/(u_in+q)
        
        u_in      = reshape(u_in, size(x[1]))
        v_in      = reshape(v_in, size(x[1]))
        w_in      = reshape(w_in, size(x[1]))
        
        return [u_in,v_in,w_in]
        
    end
    
    solver.u0fun        = FW11((x) -> u0_Oregonator(x)) 
    
    #Set boundary conditions:
    solver.BC           = [BCW(BC_Neumann), BCW(BC_Neumann), BCW(BC_Neumann), BCW(BC_Neumann)]
    
    #-----------------------------------------------------------------------------
    #INITIAL CONDITION:
    
    #Compute initial condition:
    ConvFlag            = LIRKHyp_InitialCondition!(solver)
#     CheckJacobian(solver, Plot_dQ_du=true, Plot_df_dgradu=true)
#     BC_CheckJacobian(solver, 4, Plot_df_du=true, Plot_df_dgradu=true)
#     return


    #Function to plot solution:
    figv                = Vector{Figure}(undef,3)
    if PlotFig
        figv[1]         = PyPlotSubPlots(mFig, nFig, w=wFig, h=hFig, left=0.9, right=0.4, bottom=1.1, top=1.0)
        for ii=2:length(figv)
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
            #Loop plot variables:
            for ii=1:length(PlotVars)
                PyPlot.subplot(mFig, nFig, ii)
                PyPlot.cla()
                v_plot  = PlotContourOregonator(solver, solver.model, PlotVars[ii], delta=1e-4)
#                 PlotContour(solver.u[ii], solver.fes)
                title(latexstring(PlotVars[ii],
                                "; t^n=", sprintf1("%.2e", solver.t)),
                fontsize=10)
                println(PlotVars[ii], ": min=", minimum(v_plot), ", max=", maximum(v_plot))
            end
            if SaveFig
                savefig("$(VideosUbi)Oregonator_SC$(SC)_$(nb_SaveFig).png", dpi=400, pad_inches=0)
            end

            figure(figv[2].number)
            PyPlot.cla()
            PlotContour(solver.u[1], solver.fes)
            PlotMesh!(solver.mesh, color="w")
            title(latexstring("u", "; t^n=", sprintf1("%.2e", solver.t)),
            fontsize=10)
            if SaveFig
                savefig("$(VideosUbi)Oregonator_Mesh_SC$(SC)_$(nb_SaveFig).png", dpi=400, pad_inches=0)
            end

            figure(figv[3].number)
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
                savefig("$(VideosUbi)Oregonator_errors$(SC)_$(nb_SaveFig).png", dpi=400, pad_inches=0)
            end

            t_lastFig           += Deltat_SaveFig
            ct_SaveFig          = 0
            nb_SaveFig          += 1

        end
        return

    end

    PlotSol()

    #Function to save intermediate results:
    t_lastRes           = -Deltat_SaveRes
    ct_SaveRes          = 0
    nb_SaveRes          = 0

    model                   = Oregonator()
    model.CSS               = CSS
    model.f                 = f
    model.phi               = phi
    model.q                 = q
    model.epsilon           = epsilon
    model.epsilonp          = epsilonp


    function SaveSol()

        ct_SaveRes      += 1
        if SaveRes && ( solver.t-t_lastRes>=Deltat_SaveRes ||
            ct_SaveRes==Nt_SaveRes || solver.t==tf || solver.t==0.0 )
            save("$(ResUbi)LIRKHyp_SC$(SC)_$(nb_SaveRes).jld2", "StudyCase", "Oregonator",
                 "ConvFlag", ConvFlag, "solver", save(solver),
                 "epsilon", epsilon, "epsilonp", epsilonp, "q", q, "phi", phi, "f", f,
                 "Deltat0", Deltat0, "TolS", TolS, "TolT", TolT)
            t_lastRes   += Deltat_SaveRes
            ct_SaveRes  = 0
            nb_SaveRes  += 1
        end
        return

    end

    SaveSol()

    
    #-----------------------------------------------------------------------------
    #MARCH IN TIME:
    
    while solver.t<tf
    
        ConvFlag            = LIRKHyp_Step!(solver)
        if ConvFlag<=0
            break
        end
        
        PlotSol()
        SaveSol()
        
    end


    #Save results:
    if SaveRes
        save("$(ResUbi)LIRKHyp_SC$(SC)_1000.jld2", "StudyCase", "Oregonator",
             "ConvFlag", ConvFlag, "solver", save(solver) )
    end
    
end
