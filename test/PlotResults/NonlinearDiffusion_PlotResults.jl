include("PlotResults.jl")

function CompareTMS_NonlinearDiffusion(StudyCase::String,nb::Int64,SCRef::Int64,nbRef::Int64; q::Real=2.0, 
    SaveFig::Bool=false, w::Float64=8.50, h::Float64=8.50)

    SCvv0   = NaN
    SCvv1   = NaN
    
    if StudyCase=="NCD1a"
    
        #TimeAdapt: YES
        SCvv1       = [3001:3004,
                       3005:3008,
                       3009:3012,
                       3013:3016,
                       3017:3020,
                       3021:3024]


    elseif StudyCase=="NCD1b"

        #TimeAdapt: No
        SCvv1       = [3025:3028,
                       3029:3032,
                       3033:3036,
                       3037:3040,
                       3041:3044,
                       3045:3048]
        
    elseif StudyCase=="NCD2a"
    
        #TimeAdapt: YES
        SCvv1       = [3050:3053,
                       3054:3057,
                       3058:3061,
                       3062:3065,
                       3066:3069,
                       3070:3073]
        
    elseif StudyCase=="NCD2b"

        #TimeAdapt: No
        SCvv1       = [3074:3077,
                       3078:3081,
                       3082:3085,
                       3086:3089,
                       3090:3093,
                       3094:3097]
                        
        
    
    elseif StudyCase=="NCD3a"
    
        #TimeAdapt: YES
        SCvv1       = [3099:3102,
                       3103:3106,
                       3107:3110,
                       3111:3114,
                       3115:3118,
                       3119:3122]
        
    elseif StudyCase=="NCD3b"
    
        #TimeAdapt: YES
        SCvv1       = [3123:3126,
                       3127:3130,
                       3131:3134,
                       3135:3138,
                       3139:3142,
                       3143:3146]
        
    elseif StudyCase=="NCD4a"
    
        #TimeAdapt: YES
        SCvv1       = [3148:3151,
                       3152:3155,
                       3156:3159,
                       3160:3163,
                       3164:3167,
                       3168:3171]
        
    end
        
    #------------------------------------------------------------
    
    Deltatvv1,tCPUvv1,CFLvv1,TIMethodNamevv1             = GetVbles(SCvv1, ["Deltat","tCPU","CFLmax", "TIMethodName"], nb=nb)
    errv1      = []
    errvv1     = Vector{Vector{Any}}()
    SCvv1r     = reduce(vcat,SCvv1)
    
    errv1 = RefErr_Lq(SCvv1r,nb,SCRef,nbRef;q=q)

    n = 1
    for i=1:length(SCvv1)

        k       =   length(SCvv1[i])
        serrv1  =   errv1[n:n+k-1]
        push!(errvv1,serrv1)
        n       =   n + k

    end

#     @show(errvv1)
#     @show(typeof(errvv1))
    EOCvv1                                                      = ExpOrderConv(Deltatvv1, errvv1)
    
    PyPlotFigure(w=w, h=h, bottom=1.5)
    colorv                      = PyPlotColors("jet2", length(SCvv1))
    leg                         = String[]
    for ii=1:length(SCvv1)
        loglog(Deltatvv1[ii], errvv1[ii], color=colorv[ii], linewidth=0.5, linestyle="solid", marker="s", markersize=3.5)
#         loglog(Deltatvv1[ii], etavv1[ii], color=colorv[ii], linewidth=0.5, linestyle="dashed", marker="s", markersize=3.5)

        push!(leg,TIMethodNamevv1[ii][1])
    end
    ylabel("err")
    xlabel(latexstring("\\tau"))
    legend(leg, fontsize=8)
    tick_params(axis="both", which="both", labelsize=TickSize)
    grid("on")
    if SaveFig
        savefig("$(FigUbi)NonlinearDiffusion.png", dpi=800, pad_inches=0)
    end
    
    
    PyPlotFigure(w=w, h=h, bottom=1.5)
    colorv                      = PyPlotColors("jet2", length(SCvv1))
    leg                         = String[]
    for ii=1:length(SCvv1)
        loglog(tCPUvv1[ii], errvv1[ii], color=colorv[ii], linewidth=0.5, linestyle="solid", marker="s", markersize=3.5)
#         loglog(Deltatvv1[ii], etavv1[ii], color=colorv[ii], linewidth=0.5, linestyle="dashed", marker="s", markersize=3.5)

        push!(leg,TIMethodNamevv1[ii][1])
    end
    ylabel("err")
    xlabel("tCPU")
    legend(leg, fontsize=8)
    tick_params(axis="both", which="both", labelsize=TickSize)
    grid("on")
    if SaveFig
        savefig("$(FigUbi)NonlinearDiffusion.png", dpi=800, pad_inches=0)
    end
           
    PyPlotFigure(w=w, h=h, bottom=1.5)
    colorv                      = PyPlotColors("jet2", length(SCvv1))
    leg                         = String[]
    for ii=1:length(SCvv1)
        loglog(CFLvv1[ii], errvv1[ii], color=colorv[ii], linewidth=0.5, linestyle="solid", marker="s", markersize=3.5)
#         loglog(Deltatvv1[ii], etavv1[ii], color=colorv[ii], linewidth=0.5, linestyle="dashed", marker="s", markersize=3.5)

        push!(leg,TIMethodNamevv1[ii][1])
    end
    xlabel(latexstring(GetString("CFLmax")))
    ylabel(latexstring(GetString("errL2L2")), rotation=0)
    tick_params(axis="both", which="both", labelsize=TickSize)
    if SaveFig
        savefig("$(FigUbi)NonlinearDiffusion.png", dpi=800, pad_inches=0)
    end
    
    display(EOCvv1)
    
    return
    
end


function Contour_NonlinearDiffusion(SC::Int, nb::Int; SaveFig::Bool=false, w::Float64=8.50, h::Float64=8.50, 
    PlotVars::Vector{String}=["u"], mFig::Int=2, nFig::Int=2)
    
    NCDModel        = NCD()
    FileName        = GetFileName(SC, nb)
    solver          = GetSolver(SC, nb)
    
    for ii=1:length(PlotVars)
        
        PyPlotFigure(w=w, h=h, bottom=1.5, top=1.0)
        
        #Numerical solution:
        PlotContour(solver, NCDModel, PlotVars[ii], delta=1e-4)
        PlotMesh!(SC, nb, color="w")
        title(latexstring(PlotVars[ii],"; t^n=", sprintf1("%.2e", solver.t)),fontsize=10)
        tick_params(axis="both", which="both", labelsize=TickSize)
        axis("off")
        
        if SaveFig
            savefig("$(FigUbi)SC$(SC)_Contour_$(PlotVars[ii]).png", dpi=800, pad_inches=0)
        end
    
    end
    
    return
    
end

