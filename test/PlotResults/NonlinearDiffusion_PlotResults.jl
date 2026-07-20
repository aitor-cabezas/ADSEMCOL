include("PlotResults.jl")

function CompareTMS_NonlinearDiffusion(StudyCase::String; SaveFig::Bool=false, w::Float64=8.50, h::Float64=8.50)
# function CompareTMS_NonlinearDiffusion(StudyCase::String,nb::Int64,SCRef::Int64,nbRef::Int64; q::Real=2.0, 
#     SaveFig::Bool=false, w::Float64=8.50, h::Float64=8.50)

    SCvv0   = NaN
    SCvv1   = NaN
    nb      = NaN
    
    if StudyCase=="NCD1"
    
        #TimeAdapt: No
        SCvv1       = [3000:3003,
                       3004:3007,
                       3008:3011,
                       3012:3015,
                       3016:3019,
                       3020:3023]
        nb = 1000


    elseif StudyCase=="NCD2"

        #TimeAdapt: No
        SCvv1       = [3024:3027,
                       3028:3031,
                       3032:3035,
                       3036:3039,
                       3040:3043,
                       3044:3047]
        
        nb = 1000
        
    elseif StudyCase=="NCD3"
    
        #TimeAdapt: No
        SCvv1       = [3048:3051,
                       3052:3055,
                       3056:3059,
                       3060:3063,
                       3064:3067,
                       3068:3071]
        
        nb = 1000
        
    elseif StudyCase=="NCD4"

        #TimeAdapt: No
        SCvv1       = [3072:3075,
                       3076:3079,
                       3080:3083,
                       3084:3087,
                       3088:3091,
                       3092:3095]
        
        nb = 1000
                        
        
    
    elseif StudyCase=="NCD5"
    
        #TimeAdapt: No
        SCvv1       = [3096:3099,
                       3100:3103,
                       3104:3107,
                       3108:3111,
                       3112:3115,
                       3116:3119]
        
        nb = 1000
        
    elseif StudyCase=="NCD6"
    
        #TimeAdapt: No
        SCvv1       = [3120:3123,
                       3124:3127,
                       3128:3131,
                       3132:3135,
                       3136:3139,
                       3140:3143]
        
        nb = 1000
        
    elseif StudyCase=="NCD7"
    
        #TimeAdapt: No
        SCvv1       = [3144:3147,
                       3148:3151,
                       3152:3155,
                       3156:3159,
                       3160:3163,
                       3164:3167]
        
        nb = 1000
        
    elseif StudyCase=="NCD8"
    
        #TimeAdapt: No
        SCvv1       = [3168:3171,
                       3172:3175,
                       3176:3179,
                       3180:3183,
                       3184:3187,
                       3188:3191]
        
        nb = 1000
        
    elseif StudyCase=="NCD9"
    
        #TimeAdapt: No
        SCvv1       = [3192:3195,
                       3196:3199,
                       3200:3203,
                       3204:3207,
                       3208:3211,
                       3212:3215]
        
        nb = 1000
        
    elseif StudyCase=="NCD10"
    
        #TimeAdapt: No
        SCvv1       = [3216:3219,
                       3220:3223,
                       3224:3227,
                       3228:3231,
                       3232:3235,
                       3236:3239]
        
        nb = 1000
        
    elseif StudyCase=="NCD11"
    
        #TimeAdapt: No
        SCvv1       = [3240:3243,
                       3244:3247,
                       3248:3251,
                       3252:3255,
                       3256:3259,
                       3260:3263]
        
        nb = 1000
        
    elseif StudyCase=="NCD12"
    
        #TimeAdapt: No
        SCvv1       = [3264:3267,
                       3268:3271,
                       3272:3275,
                       3276:3279,
                       3280:3283,
                       3284:3287]
        
        nb = 1000
        
    elseif StudyCase=="NCD13"
    
        #TimeAdapt: No
        SCvv1       = [3288:3291,
                       3292:3295,
                       3296:3299,
                       3300:3303,
                       3304:3307,
                       3308:3311]
        
        nb = 1000
        
    elseif StudyCase=="NCD14"
    
        #TimeAdapt: No
        SCvv1       = [3312:3315,
                       3316:3319,
                       3320:3323,
                       3324:3327,
                       3328:3331,
                       3332:3335]
        
        nb = 1000
        
    elseif StudyCase=="NCD15"
    
        #TimeAdapt: No
        SCvv1       = [3336:3339,
                       3340:3343,
                       3344:3347,
                       3348:3351,
                       3352:3355,
                       3356:3359]
        
        nb = 1000
    
    elseif StudyCase=="NCD16"
    
        #TimeAdapt: No
        SCvv1       = [3360:3363,
                       3364:3367,
                       3368:3371,
                       3372:3375,
                       3376:3379,
                       3380:3383]
        
        nb = 1000
        
    end
        
    #------------------------------------------------------------
    
    Deltatvv1,errvv1,tCPUvv1,CFLvv1,TIMethodNamevv1             = GetVbles(SCvv1, ["Deltat","errL2L2","tCPU","CFLmax", "TIMethodName"], nb=nb)
#     errv1      = []
#     errvv1     = Vector{Vector{Any}}()
#     SCvv1r     = reduce(vcat,SCvv1)
#     
#     errv1 = RefErr_Lq(SCvv1r,nb,SCRef,nbRef;q=q)
# 
#     n = 1
#     for i=1:length(SCvv1)
# 
#         k       =   length(SCvv1[i])
#         serrv1  =   errv1[n:n+k-1]
#         push!(errvv1,serrv1)
#         n       =   n + k
# 
#     end

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

