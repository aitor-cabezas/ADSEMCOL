include("PlotResults.jl")

function CompareTMS_Oregonator(StudyCase::String,nb::Int64,SCRef::Int64,nbRef::Int64; q::Real=2.0, 
    SaveFig::Bool=false, w::Float64=8.50, h::Float64=8.50)

    SCvv0   = NaN
    SCvv1   = NaN
    
    if StudyCase=="Oa"
    
        #TimeAdapt: YES
        SCvv1       = [2001:2003,
                       2005:2007,
                       2009:2011,
                       2013:2016,
                       2017:2020,
                       2021:2024]
                        
        
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

    @show(errvv1)
    @show(typeof(errvv1))
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
        savefig("$(FigUbi)Oregonator.png", dpi=800, pad_inches=0)
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
        savefig("$(FigUbi)Oregonator.png", dpi=800, pad_inches=0)
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
        savefig("$(FigUbi)Oregonator.png", dpi=800, pad_inches=0)
    end
    
    display(EOCvv1)
    
    return
    
end





function Contour_Oregonator(SC::Int, nb::Int; SaveFig::Bool=false, w::Float64=8.50, h::Float64=8.50, 
    PlotVars::Vector{String}=["u", "v", "w"], mFig::Int=2, nFig::Int=2)
    
    OregonatorModel = Oregonator()
    FileName        = GetFileName(SC, nb)
    solver          = GetSolver(SC, nb)
    
    for ii=1:length(PlotVars)
        
        PyPlotFigure(w=w, h=h, bottom=1.5, top=1.0)
        
        #Numerical solution:
        PlotContourOregonator(solver, OregonatorModel, PlotVars[ii], delta=1e-4)
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
