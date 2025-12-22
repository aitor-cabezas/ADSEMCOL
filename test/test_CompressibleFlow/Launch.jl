#Load problem:
ProblemName     = ARGS[1]
include("$(ProblemName).jl")

try
    if ProblemName=="DetonationWave"
        tf              = parse(Float64, ARGS[2])
        delta           = parse(Float64, ARGS[3])
        epsilon         = parse(Float64, ARGS[4])
        TolS            = parse(Float64, ARGS[5])
        TolT            = parse(Float64, ARGS[6])
        Deltat_SaveRes  = parse(Float64, ARGS[7])
        SC              = parse(Int, ARGS[8])
        DetonationWave(tf=tf, delta=delta, epsilon=epsilon, 
            TolS=TolS, TolT=TolT, 
            SaveRes=true, Deltat_SaveRes=Deltat_SaveRes, SC=SC)
            
            
    elseif ProblemName=="SmoothVortex"
        
        TMSName              = ARGS[2]
        RoWMethod            = ARGS[3]
        RKMethod             = ARGS[4]
        tf                   = parse(Float64,ARGS[5])
        Deltat0              = parse(Float64,ARGS[6])
        TolT                 = parse(Float64,ARGS[7])
        hp                   = parse(Float64,ARGS[8])
        TolS                 = parse(Float64,ARGS[9])
        vortex_st            = parse(Float64,ARGS[10])
        u_inf                = parse(Float64,ARGS[11])
        gamma                = parse(Float64,ARGS[12])
        SC                   = parse(Int64,ARGS[13])
        SmoothVortex(TMSName=TMSName,RoWMethod=RoWMethod,RKMethod=RKMethod,tf=tf,Deltat0=Deltat0,TolT=TolT,hp=hp,TolS=TolS,vortex_st=vortex_st,u_inf=u_inf,gamma=gamma,SC=SC,SaveRes=true,TimeAdapt=true)
        
    else
        error("Undefined problem $(ProblemName)")
    end
    
catch err
    
    println(Error2String(err))
            
end
