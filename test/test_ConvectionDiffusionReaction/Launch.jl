#Load problem:
ProblemName     = ARGS[1]
include("$(ProblemName).jl")

try
    if ProblemName=="Oregonator"
        
        TMSName              = ARGS[2]
        RoWMethod            = ARGS[3]
        RKMethod             = ARGS[4]
        tf                   = parse(Float64,ARGS[5])
        Deltat0              = parse(Float64,ARGS[6])
        TolT                 = parse(Float64,ARGS[7])
        hp                   = parse(Float64,ARGS[8])
        TolS                 = parse(Float64,ARGS[9])
        TimeAdapt            = parse(Bool,ARGS[10])
        SpaceAdapt           = parse(Bool,ARGS[11])
        phi                  = parse(Float64,ARGS[12])
        epsilon              = parse(Float64,ARGS[13])
        epsilonp             = parse(Float64,ARGS[14])
        Du                   = parse(Float64,ARGS[15])
        Dw                   = parse(Float64,ARGS[16])
        q                    = parse(Float64,ARGS[17])
        f                    = parse(Float64,ARGS[18])
        A                    = parse(Float64,ARGS[19])
        SC                   = parse(Int64,ARGS[20])
        Oregonator_test(TMSName=TMSName,RoWMethod=RoWMethod,RKMethod=RKMethod,tf=tf,Deltat0=Deltat0,TolT=TolT,hp=hp,TolS=TolS,TimeAdapt=TimeAdapt,SpaceAdapt=SpaceAdapt,phi=phi,epsilon=epsilon,epsilonp=epsilonp,Du=Du,Dw=Dw,q=q,f=f,A=A,SC=SC,SaveRes=true)
        
    else
        error("Undefined problem $(ProblemName)")
    end
    
catch err
    
    println(Error2String(err))
            
end
