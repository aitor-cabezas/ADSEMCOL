
#Dirichlet Conditions

function bflux!(model::Oregonator, BC::Dirichlet, _bqp::TrBintVars, ComputeJ::Bool)
    
        x                       =   _bqp.x
        u                       =   _bqp.u
        du                      =   _bqp.gradu
        ParentElems             =   _bqp.Binteg2D.bmesh.ParentElems
        metric                  =   _bqp.Binteg2D.mesh.metric
        h                       =   1.0./sqrt.(metric.lambda_bar[ParentElems])*ones(1,_bqp.nqp)
        bflux                   =   _bqp.f
        dbflux_du               =   _bqp.df_du
        dbflux_dgradu           =   _bqp.df_dgradu
    
end














#Neumann conditions:
function bflux!(model::Oregonator, BC::Neumann, _bqp::TrBintVars, ComputeJ::Bool)

    t                           = _bqp.t
    x                           = _bqp.x
    
    #Since u is extrapolated, penalty is zero.
    
    #Impose flux _bqp.f = g_alpha = f_alphai*nb:
    
    fn                          = BC.q(t,x)
    @tturbo @. _bqp.f[1]        += fn[1]
    @tturbo @. _bqp.f[2]        += fn[2]
    @tturbo @. _bqp.f[3]        += fn[3]
     
    return
    
end
