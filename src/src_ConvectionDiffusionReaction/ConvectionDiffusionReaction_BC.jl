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
