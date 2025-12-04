#Neumann conditions:
function bflux!(model::Oregonator, BC::Neumann, _bqp::TrBintVars, ComputeJ::Bool)

    #Since u is extrapolated, penalty is zero.
    
    #Impose flux _bqp.f = g_alpha = f_alphai*nb:
     @tturbo @. _bqp.f[1]          += 0.0
     @tturbo @. _bqp.f[2]          += 0.0
     @tturbo @. _bqp.f[3]          += 0.0
     
    return
    
end
