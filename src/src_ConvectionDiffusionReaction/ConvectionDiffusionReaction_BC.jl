#--------------------------------------------------------------------------------
#BOUNDARY CONDITIONS:

#Impose penalty flux fn_I = sigma*(u_I - uBC_I):
function penalty!(model::NCD,
                  sigma::MFloat,
                  u::Vector{MFloat}, uBC::Vector{MFloat}, duBC_du::Matrix{MFloat},
                  ComputeJ::Bool,
                  bflux::Vector{MFloat}, dbflux_du::Matrix{MFloat}) where MFloat<:Matrix{Float64}

#Update boundary flux:
for II=1:model.nVars
    @mlv    bflux[II]                   += sigma*(u[II]-uBC[II])
    if ComputeJ
        for JJ=1:model.nVars
            @mlv    dbflux_du[II,JJ]    += sigma*((II==JJ)-duBC_du[II,JJ])
        end
    end
end

return

end

#Dirichlet Conditions

function bflux!(model::NCD, BC::Dirichlet, _bqp::TrBintVars, ComputeJ::Bool)
        
        t                       =   _bqp.t
        nb                      =   _bqp.nb
        x                       =   _bqp.x
        u                       =   _bqp.u
        du                      =   _bqp.gradu
        ParentElems             =   _bqp.Binteg2D.bmesh.ParentElems
        metric                  =   _bqp.Binteg2D.mesh.metric

        #Dirichlet condition:
        uBC                     = BC.uDir(t,x)
        duBC_du                 = Matrix{Matrix{Float64}}(undef,1,1)
        duBC_du[1,1]            = zeros(size(u[1]))

        #Hyperbolic term:
        a                           = model.a(t,x,u)
        a_n                         = @tturbo @. a[1]*nb[1]+a[2]*nb[2]
        da_du                       = model.da_du(t,x,u)
        ahat_1                      = @tturbo @. da_du[1]*u[1] + a[1]
        ahat_2                      = @tturbo @. da_du[2]*u[1] + a[2]
        ahat_n                      = @tturbo @. ahat_1*nb[1] + ahat_2*nb[2]
        inflow                      = @tturbo @. ahat_n <= 0.0
        outflow                     = @tturbo @. !inflow
        #
        aBC                             = model.a(t,x,uBC)
        aBC_n                           = @tturbo @. aBC[1]*nb[1] + aBC[2]*nb[2]
        @tturbo @. _bqp.f[1][inflow]    += aBC_n[inflow]*uBC[1][inflow]
        @tturbo @. _bqp.f[1][outflow]   += a_n[outflow]*u[1][outflow]
        if ComputeJ
            @tturbo @. _bqp.df_du[1,1][inflow]    += 0.0
            @tturbo @. _bqp.df_du[1,1][outflow]   += ahat_n[outflow]
        end

        #Allocate viscous fluxes:
        flux, dflux_du, dflux_dgradu = FluxAllocate(1, size(u[1]), ComputeJ)
        DT, dDT_du, dDT_dgradu = ViscosityAllocate(1, size(u[1]), ComputeJ)

        #Natural viscosity:
        DT                             = model.DT(t,x,u)
        dDT_du                         = Vector{Matrix{Float64}}() 
        if ComputeJ
            dDT_du                     = model.dDT_du(t,x,u)
        end

        #Extrapolate natural viscous flux:
        SSDiffusiveFlux!(model, DT, dDT_du, u, du, ComputeJ, flux, dflux_du, dflux_dgradu)
        fn, dfn_du, dfn_dgradu = ProjectFlux(flux, dflux_du, dflux_dgradu, nb, ComputeJ)
        #
        @mlv _bqp.f[1]                  += fn[1]
        if ComputeJ
            @mlv _bqp.df_du[1,1]        += dfn_du[1,1]
            @mlv _bqp.df_dgradu[1,1,1]  += dfn_dgradu[1,1,1]
            @mlv _bqp.df_dgradu[1,1,2]  += dfn_dgradu[1,1,2]
        end
        #Add penalty terms:
        h_Elems                 = @mlv 1.0/sqrt(metric.lambda_bar[ParentElems])
        hp                      = h_Elems./_bqp.FesOrder * ones(1, _bqp.nqp)
        sigma                   = @mlv model.CW .* DT[1] ./ hp
        penalty!(model, sigma, u, uBC, duBC_du, ComputeJ, _bqp.f, _bqp.df_du)

        return
    
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
