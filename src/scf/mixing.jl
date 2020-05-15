# Mixing rules: (ρin, ρout) => ρnext, where ρout is produced by diagonalizing the Hamiltonian at ρin
# These define the basic fix-point iteration, that are then combined with acceleration methods (eg anderson)
# All these methods attempt to approximate the inverse jacobian of the SCF step, J^-1 = (1 - χ0 vc)^-1
# Note that "mixing" is sometimes used to refer to the combined process of formulating the fixed-point
# and solving it; we call "mixing" only the first part

# The interface is `mix(m, basis, ρin, ρout) -> ρnext`

using LinearMaps
using IterativeSolvers

@doc raw"""
Kerker mixing: ``J^{-1} ≈ \frac{α G^2}{(G0^2 + G^2}``

Abinit calls ``1/G0`` the dielectric screening length (parameter *dielng*)
"""
struct KerkerMixing{T <: Real}
    α::T
    G0::T
end

# Default parameters suggested by Kresse, Furthmüller 1996 (α=0.8, G0=1.5 Ǎ^{-1})
# DOI 10.1103/PhysRevB.54.11169
KerkerMixing() = KerkerMixing(0.8, 0.8)

function mix(m::KerkerMixing, basis, ρin::RealFourierArray, ρout::RealFourierArray; kwargs...)
    Gsq = [sum(abs2, basis.model.recip_lattice * G)
           for G in G_vectors(basis)]
    ρin = ρin.fourier
    ρout = ρout.fourier
    ρnext = @. ρin + m.α * (ρout - ρin) * Gsq / (m.G0^2 + Gsq)
    # take the correct DC component from ρout; otherwise the DC component never gets updated
    ρnext[1] = ρout[1]
    from_fourier(basis, ρnext; assume_real=true)
end

"""
Simple mixing: J^-1 ≈ α
"""
struct SimpleMixing{T <: Real}
    α::T
end
SimpleMixing() = SimpleMixing(1)
function mix(m::SimpleMixing, basis, ρin::RealFourierArray, ρout::RealFourierArray; kwargs...)
    if m.α == 1
        return ρout # optimization
    else
        ρin + m.α * (ρout - ρin)
    end
end

struct HybridMixing
    α               # Damping parameter
    ldos_nos        # Minimal NOS value in for LDOS computation
    ldos_maxfactor  # Maximal factor between electron temperature and LDOS temperature
    G_blur          # Width of Gaussian filter applied to LDOS in reciprocal space.
end
function HybridMixing(α=1; ldos_nos=20, ldos_maxfactor=10, G_blur=Inf)
    HybridMixing(α, ldos_nos, ldos_maxfactor, G_blur)
end

function mix(m::HybridMixing, basis, ρin::RealFourierArray, ρout::RealFourierArray;
             LDOS=nothing, kwargs...)
    LDOS === nothing && return ρin + m.α * (ρout - ρin)  # Fallback to simple mixing

    # blur the LDOS
    if m.G_blur < Inf
        blur_factor(G) = exp(-(norm(G)/m.G_blur)^2)
        LDOS_fourier = r_to_G(basis, complex.(LDOS))
        LDOS_fourier .*= blur_factor.(basis.model.recip_lattice * G for G in G_vectors(basis))
        LDOS = real.(G_to_r(basis, LDOS_fourier))
    end

    # F : ρin -> ρout has derivative χ0 vc
    # a Newton step would be ρn+1 = ρn + (1 -χ0 vc)^-1 (F(ρn) - ρn)
    # We approximate -χ0 by a real-space multiplication by LDOS
    # We want to solve J Δρ = ΔF with J = (1 - χ0 vc)
    ΔF = ρout.real - ρin.real
    devec(x) = reshape(x, size(ρin))
    function Jop(x)
        den = devec(x)
        Gsq = [sum(abs2, basis.model.recip_lattice * G)
               for G in G_vectors(basis)]
        Gsq[1] = Inf # Don't act on DC
        den_fourier = from_real(basis, den).fourier  # TODO r_to_G ??
        pot_fourier = 4π ./ Gsq .* den_fourier
        pot_real = from_fourier(basis, pot_fourier).real  # TODO G_to_r ??

        # apply χ0
        den_real = real(LDOS .* pot_real - sum(LDOS .* pot_real) .* LDOS ./ sum(LDOS))
        vec(den + den_real)
    end
    J = LinearMap(Jop, length(ρin))
    x = gmres(J, ΔF)
    Δρ = devec(x)
    from_real(basis, real(ρin.real + m.α * Δρ))
end

struct χ0Mixing
    α               # Damping parameter
    ldos_nos        # Minimal NOS value in for LDOS computation
    ldos_maxfactor  # Maximal factor between electron temperature and LDOS temperature
    droptol         # Tolerance for dropping contributions in χ0
    sternheimer_contribution  # Use Sternheimer for contributions of unoccupied orbitals
end
function χ0Mixing(α=1; ldos_nos=20, ldos_maxfactor=10, droptol=Inf, sternheimer_contribution=false)
    χ0Mixing(α, ldos_nos, ldos_maxfactor, droptol, sternheimer_contribution)
end

function mix(m::χ0Mixing, basis, ρin::RealFourierArray, ρout::RealFourierArray;
             LDOS, ham, ψ, occupation, εF, eigenvalues, temperature)
    # TODO Duplicate code with HybridMixing
    #
    # F : ρin -> ρout has derivative χ0 vc
    # a Newton step would be ρn+1 = ρn + (1 -χ0 vc)^-1 (F(ρn) - ρn)
    # We approximate -χ0 by a real-space multiplication by LDOS
    # We want to solve J Δρ = ΔF with J = (1 - χ0 vc)
    ΔF = ρout.real - ρin.real
    devec(x) = reshape(x, size(ρin))
    function Jop(x)
        den = devec(x)
        Gsq = [sum(abs2, basis.model.recip_lattice * G)
               for G in G_vectors(basis)]
        Gsq[1] = Inf # Don't act on DC
        den_fourier = from_real(basis, den).fourier  # TODO r_to_G ??
        pot_fourier = 4π ./ Gsq .* den_fourier
        pot_real = from_fourier(basis, pot_fourier).real  # TODO G_to_r ??

        # apply χ0
        den_real = apply_χ0(ham, real(pot_real), ψ, occupation, εF, eigenvalues;
                            droptol=m.droptol,
                            sternheimer_contribution=m.sternheimer_contribution,
                            temperature=temperature)
        vec(den - den_real)
    end
    J = LinearMap(Jop, length(ρin))
    x = gmres(J, ΔF)
    Δρ = devec(x)
    from_real(basis, real(ρin.real + m.α * Δρ))
end
