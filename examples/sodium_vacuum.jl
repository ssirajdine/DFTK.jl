using PyCall
using DFTK
using Printf
using PyPlot
using Random

# using FFTW
# FFTW.set_num_threads(4)
# BLAS.set_num_threads(4)



kgrid = [1, 1, 1]        # k-Point grid
Ecut = 30                 # kinetic energy cutoff in Hartree

# # System from https://math.berkeley.edu/~linlin/publications/EllipticPreconditioner.pdf
# n_sodium = 10
# n_vacuum = 10
# n_cells = div(n_sodium, 2) + n_vacuum
# x_pos = collect(range(0, div(n_sodium, 2)/n_cells, length=n_sodium))
# x_pos .+= 1/2 .- x_pos[div(end, 2)]
# Random.seed!(0)
# x_pos += .03*randn(length(x_pos)) / n_cells
# # positions = [[x, 0, 0] + .1*randn(3)/n_cells for x in x_pos]
# positions = [[x, 0, 0] for x in x_pos]

# # 8 bohr unit cell. 2 sodium atoms per unit cell
# lattice = Diagonal([8*n_cells, 8., 8.])


# # Tsmear = 0.01            # Smearing temperature in Hartree
# Tsmear = 0.001            # Smearing temperature in Hartree. 300K=0.001 Ha

# Na = Species(11, psp=load_psp("na-lda-q1.hgh"))
# atoms = [Na => positions]








# Sodium-silicon system
n_sodium = 10
n_silicon = 10

# # mixed
# na_pos = collect(range(0, 1, length=n_sodium+1)[1:end-1])
# si_pos = collect(range(0, 1, length=n_silicon+1)[1:end-1])
# si_pos .+= (si_pos[2] - si_pos[1])/2
# Random.seed!(0)
# na_pos = [p + .01*randn()/n_sodium for p in na_pos]
# si_pos = [p + .01*randn()/n_silicon for p in si_pos]

# separated
# na_pos = collect(range(0, 0.5, length=n_sodium+1)[1:end-1])
na_pos = collect(range(0, .5, length=n_sodium+1)[1:end-1])
si_pos = collect(range(0.5, 1, length=div(n_silicon, 2)+1)[1:end-1])

na_positions = [[x, 0, 0] for x in na_pos]
si_positions = [[x, 0, 0] for x in si_pos]


# 8 bohr unit cell. 2 sodium atoms per unit cell
lattice = Diagonal([8*div(n_sodium+n_silicon, 2), 8., 8.])

Tsmear = 0.001            # Smearing temperature in Hartree

Na = ElementPsp(:Na, psp=load_psp("hgh/lda/na-q1"))
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/si-q4"))
He = ElementPsp(:He, psp=load_psp("hgh/lda/he-q2"))
atoms = [Na => na_positions, He => si_positions]
# atoms = [Na => na_positions]




# Setup PBE model with Methfessel-Paxton smearing and its discretisation
model = model_DFT(lattice, atoms, :lda_xc_teter93;
                  temperature=Tsmear,
                  smearing=Smearing.FermiDirac())
n_bands = model.n_electrons

# kcoords, ksymops = bzmesh_ir_wedge(kgrid, lattice, atoms)
kcoords, ksymops = bzmesh_uniform(kgrid)
fft_size = determine_grid_size(model, Ecut)
fft_size = [fft_size[1], 1, 1]
basis = PlaneWaveBasis(model, Ecut, kcoords, ksymops, fft_size=fft_size)

close(:all)
figure()

OUTPUTS = []
function mixing_diagnostics(info)
    E = sum(values(info.energies))
    res = norm(info.ρout.fourier - info.ρin.fourier)
    neval = info.neval
    if neval == 1
        println("Iter   Energy             ρout-ρin")
        println("----   ------             --------")
    end
    @printf "%3d    %-15.12f    %E\n" neval E res

    toplot = sum(info.ρin.real, dims=(2, 3))[:]
    push!(OUTPUTS, toplot)
    plot(-toplot ./ maximum(abs.(toplot)), "b", linewidth=0.5)

    if !isnothing(info.ldos)
        toplot = sum(info.ldos, dims=(2, 3))[:]
        toplot /= maximum(abs.(toplot))
        plot(toplot, "r", linewidth=0.5)

        summed = sum(info.ldos[:]) / length(info.ldos[:])
        println("        avg(ldos) = ", summed)
        println("        scrlng    = ", sqrt(1/(4π * summed)))
        println()
    end

    # plot("ρin", sum(real(ρin), dims=(2, 3))[:])
    # plot("Δρ", sum(real(ρnext - ρin), dims=(2, 3))[:])
    # plot("ldos", sum(real(LDOS), dims=(2, 3))[:])
end


# Run SCF
tol = 1e-10
diagtol = 1e-8
# mixing = DFTK.SimpleMixing(.3)
mixing = DFTK.HybridMixing()
# mixing = DFTK.KerkerMixing()
scfres = self_consistent_field(basis, n_bands=n_bands, solver=scf_nlsolve_solver(m=7),
                               mixing=mixing, tol=tol, diagtol=diagtol,
                               callback=mixing_diagnostics)

figure()
for i in 2:length(OUTPUTS)
    ρ = OUTPUTS[i]
    plot(ρ - OUTPUTS[end])
end

includet("../src/postprocess/chi0.jl")
χ0 = compute_χ0(scfres.ham)




# # Plot DOS
# εs = range(minimum(minimum(scfres.orben)) - 1, maximum(maximum(scfres.orben)) + 1, length=1000)
# # Ds = DOS.(εs, Ref(basis), Ref(scfres.orben), T=Tsmear*10, smearing=DFTK.smearing_methfessel_paxton_1)
# Ds = DOS.(εs, Ref(basis), Ref(scfres.orben), T=Tsmear*5, smearing=DFTK.smearing_fermi_dirac)
# plot(εs, Ds)
# axvline(scfres.εF)

# # figure()
# # for T in (0.001, 0.01, 0.02, 0.04)
# #     ldos = LDOS(scfres.εF, basis, scfres.orben, scfres.Psi, smearing=smearing_fermi_dirac, T=T)
# #     toplot = ldos[:, 1, 1]
# #     # toplot = sum(ldos, dims=(2, 3))[:]
# #     # toplot /= maximum(abs.(toplot))
# #     plot(toplot, label=string(T))
# # end
# # legend()

# # # for ψ in eachcol(scfres.Psi[1])
# # #     toplot = sum(G_to_r(basis, basis.kpoints[1], ψ), dims=(2, 3))[:]
# # #     plot(abs.(toplot))
# # # end


# # figure()
# # toplot = sum(real(scfres.ρ), dims=(2, 3))[:]
# # plot(abs.(toplot))
