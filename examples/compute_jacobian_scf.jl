using Test
using PyCall
using DFTK


mg = pyimport("pymatgen")
symmetry = pyimport("pymatgen.symmetry")
elec_structure = pyimport("pymatgen.electronic_structure")
plotter = pyimport("pymatgen.electronic_structure.plotter")

#
# Calculation parameters
#
kgrid = [1,1,1]
Ecut = 3  # Hartree
n_bands = 8
tol = 1e-10


#
# Setup silicon structure in pymatgen
#
a = 5.431020504 * mg.units.ang_to_bohr
A = mg.ArrayWithUnit(a / 2 .* [[0 1 1.];
                               [1 0 1.];
                               [1 1 0.]], "bohr")
lattice = mg.lattice.Lattice(A)
recip_lattice = lattice.reciprocal_lattice
structure = mg.Structure(lattice, ["Si", "Si"], [ones(3)/8, -ones(3)/8])

# Get k-Point mesh for Brillouin-zone integration
spgana = symmetry.analyzer.SpacegroupAnalyzer(structure)
bzmesh = spgana.get_ir_reciprocal_mesh(kgrid)
kpoints = [mp[1] for mp in bzmesh]
kweigths = [mp[2] for mp in bzmesh]
kweigths = kweigths / sum(kweigths)

#
# SCF calculation in DFTK
#
# Construct basis: transpose is required, since pymatgen uses rows for the
# lattice vectors and DFTK uses columns
grid_size = DFTK.determine_grid_size(A', Ecut, kpoints=kpoints) * ones(Int, 3)
basis = PlaneWaveBasis(A', grid_size, Ecut, kpoints, kweigths)

# Setup model for silicon and list of silicon positions
Si = Species(mg.Element("Si").number, psp=load_psp("si-pade-q4.hgh"))
composition = [Si => [s.frac_coords for s in structure.sites if s.species_string == "Si"]]
n_electrons = sum(length(pos) * n_elec_valence(spec) for (spec, pos) in composition)

# Construct Hamiltonian
ham = Hamiltonian(basis, pot_local=build_local_potential(basis, composition...),
                  pot_nonlocal=build_nonlocal_projectors(basis, composition...),
                  pot_hartree=PotHartree(basis),
                  # pot_xc=PotXc(basis, Functional.([:lda_x, :lda_c_vwn]))
                  )

# Build a guess density and run the SCF
ρ = guess_gaussian_sad(basis, composition...)
scfres = self_consistent_field(ham, Int(n_electrons / 2 + 2), n_electrons, ρ=ρ, tol=tol,
                               lobpcg_prec=PreconditionerKinetic(ham, α=0.1))
ρ_eq = scfres[1]

den_scaling = 0.0

Psi = [Matrix(qr(randn(ComplexF64, length(basis.basis_wf[ik]), n_bands)).Q)
       for ik in 1:length(basis.kpoints)]
Gsq = vec([4π * sum(abs2, basis.recip_lattice * G)
           for G in basis_ρ(basis)])
Gsq[basis.idx_DC] = 1.0 # do not touch the DC component
den_to_mixed = Gsq.^(-den_scaling)
mixed_to_den = Gsq.^den_scaling

function compute_occupation(basis, energies, Psi)
    DFTK.occupation_zero_temperature(basis, energies, Psi, n_electrons)
end

function foldρ(ρ)
    ρ = den_to_mixed .* ρ
    # Fold a complex array representing the Fourier transform of a purely real
    # quantity into a real array
    half = Int((length(ρ) + 1) / 2)
    ρcpx =  ρ[1:half]
    vcat(real(ρcpx), imag(ρcpx))
end
function unfoldρ(ρ)
    # Undo "foldρ"
    half = Int(length(ρ) / 2)
    ρcpx = ρ[1:half] + im * ρ[half+1:end]
    ρ_unfolded = vcat(ρcpx, conj(reverse(ρcpx)[2:end]))
    ρ_unfolded .* mixed_to_den
end

fp_map(ρ) = foldρ(DFTK.new_density(ham, n_bands, compute_occupation, unfoldρ(ρ), tol, lobpcg_prec=PreconditionerKinetic(ham, α=0.1), Psi=Psi))

function jac(f, x, ε)
    J = zeros(length(x), length(x))
    x_curr = copy(x)
    fx = f(x)
    for i=1:length(x)
        println("$i of $(length(x))")
        x_curr = copy(x)
        x_curr[i] += ε
        fx_curr = f(x_curr)
        J[:,i] = (fx_curr - fx)/ε
    end
    J
end

J = jac(fp_map, foldρ(ρ_eq), sqrt(tol))

# sort in increasing kinetic energy order
perm = sortperm(vec([norm(basis.recip_lattice * G) for G in basis_ρ(basis)]))

using PyPlot
pcolormesh(log10.(abs.(J[perm,perm])))
colorbar()
