using Test
using DFTK

include("testcases.jl")

# Silicon redHF (without xc) is a metal, so we add a bit of temperature to it


function run_silicon_redHF_compare(T; Ecut=5, test_tol=1e-6, n_ignored=0, grid_size=15, scf_tol=1e-6)
    # T + Vloc + Vnloc + Vhartree
    # These values were computed using ABINIT with the same kpoints as testcases.jl
    # and Ecut = 25
    ref_redHF = [
        [0.17899118507651615, 0.6327279881297371, 0.6327279881326648, 0.6327279881356039,
         0.706557757783828, 0.7065577577877139, 0.7065577577915956, 0.7397951816714727,
         0.8532089291297222, 0.8978914445971602],
        [0.23220003663858457, 0.42189409862896016, 0.5921574659414509, 0.5921574659446628,
         0.672858189872362, 0.7372271903827399, 0.7372271903861028, 0.8643640848936627,
         0.9011792204214196, 0.9011792204356576],
        [0.2517502116803524, 0.445206025448218, 0.5328870916963034, 0.532887091701182,
         0.6211365856991057, 0.661989858948651, 0.8863951918546257, 0.8863951918584175,
         0.973261179805555, 0.9771287508158364],
        [0.30685586314464863, 0.376375429632464, 0.4438764716222098, 0.5459065154292047,
         0.651122698647485, 0.8164293660861612, 0.8515978828421051, 0.8735213568005982,
         0.8807275612483988, 0.8886454931307763]
    ]
    ref_etot = -5.440593269861395

    n_bands = length(ref_redHF[1])
    fft_size = grid_size * ones(3)
    Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
    model_none = model_reduced_hf(Array{T}(silicon.lattice), [Si => silicon.positions],
                             temperature=0.05)
    model_collinear = model_reduced_hf(Array{T}(silicon.lattice), [Si => silicon.positions],
                              temperature=0.05, spin_polarisation=:collinear)
    basis_none = PlaneWaveBasis(model_none, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)
    basis_collinear = PlaneWaveBasis(model_collinear, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)
    ham_none = Hamiltonian(basis_none, guess_density(basis_none, [Si => silicon.positions]))
    ham_collinear = Hamiltonian(basis_collinear, guess_density(basis_collinear, [Si => silicon.positions]))

    scfres_none = self_consistent_field(ham_none, n_bands, tol=scf_tol)
    scfres_collinear = self_consistent_field(ham_collinear, n_bands, tol=scf_tol)

    for ik in 1:length(silicon.kcoords)
        @test eltype(scfres.orben[ik]) == T
        @test eltype(scfres.Psi[ik]) == Complex{T}
        println(ik, "  ", abs.(scfres_none.orben[ik] - scfres_collinear.orben[ik]))
    end
    for ik in 1:length(silicon.kcoords)
        # Ignore last few bands, because these eigenvalues are hardest to converge
        # and typically a bit random and unstable in the LOBPCG
        diff = abs.(scfres_none[ik] - scfres_collinear.orben[ik])
        @test maximum(diff[1:n_bands - n_ignored]) < test_tol
    end
end


function run_silicon_redHF_none(T; Ecut=5, test_tol=1e-6, n_ignored=0, grid_size=15, scf_tol=1e-6)
    # T + Vloc + Vnloc + Vhartree
    # These values were computed using ABINIT with the same kpoints as testcases.jl
    # and Ecut = 25
    ref_redHF = [
        [0.17899118507651615, 0.6327279881297371, 0.6327279881326648, 0.6327279881356039,
         0.706557757783828, 0.7065577577877139, 0.7065577577915956, 0.7397951816714727,
         0.8532089291297222, 0.8978914445971602],
        [0.23220003663858457, 0.42189409862896016, 0.5921574659414509, 0.5921574659446628,
         0.672858189872362, 0.7372271903827399, 0.7372271903861028, 0.8643640848936627,
         0.9011792204214196, 0.9011792204356576],
        [0.2517502116803524, 0.445206025448218, 0.5328870916963034, 0.532887091701182,
         0.6211365856991057, 0.661989858948651, 0.8863951918546257, 0.8863951918584175,
         0.973261179805555, 0.9771287508158364],
        [0.30685586314464863, 0.376375429632464, 0.4438764716222098, 0.5459065154292047,
         0.651122698647485, 0.8164293660861612, 0.8515978828421051, 0.8735213568005982,
         0.8807275612483988, 0.8886454931307763]
    ]
    ref_etot = -5.440593269861395

    n_bands = length(ref_redHF[1])
    fft_size = grid_size * ones(3)
    Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
    model = model_DFT(Array{T}(silicon.lattice), [Si => silicon.positions], [];
                      temperature=0.05)
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)

    scfres = self_consistent_field(basis, tol=scf_tol, n_bands=n_bands)

    for ik in 1:length(silicon.kcoords)
        @test eltype(scfres.eigenvalues[ik]) == T
        @test eltype(scfres.ψ[ik]) == Complex{T}
        println(ik, "  ", abs.(ref_redHF[ik] - scfres.eigenvalues[ik][1:n_bands]))
    end
    for ik in 1:length(silicon.kcoords)
        # Ignore last few bands, because these eigenvalues are hardest to converge
        # and typically a bit random and unstable in the LOBPCG
        diff = abs.(ref_redHF[ik] - scfres.eigenvalues[ik][1:n_bands])
        @test maximum(diff[1:n_bands - n_ignored]) < test_tol
    end 
end


function run_silicon_redHF_collinear(T; Ecut=5, test_tol=1e-6, n_ignored=0, grid_size=15, scf_tol=1e-6)
    # T + Vloc + Vnloc + Vhartree
    # These values were computed using ABINIT with the same kpoints as testcases.jl
    # and Ecut = 25
    ref_redHF = [
        [0.17899118507651615, 0.6327279881297371, 0.6327279881326648, 0.6327279881356039,
         0.706557757783828, 0.7065577577877139, 0.7065577577915956, 0.7397951816714727,
         0.8532089291297222, 0.8978914445971602],
        [0.23220003663858457, 0.42189409862896016, 0.5921574659414509, 0.5921574659446628,
         0.672858189872362, 0.7372271903827399, 0.7372271903861028, 0.8643640848936627,
         0.9011792204214196, 0.9011792204356576],
        [0.2517502116803524, 0.445206025448218, 0.5328870916963034, 0.532887091701182,
         0.6211365856991057, 0.661989858948651, 0.8863951918546257, 0.8863951918584175,
         0.973261179805555, 0.9771287508158364],
        [0.30685586314464863, 0.376375429632464, 0.4438764716222098, 0.5459065154292047,
         0.651122698647485, 0.8164293660861612, 0.8515978828421051, 0.8735213568005982,
         0.8807275612483988, 0.8886454931307763]
    ]
    ref_etot = -5.440593269861395

    n_bands = length(ref_redHF[1])
    fft_size = grid_size * ones(3)
    Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
    model = model_DFT(Array{T}(silicon.lattice), [Si => silicon.positions], [];
                      temperature=0.05, spin_polarisation=:collinear)
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)

    scfres = self_consistent_field(basis, tol=scf_tol, n_bands=n_bands)

    for ik in 1:length(silicon.kcoords)
        @test eltype(scfres.eigenvalues[ik]) == T
        @test eltype(scfres.ψ[ik]) == Complex{T}
        println(ik, "  ", abs.(ref_redHF[ik] - scfres.eigenvalues[ik][1:n_bands]))
    end
    for ik in 1:length(silicon.kcoords)
        # Ignore last few bands, because these eigenvalues are hardest to converge
        # and typically a bit random and unstable in the LOBPCG
        diff = abs.(ref_redHF[ik] - scfres.eigenvalues[ik][1:n_bands])
        @test maximum(diff[1:n_bands - n_ignored]) < test_tol
    end
end
