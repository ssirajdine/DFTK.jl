using CuArrays

function as_cuarray(block::HamiltonianBlock)
    n_bas = length(G_vectors(block.kpt))
    T = eltype(block)
    mat = CuArray{T}(undef, n_bas, n_bas)
    @inbounds for i = 1:n_bas
        v = CuArrays.fill(zero(T), n_bas)
        v[i] = one(T)
        mul!(view(mat, :, i), block, v)
    end
    mat
end


function diag_full(A, X0::CuArray; kwargs...)
    Neig = size(X0, 2)
    Afull = as_cuarray(A)
    λ, X = CuArrays.CUSOLVER.heevd!('V', 'U', Afull)
    X = X[:, 1:Neig]
    λ = λ[1:Neig]
    (λ=λ, X=X, residual_norms=zeros(Neig), iterations=0, converged=true)
end
