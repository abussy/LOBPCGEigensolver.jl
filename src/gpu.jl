# GPU specific utilities and performance-optimized implementations.

"""
Transfer an array from a device (typically a GPU) to the CPU.
"""
to_cpu(x::AbstractArray) = Array(x)
to_cpu(x::Array) = x

# GPU-specific implementation: the massive parallelism of the GPU is only fully
# exploited by operating on whole arrays rather than looping over columns.
function columnwise_dots(A::AbstractGPUArray{T}, B::AbstractGPUArray{T}) where {T}
    vec(sum(conj(A) .* B; dims=1))
end

# Apply the Cholesky factorization N times to orthogonalize X. If it fails, e.g. because
# it might be ill-conditioned, fall back to the slower but more robust ortho! function.
function ortho_chol_n!(X::AbstractArray{T}; N::Int=2) where {T}
    nchol_total = 0
    try
        for _ = 1:N
            O = mul_hermi(X', X)
            R = cholesky(O).U
            rdiv!(X, R)
            nchol_total += 1
        end
    catch err
        (X, nchol_total, _) = ortho!(X)
    end

    (; X, nchol=nchol_total)
end

# Faster orthogonalization of X against Y compared to ortho! from lobpcf_impl.jl.
# Constant checks on column norms, estimated errors, etc. create a lot of implicit GPU
# synchronization, which is avoided here. In practice, ortho! usually converges after 2
# iterations or more. Here, we only start checking for convergence after 2 iterations,
# in order to streamline GPU execution
function ortho!(X::AbstractArray{T}, Y, BY; tol=2eps(real(T)),
                     timer=disabled_timer) where {T}

    # normalize to try to cheaply improve conditioning
    X ./= columnwise_norms(X)'

    niter = 1
    while true
        BYX = BY' * X

        if niter > 2
            norm(BYX) < tol && break
        end

        mul!(X, Y, BYX, -1, 1)  # X -= Y*BY'X

        if niter > 2
            @timeit timer "ortho!" X, ninner, growth_factor = ortho!(X; tol)
            estimated_error = growth_factor * eps(real(T))
            estimated_error < tol && break
        else
            @timeit timer "ortho_chol_n!" X, ninner = ortho_chol_n!(X; N=2)
        end

        if niter > 10
            U, _, V = svd(X)  # Fall back to gold standard
            X = U*V'
            @error("Ortho(X, Y) is failing badly, falling back to SVD",
                   error=round(norm(BY'X); sigdigits=2), tol=tol,
                   estimated_error=round(estimated_error; sigdigits=2))
            return X
        end

        niter += 1
    end

    # If the orthogonalization has produced results below tol (unlikely),
    # we revert to the slower ortho!, which knows how to deal with this.
    dropped = drop_small!(X; tol)
    if !isempty(dropped)
        X = safe_ortho!(X, Y, BY; tol=tol, timer=timer)
    end

    X
end
