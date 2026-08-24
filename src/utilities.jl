# Small self-contained helpers used by the LOBPCG implementation. They are
# intentionally kept minimal (only what the solver needs) so the package stays
# free of any application-specific dependencies. Device-agnostic versions are
# defined for generic `AbstractArray`s, with GPU-optimized variants dispatching
# on `AbstractGPUArray`.

"""
Transfer an array from a device (typically a GPU) to the CPU.
"""
NVTX.@annotate to_cpu(x::AbstractArray) = Array(x)
NVTX.@annotate to_cpu(x::Array) = x

"""
Create an array of the same "array type" as `X` filled with zeros, minimizing the
number of allocations. This unifies CPU and GPU code, as the output always lands
on the same device as the input.
"""
NVTX.@annotate function zeros_like(X::AbstractArray, T::Type=eltype(X), dims::Integer...=size(X)...)
    Z = similar(X, T, dims...)
    Z .= zero(T)
    Z
end
NVTX.@annotate zeros_like(X::AbstractArray, dims::Integer...) = zeros_like(X, eltype(X), dims...)
NVTX.@annotate zeros_like(X::Array, T::Type=eltype(X), dims::Integer...=size(X)...) = zeros(T, dims...)

# Calculate the norms of the columns of an array
NVTX.@annotate function columnwise_norms(X::AbstractArray)
    vec(sqrt.(sum(abs2, X; dims=1)))
end

# Returns a vector of dot(A[:, i], B[:, i]), for all columns of A, B
NVTX.@annotate function columnwise_dots(A::AbstractArray{T}, B::AbstractArray{T}) where {T}
    @views [dot(A[:, i], B[:, i]) for i = 1:size(A, 2)]
end

# GPU-specific implementation: the massive parallelism of the GPU is only fully
# exploited by operating on whole arrays rather than looping over columns.
NVTX.@annotate function columnwise_dots(A::AbstractGPUArray{T}, B::AbstractGPUArray{T}) where {T}
    vec(sum(conj(A) .* B; dims=1))
end

NVTX.@annotate format_log8(e) = @sprintf "%8.2f" log10(abs(e))

# Compact human-readable rendering of a duration given in nanoseconds, to ~3 significant
# figures with an adaptive unit, e.g. "4.56s", "27.7ms", "812μs". Used by the default
# callback's time column; the stdlib alternative (`Dates.canonicalize`) is coarser and
# verbose ("4 seconds, 560 milliseconds"), so we format it here.
NVTX.@annotate function prettytime(t)
    value, units = t < 1e3 ? (t,       "ns") :
                   t < 1e6 ? (t / 1e3, "μs") :
                   t < 1e9 ? (t / 1e6, "ms") :
                             (t / 1e9, "s")
    if round(value) >= 100
        @sprintf("%.0f%s", value, units)
    elseif round(value * 10) >= 100
        @sprintf("%.1f%s", value, units)
    else
        @sprintf("%.2f%s", value, units)
    end
end

# Preconditioner API (also used in Optim.jl): if the solver supports adaptive
# preconditioning it will call `precondprep!(P, X)` right before applying the
# preconditioner. The default is a no-op; concrete preconditioners may override it.
NVTX.@annotate precondprep!(P, X) = P
