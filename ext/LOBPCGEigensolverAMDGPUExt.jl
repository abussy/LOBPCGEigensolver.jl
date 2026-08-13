module LOBPCGEigensolverAMDGPUExt
using AMDGPU
using LinearAlgebra

# Workarounds for LinearAlgebra routines that LOBPCG relies on, but that are
# currently broken or slow for AMDGPU `ROCArray`s. As of August 2026, it's
# only the 5-argument mul!

# Temporary workaround for 5-argument mul!, where performance is very bad when array
# element types and scaling factors types differ.
# See https://github.com/JuliaGPU/AMDGPU.jl/issues/866#issuecomment-3636981853
# Scaling a Float/Complex matrix with an Integer:
function LinearAlgebra.mul!(C::AMDGPU.ROCArray{T}, A::AMDGPU.ROCArray{T}, B::AMDGPU.ROCArray{T},
                            α::U, β::U) where {T<:Union{AbstractFloat,Complex}, U<:Integer}
    LinearAlgebra.mul!(C, A, B, T(α), T(β))
end
# Scaling a Complex matrix with a Float:
function LinearAlgebra.mul!(C::AMDGPU.ROCArray{T}, A::AMDGPU.ROCArray{T}, B::AMDGPU.ROCArray{T},
                            α::U, β::U) where {T<:Complex, U<:AbstractFloat}
    LinearAlgebra.mul!(C, A, B, T(α), T(β))
end

end
