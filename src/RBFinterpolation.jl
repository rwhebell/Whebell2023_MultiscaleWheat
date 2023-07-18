
using LinearAlgebra, Distances, LoopVectorization

include("MultiVarPolynomials.jl")


# ==========================================================================
# CORE FUNCTIONALITY

struct RBFinterpolant{T}
    x :: Array{T, 2}                # points
    f :: Vector{T}                  # samples
    d :: Int                        # dimension
    N :: Int                        # num points
    φ :: Function                   # radial basis function
    ρ :: T                          # smoothing parameter
    λ :: Vector{T}                  # interpolation weights
    p :: MPoly                      # multivariate polynomials
end

function RBFinterpolant(x::Matrix{T}, f::Vector, φ::Function, 
    ρ::Real, polydeg::Integer) where T
    (d,N) = size(x)
    λ = zeros(T,N)
    p = MPoly(polydeg, d) # zero polynomial
    return RBFinterpolant{T}(x, f, d, N, φ, ρ, λ, p)
end


"""
"""
function localfit!(F::RBFinterpolant)
    D = pairwise(Euclidean(1e-12), F.x, dims=2)
    P = evalMonomials(F.p, F.x)
    n = F.p.numTerms
    N = F.N 

    A = Symmetric([F.φ.(D) + F.ρ*N*I transpose(P); P zeros(n,n)])
    b = [ F.f; zeros(n) ]

    soln = A \ b

    F.λ .= soln[1:F.N]
    setcoeffs!(F.p, soln[N+1:end])

    return soln
end


function localfit!(F::RBFinterpolant{T}, A, D, P, b, soln) where T

    n = F.p.numTerms
    N = F.N

    pairwise!(D, Euclidean(), F.x, dims=2)

    # A[N+1:N+n, 1:N] .= P
    evalMonomials!(F.p, F.x, view(A, N+1:N+n, 1:N))
    A[1:N, N+1:N+n] .= transpose(A[N+1:N+n, 1:N])

    # A[1:N, 1:N] .= F.φ.(D) + (F.ρ*N)*I
    map!(F.φ, view(A, 1:N, 1:N), D)
    for i in 1:N
        A[i,i] += F.ρ*N
    end

    A[N+1:N+n, N+1:N+n] .= 0.0
    b[1:N] .= F.f
    b[N+1:N+n] .= 0.0
    
    ldiv!(soln, lu!(A), b)

    F.λ .= soln[1:F.N]
    setcoeffs!(F.p, soln[N+1:end])

    return soln

end


"""
"""
function sample(F::RBFinterpolant, y::AbstractMatrix{T}) where T
    
    @assert size(y,1) == F.d

    Ny = size(y,2)
    Fy = zeros(Ny)
    blockLen = 1
    blockIter = BlockIterator(Ny, blockLen)

    D_full = zeros(blockLen, F.N)

    ys = reinterpret( SVector{F.d, T}, y )

    φ = F.φ
    x = F.x
    λ = F.λ
    p = F.p

    @views @inbounds @fastmath for block in blockIter

        D = D_full[ 1:length(block), : ]

        pairwise!(D, Euclidean(), y[:,block], x; dims=2) # distance matrix

        map!(φ, D, D) # apply RBF to distance matrix

        mul!(Fy[block], D, λ)
        
        addPolynomial!(Fy[block], p, ys[block])

    end

    return Fy

end



function sample2(F::RBFinterpolant, y::AbstractMatrix{T}) where T
    
    @assert size(y,1) == F.d

    Ny = size(y,2)
    Fy = zeros(Ny)

    D = zeros(F.N)

    φ = F.φ
    x = F.x
    λ = F.λ
    p = F.p

    @inbounds for (i,yi) in enumerate(eachcol(y))

        colwise!(D, Euclidean(1e-10), yi, x) # distances

        @turbo D .= φ.(D)

        Fy[i] = D ⋅ λ
        Fy[i] += p(yi)

    end

    return Fy

end



function sample(F::RBFinterpolant, y::AbstractVector{<:AbstractVector})
    
    @assert length(y[1]) == F.d

    Ny = length(y)
    Fy = zeros(Ny)
    blockLen = min(100, length(y))
    blockIter = BlockIterator(Ny, blockLen)

    D_full = zeros(blockLen, F.N)

    x = reinterpret( SVector{F.d, eltype(F.x)}, F.x )

    @views @inbounds @fastmath for block in blockIter

        D = D_full[ 1:length(block), : ]

        pairwise!(D, Euclidean(), y[block], x)

        map!(F.φ, D, D)
        mul!(Fy[block], D, F.λ)
        
        addPolynomial!(Fy[block], F.p, y[block])

    end

    return Fy

end


function smallSample!(F::RBFinterpolant{T}, y, Fy, D) where T

    Ny = length(y)

    x = reinterpret( SVector{F.d, T}, F.x )

    pairwise!(D, Euclidean(), y, x)
    map!(F.φ, D, D)
    mul!(Fy, D, F.λ)
    
    addPolynomial!(Fy, F.p, y)

    return Fy

end 


function sample_and_grad(F::RBFinterpolant{S}, y::AbstractVector{T}, φ, dφdr) where {S, T <: Number}

    Fx = zero(T)
    ∇Fx = 0 * y

    x = F.x
    λ = F.λ
    
    @fastmath @inbounds for j in 1:F.N

        yxj = y - SVector{ length(y), T }(view(x,:,j))
        r = norm(yxj)

        Fx += λ[j] * φ(r)

        ∇Fx += λ[j] / r * dφdr(r) * yxj

    end

    Fx += F.p(y)
    ∇Fx += ∇p(F.p, y)

    return Fx, ∇Fx

end


function Base.show(io::IO, F::RBFinterpolant)
    println(io, typeof(F))
    println(io, "  Dimension: $(F.d)")
    println(io, "  Num points: $(F.N)")
    println(io, "  Basis fun: $(F.φ)")
    print(io, "  Smoothing parm: $(F.ρ)")
end