
# You could use just about any other interpolation method in place 
# of RBFinterpolant here, but I am too lazy for that. 

using LinearAlgebra, NearestNeighbors, Distances, ProgressBars, Base.Threads, 
LoopVectorization, StructArrays, StaticArrays, FLoops

include("DomainPartitionTools.jl")
include("RBFinterpolation.jl")

export RBFPUMinterpolant, fit!, sample, sample_and_grad


"""
RBFPUMinterpolant
"""
struct RBFPUMinterpolant{T, INTERP_TYPE}
    Nbounds
    expand
    Ω :: BallPartition
    Fi :: Vector{INTERP_TYPE}
    function RBFPUMinterpolant(x, f, φ, ρ, pdeg, w, Nbounds, expand)
        (Ω, tree) = BallPartition(x, w, Nbounds, expand)
        Fi = RBFinterpolant{eltype(x)}[]
        for (c, r) in Ω
            J = inrange(tree, c, r)
            push!(Fi, RBFinterpolant(x[:,J], f[J], φ, ρ, pdeg))
        end
        return new{eltype(x), eltype(Fi)}(Nbounds, expand, Ω, Fi)
    end
end


"""
"""
function fit!(F::RBFPUMinterpolant)
    
    println("Fitting RBFPUM interpolant... ")
    
    Nmax = F.Nbounds[2]         # the upper bound on num points in subdomain
    np = F.Fi[1].p.numTerms     # num polynomial terms
    nt = nthreads()
    println("Using $nt threads.")
    
    @floop for i in 1:length(F.Fi) # for each RBFinterpolant in the partition
        @init begin
            A = zeros(Nmax+np,Nmax+np)      # prealloc interpolation matrices
            D = zeros(Nmax,Nmax)
            P = zeros(np,Nmax)              # prealloc polynomial matrices
            b = zeros(Nmax+np)              # prealloc rhs vectors
            soln = zeros(Nmax+np)           # prealloc solution vectors
        end
        Fi = F.Fi[i]
        Ni = Fi.N
        Ai = view(A, 1:Ni+np, 1:Ni+np)
        Di = view(D, 1:Ni, 1:Ni)
        Pi = view(P, 1:np, 1:Ni)
        bi = view(b, 1:Ni+np)
        solni = view(soln, 1:Ni+np)
        try
            localfit!(Fi, Ai, Di, Pi, bi, solni)
        catch ex
            if ex isa SingularException
                println("Singular exception in subdomain $i; info $(ex.info)")
            end
            throw(ex)
        end
    end
    return nothing
end


"""
"""
function sample(F::RBFPUMinterpolant, y::AbstractMatrix; return_wsums=false, doProgressBar=size(y,2)>10_000)
    
    @assert size(y,1) == F.Ω.dim
    yTree = KDTree(y)
    Ny = size(y, 2)
    
    itr = 1:F.Ω.n
    if doProgressBar
        println("Evaluating RBFPUM interpolant at $Ny points...")
        itr = ProgressBar(itr)
    end
    
    wsums = zeros(Ny)
    results = zeros(Ny)
    dists_store = zeros(Ny)
    samples_store = zeros(Ny)

    w = F.Ω.w

    for i in itr

        cᵢ = F.Ω.C[:,i]
        rᵢ = F.Ω.R[i]

        Ji = inrange(yTree, cᵢ, rᵢ)
        
        if isempty(Ji)
            continue
        end

        Fᵢ = F.Fi[i]

        yJ = y[:, Ji]
        dists = @view dists_store[1:length(Ji)]
        samples = @view samples_store[1:length(Ji)]

        colwise!(dists, Euclidean(), yJ, cᵢ)
        
        map!(dists, dists) do d
            w.(d / rᵢ)
        end
        
        wsums[Ji] .+= dists
        
        samples .= sample2(Fᵢ, yJ)

        results[Ji] .+= dists .* samples
        
    end
    
    if return_wsums
        return ( results, wsums )
    else
        map!(/, results, results, wsums)
        return results
    end
    
end


function sample(F::RBFPUMinterpolant, y::AbstractVector{<:AbstractVector{T}}; 
    return_wsums=false, doProgressBar=false) where T

    dim = length(y[1])
    n = length(y)
    Y = reshape( reinterpret(T, y), (dim, n) )

    return sample(F, Y; return_wsums, doProgressBar)

end



function sample_and_grad(F::RBFPUMinterpolant, y::AbstractVector{T}, dφdr, dwdτ) where T <: Number

    Σwᵢfᵢ = 0.0
    ∇Σwᵢfᵢ = @SVector zeros(F.Ω.dim)

    Σw = 0.0
    ∇Σw = @SVector zeros(F.Ω.dim)

    C = F.Ω.C
    R = F.Ω.R
    w = F.Ω.w
    Fi_list = F.Fi

    @inbounds for i in 1:length(Fi_list)

        Fi = Fi_list[i]

        y_ci = y - @view(C[:,i])
        norm_y_ci = norm(y_ci)
        τ = norm_y_ci / R[i]

        if τ ≥ 1
            continue
        end

        wi = w(τ)
        ∇wi = dwdτ(τ) * y_ci / R[i] / norm_y_ci

        Σw += wi
        ∇Σw += ∇wi

        fi, ∇fi = sample_and_grad(Fi, y, Fi.φ, dφdr)

        Σwᵢfᵢ += wi * fi
        ∇Σwᵢfᵢ += ∇wi*fi + wi*∇fi # hehe, wifi

    end

    if Σw == 0
        fx = NaN
    else
        fx = Σwᵢfᵢ / Σw
    end

    ∇fx = -∇Σw / (Σw^2) * Σwᵢfᵢ + 1/(Σw) * ∇Σwᵢfᵢ

    return fx, ∇fx

end


function Base.show(io::IO, F::RBFPUMinterpolant)
    println(io, typeof(F))
    println(io, "  Dimension: $(F.Ω.dim)")
    println(io, "  Num points (incl. repeats): $(sum([Fi.N for Fi in F.Fi]))")
    print(io, "  Num subdomains: $(F.Ω.n)")
end
