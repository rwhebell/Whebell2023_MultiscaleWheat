export affineTform

using NearestNeighbors, Distances, Statistics, Random, StaticArrays, LinearAlgebra


"""
BallPartition

Struct to represent a partition of unity with hyperspheres.

C: centres of spheres

R: radii of spheres

w: weight function
"""
struct BallPartition{T}
    C :: Array{T,2}        # ball centres
    R :: Array{T,1}        # ball radii
    w :: Function               # weight function
    n :: Int64                  # number of subdomains
    dim :: Int64                # dimension of pts
    function BallPartition(C,R,w)
        @assert size(C,2) == length(R)
        (dim, n) = size(C)
        return new{eltype(C)}(C,R,w,n,dim)
    end
end


"""
PU(i::Int64, x::Array{Real,2})

Partition of unity weight function at points x.
"""
function (PU::BallPartition)(i::Int64, x::Array{<:AbstractFloat,2}; tree::KDTree = KDTree(x))
    J = inrange(tree, PU.C[:,i], PU.R[i])
    wx = zeros(size(x,2))
    c = reshape(PU.C[:,i], (PU.dim,1))
    d = pairwise(Euclidean(1e-12), x[:,J], c)[:]
    wx[J] .= PU.w.( d )
    return wx
end


"""
(PU, T) = BallPartition(x, w, Nbounds, expand)

builds an octree-like domain partition using the sample points x, and
returns the tuple (PU, T) where T is the KDTree used to do space searching.
"""
function BallPartition(x::Array{<:AbstractFloat,2}, w, Nbounds, expand)
    T = KDTree(x)
    PU = BallPartition(T, w, Nbounds, expand)
    return (PU, T)
end


"""
PU = BallPartition(T, w, Nbounds, expand)

builds an octree-like domain partition using the sample points in the KDTree T.
"""
function BallPartition(T::KDTree, w, Nbounds, expand)
    (C, R) = buildOctreePU(T, Nbounds, expand)
    return PU = BallPartition(C, R, w)
end


"""
buildOctreePU(T, Nbounds, expand)

Internal function to do octree-like domain partition into hyper-spheres
"""
function buildOctreePU(T::KDTree, Nbounds, expand)

    dim = length(T.data[1])

    r(s) = expand * sqrt(dim) * (s/2)

    Nmin = Nbounds[1]; Nmax = Nbounds[2]

    centres = [ 0.5.*(T.hyper_rec.mins .+ T.hyper_rec.maxes) ]
    sideLens = [ maximum( T.hyper_rec.maxes .- T.hyper_rec.mins ) ]
    radii = [ r(sideLens[1]) ]

    counts = [length(T.data)]

    # Subdivide until no subdomain contains more than Nmax pts
    while maximum(counts) > Nmax
        (boop, i) = findmax(counts)
        L = sideLens[i]
        newL = L/2

        shifts = collect.(collect(setprod([-0.25L, 0.25L], dim))[:])
        newC = [centres[i]] .+ shifts
        
        deleteat!(centres, i)
        append!(centres, newC)

        deleteat!(sideLens, i)
        append!(sideLens, fill(newL, 2^dim))

        deleteat!(radii, i)
        append!(radii, fill(r(newL), 2^dim))

        newCounts = zeros(2^dim)
        for (j,c) in enumerate(newC)
            newCounts[j] = length( inrange(T, newC[j], r(newL)) )
        end

        deleteat!(counts, i)
        append!(counts, newCounts)
    end

    # Trim empty partitions
    trim = counts .== 0
    deleteat!(centres, trim)
    deleteat!(sideLens, trim)
    deleteat!(radii, trim)
    deleteat!(counts, trim)

    # Expand too-small partitions to include at least Nmin pts
    if length(counts) > 1
        for (i, c) in enumerate(centres)
            if counts[i] < Nmin
                (ind, dists) = knn( T, c, Nmin )
                radii[i] = maximum(dists)
            end
        end
    end

    centres = hcat(centres...)

    p = randperm(length(radii))
    
    return (centres[:,p], radii[p])

end


function Base.show(io::IO, PU::BallPartition)
    println(io, typeof(PU))
    println(io, "  Number of subdomains: $(PU.n)")
    println(io, "  Dimensions: $(size(PU.C, 1))")
end

function Base.iterate(Ω::BallPartition, i::Int64=1)
    if i > Ω.n
        return nothing
    end
    return ((Ω.C[:,i], Ω.R[i]), i+1)
end

function Base.length(PU::BallPartition)
    return PU.n
end