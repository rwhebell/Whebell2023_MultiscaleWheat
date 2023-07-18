using LinearAlgebra, Distances, MiniQhull,
    Random, StaticArrays, NearestNeighbors

using StatsBase: sample as randomSample

export AlphaShape, inshape, boundaryFacets, findBoundary


struct AlphaShape{TypeX, TypeT, d}
    X::Matrix{TypeX}
    T::Matrix{TypeT}
    circumcenters::Vector{SVector{d, TypeX}}
    radii::Vector{TypeX}
    alpha::TypeX
    elemInShape::Vector{Bool}
    edge2elem::Dict{Vector{TypeT}, Vector{Int}}
    boundary::Matrix{TypeT}
    normals::Vector{SVector{d, TypeX}}
    origins::Vector{SVector{d, TypeX}}
    barycentricTforms::Vector{SMatrix{d, d, TypeX}}

    function AlphaShape(X, T, alpha)

        d, nx = size(X)
        s, nt = size(T)
        @assert d+1 == s

        radii = zeros(nt)
        circumcenters = [ @SVector zeros(d) for _ in 1:nt ]
        for (i,t) in enumerate(eachcol(T))
            Xt = X[:,t]
            Xt = Xt .- mean(Xt, dims=2)
            circumcenters[i], radii[i] = circumsphere(Xt)
        end

        elemInShape = radii .< alpha

        edge2elem = buildIncidence(T)

        boundary = findBoundary(T[:, elemInShape])
        normals = [ zeros(d) for i in eachcol(boundary) ]
        origins = [ zeros(d) for i in eachcol(boundary) ]
        barycentricTforms = [ zeros(d,d) for i in eachcol(boundary) ]

        for (i,t) in enumerate(eachcol(boundary))

            origins[i] = X[:, t[1]]
            P = X[:, t[2:end]]
            P .-= origins[i]
            
            n = nullspace( transpose( P ) )
            normals[i] = size(n,2)==1 ? vec(n) : zeros(d)
            
            barycentricTforms[i] = pinv([P normals[i]])

        end

        return new{eltype(X), eltype(T), d}(
            X, T, circumcenters, radii, alpha, elemInShape, edge2elem, boundary, normals,
            origins, barycentricTforms
        )

    end

end


"""
    ashape = AlphaShape(X, alpha=Inf)
"""
function AlphaShape(X, alpha=Inf)
    T = delaunay(X)
    return AlphaShape(X, T, alpha)
end



"""
    tf = inshape(ashape, y)
"""
function inshape(ashape::AlphaShape, y::AbstractVector)
    id = inTriangulation(ashape.X, ashape.T, ashape.edge2elem, ashape.circumcenters, ashape.radii, y)
    if id < 0
        # @warn "intriangulation failed, last id = $(-id)"
        return true
    else
        return (id != 0) && ashape.elemInShape[id]
    end
end


"""
    V, F = boundaryFacets(ashape)
"""
function boundaryFacets(ashape::AlphaShape)

end


"""
    c, r = circumsphere(V::Matrix)

Calculate the circumcenter, `c`, and the circumradius, `r`, of
a simplex with points defined by the columns of `V`.

Results not guaranteed if the number of points is not equal to 
their dimension plus one.

Ref: https://math.stackexchange.com/questions/4056099/circumcenter-of-the-n-simplex
"""
function circumsphere(V::Matrix{S}) where S
    s, nv = size(V)
    A = [ 2*transpose(V)*V  ones(S,nv,1);  ones(S,1,nv)  zero(S) ]
    b = [ [ norm(V[:,i])^2 for i in 1:nv ]; one(S) ]
    alpha = (A\b)[1:nv]
    c = V*alpha
    r = norm(V[:,1] - c)
    return c, r
end


function findBoundary(T::Matrix{S}) where S<:Integer
    # todo: this could call buildIncidence
    d, n = size(T)
    occurence = Dict{Vector{S}, Int}()
    for i in 1:d
        inds = setdiff(1:d, i)
        for j in 1:n
            edge = T[inds,j] |> vec |> sort
            occurence[edge] = get(occurence, edge, 0) + 1
        end
    end
    boundaries = filter(p -> p.second == 1, occurence) |> keys |> collect
    return [ boundaries[i][j] for j in 1:d-1, i in 1:length(boundaries) ]
end


function pickStartingEdge(X, T, q)

    d, nx = size(X)
    k, nt = size(T)

    m = ceil(Int, nx^(1/4))

    sampleIDs = randomSample(1:nt, m, replace=false)

    _, closestInd = findmin(sampleIDs) do i
        minDist = minimum(1:k) do j
            SqEuclidean()(@view(X[:, T[j,i]]), q)
        end
    end

    closest = sampleIDs[closestInd]
    
    # Starting edge
    edge = T[1:k-1, closest] |> sort
    
    return edge
    
end

# This function is /far/ from perfect, and not well optimised, but it works well for 2D alpha-shapes
function inTriangulation(X, T, incidence, C, R, q)::Int
    # Ernst P. Mücke, Isaac Saias, Binhai Zhu (2009).
    # "Fast randomized point location without preprocessing in two- and three-dimensional
    # Delaunay triangulations"
    # Computational Geometry.

    dolog = false

    # start with m random points in the triangulation
    
    k, nt = size(T)
    dim, _ = size(X)
    
    edge = pickStartingEdge(X, T, q)
    new_edge = zero(edge)
    inds = zeros(Int, k-1)

    id = incidence[edge][1]
    elem = @view T[:,id]
    oppPoint = getOppPoint(elem, edge)

    # TODO: abolish this storage of n
    # Just reverse the order of P's columns
    # or better still, reverse `edge`
    P = MMatrix{size(X[:, edge])...}(X[:, edge])
    n = X[:, oppPoint]

    Pq = 0.0 .* P
    Pn = 0.0 .* P

    A = zeros(dim+1, k)
    b = zeros(dim+1)

    if length(incidence[edge]) > 1
        if !testPointSign!(Pq, Pn, P, n, q)
            id = incidence[edge][2]
            elem = @view T[:,id]
            oppPoint = getOppPoint(elem, edge)
            n .= @view X[:, oppPoint]
        end
    end

    # Is this the one? Could it be?
    if inSimplex!( A, b, C[id], R[id], @views(X[:, elem]), q )
        dolog && println("\t elem $id = $elem contains the point!")
        return id
    else
        dolog && println("\t elem $id does not contain the point")
    end

    # note: face k of a simplex has all the vertices but the kth.
    # eg., for triangle (a,b,c), edge 1 is (b,c), edge 2 is (a,c), and edge 3 is (a,b).

    it = 0
    while true

        it += 1
        if it > 1000
            dolog = true
        end
        if it > 1005
            dolog = false
        end

        dolog && println("start loop: edge $edge, elem $id = $(T[:,id])")
        dolog && println("q = $q")

        if length(incidence[edge]) == 1

            dolog && println("\t boundary edge")

            elem = @view T[:,id]
            oppPoint = getOppPoint(elem, edge)
            P .= @view X[:, edge]
            n .= @view X[:, oppPoint]

            if testPointSign!(Pq, Pn, P, n, q)
                dolog && println("\t point outside, exiting")
                # this is a boundary edge and the point is outside
                return 0
            end

            dolog && println("\t point not outside")

        else

            # Pick the next element
            dolog && println("\t pivot to the other elem containing this edge")

            if id == incidence[edge][1]
                id = incidence[edge][2]
            else
                id = incidence[edge][1]
            end

            elem = @view T[:,id]
            oppPoint = getOppPoint(elem, edge)
            P .= @view X[:, edge]
            n .= @view X[:, oppPoint]

            dolog && println("\t picked elem $id = $(T[:,id])")
            dolog && println("\t norm(n - q) = $(norm(n - q))")
            dolog && print("\t this should be false: ")
            dolog && @show testPointSign!(Pq, Pn, P, n, q)

        end

        # Is this the one? Could it be?
        if inSimplex!( A, b, C[id], R[id], @views(X[:, T[:,id]]), q )

            dolog && println("\t elem $id = $(T[:,id]) contains the point!")
            return id

        else

            dolog && println("\t elem $id does not contain the point")

        end

        # If not, pick a new edge of the new element
        thisIsTrueItsBad = true
        dolog && println("\t choosing new edge")

        for f in shuffle(1:k)

            inds[1:f-1] .= 1:f-1
            inds[f:end] .= f+1:k
            new_edge .= @view T[inds, id]
            sort!(new_edge)
            if new_edge == edge
                continue
            end

            new_oppPoint = getOppPoint(elem, new_edge)
            P .= @view X[:, new_edge]
            n .= @view X[:, new_oppPoint]
            
            dolog && print("\t testing edge $new_edge...")
            if testPointSign!(Pq, Pn, P, n, q) # this has to be true for one of the edges
                thisIsTrueItsBad = false
                edge .= new_edge
                dolog && println(" point is on positive side! goto start")
                break
            else
                dolog && println(" query point on negative side")
            end

        end

        if thisIsTrueItsBad
            dolog && println("Ah!")
            return -id
        end

    end

end


function findNbrs(incidence, id, T)

    nbrs = Set{Int}([])
    for k in 1:size(T,1)
        edgek = T[[1:k-1; k+1:end], id] |> vec |> sort!
        union!(nbrs, incidence[edgek])
    end
    setdiff!(nbrs, id)
    return collect(nbrs)

end


function getOppPoint(elem, edge)

    return elem[ findfirst(!in(edge), elem) ]

end


function buildIncidence(T::Matrix{TypeT}) where TypeT
    d, nt = size(T)
    incidence = Dict{Vector{TypeT}, Vector{Int}}()
    for i in 1:d
        inds = [1:i-1; i+1:d]
        for t in 1:nt
            edge = T[inds,t] |> sort
            if edge ∈ keys(incidence)
                push!(incidence[edge], t)
            else
                incidence[edge] = [t]
            end
        end
    end
    return incidence
end


function testPointSign(P, n, q)

    Pq = 0 .* P
    Pn = 0 .* P

    return testPointSign!(Pq, Pn, P, n, q)

end

function testPointSign!(Pq, Pn, P, n, q)

    # P has d-1 points in R^d as its columns.
    # (these uniquely define a hyperplane)
    # n is a point on the negative side of the plane.
    # q is the query point.
    # Is q on the positive side of the plane?

    d, m = size(P)
    
    # In dimension d, the face has d points.
    @assert d == m
    @assert size(Pq) == size(Pn) == (d,m)

    leaveOut = 1
    smallDetWarning = true
    detq = 0.0
    detn = 0.0

    while smallDetWarning && leaveOut ≤ m

        @inbounds begin
            
            Pq .*= 0.0
            Pn .*= 0.0

            Pq[:, 1:leaveOut-1] .= P[:, 1:leaveOut-1]
            Pn[:, 1:leaveOut-1] .= P[:, 1:leaveOut-1]

            Pq[:, leaveOut:m-1] .= P[:, leaveOut+1:m]
            Pn[:, leaveOut:m-1] .= P[:, leaveOut+1:m]

            Pq[:,m] .= q
            Pn[:,m] .= n

            Pq .-= P[:, leaveOut]
            Pn .-= P[:, leaveOut]

        end

        detq = det(Pq)
        detn = det(Pn)

        smallDetWarning = (abs(detq) < sqrt(eps())) || (abs(detn) < sqrt(eps()))

        leaveOut += 1

    end

    q_sign = sign(detq)
    n_sign = sign(detn)

    return q_sign != n_sign

    # determinant-based check:
    # https://math.stackexchange.com/questions/2214825/determinant-connection-to-deciding-side-of-plane

end


function inSimplex!(A, b, c, r, P, q)::Bool

    if SqEuclidean()(c, q) > r^2
        return false
    end

    d, n = size(P)
    @assert n == d+1
    @assert size(A,1) == d+1
    @assert size(A,2) == n
    @assert size(b) == size(q).+1

    A[1:end-1, :] = P .- P[:,1]
    A[end, :] .= 1

    b[1:end-1] = q - P[:,1]
    b[end] = 1

    @fastmath A_fac = lu!(A, check=false)

    if !issuccess(A_fac)
        A[1:end-1, :] = P .- P[:,1]
        A[end, :] .= 1
        A_fac = qr!(A)
    end

    @fastmath lambda = ldiv!(A_fac, b)
    return all(>(-1e-2), lambda)

end



function listview(X::AbstractMatrix)
    return [ view(X,:,i) for i in axes(X,2) ]
end
