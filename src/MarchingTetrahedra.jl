
using StaticArrays, LoopVectorization, ProgressBars, GroupSlices, ThreadsX

export writeOBJ, compressTriangulation, getLargestComponents, marchTet_grid_memoryHungry


function marchTet_grid_memoryHungry(f::Function, mins::Vector, maxes::Vector, steps;
    levels=0.0, 
    interior = (z -> fill(true, size(z,2))), 
    onSurface = (z -> true)
)

    CoordType = Float32
    PointType = SVector{3, CoordType}
    TriangleType = SVector{3, PointType}

    #// Don't heck with this bit, trust me -- it took lots of scribbling to work out
    binTable = Iterators.product((0,1),(0,1),(0,1)) |> collect |> vec .|> reverse
    indexDiffs = [ CartesianIndex(b...) for b in binTable ]
    tetTable = [[2,1,3,5], [4,2,3,5], [4,6,2,5], [4,7,6,5], [4,3,7,5], [4,8,6,7]]
    #//

    if isa(levels, Number)
        levels = [levels]
    end

    if length(steps) == 1
        R = [range(mins[i], maxes[i], step=steps) for i in 1:3]
    elseif length(steps) == 3
        R = [range(mins[i], maxes[i], step=steps[i]) for i in 1:3]
    end

    I = CartesianIndices( ((length.(R)...,) .- 1) )

    x1 = R[1]; x2 = R[2]; x3 = R[3]

    X_list = Iterators.product(R...) |> collect .|> collect
    X = reduce(hcat, X_list |> vec)
    
    println("Finding points interior:")
    Inside = zeros(Bool, length(X_list))
    @time Threads.@threads for k in ProgressBar(axes(X,2))
        Inside[k] = interior(view(X, :, k))
    end
    Inside = reshape( Inside, Tuple(length.(R)) )
    insideIndices = findall(vec(Inside))

    println("Evaluating function ($(length(insideIndices)) points):")

    F = fill(NaN, Tuple(length.(R)))
    # F[vec(Inside)] = f(X_inside)

    numBlocks = Threads.nthreads()
    numPoints = length(insideIndices)
    blockSize = ceil(Int, numPoints / numBlocks)
    splitInsideIndices = [ 
        view(
            insideIndices, 
            range(b*blockSize + 1, min((b+1)*blockSize, numPoints))
        ) 
        for b in 0:numBlocks-1
    ]

    @time Threads.@threads for b in 1:numBlocks
        threadIndices = splitInsideIndices[b]
        F[threadIndices] = f( view(X, :, threadIndices) )
    end

    cubeIndices = [I[1] + d for d in indexDiffs]
    cubePoints = X_list[cubeIndices]
    cubeVals = F[cubeIndices]

    tetPoints = cubePoints[1:4]
    tetVals = cubeVals[1:4]

    # this was a typo but it's too funny to fix
    TRINGLES = Dict{eltype(levels), Vector{TriangleType}}()
    for L in levels
        TRINGLES[L] = Vector{TriangleType}()
    end
    
    println("Creating triangles...")

    @inbounds for i in ProgressBar(I)

        cubeIndices .= [i + d for d in indexDiffs]
        cubePoints .= X_list[cubeIndices]
        cubeVals .= F[cubeIndices]

        if all(Inside[cubeIndices])
            for tetIndices in tetTable
                tetPoints .= cubePoints[tetIndices]
                for L in levels
                    map!(j -> (cubeVals[j] - L), tetVals, tetIndices)
                    addTriangles!(TRINGLES[L], tetPoints, tetVals, onSurface)
                end
            end
        end

    end

    return TRINGLES

end


function interp(x,y,a,b)
    @fastmath z = ( (abs(b)*x + abs(a)*y) / abs(b-a) )
    return z
end

function addTriangles!(triangles, p, v, onSurface=z->true)
    if all(x -> sign(x) == sign(v[1]), v)
        return nothing
    end

    signs = sign.(v)
    signs[signs.==0] .= 1
    if count(isequal(1), signs) < 2
        v *= -1
        signs *= -1
    end

    i = sortperm(v)
    # p = p[i]
    # v = v[i]

    @inbounds if count(isequal(-1), signs) == 1 # vals are [- + + +]
        tri = [ interp(p[i[1]],p[i[2]],v[i[1]],v[i[2]]),
        interp(p[i[1]],p[i[3]],v[i[1]],v[i[3]]),
        interp(p[i[1]],p[i[4]],v[i[1]],v[i[4]]) ]
        if all(onSurface, tri)
            push!(triangles, tri)
        end
        return nothing
    end

    @inbounds if count(isequal(-1), signs) == 2 # vals are [- - + +]
        q = [ interp(p[i[1]],p[i[3]],v[i[1]],v[i[3]]),
            interp(p[i[1]],p[i[4]],v[i[1]],v[i[4]]),
            interp(p[i[2]],p[i[4]],v[i[2]],v[i[4]]),
            interp(p[i[2]],p[i[3]],v[i[2]],v[i[3]]) ]
        if norm(q[1]-q[3]) > norm(q[2]-q[4])
            tri1 = [q[1], q[2], q[4]]
            tri2 = [q[2], q[3], q[4]]
        else
            tri1 = [q[1], q[2], q[3]]
            tri2 = [q[1], q[3], q[4]]
        end
        if all(onSurface, tri1)
            push!(triangles, tri1)
        end
        if all(onSurface, tri2)
            push!(triangles, tri2)
        end
        return nothing
    end

end

function writeOBJ(V::Matrix, F::Matrix{<:Integer}, filename::String)
    fout = open(filename, "w")
    for i = 1:size(V,2)
        println(fout, "v ", V[1,i], " ", V[3,i], " ", V[2,i])
    end
    for j = 1:size(F,2)
        println(fout, "f ", F[1,j], " ", F[3,j], " ", F[2,j])
    end
    close(fout)
end

function writeOBJ(X::Vector{<:AbstractVector}, F::Matrix{<:Integer}, filename::String)
    open(filename, "w") do fout
        for i = 1:length(X)
            println(fout, "v ", X[i][1], " ", X[i][3], " ", X[i][2])
        end
        for j = 1:size(F,2)
            println(fout, "f ", F[1,j], " ", F[3,j], " ", F[2,j])
        end
    end
end

function writeOBJ(triangles::Vector{<:AbstractVector{<:AbstractVector{T}}},
    filename::String) where T<:AbstractFloat

    if isempty(triangles)
        return nothing
    else
        P, F = compressTriangulation(triangles)
        writeOBJ(P, F, filename)
        return nothing
    end

end

function compressTriangulation(triangles::Vector{<:AbstractVector{<:AbstractVector{T}}}) where T

    points = reduce(vcat, triangles)

    uniqueInds = groupslices(points)
    usedPts = unique(uniqueInds)

    n = length(points)

    newIdx = zeros(Int, n)
    newIdx[usedPts] .= 1:length(usedPts)

    faces = collect(reshape(1:n, (3, Int(n/3))))
    faces = uniqueInds[faces]

    return points[usedPts], newIdx[faces]

end

function getLargestComponents(X, T, kmax)

    labels, counts = groupTris(X, T)
    K = sortperm(counts, rev=true)[1:kmax]
    T_k = [T[:, labels .== k] for k in K]
    return T_k

end

function labelTriAndNbrs!(labels, seed, nbrs, colour)

    @assert labels[seed] == 0
    q = [seed]
    labels[seed] = colour
    count = 1

    while !isempty(q)

        i = pop!(q)
        for j in nbrs[i]
            labels[j] == 0 || continue
            labels[j] = colour
            count += 1
            push!(q, j)
        end

    end

    return count

end

function groupTris(X,T)

    nbrs = buildTriNbrs(T)
    labels = zeros(Int, size(T,2))
    colour = 0
    counts = Dict{Int, Int}()
    while any(==(0), labels)
        seed = findfirst(==(0), labels)
        colour += 1
        counts[colour] = labelTriAndNbrs!(labels, seed, nbrs, colour)
    end
    countsVec = zeros(Int, maximum(keys(counts)))
    for (label,count) in counts
        countsVec[label] = count
    end
    return labels, countsVec

end

function buildEdge2Tris(T::Matrix{S}) where S

    edgeIndices = [ setdiff(1:3, j) for j in 1:3 ]
    edge2Tris = Dict{Set{S}, Vector{Int}}()
    for i in axes(T,2)
        for j in 1:3
            edge = Set( T[edgeIndices[j], i] )
            tris = get!(edge2Tris, edge, sizehint!(Int[], 2))
            push!(tris, i)
        end
    end
    return edge2Tris

end

function buildTriNbrs(T::Matrix)

    edge2Tris = buildEdge2Tris(T)

    triNbrs = [ Set{Int}() for _ in axes(T,2) ]
    for tris in values(edge2Tris)
        for i in tris
            union!(triNbrs[i], tris)
            if length(triNbrs) == 1
                union!(triNbrs[i], 0)
            end
        end
    end

    for i in axes(T,2)
        setdiff!(triNbrs[i], i)
    end

    return triNbrs

end

function fixFlippedOrThinFaces(T_in::Matrix)

    n = size(T_in, 2)

    thin = zeros(Bool, n)
    for (i,t) in enumerate(eachcol(T_in))
        if length(unique(t)) < 3
            thin[i] = true
        end
    end

    T = T_in[:, .!thin]
    n = size(T, 2)

    seed = 1

    nbrs_sets = buildTriNbrs(T)
    nbrs = collect.(nbrs_sets)

    passed = zeros(Bool, n)
    queue = [seed]

    while !isempty(queue)

        i = popfirst!(queue)

        if passed[i]
            continue
        end
        
        passedNbrIndex = findfirst(k -> passed[k], nbrs[i])

        if passedNbrIndex === nothing
            # triangle i doesn't have any neighbours setting an orientation
        else
            j = nbrs[i][passedNbrIndex]
            # make sure the edge common to triangle i and j appears in opposite orders
            edge = intersect(T[:,i], T[:,j])
            edge_i_locs = [ findfirst(==(vert), T[:,i]) for vert in edge ]
            edge_j_locs = [ findfirst(==(vert), T[:,j]) for vert in edge ]
            if diff(edge_i_locs)[1] % 3 != diff(edge_j_locs)[1] % 3
                T[:,i] = reverse(T[:,i])
            end
        end

        passed[i] = true

        for k in nbrs[i]
            if !passed[k]
                push!(queue, k)
            end
        end

        if isempty(queue) && !all(passed)
            push!(queue, findfirst(!, passed))
        end

    end

    return T

end