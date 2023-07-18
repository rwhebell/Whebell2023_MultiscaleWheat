using PlyIO, NearestNeighbors, Distances, LinearAlgebra, StaticArrays,
Statistics, Graphs, SimpleWeightedGraphs, LoopVectorization, 
Base.Threads

export denoise, PointCloud, readPointCloud, writePointCloud, 
list2Mat, estimateNormals!, orientNormals!, downsample, mat2List,
readMatPointCloudWithNormals


"""
PointCloud(points::Matrix[; normals=zeros(size(points)), tree=KDTree(points)])

Create an instance of PointCloud with a d x N matrix points; and
optionally a d x N matrix normals, and/or a KDTree indexing the points.
"""
struct PointCloud{T<:AbstractFloat, D}
    points :: Vector{SVector{D,T}}
    normals :: Vector{SVector{D,T}}
    tree :: U where U<:KDTree
    function PointCloud(points::Matrix{T}; 
        normals = zeros(size(points)),
        tree = KDTree(points) ) where T <: AbstractFloat
        
        dim = size(points,1)
        @assert size(points) == size(normals)
        
        points = mat2List(points)
        normals = mat2List(normals)
        
        return new{T, dim}(points, normals, tree)
    end
end



"""
downsample(PC::PointCloud, step)

Grid average downsample a point cloud.
"""
function downsample(PC::PointCloud, step)
    
    x = list2Mat(PC.points)
    x0 = minimum(x, dims=2)
    d = length(x0) # spatial dimensions

    cellIDs = Tuple.( mat2List( floor.( Int, (x .- x0) ./ step ) ) )
    
    cell2point = Dict{eltype(cellIDs), Vector{Int32}}()

    for (j, id) in enumerate(cellIDs)
        if haskey(cell2point, id)
            push!(cell2point[id], j)
        else
            cell2point[id] = [j]
        end
    end

    new_n = length(keys(cell2point))
    y = fill( SVector{d}(zeros(d)), new_n ) 
    yn = fill( SVector{d}(zeros(d)), new_n )

    inCells = collect(values(cell2point))
    
    @threads for k in 1:length(inCells)
        y[k] = mean(PC.points[inCells[k]])
        yn[k] = mean(PC.points[inCells[k]])
    end

    yn .= normalize.(yn)
    
    return PointCloud(list2Mat(y), normals=list2Mat(yn))

end


"""
denoise(PC::PointCloud, k, sd)
"""
function denoise(PC::PointCloud; k=4, thresh)
    avgDistToNbrs = zeros(length(PC.points))
    _, dists = knn(PC.tree, list2Mat(PC.points), k)
    avgDistToNbrs = sum.(dists) ./ (k-1)
    if thresh === nothing
        thresh = std(avgDistToNbrs)
    end
    inlier = avgDistToNbrs .< (mean(avgDistToNbrs) + thresh)
    return PointCloud( list2Mat(PC.points[inlier]), 
        normals = list2Mat(PC.normals[inlier]) )
end


"""
estimateNormals!(PC::PointCloud, k)

Use PCA to approximate point normals using `k` nearest neighbours.
"""
function estimateNormals!(PC::PointCloud, k)
    x = []
    for (i,p) in enumerate(PC.points)
        j, _ = knn(PC.tree, p, k)
        x = list2Mat(PC.points[j])
        x = x .- mean(x, dims=2)
        C = x*transpose(x)
        PC.normals[i] = normalize(eigvecs(C)[:,1])
    end
    return PC.normals
end


"""
orientNormals!(PC::PointCloud, nbrs[, step])

Ensure the point normals of a surface define a consistent orientation.
"""
function orientNormals!(PC::PointCloud, nbrs, step)
    PCds = downsample(PC, step)
    estimateNormals!(PCds, nbrs)
    orientNormals!(PCds, nbrs)
    for i in 1:length(PC.points)
        j, _ = nn(PCds, PC.points[i])
        if dot(PC.normals[i], PCds.normals[j]) < 0
            PC.normals[i] = -PC.normals[i]
        end
    end
    return PC.normals
end

function orientNormals!(PC::PointCloud, nbrs)
    
    # Make sure we have normals to orient
    if all(PC.normals[1] .== 0.0)
        error("Missing normals to orient!")
    end
    
    # Initialise weighted graph of points
    G = SimpleWeightedGraph(length(PC.points))
    
    # Find nearest neighbours of each point
    J, _ = knn(PC.tree, PC.points, nbrs)
    
    # Weights indicate how similar normals are
    for i in 1:length(PC.points)
        for j in J[i]
            i == j ? continue : nothing
            w = max(1e-12, 1 - abs(dot(PC.normals[i], PC.normals[j])))
            SimpleWeightedGraphs.add_edge!(G, SimpleWeightedEdge(i, j, w))
        end
    end
    
    # Traverse a minimal spanning tree of G, flipping normals as required
    function propogateOrientation!(dfs, i, PC, vlist)
        dsts = dfs.fadjlist[i]
        for j in dsts
            dotProd = dot(PC.normals[vlist[i]], PC.normals[vlist[j]])
            if dotProd < 0
                PC.normals[vlist[j]] = -PC.normals[vlist[j]]
            end
            propogateOrientation!(dfs, j, PC, vlist)
        end
        return nothing
    end

    vertex_sets = connected_components(G)
    for k in 1:length(vertex_sets)
    
        Gk, vlist = induced_subgraph(G, vertex_sets[k])
        mst_edges = kruskal_mst(Gk)
        MST = SimpleWeightedGraph(nv(Gk))
        for e in mst_edges
            add_edge!(MST, e)
        end
        dfs = dfs_tree(MST, 1)

        propogateOrientation!(dfs, 1, PC, vlist)

    end
    
    return nothing
    
end

"""
readPointCloud(filename::AbstractString)

Read a point cloud in PLY format.

Expects the element "vertex" with properties "x", "y", "z"; and optionally
the element "normal" with properties "u", "v", "w".
"""
function readPointCloud(filename::AbstractString)
    ply = load_ply(filename)
    x = collect( ply["vertex"]["x"] )
    y = collect( ply["vertex"]["y"] )
    z = collect( ply["vertex"]["z"] )
    points = [x'; y'; z']
    nrmls = zeros(size(points))
    if "normal" ∈ ply.elements
        u = collect( ply["normal"]["u"] )
        v = collect( ply["normal"]["v"] )
        w = collect( ply["normal"]["w"] )
        nrmls .= [u'; v'; w']
    end
    return PointCloud(points, normals = nrmls)
end


"""
readMatPointCloudWithNormals(filename::AbstractString)

Read a point cloud in PLY format.

Expects the element "vertex" with properties "x", "y", "z", "nx", "ny", "nz"
"""
function readMatPointCloudWithNormals(filename::AbstractString)
    ply = load_ply(filename)
    x = collect( ply["vertex"]["x"] )
    y = collect( ply["vertex"]["y"] )
    z = collect( ply["vertex"]["z"] )
    points = [x'; y'; z']
    nrmls = zeros(size(points))
    nx = collect( ply["vertex"]["nx"] )
    ny = collect( ply["vertex"]["ny"] )
    nz = collect( ply["vertex"]["nz"] )
    nrmls .= [nx'; ny'; nz']
    return PointCloud(points, normals = nrmls)
end


"""
readPointCloud(filename::AbstractString)

Write a point cloud in PLY format.
"""
function writePointCloud(PC::PointCloud, filename)
    ply = Ply()
    
    points = list2Mat(PC.points)
    vertexElem = PlyElement("vertex",
    ArrayProperty("x", points[1,:]),
    ArrayProperty("y", points[2,:]),
    ArrayProperty("z", points[3,:]) )
    push!(ply, vertexElem)
    
    normals = list2Mat(PC.normals)
    normalElem = PlyElement("normal",
    ArrayProperty("u", normals[1,:]),
    ArrayProperty("v", normals[2,:]),
    ArrayProperty("w", normals[3,:]) )
    push!(ply, normalElem)
    
    save_ply(ply, filename, ascii=true)
end


import NearestNeighbors: knn, nn
knn(PC::PointCloud, points, k) = knn(PC.tree, points, k)
nn(PC::PointCloud, points) = nn(PC.tree, points)


import Base.show
function show(io::IO, PC::PointCloud)
    println(io, typeof(PC))
    println(io, "  Number of points: $(length(PC.points))")
    println(io, "  Normals: $(!ismissing(PC.normals[1][1]))")
end
