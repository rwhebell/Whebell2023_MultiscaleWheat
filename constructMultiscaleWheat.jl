using Whebell2023_MultiscaleWheat, PlyIO, NearestNeighbors, LinearAlgebra, StaticArrays
using Dates: format, now


BLAS.set_num_threads(1)


outputsPath = "./outputs"
mkpath(outputsPath)

ctFilePath = "ctScanPts_24um_rotated_cropped.ply"
wheatLeafPath = "wheatLeaf.ply"
wheatLeafMidPtsPath = "wheatLeaf_midPts.ply"


# Scale to sample leaf model: 
# {"tiny", "crossSection", "seam", "onesegment", "foursegments", "whole"} 
# Defaults to whole leaf.
scaleString = "whole"


## Read data
ctFile = load_ply(ctFilePath)

N = length(ctFile["point"])
xyz = zeros(3,N)

xyz[1,:] = collect( ctFile["point"]["x"] )
xyz[2,:] = collect( ctFile["point"]["y"] )
xyz[3,:] = collect( ctFile["point"]["z"] )
f = collect( ctFile["point"]["f"] )

ct_mask = 1600 .< xyz[3,:] .< 1900

xyz .= xyz .- minimum(xyz,dims=2)

xyz[3,:] .-= 370 # so the main surface is at z=0

xyz = xyz[:,ct_mask]
f = f[ct_mask]

ctMins = minimum(xyz, dims=2) |> vec
ctMaxs = maximum(xyz, dims=2) |> vec
ctRange = ctMaxs - ctMins


## Fit implicit RBFPUM surface at CT resolution
rCubed(r) = r^3
ρ = 3e-5
pdeg = 2
wendland(r) = max(0, 1-r)^4 * (4r + 1)
Nbounds = (2000, 5000)
expand = 1.1

F = RBFPUMinterpolant(xyz, f, rCubed, 96π*ρ, pdeg, wendland, Nbounds, expand)

println("\nmicro: fit $(length(F.Fi)) subdomains to $(size(xyz,2)) pts")
@time fit!(F)


## Get some leaf-level data
# Loading points
leaf_ply = load_ply(wheatLeafPath)
worldPoints = zeros(3, length(leaf_ply["vertex"]))
worldPoints[1,:] = collect( leaf_ply["vertex"]["x"] )
worldPoints[2,:] = collect( leaf_ply["vertex"]["y"] )
worldPoints[3,:] = collect( leaf_ply["vertex"]["z"] )
worldPoints .*= 1000

midPts_ply = load_ply(wheatLeafMidPtsPath)
midPoints = zeros(3, length(midPts_ply["vertex"]))
midPoints[1,:] = collect( midPts_ply["vertex"]["x"] )
midPoints[2,:] = collect( midPts_ply["vertex"]["y"] )
midPoints[3,:] = collect( midPts_ply["vertex"]["z"] )
midPoints .*= 1000

# Fit leaf coord system

println("\nFit leaf coord system:")
@time LCS = fitLCS(worldPoints, midPoints)

println("\nMap world to leaf coords, $(size(worldPoints,2)) points:")
@time leafPoints = map(x -> world2leaf(LCS, x), listView(worldPoints))

# leafPoints are like (t, v, w) right now;
# which is like ("dist along leaf", "height above leaf", "dist from medial axis of leaf")

# but just for conveniance we make it (t, w, v) to align with the third coord being "up"
leafPoints = [ leafPoints[i][j] for j in [1,3,2], i in eachindex(leafPoints) ]

# now leaf points are (t, w, v)
# and ct points are (x, y, z) where stripes align with x

leafMins = minimum(leafPoints, dims=2) |> vec
leafMaxs = maximum(leafPoints, dims=2) |> vec
leafRange = leafMaxs - leafMins




## Fit explicit RBFPUM surface at leaf resolution
using IfElse
function tps(r) 
    return IfElse.ifelse(r == 0, zero(r), r^2 * log(r))
end

G = RBFPUMinterpolant(
    leafPoints[1:2,:], leafPoints[3,:], 
    tps, 8π * 400, 1, wendland, Nbounds, expand
)

println("\nmacro: fit $(length(G.Fi)) subdomains to $(size(leafPoints,2)) pts")
@time fit!(G)





## Make ghost copies
tileNums = ceil.(Int, leafRange ./ ctRange)
tileNums[3] = 1 # no tiling normal to the leaf surface

tforms = affineTform[]
for i in 0:tileNums[1]-1
    for j in 0:tileNums[2]-1
        T = affineTform(
            Matrix{Float64}(I,3,3),
            [ leafMins[1] + ctRange[1]*i,
            leafMins[2] + ctRange[2]*j, 
            0 ]
        )
        push!(tforms, T)
    end
end

F_multiscale = CopiedInterpolant(F, tforms)


if lowercase(scaleString) == "tiny"

    mins = [leafMins[1] + 0.41*leafRange[1]; leafMins[2] + 0.2*leafRange[2]; ctMins[3]]
    maxs = mins .+ [ 0.4*ctRange[1], 0.4*ctRange[2], ctRange[3] ]
    steps = 4

elseif lowercase(scaleString) == "crosssection"

    mins = [leafMins[1] + 0.40*leafRange[1]; leafMins[2]; ctMins[3]]
    maxs = [leafMins[1] + 0.46*leafRange[1]; leafMaxs[2]; ctMaxs[3]]
    steps = 12

elseif lowercase(scaleString) == "seam"

    mins = [leafMins[1] + 37.9*ctRange[1]; leafMins[2] + 1.5*ctRange[2]; ctMins[3]]
    maxs = mins .+ [0.2, 1, 1] .* ctRange
    steps = 6

elseif lowercase(scaleString) == "onesegment"

    mins = [leafMins[1] + 37*ctRange[1]; leafMins[2] + 2*ctRange[2]; ctMins[3]]
    maxs = mins .+ ctRange
    steps = 4

elseif lowercase(scaleString) == "foursegments"

    mins = [leafMins[1] + 37*ctRange[1]; leafMins[2] + 2*ctRange[2]; ctMins[3]]
    maxs = mins .+ [2, 2, 1] .* ctRange
    steps = 8

else

    mins = [leafMins[1]; leafMins[2]; ctMins[3]]
    maxs = [leafMaxs[1]; leafMaxs[2]; ctMaxs[3]]
    steps = 100

end


level = 8/255

evalFunc(x) = sample(F_multiscale, x)

ashape = AlphaShape(leafPoints[1:2,:], 1_000)
function isInterior(u)
    return inshape(ashape, u[1:2])
end

println("\nrealise surface, scale = $scaleString:")

triangles = nothing
@time triangles = marchTet_grid_memoryHungry(
    evalFunc, 
    mins, 
    maxs, 
    steps; 
    levels = level, 
    interior = isInterior
)

println("made $(length(triangles[level])) triangles")




## Un-transform back to world coords

verts, faces = compressTriangulation(triangles[level])
vertsMat = reinterpret(reshape, eltype(verts[1]), verts) |> collect

println("\nsample height map at $(size(vertsMat,2)) pts:")
@time vertsMat[3,:] .+= sample(G, vertsMat[1:2,:])

println("\nconvert triangles back to world coords:")
vertsMat = vertsMat[[1;3;2], :]
@time for i in axes(vertsMat,2)
    point_leaf = vertsMat[:,i]
    point_world = leaf2world(LCS, point_leaf)
    vertsMat[:,i] .= point_world
end

writeOBJ(1e-6 .* vertsMat, faces, 
    joinpath(outputsPath, "wheat_" * scaleString * format(now(), "_YYYY-u-dd_HHMM") * ".obj"))