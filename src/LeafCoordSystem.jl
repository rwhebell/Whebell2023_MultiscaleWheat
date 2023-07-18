using Dierckx, LinearAlgebra, NearestNeighbors, Statistics, Optim

export LeafCoordSystem, fitLCS, world2leaf, leaf2world, listView



struct LeafCoordSystem
    veinSpline::ParametricSpline
    normalSpline::ParametricSpline
end


"""
    fitLCS(worldPoints, midPoints; nbrRadius=nothing)
"""
function fitLCS(worldPoints, midPoints; nbrRadius=nothing)
    
    # Fitting spline U(t) with t = 0, ..., n
    t = 0:size(midPoints,2)
    veinSpline = ParametricSpline(t, midPoints, bc="error")
    
    # Reparameterise so t represents the arc length
    arcLens = zeros(size(t))
    numSteps = 1000
    for j in 1:size(midPoints,2)
        mt = veinSpline(range(t[j], t[j+1], length=numSteps))
        difs = diff(mt, dims=2)
        arcLens[j+1] = sum( sqrt.( sum( difs.^2, dims=1 ) ) )
    end
    
    # Refit
    t = cumsum(arcLens)
    veinSpline = ParametricSpline(t, midPoints, bc="error")
    
    # Fitting V(t), the normal spline
    if nbrRadius === nothing
        nbrRadius = mean(arcLens[2:end]) / 2
    end
    midPointNrmls = zeros(size(midPoints))
    tree = KDTree(worldPoints)
    for j in 1:size(midPoints,2)
        inds = inrange(tree, midPoints[:,j], nbrRadius)
        nbrs = worldPoints[:,inds]
        nbrs .-= mean(nbrs, dims=2)
        COV = nbrs * transpose(nbrs)
        midPointNrmls[:,j] = normalize(eigvecs(COV)[:,1])
        if j>1 && dot( midPointNrmls[:,j], midPointNrmls[:,j-1] ) < 0
            midPointNrmls[:,j] *= -1
        end
    end
    
    normalSpline = ParametricSpline(t, midPointNrmls, bc="error")

    return LeafCoordSystem(veinSpline, normalSpline)
    
end


"""
    world2leaf(LCS::LeafCoordSystem, x)

Transform a point `x`, in world coords, to leaf coords, [t, v, w]^T,
as defined by the leaf coordinate system `LCS`.
"""
function world2leaf(LCS::LeafCoordSystem, x)

    loss(s::Real) = sum( (LCS.veinSpline(s) - x).^2 )

    res = optimize(loss, 0.0, maximum(LCS.veinSpline.t))
    t = Optim.minimizer(res)

    if !Optim.converged(res)
        println("Oh no!")
    end
    
    mt = LCS.veinSpline(t)
    Ut = normalize(derivative(LCS.veinSpline, t; nu=1))

    Vt = LCS.normalSpline(t)
    Vt .-= (Vt⋅Ut)*Ut
    normalize!(Vt)
    Wt = normalize(Ut × Vt)

    return [ t; (x-mt)⋅Vt; (x-mt)⋅Wt ]

end


"""
    leaf2world(LCS::LeafCoordSystem, y)

Transform a point `y`, in leaf coords, to world coords, [x1, x2, x3]^T.
Leaf coords are defined by the leaf coordinate system `LCS`.
"""
function leaf2world(LCS::LeafCoordSystem, y)

    t = y[1]
    v = y[2]
    w = y[3]

    mt = LCS.veinSpline(t)
    Ut = normalize!(derivative(LCS.veinSpline, t; nu=1))
    Vt = LCS.normalSpline(t)
    Vt -= (Vt⋅Ut)*Ut
    normalize!(Vt)
    Wt = normalize!(Ut × Vt)

    return mt + v*Vt + w*Wt

end




# ===== Utility funcs =====

function listView(pointMat)
    @views list = [ pointMat[:,i] for i in axes(pointMat,2) ]
    return list
end

