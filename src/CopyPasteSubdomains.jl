
export affineTform, heightMapTform, invTform, CopiedInterpolant, sample


abstract type Tform end

struct affineTform{T,d} <: Tform
    A :: SMatrix{d,d,T}
    b :: SVector{d,T}
    pinvA :: SMatrix{d,d,T}
    function affineTform(A::AbstractMatrix{T}, b::AbstractVector{T}) where T
        d = length(b)
        if rank(A) < d
            @warn "The matrix is singular. Inverse transforms will be ambiguous."
        end
        new{T,d}(A,b,pinv(A))
    end
end

function (Tr::affineTform)(x)
    Tr.A*x .+ Tr.b
end

function invTform(Tr::affineTform, y)
    Tr.pinvA*(y .- Tr.b)
end


struct heightMapTform{T} <: Tform
    f :: Function # z = f(x,y) is the height map
    x1off :: T
    x2off :: T # offsets; translations
    function heightMapTform(f::Function, x0::T, y0::T) where T
        new{T}(f, x0, y0)
    end
end

function (H::heightMapTform)(x)
    y = zeros(3)
    y[1:2] = x[1:2] + [H.x1off; H.x2off]
    y[3] = x[3] + H.f(y[1],y[2])[1]
    return y
end

function invTform(H::heightMapTform, y)
    x = zeros(3)
    x[1] = y[1] - H.x1off
    x[2] = y[2] - H.x2off
    x[3] = y[3] - H.f(y[1],y[2])[1]
    return x
end


function Base.show(io::IO, Tr::Tform)
    print(io, typeof(Tr))
end


struct CopiedInterpolant
    F :: RBFPUMinterpolant              # original RBFPUM interpolant
    C :: Matrix                         # centres
    R :: Vector                         # radii
    tforms :: Vector{Tform}             # affine transforms applied to subdomains
    tformIDs :: Vector{Int}             # which transform was applied to which subdomain?
    hyperRecs                           # bounding rectangles for each copy
    function CopiedInterpolant(F, C, R, tforms, tformIDs, hyperRecs)
        @assert maximum(tformIDs) <= length(tforms)
        @assert length(hyperRecs) == length(tforms)
        @assert size(C,2) == size(R,1) == length(tformIDs)
        new(F, C, R, tforms, tformIDs, hyperRecs)
    end
end

function CopiedInterpolant(F::RBFPUMinterpolant, tforms)
    
    n = F.Ω.n
    C = []
    R = []
    tformIDs = []
    hyperRecs = []
    for (i,T) in enumerate(tforms)
        newC = T.(listview(F.Ω.C)) |> list2Mat
        newR = F.Ω.R
        maxs = maximum(newC .+ reshape(newR, (1,n)), dims=2)
        mins = minimum(newC .- reshape(newR, (1,n)), dims=2)

        append!(C, mat2List(newC))
        append!(R, newR)
        append!(tformIDs, fill(i,n))
        push!(hyperRecs, (mins=mins, maxs=maxs))
    end

    return CopiedInterpolant(F, hcat(C...), R, tforms, tformIDs, hyperRecs)

end

function sample(🐱::CopiedInterpolant, y::AbstractArray{<:AbstractArray})
    
    ny = length(y)
    Fy = zeros(ny)
    wsums = zeros(ny)
    
    for (i,T) in enumerate(🐱.tforms)

        rec = 🐱.hyperRecs[i]
        inRec(p) = all( dm -> rec.mins[dm] < p[dm] < rec.maxs[dm], 1:length(p) )
        J = findall(inRec, y)

        if isempty(J)
            continue
        end

        z = [ SVector{🐱.F.Ω.dim}(invTform(T, y[j])) for j in J ]

        Fy_part, wsums_part = sample(🐱.F, z, return_wsums=true, doProgressBar=false)

        @assert length(Fy_part) == length(wsums_part) == length(z)
        Fy[J] .+= Fy_part
        wsums[J] .+= wsums_part

    end

    map!(/, Fy, Fy, wsums)

    return Fy

end


function sample(🐱::CopiedInterpolant, Y::AbstractMatrix{T}) where T

    (dim, n) = size(Y)
    y = reinterpret(SVector{dim, T}, Y) |> vec
    return sample(🐱, y)

end


function Base.show(io::IO, 🐱::CopiedInterpolant)
    println(io, typeof(🐱), ": contains $(length(🐱.tforms)) copies of:")
    print(io, "  ", 🐱.F)
end