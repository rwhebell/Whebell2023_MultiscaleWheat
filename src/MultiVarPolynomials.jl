using Printf


"""
A struct representing a multi-variate polynomial.

The best way to build one is to initialise a zero-coefficient one:

    p = MPoly(deg, dim)

then mutate its coefficients as you please:

    setcoeffs!(p::MPoly, c)
"""
struct MPoly
    degree :: Int64
    dimension :: Int64
    coeffs  :: Array{Float64,1}
    expons :: Vector{Vector{Int64}}
    numTerms :: Int64
    function MPoly(deg, dim, c, e, n)
        new(deg, dim, c, e, n)
    end
end

# Ugly function to make nice output in REPL
function Base.show(io::IO, p::MPoly)
    println(io, typeof(p))
    println(io, "  Degree: $(p.degree)")
    println(io, "  Dimension: $(p.dimension)")
    
    if all(p.coeffs .== 0)
        print(io, "  p = 0")
        return nothing
    end
    
    print(io, "  p(")
    firstVar = true
    for d = 1:p.dimension
        firstVar ? (firstVar = false) : print(io, ", ")
        print(io, "x_$d")
    end
    print(io, ") = ")
    
    firstTerm = true
    for k = 1:p.numTerms
        p.coeffs[k] == 0 ? continue : nothing
        firstTerm ? (firstTerm = false) : print(io, " + ")
        @printf(io, "%.2g", p.coeffs[k])
        for d = 1:p.dimension
            p.expons[k][d] > 0 ? print(io, " x_$d^$(p.expons[k][d])") : continue
        end
    end
end

# Construct a polynomial with all zero coeffs
function MPoly(deg, dim)
    e_tup = collect(setprod(0:deg, dim))[:]
    e = [ [ek...] for ek in e_tup ]
    c = fill(0.0, size(e))
    n = length(e)
    return MPoly(deg,dim,c,e,n)
end

# Evaluate polynomial at points
function (p::MPoly)(x::Matrix)
    @assert size(x,1) == p.dimension "Points are wrong dimension."
    px = zeros(size(x,2))
    for k = 1:p.numTerms
        px .+= p.coeffs[k] .* prod(x.^p.expons[k], dims=1)[:]
    end
    return px
end

function addPolynomial!(out, p::MPoly, x::AbstractMatrix)
    for k = 1:p.numTerms
        for i in 1:size(x,2)
            out[i] += p.coeffs[k] * prod(s -> x[s,i]^p.expons[k][s], 1:p.dimension)
        end
    end
end

function addPolynomial!(out, p::MPoly, x)
    for k = 1:p.numTerms
        for i in 1:length(x)
            out[i] += p.coeffs[k] * prod(s -> x[i][s]^p.expons[k][s], 1:p.dimension)
        end
    end
end

function (p::MPoly)(x::AbstractVector{T}) where T<:Number
    @assert size(x,1) == p.dimension "Points are wrong dimension."
    px = 0.0
    for k = 1:p.numTerms
        px += p.coeffs[k] * prod(s -> x[s]^p.expons[k][s], 1:p.dimension)
    end
    return px
end

function ∇p(p::MPoly, x::AbstractVector{T}) where T<:Number

    dpx = @MVector zeros(T, length(x))

    for k = 1:p.numTerms

        expons = p.expons[k]
        # p_k(x) = a_k * (x_1^expons[1] * ... * x_d^expons[d])

        for i = 1:p.dimension # concerned with the iᵗʰ element of pdx

            if expons[i] == 0
                continue
            end

            dpk_dxi = 1.0

            for j = 1:p.dimension

                if i == j
                    dpk_dxi *= expons[j] * x[j] ^ (expons[j] - 1)
                else
                    dpk_dxi *= x[j] ^ expons[j]
                end

            end

            dpx[i] += p.coeffs[k] * dpk_dxi

        end

    end

    return dpx

end

# Generate coefficient matrix, basically
function evalMonomials(p::MPoly, x)
    @assert size(x,1) == p.dimension "Points are wrong dimension!"
    pkx = zeros(Float64, p.numTerms, size(x,2))
    for k = 1:p.numTerms
        pkx[k,:] = prod(x.^p.expons[k], dims=1)
    end
    return pkx
end

function evalMonomials!(p::MPoly, x, pkx)
    @assert size(x,1) == p.dimension "Points are wrong dimension!"
    for k = 1:p.numTerms
        for i in 1:size(x,2)
            pkx[k,i] = prod( s -> x[s,i]^p.expons[k][s], 1:p.dimension )
        end
    end
end

# Mutate coefficients
function setcoeffs!(p::MPoly, c)
    p.coeffs .= c
end