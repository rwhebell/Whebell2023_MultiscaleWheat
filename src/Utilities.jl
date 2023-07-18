"""
list2Mat(points::Vector{<:SVector})

Convert an array of equal length vectors to a matrix.
"""
function list2Mat(points::Vector{<:AbstractVector})
    return [ points[j][i] for i in 1:length(points[1]), j in 1:length(points) ]
end


"""
mat2List(points::Matrix)

Convert a matrix to an array of vectors (the matrix's columns).
"""
function mat2List(points::Matrix)
    return [ points[:,i] for i in 1:size(points,2) ]
end


"""
setprod(elems, n)

Cartesian set product
"""
function setprod(elems, n)
    return Iterators.product(ntuple(i->elems, n)...)
end


"""
BlockIterator(len, blockSize)

Acts like `range`, but iterations are blocks of indices.
"""
struct BlockIterator
    len :: Int
    blockSize :: Int
    numBlocks :: Int
    function BlockIterator(len, blockSize)
        numBlocks = ceil(len / blockSize)
        if blockSize > len
            blockSize = len
        end
        return new(len, blockSize, numBlocks)
    end
end

function Base.iterate(B::BlockIterator, last_end=0)
    if last_end ≥ B.len
        return nothing
    end
    new_start = last_end + 1
    new_end = min( new_start + B.blockSize - 1, B.len )
    return ( new_start:new_end, new_end )
end


"""
`listview(mat)`

Return an array of views into the columns of `mat`.
"""
function listview(A)
    @views return [ A[:,i] for i in axes(A,2) ]
end