module WJFDPC
using WJFKNN
using WJFParallelTask

export DPCNode, WjfDPC

struct DPCNode
    index::UInt32
    density::Float64
    parent::Vector{DPCNode}
    children::Vector{DPCNode}
end

function DPCNode(index::UInt32, density::Float64)::DPCNode
    return DPCNode(index, density, Vector{DPCNode}(), Vector{DPCNode}())
end

function markClusterID!(
    clusterIDs::Vector{UInt32},
    clusterID::UInt32,
    node::DPCNode,
)
    clusterIDs[node.index] = clusterID
    for child in node.children
        markClusterID!(clusterIDs, clusterID, child)
    end
end

function computeClusterIDs(m::UInt32, roots::Vector{DPCNode},
    parallel::Bool)::Vector{UInt32}
    clusterIDs = Vector{UInt32}(undef, m)
    numRoots = length(roots)
    function mapFun(i1, i2)
        for i::UInt32 = i1:i2
            markClusterID!(clusterIDs, i, roots[i])
        end
    end
    mapOnly(1, numRoots, 1, mapFun, parallel)
    return clusterIDs
end

function getClusterDensityRanks!(
    densityRanks::Vector{Tuple{Float64,UInt32}},
    node::DPCNode,
)
    value = (node.density, node.index)
    pos = searchsortedfirst(densityRanks, value, rev = true)
    insert!(densityRanks, pos, value)
    for child in node.children
        getClusterDensityRanks!(densityRanks, child)
    end
end

function getClusterDensityRanks(
    roots::Vector{DPCNode},
    parallel::Bool,
)::Vector{Vector{Tuple{Float64,UInt32}}}
    numRoots = length(roots)
    densityRanks = Vector{Vector{Tuple{Float64,UInt32}}}(undef, numRoots)
    function mapFun(i1, i2)
        for i::UInt32 = i1:i2
            densityRanks[i] = Vector{Tuple{Float64,UInt32}}()
            getClusterDensityRanks!(densityRanks[i], roots[i])
        end
    end
    mapOnly(1, numRoots, 1, mapFun, parallel)
    return densityRanks
end

struct WjfDPC
    forest::Vector{DPCNode}
    clusterIDs::Vector{UInt32}
end

function WjfDPC(
    data::Vector,
    cardinalities::Vector{UInt32},
    distSqrFun::Function,
    k::UInt32,
    eps::Float64,
    parallel::Bool = true,
)::WjfDPC
    m = UInt32(length(data))
    densities = Vector{Float64}(undef, m)
    wjfKNN = WjfKNN(data, distSqrFun)
    neighbors = knnSearch(wjfKNN, k, 0.5)
    function mapFun1(i1::UInt32, i2::UInt32)
        for i::UInt32 = i1:i2
            numPoints::UInt32 = cardinalities[i] - 1
            sumDSqr::Float64 = 0.0
            for (distSqr, neighbor) in neighbors[i]
                sumDSqr += distSqr
                numPoints += cardinalities[neighbor]
            end
            densities[i] = exp(-sumDSqr / numPoints)
        end
    end
    mapOnly(UInt32(1), m, UInt32(100), mapFun1, parallel)
    nodes = [DPCNode(i, densities[i]) for i::UInt32 = 1:m]

    epsSqr = eps * eps
    function mapFun2(i1::UInt32, i2::UInt32)::Dict{UInt32,Vector{UInt32}}
        myResult = Dict{UInt32,Vector{UInt32}}()
        for i::UInt32 = i1:i2
            hdDistSqr = Inf64
            hdNeighbor = 0
            node = nodes[i]
            myDensity = node.density
            for (distSqr, neighbor) in neighbors[i]
                if distSqr < epsSqr
                    neighborNode = nodes[neighbor]
                    neighborDensity = neighborNode.density
                    if neighborDensity > myDensity && distSqr < hdDistSqr
                        hdDistSqr = distSqr
                        hdNeighbor = neighbor
                    end
                end
            end
            if hdNeighbor > 0
                push!(node.parent, nodes[hdNeighbor])
                if hdNeighbor in keys(myResult)
                    push!(myResult[hdNeighbor], i)
                else
                    myResult[hdNeighbor] = [i]
                end
            elseif length(neighbors[i]) > 0
                if neighbors[i][end][1] < epsSqr
                    # mark this node for further exploration
                    push!(node.parent, node)
                end
            end
        end
        return myResult
    end
    function reduceFun2(
        values::Vector{Dict{UInt32,Vector{UInt32}}},
    )::Dict{UInt32,Vector{UInt32}}
        myResult = Dict{UInt32,Vector{UInt32}}()
        for value in values
            for (k, v) in value
                if k in keys(myResult)
                    append!(myResult[k], v)
                else
                    myResult[k] = copy(v)
                end
            end
        end
        return myResult
    end
    childrenToAdd = mapReduce(
        UInt32(1),
        UInt32(length(nodes)),
        UInt32(100),
        mapFun2,
        reduceFun2,
        Dict{UInt32,Vector{UInt32}}(),
        parallel,
    )

    preRoots = Vector{DPCNode}()
    for i::UInt32 = 1:length(nodes)
        node = nodes[i]
        if length(node.parent) == 0
            push!(preRoots, node)
        elseif node.parent[1] == node
            push!(preRoots, node)
        end
        if i in keys(childrenToAdd)
            for j in childrenToAdd[i]
                push!(node.children, nodes[j])
            end
        end
    end
    preClusterIDs = computeClusterIDs(m, preRoots, parallel)
    clusterDensityRanks = getClusterDensityRanks(preRoots, parallel)

    function mapFun3(i1::UInt32, i2::UInt32)
        myResult = Dict{UInt32,Set{UInt32}}()
        for i::UInt32 = i1:i2
            preClusterID = preClusterIDs[i]
            for (distSqr, neighbor) in neighbors[i]
                neighborPreClusterID = preClusterIDs[neighbor]
                if preClusterID != neighborPreClusterID
                    if preClusterID in keys(myResult)
                        push!(myResult[preClusterID], neighborPreClusterID)
                    else
                        myResult[preClusterID] = Set(neighborPreClusterID)
                    end
                    if neighborPreClusterID in keys(myResult)
                        push!(myResult[neighborPreClusterID], preClusterID)
                    else
                        myResult[neighborPreClusterID] = Set(preClusterID)
                    end
                end
            end
        end
        return myResult
    end
    function reduceFun3(
        values::Vector{Dict{UInt32,Set{UInt32}}},
    )::Dict{UInt32,Set{UInt32}}
        myResult = Dict{UInt32,Set{UInt32}}()
        for value in values
            for (k, v) in value
                if k in keys(myResult)
                    union!(myResult[k], v)
                else
                    myResult[k] = copy(v)
                end
            end
        end
        return myResult
    end
    connectivities = mapReduce(
        UInt32(1),
        UInt32(length(nodes)),
        UInt32(100),
        mapFun3,
        reduceFun3,
        Dict{UInt32,Set{UInt32}}(),
        parallel,
    )
    connectivities2ndLevel = Dict{UInt32,Set{UInt32}}()

    numPreRoots = UInt32(length(preRoots))
    function mapFun4(i1::UInt32, i2::UInt32)::Dict{UInt32,Vector{UInt32}}
        myResult = Dict{UInt32,Vector{UInt32}}()
        for i::UInt32 = i1:i2
            node = preRoots[i]
            if length(node.parent) == 0
                continue
            end
            if i ∉ keys(connectivities)
                pop!(node.parent)
                continue
            end
            myDensity = node.density
            hdDistSqr = Inf64
            hdNeighbor = 0
            for j in connectivities[i]
                jDensityRanks = clusterDensityRanks[j]
                for k = 1:length(jDensityRanks)
                    if jDensityRanks[k][1] <= myDensity
                        break
                    end
                    neighbor = jDensityRanks[k][2]
                    distSqr = distSqrFun(data[node.index], data[neighbor])
                    if distSqr < epsSqr && distSqr < hdDistSqr
                        hdDistSqr = hdDistSqr
                        hdNeighbor = neighbor
                    end
                end
            end
            if hdNeighbor > 0
                node.parent[1] = nodes[hdNeighbor]
                if hdNeighbor in keys(myResult)
                    push!(myResult[hdNeighbor], node.index)
                else
                    myResult[hdNeighbor] = [node.index]
                end
            else
                pop!(node.parent)
            end
        end
        return myResult
    end
    childrenToAdd = mapReduce(
        UInt32(1),
        numPreRoots,
        UInt32(100),
        mapFun4,
        reduceFun2,
        Dict{UInt32,Vector{UInt32}}(),
        parallel,
    )
    roots = Vector{DPCNode}()
    for i::UInt32 = 1:numPreRoots
        node = preRoots[i]
        if length(node.parent) == 0
            push!(roots, node)
        end
    end
    for (i, newChildren) in childrenToAdd
        for j in newChildren
            push!(nodes[i].children, nodes[j])
        end
    end
    clusterIDs = computeClusterIDs(m, roots, parallel)
    return WjfDPC(roots, clusterIDs)
end

end # module
