using Test
using PlotlyJS, CSV, DataFrames
include("../src/WJFDPC.jl")
csv_reader = CSV.File("data/EngyTime.csv")
x = [(csv_reader[i][2],csv_reader[i][3]) for i in 1:length(csv_reader)]
y = [csv_reader[i][1] for i in 1:length(csv_reader)]
cards = [UInt32(1) for i in 1:length(csv_reader)]
function distSqrFun(a::NTuple{2,Float64}, b::NTuple{2,Float64})::Float64
    return sum((a .- b).^2)
end
#include("../src/WJFDPC.jl")
@time dpc = WJFDPC.WjfDPC(x, cards, distSqrFun, UInt32(10), 5.0, true)
clusterIDs = dpc.clusterIDs
x1 = [x[i][1] for i in 1:length(x)]
x2 = [x[i][2] for i in 1:length(x)]
#x3 = [x[i][3] for i in 1:length(x)]
a = DataFrame("x1"=>x1,"x2"=>x2,"Class"=>clusterIDs)
plot(a, x=:x1,y=:x2,color=:Class,mode="markers")
