import collectConstStats

res = collectConstStats.countOverlaps([[0,0,1,2,2],[1,0,3,2,0], [1,1,1,0,4]], [1,1,2,1,1])
assert(res == 2)
