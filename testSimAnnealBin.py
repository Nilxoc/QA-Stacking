import stacking
import collectConstStats
import numpy as np

instances = [[[0,1],[1,0]],
        [[0,2,1],[1,0,2]],
        [[1,0,2,1,2],[0,1,0,2]],
        [[3,0,3,4,1,3,4,0,2],[3,2,4,0,2,4,1,2,1,0,1]]] 


for instance in instances:
    print(instance)
    res = stacking.solveSimAnneal(instance, 1000, dec_bound=1)
    ss = res[1]
    correct = np.sum(ss.record['energy'] < 10)
    print(correct)
    print(res[0])
    print(collectConstStats.calcConstraintStats(ss))
