import stackingPallet
import collectConstStatsPallet
import numpy as np

instances = [[[0,1],[1,0]],
        [[0,1,3,2],[3,1,0,2]],
        [[0,2,5,1,3,4,3,1,2,3,4,5,3,1,5],[4,5,1,2,3,4,3,1,5,4,5,1,1,3,4]],
        [[0,3,6,2,1,7,6,5,0,3,4,2],[0,5,3,4,1,6,5,2,7,4,1,7]]] 


for instance in instances:
    print(instance)
    res = stackingPallet.solveSimAnneal(instance, 1000)
    ss = res[1]
    correct = np.sum(ss.record['energy'] < 10)
    print(correct)
    print(res[0])
    print(collectConstStatsPallet.calcConstraintStats(ss))
