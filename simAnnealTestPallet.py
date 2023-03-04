import stackingPallet
import random
import math
import time
import pickle
from pandas import DataFrame

def generateSequences(labelCount, labelSize):
    seq1 = []
    seq2 = []
    for j in range(0, math.floor(labelSize/2)):
        seq1 += [i for i in range(0, labelCount)]
        seq2 += [i for i in range(0, labelCount)]

    if labelSize % 2 != 0:
        seq1 += [i for i in range(0, labelCount)]

    random.shuffle(seq1)
    random.shuffle(seq2)
    return [seq1, seq2]
    
def countCorrect(sampleset, gen):
    count = 0
    for sample in sampleset.data(sorted_by='energy', fields=['energy']):
        if sample.energy < gen.penaltyFactor:
            count += 1
    return count

resFrame = DataFrame(columns=['labelCount', 'labelSize', 'time', 'varCount', 'correctCount'])
outBin = open('pallet-simAnneal.dmp', 'wb')
print('=====Bin Solution=====')
for labelCount in range(2, 8):
    for labelSize in range(2, 6):
        #TODO: Average over multiple runs
        sequences = generateSequences(labelCount, labelSize)
        print(labelCount, labelSize, sequences)
        res = stackingPallet.solveSimAnneal(sequences,1000)
        correct = countCorrect(res[1], res[2])
        resFrame = resFrame.append([{'labelCount': labelCount, 'labelSize':labelSize, 'time': res[0], 'varCount': len(res[2].bqm), 'correctCount':correct}])
        print(resFrame)
        print("----------")

pickle.dump(resFrame, outBin)
outBin.close()
