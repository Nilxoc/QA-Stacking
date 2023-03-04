import numpy as np
import qaUtils
from collectConstStatsPallet import calcConstraintStats
from matplotlib import pyplot as plt

stackedWidth = 0.6 #Width of stacked bars
groupedWidth = stackedWidth/3 #Width of inner bars

def embeddingStats(embedding):
    max_len = 0;
    chain_count = 0;
    var_count = 0;
    for source, target in embedding.items():
        chain_len = len(target)
        var_count += chain_len
        if chain_len > max_len:
            max_var = source
            max_len = chain_len

        if chain_len > 1:
            chain_count += 1

    return max_var, max_len, chain_count, var_count

def plotResults(samplesets, xAxisLabels):
    labels = list((np.arange(len(samplesets))+1).astype(str))
    x = np.arange(len(samplesets))

    correct = []
    incorrect = []
    permut = []
    yjc = []
    count = []

    statList = []
    idx = 0

    for ss in samplesets:
        #ss = qaUtils.loadSampleset(instance)
        stats = calcConstraintStats(ss)

        print('==========')
        print("Sequences: " + str(ss.info['sequences']))
        print("Number of constraint violations: "+ str(stats))
        print("Number of variables(before embedding):" +  str(len(ss.info['bqm'])))
        
        max_chain_var, max_chain_len, chain_count, var_count = embeddingStats(ss.info['embedding_context']['embedding'])
        print("Max chain length: ", max_chain_len, "for", max_chain_var)
        print("Chain count: ", chain_count)
        print("Qubits used: ", var_count)
        permut.append(stats['Permutation'])
        yjc.append(stats['Y(j,c)'])
        count.append(stats['Count'])
        
        penaltyFactor = 10 #Backwards compatibility
        if 'penaltyFactor' in ss.info:
          penaltyFactor = ss.info['penaltyFactor']
        
        correctCount = np.sum(ss.record[ss.record['energy'] < penaltyFactor]['num_occurrences'])
        print("Number of samples without violated constraints: " + str(correctCount))
        correct.append(correctCount)
        incorrect.append(np.sum(ss.record['num_occurrences'])-correctCount)

        stats['instance'] = xAxisLabels[idx]
        stats['varCount'] = str(len(ss.info['bqm']))
        stats['correct'] = correctCount
        stats['opt?'] = np.sum(ss.record[ss.record['energy'] == np.min(ss.record['energy'])]['num_occurrences']) 
        statList.append(stats)
        stats['minEnergy'] = np.min(ss.record['energy'])
        
        idx += 1

    #LaTeX tabular output
    for stats in statList:
      print(str(stats['instance']) + " & " + str(stats['varCount']) + " & " + str(stats['correct']) + " & " + str(stats['opt?']) + "(" + str(stats['minEnergy']) + "?) & " + str(stats['Permutation']) + " & " + str(stats['Y(j,c)']) + " & " + str(stats['Count']) + "\\\\ \\hline")

    fig, ax = plt.subplots()
    ax.bar(labels, incorrect, stackedWidth+.1, label='Invalid', color='tab:red')
    ax.bar(labels, correct, stackedWidth+.1, bottom=incorrect, label='Valid', color='tab:olive')

    ax.bar(x-groupedWidth, permut, groupedWidth, label='PERMUATION', color='aqua')
    ax.bar(x, yjc, groupedWidth, label='Y(j,c)', color='cadetblue')
    ax.bar(x+groupedWidth, count, groupedWidth, label='Inequality', color='magenta')

    plt.xlabel("Instance ID")
    plt.ylabel("Number of Samples")

    plt.legend(bbox_to_anchor=(.95, 1), loc='upper left', fontsize='small')

    ax.set_xticklabels(xAxisLabels)

    plt.show()

if __name__ == '__main__':
    files = ['data/pallet/results/2Seq2Lab4Bin.dat', 
             'data/pallet/results/2Seq2Lab6Bin.dat',
             'data/pallet/results/2Seq3Lab6Bin.dat',
             'data/pallet/results/2Seq2Lab8Bin.dat', 
             'data/pallet/results/3Seq3Lab6Bin.dat',
             'data/pallet/results/3Seq3Lab8Bin.dat', 
             'data/pallet/results/3Seq3Lab9Bin.dat',
             'data/pallet/results/2Seq3Lab8Bin.dat', 
             'data/pallet/results/2Seq4Lab8Bin.dat']

    writtenLabels=[1,2,3,4,5,6,7,8,9]#x-Axis Labels for plotted instances(e.g Instance Identifiers)
    samplesets = [qaUtils.loadSampleset(file) for file in files]
    plotResults(samplesets, writtenLabels)


