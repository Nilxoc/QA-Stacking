import numpy as np
import qaUtils
from collectConstStats import calcConstraintStats
from matplotlib import pyplot as plt

def embeddingStats(embedding):
    max_len = 0
    chain_count = 0
    var_count = 0
    for source, target in embedding.items():
        chain_len = len(target)
        var_count += chain_len
        if chain_len > max_len:
            max_var = source
            max_len = chain_len

        if chain_len > 1:
            chain_count += 1

    return max_var, max_len, chain_count, var_count

if __name__ == '__main__':
    stackedWidth = 0.6
    groupedWidth = stackedWidth/4

    files = [('data/results/2Lab2Bin1Dec.dat',1), ('data/results/3Seq2Lab3Bin1Dec.dat',1), ('data/results/3Lab2Bin1Dec.dat',1), ('data/results/3Lab2Bin2Dec.dat',2), ('data/results/2Seq2Lab4Bin1Dec.dat',1), ('data/results/3Seq3Lab2Bin1Dec.dat',1), ('data/results/3Seq3Lab2Bin2Dec.dat',2), ('data/results/3Seq3Lab8BinTot1Dec.dat',1), ('data/results/3Seq3Lab8BinTot2Dec.dat',2)]
    #labels = list((np.arange(len(files))+1).astype(str))
    labels = ['1,1','2,1','3,1','3,2','4,1','5,1','5,2','6,1','6,2']
    x = np.arange(len(files))

    correct = []
    incorrect = []
    permut = []
    seq = []
    ftc = []
    count = []


    for instance in files:
        ss = qaUtils.loadSampleset(instance[0])

        stats = calcConstraintStats(ss, instance[1])

        print('==========')
        print(ss.info['sequences'])
        print(stats)

        permut.append(stats['Permutation'])
        seq.append(stats['SequenceOrder'])
        ftc.append(stats['f(t,c)'])
        count.append(stats['Count'])

        correctCount = np.sum(ss.record[ss.record['energy'] < 10]['num_occurrences'])

        max_chain_var, max_chain_len, chain_count, var_count = embeddingStats(ss.info['embedding_context']['embedding'])
        print("Max chain length: ", max_chain_len, "for", max_chain_var)
        print("Chain count: ", chain_count)
        print("Qubits used: ", var_count)
        print("Correct results: ", correctCount)
        correct.append(correctCount)
        incorrect.append(10000-correctCount)

    fig, ax = plt.subplots()
    ax.bar(labels, incorrect, stackedWidth+.1, label='Invalid', color='tab:red')
    ax.bar(labels, correct, stackedWidth+.1, bottom=incorrect, label='Valid', color='tab:olive')

    ax.bar(x-groupedWidth*1.5, permut, groupedWidth, label='PERMUTATION', color='aqua')
    ax.bar(x-groupedWidth*0.5, seq, groupedWidth, label='SEQUENCE_ORDER', color='darkturquoise')
    ax.bar(x+groupedWidth*0.5, ftc, groupedWidth, label='f(t,c)', color='cadetblue')
    ax.bar(x+groupedWidth*1.5, count, groupedWidth, label='Inequality', color='magenta')

    plt.xlabel("Instance ID, Value of k")
    plt.ylabel("Number of Samples")

    plt.legend(bbox_to_anchor=(.95, 1), loc='upper left', fontsize='small')

    plt.show()
