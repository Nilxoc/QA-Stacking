import sys
from glob import glob
import numpy as np
from pprint import pprint
import math
from dataclasses import dataclass

import numpy as np

import qaUtils
from collectConstStats import calcConstraintStats
from plotResultsBin import embeddingStats

@dataclass(init=True)
class Instance:
    file_pattern: str 
    dec_bound: int
    optimal_energy: int

def aggregate_stats(files, dec_bound, opt_energy):
    samplesets = [qaUtils.loadSampleset(file) for file in files]

    print(samplesets[0].info['sequences'])
    print("k: ", dec_bound)

    print(f'{len(files)} samplesets\n')

    stats = []

    for ss in samplesets:
        curr_stats = calcConstraintStats(ss, dec_bound)
        curr_stats['num_var'] = len(ss.info['bqm'])
        curr_stats['correct'] = np.sum(ss.record[ss.record['energy'] < 10]['num_occurrences'])

        curr_stats['samples_with_optimal_energy'] = np.sum(ss.record[ss.record['energy'] == opt_energy]['num_occurrences'])
        curr_stats['optimal_energy'] = opt_energy

        max_var, max_len, chain_count, var_count = embeddingStats(ss.info['embedding_context']['embedding'])
        curr_stats['num_embedded_var'] = var_count
        curr_stats['max_chain_len'] = max_len
        curr_stats['chain_count'] = chain_count

        stats.append(curr_stats)

    sum_dict = {}
    for key in stats[0]:
        sum_dict[key] = [stat[key] for stat in stats]

    avg_dict = {}
    for key in sum_dict:
        avg_dict[key] = np.mean(sum_dict[key])

    print('--Average--')
    pprint(avg_dict)

    dev_dict = {}

    #Compute variance
    for key in sum_dict:
        dev_dict[key] = np.std(sum_dict[key])

    print('--Standard Deviation--')
    pprint(dev_dict)

if __name__ == '__main__':
    instances = [
            Instance('data/batched/0-*',1,2),
            Instance('data/batched/1-*',1,2),
            Instance('data/batched/2-*',1,2),
            Instance('data/batched/3-*',2,2),
            Instance('data/batched/4-*',1,2),
            Instance('data/batched/5-*',1,2),
            Instance('data/batched/6-*',2,2)
            #Instance('data/batched_2000/7-*',1,2),
            #Instance('data/batched_2000/8-*',2,2)
    ]

    for instance in instances:
        files = glob(instance.file_pattern)
        dec_bound = instance.dec_bound
        opt = instance.optimal_energy
        print(f'{opt=}')
        aggregate_stats(files, dec_bound, opt)
        print('======')
