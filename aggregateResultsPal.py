import sys
from glob import glob
import numpy as np
from pprint import pprint
import math
from dataclasses import dataclass

import qaUtils
from collectConstStatsPallet import calcConstraintStats
from plotResultsPal import embeddingStats

@dataclass(init=True)
class Instance:
    file_pattern: str
    optimal_energy: int

def aggregate_stats(files, opt_energy):
    samplesets = [qaUtils.loadSampleset(file) for file in files]

    print(samplesets[0].info['sequences'])
    print(f'{len(files)} samplesets')

    stats = []

    for ss in samplesets:
        print(ss.info['solverId'])
        curr_stats = calcConstraintStats(ss)
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
    for key in sum_dict:
        dev_dict[key] = np.std(sum_dict[key])

    print('--Standard Deviation--')
    pprint(dev_dict)

if __name__ == '__main__':
    instances = [
            Instance('data/pallet/batched/0-*',3),
            Instance('data/pallet/batched/1-*',3),
            Instance('data/pallet/batched/2-*',3),
            Instance('data/pallet/batched/3-*',3),
            Instance('data/pallet/batched/4-*',3),
            Instance('data/pallet/batched/5-*',3),
            Instance('data/pallet/batched/6-*',3),
            Instance('data/pallet/batched/7-*',3),
            Instance('data/pallet/batched/8-*',3)
    ]

    for instance in instances:
        files = glob(instance.file_pattern) 

        aggregate_stats(files, instance.optimal_energy)
        print('======')
