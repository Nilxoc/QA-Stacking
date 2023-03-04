import numpy as np
import qaUtils
import sys

ss = qaUtils.loadSampleset(sys.argv[1])
print(ss.info['sequences'])
return np.sum(ss.record[ss.record['energy'] < 10]['num_occurrences'])
