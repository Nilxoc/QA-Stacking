"""
    Convenience script to run through multiple plots and generate a plot and LaTeX table
    samplesets are still saved to data/pallet/QA-{TIMESTAMP}.dat
"""

import stacking

print("Running this script will run 10 instances using approx. 7.5 seconds of computation time(depending on parameters, num_reads etc")
input("Press Enter to continue")

additional_params = {} #Add additional parameters here
                       #e.g additional_params = {'anneal_schedule':somevalue}
instances = [
        ([[0,1],[1,0]],1),
        ([[0,1,1],[1,0,1]],1),
        ([[0,2,1],[1,0,2]],1),
        ([[0,2,1],[1,0,2]],2),
        ([[0,1,0,1],[1,1,0,0]],1),
        ([[0,2],[1,1],[2,0]],1),
        ([[0,2],[1,1],[2,0]],2)
        #([[0,2,1],[1,0,2],[1,2]],1),
        #([[0,2,1],[1,0,2],[1,2]],2)
]#Problem instances to solve

instanceIds = [1,2,3,4,5,6,7,8,9]#x-Axis labels and leftmost table column

path_format = "data/batched/%d-QA-"
num_reads = 2000
resultSamplesets = []

for count, instance in enumerate(instances):
    save_path = path_format%count
    for i in range(0,10):
        print("Run",i,"of instance",count)
        problem = instance[0]
        dec_bound = instance[1]
        print(problem)
        print(dec_bound)
        stacking.solveDWave(problem, num_reads, dec_bound=dec_bound,prefix=save_path, **additional_params)
