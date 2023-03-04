"""! This Script collects statistics about which constraints are being violated how much in a given sampleset"""
from stackingPallet import PalletQUBOGenerator
import numpy as np
import qaUtils

def addMissingVariables(bqm, sampleset):
    """! Adds variables that are present in the sampleset but not the bqm to the bqm o facilitate the calculation of energy with
    partial constraints"""
    
    for var in bqm.variables:
        if var not in sampleset.variables:
            print(var)

    for var in sampleset.variables:
        if var not in bqm.variables:
            bqm.add_variable(var, 0)

def countGreaterZero(energies, occs):
    acc = 0

    for i,energy in enumerate(energies):
        if energy > 0:
            acc += occs[i]
    
    return acc
    

def completePartialBQM(sampleset, generator):
    """!   Completes BQM not using all conditions for testing"""
    addMissingVariables(generator.bqm, sampleset)

def calcConstraintStats(sampleset):
    """! Calculates the number of violations of each contstraint in a sampletset 
    
    @param sampleset The dimod.SampleSet to investigate 

    @returns dict{String:List} Dictionary of constraint names and number of violations"""
    res = {}
    sequences = sampleset.info['sequences']
    
    permutGen = PalletQUBOGenerator(sequences, autoGenerate = False)
    permutGen.permutationConstraint()
    completePartialBQM(sampleset, permutGen) 

    yjcGen = PalletQUBOGenerator(sequences, autoGenerate = False)
    yjcGen.constructSequenceGraph()
    yjcGen.yjc()
    completePartialBQM(sampleset, yjcGen)

    countGen = PalletQUBOGenerator(sequences, autoGenerate = False)
    countGen.inequalityConstraints()
    completePartialBQM(sampleset, countGen)
    
    occs = sampleset.record['num_occurrences']
    res['Permutation'] = countGreaterZero(permutGen.bqm.energies(sampleset),occs)
    res['Y(j,c)'] =  countGreaterZero(yjcGen.bqm.energies(sampleset),occs)
    res['Count'] =  countGreaterZero(countGen.bqm.energies(sampleset),occs)
    
    return res


if __name__=='__main__':
    import sys
    ss = qaUtils.loadSampleset(sys.argv[1])
    print(calcConstraintStats(ss))
    ss = qaUtils.loadSampleset(sys.argv[1])

    print(np.sum(ss.record[ss.record['energy']==ss.first.energy]['num_occurrences'])
, "solutions at lowest energy(", ss.first.energy, ")")
    print(np.sum(ss.record[ss.record['energy'] < 10]['num_occurrences']), "correct solutions") #Nur für bis zu 9 Label und entsprechend hohen penalty factor(>10)!!!!

