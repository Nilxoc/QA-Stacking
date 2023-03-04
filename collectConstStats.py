"""! This Script collects statistics about which constraints are being violated how much in a given sampleset"""
from stacking import StackingQUBOGenerator
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
    generator.generateLinears()
    generator.fixPlanVariables()

    for var in generator.toFix:
        generator.bqm.fix_variable(generator.variableName(var[0], var[1]),0)

    addMissingVariables(generator.bqm, sampleset)

def countOverlaps(inList, occs):
    res = 0
    for i in range(0, len(inList[0])):
        allGreaterZero = True
        for energyList in inList:
            if energyList[i] <= 0:
                allGreaterZero = False
        if allGreaterZero:
            res += occs[i]

    return res

def calcConstraintStats(sampleset, dec_bound=1):
    """! Berechnet, wie oft die einzelnen Constraint des FIFO-Stack up Problems im angegebenen Sampleset verletzt werden.
    
    @param sampleset Das dimod.SampleSet über das die Statistik erhoben werden soll

    @returns dict{String:List} Dictionary mit den einzelnen Constraints als Keys und Statistiken über diese Constraints"""
    res = {}
    sequences = sampleset.info['sequences']
    
    permutGen = StackingQUBOGenerator(sequences, dec_bound)
    permutGen.permutationConstraint()
    completePartialBQM(sampleset, permutGen) 

    orderGen = StackingQUBOGenerator(sequences, dec_bound)
    orderGen.sequenceOrder()
    completePartialBQM(sampleset, orderGen)
    
    ftcGen = StackingQUBOGenerator(sequences, dec_bound)
    ftcGen.fixPlanVariables()
    ftcGen.ftcConstraint()
    completePartialBQM(sampleset, ftcGen)

    countGen = StackingQUBOGenerator(sequences, dec_bound)
    countGen.countStackingPlacesConstraint()
    completePartialBQM(sampleset, countGen)
    
    occs = sampleset.record['num_occurrences']
    permEnergies = permutGen.bqm.energies(sampleset)
    seqEnergies = orderGen.bqm.energies(sampleset)
    ftcEnergies = ftcGen.bqm.energies(sampleset)
    countEnergies = countGen.bqm.energies(sampleset)

    res['Permutation'] = countGreaterZero(permEnergies,occs)
    res['SequenceOrder'] =  countGreaterZero(seqEnergies,occs)
    res['f(t,c)'] =  countGreaterZero(ftcEnergies,occs)
    res['Count'] =  countGreaterZero(countEnergies,occs)
    
    """
    print(f"Overlap perm,seq:{countOverlaps([permEnergies, seqEnergies], occs)}")
    print(f"Overlap perm,ftc:{countOverlaps([permEnergies, ftcEnergies], occs)}")
    print(f"Overlap perm,count:{countOverlaps([permEnergies, countEnergies], occs)}")
    print(f"Overlap perm,seq,ftc:{countOverlaps([permEnergies, seqEnergies,ftcEnergies], occs)}")
    print(f"Overlap seq, ftc:{countOverlaps([seqEnergies,ftcEnergies], occs)}")
    print(f"Overlap seq, count:{countOverlaps([seqEnergies,countEnergies], occs)}")
    print(f"Overlap seq, ftc,count:{countOverlaps([seqEnergies,ftcEnergies, countEnergies], occs)}")
    print(f"Overlap ftc, count:{countOverlaps([ftcEnergies,countEnergies], occs)}")
    print(f"Overlap all: {countOverlaps([permEnergies, seqEnergies, ftcEnergies, countEnergies], occs)}") 
    """

    return res


if __name__=='__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-db', type=int, action='store', dest='dec_bound', metavar='Boundary for decision problem', default=1)

    args = parser.parse_args(sys.argv[2:])
    ss = qaUtils.loadSampleset(sys.argv[1])
    print(np.sum(ss.record[ss.record['energy']==ss.first.energy]['num_occurrences'])
, "solutions at lowest energy(", ss.first.energy, ")")
    print(np.sum(ss.record[ss.record['energy'] < 10]['num_occurrences']), "correct solutions") #Nur für bis zu 9 Label und entsprechend hohen penalty factor(>10)!!!!
    print(calcConstraintStats(ss, args.dec_bound))
