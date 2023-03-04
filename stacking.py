###
#@mainpage Test
#
#@section description_main Description
#A script to generate QUBO-Formulations of instances of the stacking problem
##
import dimod
import math
from dwave.system import DWaveSampler, EmbeddingComposite
import pickle
from datetime import datetime
from qaUtils import saveSampleset
from neal.sampler import SimulatedAnnealingSampler
import argparse
import sys
import time
from icecream import ic

def iterN(items, n):
    """! Generator that iterates over a given collection in slices of size n.
    If len(items) is not divisible by n the last slice returned will contain
    the remaining elements.

    @params items The value to slice
    @params n Size of slices
    """
    while len(items) > n :
        yield items[:n]
        items = items[n:]
    if len(items) > 0:
        yield items
        
class StackingQUBOGenerator:
    """! Class to convert an instance of the stacking problem to a QUBO Formulation of that instance."""

    def __init__(this, sequences, dec_bound=1):
        """! Initialize the generator
        @param sequences List of sequences. Each sequence lists the labels of the bins it contains
        """
        this.bqm = dimod.BinaryQuadraticModel(dimod.Vartype.BINARY) #The resulting matrix

        i = 0; #Sequence index
        j = 0; #Element index
        
        this.bySequence = []
        this.byLabel = {}
        this.labels = []

        for sequence in sequences:
            this.bySequence.append([])
            for label in sequence:
                this.bySequence[i].append(j)

                if(label not in this.byLabel):
                    this.byLabel[label] = [j]
                    this.labels.append(label)
                else:
                    this.byLabel[label].append(j)

                j += 1

            i += 1

        this.binCount = j
        
        #this.auxSize = math.ceil(math.log(len(this.byLabel))+1)
        this.auxSize = math.floor(math.log(len(this.byLabel),2))+1
        this.penaltyFactor = (pow(2,this.auxSize)+1)*50 #Penalty larger than maximum possible p

        this.boolVarCount = 0

        this.planCount = this.binCount**2

        this.toFix = {}

        #The request for the decision problem version, e.g. dec_bound=2: Can these sequences be stacked with 2 stacking places?
        #Values lower than dec_bound then only confirm that stacking with 2 stacking places is possible
        this.dec_bound = dec_bound
    
    def generateLinears(this):
        """! Helper function to generate every linear entry according to binCount.
        Mostly useful for testing."""

        for elem in range(0, this.binCount):
            for time in range(0, this.binCount):
                this.bqm.add_variable(this.variableName(elem,time),0)

    def variableName(this, index, time):
        """! Return the name of the bin with the given index at the given time
             @param index The index of the bin
             @param time The step in the plan
        """

        return 'x('+str(index)+','+str(time)+')'
    
    def fName(this, label, time):
        """!Returns the variable containing the result of f(label,time)
        """
        return 'f('+str(label)+','+str(time)+')'
    
    def generateOr(this, values):
        """! Models an OR statement over the given values. Up to len(values) auxiliary variables will be created.
      
        @param values The values in the or statement

        @result The name of the auxiliary variable containing the result of the expression
        """
        #This is a likely candidate for reducing the number of auxiliary variables
        #As of now it approximately doubles the number from n^2 to 2n^2
        while(len(values)>1):
            pair = []
            pair.append(values.pop(0))
            pair.append(values.pop(0))
            auxName = str(pair[0])+'or'+str(pair[1])
            values.insert(0,auxName)
            #Constraint term: a v b = c => a+b+c+ab-2ac-2bc
            if auxName not in this.bqm.variables:
                this.boolVarCount += 1
                this.bqm.add_variable(pair[0], this.penaltyFactor)
                this.bqm.add_variable(pair[1], this.penaltyFactor)
                this.bqm.add_variable(auxName, this.penaltyFactor)
                this.bqm.add_interaction(pair[0], pair[1], this.penaltyFactor)
                this.bqm.add_interaction(pair[0], auxName, -2*this.penaltyFactor)
                this.bqm.add_interaction(pair[1], auxName, -2*this.penaltyFactor)
        return values[0]
                    
            
        
    def f(this, t):
        """! Generate term for f(t,c), which indicates whether bins with label t require a stacking place at time c.
        After execution, bqm will contain a variable 'f(t,c)' for every c and constraints will be modeled so 'f(t,c)'
        contains 1 if a stacking place is required and 0 if no stacking place is required.

        @param t Label"""

        if this.dec_bound >= len(this.byLabel):
            return

        timeSubs = []
        for c in range(0, this.binCount):
            values = []
            for index in this.byLabel[t]:
                if (index, c) not in this.toFix:
                    values.append(this.variableName(index,c))
            if(len(values) > 0):
                timeSubs.append(this.generateOr(values))
            else:
                #This is neccessary to identify the blocks correctly
                timeSubs.append('s')
            
        for c in range(this.dec_bound,this.binCount-(1+this.dec_bound)):
            #a AND b is simply modeled by a*b
            #Constraint for c = ab : ab-2ac-2bc+3c
            varName = 'f('+str(t)+','+str(c)+')'
            leftList = [val for val in timeSubs[0:c+1] if val != 's']
            rightList = [val for val in timeSubs[c+1:] if val != 's']

            if len(leftList) <= 0 or len(rightList) <= 0:
                continue

            leftTerm = this.generateOr(leftList)

            rightList.reverse() #Fewer auxilliary variables
            rightTerm = this.generateOr(rightList)
            
            this.bqm.add_interaction(leftTerm, rightTerm, this.penaltyFactor)
            this.bqm.add_interaction(leftTerm, varName, -2*this.penaltyFactor)
            this.bqm.add_interaction(rightTerm, varName, -2*this.penaltyFactor)
            this.bqm.add_variable(varName, 3*this.penaltyFactor)
            this.boolVarCount += 1
    
    def squareAux(this, auxName, factor=1):
        """! Calculate the square of an auxiliary variable,
        which is a natural number represented by multiple qubits
        in binary notation.
        """
        auxName+='_'
        for i in range(0, this.auxSize):
            this.bqm.add_variable(auxName+str(i), (pow(2, i)**2)*factor)
            for j in range(i+1, this.auxSize):
                this.bqm.add_interaction(auxName+str(i), auxName+str(j), pow(2,i+j+1)*factor)

    def sequenceOrderForSequence(this, time, sequence):
        """! Models the SEQUENCE_ORDER constraint for one sequence.
        See also sequenceOrder().
        """
        for laterTime in range(time+1, this.binCount):
            for i in range(0, len(sequence)-1):
                for elem in sequence[i+1:]:
                        #If an element is removed at time t 
                        #elements later in the sequence can't be removed earlier than t
                        this.bqm.add_interaction(this.variableName(sequence[i], laterTime), 
                                this.variableName(elem, time), this.penaltyFactor)
    
    def permutationConstraint(this):
        """! Models the PERMUTATION constraint, which ensures that 
        Each bin is only removed once and only one removal is performed
        at each point in time
        """
        #Exactly one true over each bin(each bin only gets removed once)
        #Exactly one true term: abcd => (-a-b-c-d+2ab+2ac+2ad+2bc+2bd+2cd+1)
        #This term has a constant, meaning that that minimum energy will be reduced by -n
        for elem in range(0, this.binCount):
            for i in range(0, this.binCount):
                iName = this.variableName(elem,i)
                this.bqm.add_variable(iName, -this.penaltyFactor)
                for  j in range(i+1, this.binCount):
                    this.bqm.add_interaction(iName, this.variableName(elem, j), 2*this.penaltyFactor)

        #Exactly one true over each time(only one bin gets removed at each point in time)
        #This loop has the same structure as the one above, so they could be combined
        #The loops are split for readability
        for time in range(0, this.binCount):
            for i in range(0, this.binCount):
                iName = this.variableName(i,time)
                this.bqm.add_variable(iName, -this.penaltyFactor)
                for j in range(i+1, this.binCount):
                    this.bqm.add_interaction(iName, this.variableName(j, time), 2*this.penaltyFactor)

        this.bqm.offset = 2*this.binCount*this.penaltyFactor

    def sequenceOrder(this):
        """! Models the SEQUENCE_ORDER constraint.
        This constraint ensures, that the bins of each
        sequence are removed in order.
        """
        for time in range(0, this.binCount-1):
            for sequence in this.bySequence:
                this.sequenceOrderForSequence(time, sequence) 
    
    def ftcConstraint(this):
        """! Model the boolean expressions to set the values of each f(t,c) correctly"""
        for label in this.byLabel:
            this.f(label)

    def countStackingPlacesConstraint(this):
        """! Model the inequalities which ensure that p is set to
        the highest number of stacking places required at the same time
        """

        for c in range(this.dec_bound, this.binCount-(this.dec_bound+1)):
            #Square sum_t(f(t,c))
            for i in range(0, len(this.byLabel)):
                iLabel = this.labels[i]
                this.bqm.add_variable(this.fName(iLabel, c), this.penaltyFactor)
                for j in range(i+1, len(this.byLabel)):
                    jLabel = this.labels[j]
                    this.bqm.add_interaction(this.fName(iLabel,c),this.fName(jLabel,c), 2*this.penaltyFactor)

            this.squareAux('s'+str(c), this.penaltyFactor)
            this.squareAux('p', this.penaltyFactor)
            
            #This could be done in the upper loop but doing it here makes the code easier to read
            for label in this.byLabel:
                for i in range(0, this.auxSize):
                    this.bqm.add_interaction(this.fName(label,c),'s'+str(c)+'_'+str(i), this.penaltyFactor*2*pow(2,i))
                    this.bqm.add_interaction(this.fName(label,c),'p_'+str(i), -this.penaltyFactor*2*pow(2,i))

            for i in range(0, this.auxSize):
                for j in range(0, this.auxSize):
                    this.bqm.add_interaction('s'+str(c)+'_'+str(i), 'p_'+str(j), -this.penaltyFactor*2*pow(2,i)*pow(2,j))

    
    def fixPlanVariables(this):
        """!Fixes plan variables that can never be 1 because of their position in the sequence"""
        fixed = 0
        for sequence in this.bySequence:
            #A bin that's after the first position can't be removed on the first step and so on
            for i in range(1, len(sequence)):
                index = sequence[i]
                for c in range(0, i):
                    #this.bqm.fix_variable(this.variableName(index,c), 0)
                    this.planCount -= 1
                    this.toFix[(index,c)] = True
                    fixed+=1
            #By the same logic, the first bin has to be removed at the latest after the other sequences are exhausted and so on
            binsOutsideSequence = this.binCount - len(sequence)
            for i, index in enumerate(sequence):
                for c in range(binsOutsideSequence+i+1, this.binCount):
                    #this.bqm.fix_variable(this.variableName(index, c), 0)
                    this.planCount -= 1
                    this.toFix[(index,c)] = True
                    fixed += 1
        #print("Fixed", fixed)

    def generateBQM(this):
        this.permutationConstraint()
        this.fixPlanVariables()
        this.sequenceOrder()
        this.ftcConstraint()
        this.countStackingPlacesConstraint()

        for var in this.toFix:
            this.bqm.fix_variable(this.variableName(var[0],var[1]),0)

        #Optimize p(Number of stacking places)
        for i in range(0, this.auxSize):
            this.bqm.add_variable('p_'+str(i), pow(2,i))

    def breakDownVariables(this):
        """! Output a breakdown of how many variables are created for what purpose"""
        varCount = len(this.bqm)
        print("Total number of variables: " + str(varCount))
        coreCount = this.planCount
        print("Number of plan variables: " + str(coreCount))
        varCount -= coreCount
        auxCount = (this.binCount-2*this.dec_bound)*this.auxSize
        print("Number of variables that model numbers: " + str(auxCount))
        varCount -= auxCount
        print("Number of variables that model OR and AND statements: " + str(this.boolVarCount))

def embeddingStats(embedding):
    max_len = 0
    chain_count = 0
    var_count = 0
    for source, target in embedding.items():
        chain_len = len(target)
        ic(chain_len)
        var_count += chain_len
        if chain_len > max_len:
            max_var = source
            max_len = chain_len

        if chain_len > 1:
            chain_count += 1

    return max_var, max_len, chain_count, var_count


def solveDWave(sequences, num_reads, dec_bound, prefix="data/QA-"):
    """! Approximate a solutions of the Stacking Problem with the given sequences
    using a DWave Quantum Annealer"""
    test = StackingQUBOGenerator(sequences, dec_bound)
    test.generateBQM()
    print("Generated bqm")
    test.breakDownVariables()

    sampler = EmbeddingComposite(DWaveSampler(solver='Advantage_system6.1'))
    sampleset = sampler.sample(test.bqm, num_reads=num_reads, return_embedding=True,warnings='save')
    sampleset.info['bqm'] = test.bqm
    sampleset.info['sequences'] = sequences
    sampleset.info['solverId'] = sampler.child.solver.id

    #ic(embeddingStats(sampleset.info['embedding_context']['embedding']))
    saveSampleset(sampleset, prefix)

    print('Lowest energy:', sampleset.first.energy)
    interpretSolution(sampleset.first, test.binCount)
    print('')

def solveSimAnneal(sequences,num_reads, dec_bound):
    """! Approximate a solution of the Stacking Problem with the given sequences
        using Simulated Annealing with a QUBO-Formulation of the Energy Function"""
    test = StackingQUBOGenerator(sequences, dec_bound)
    test.generateBQM()

    print("Generated bqm")
    test.breakDownVariables()

    sampler = SimulatedAnnealingSampler()
    start = time.time()
    sampleset = sampler.sample(test.bqm, num_reads=num_reads)
    end = time.time()
    sampleset.info['bqm'] = test.bqm
    sampleset.info['sequences'] = sequences
    saveSampleset(sampleset, "data/SA-")

    print('Lowest energy:', sampleset.first.energy)
    print('')
    interpretSolution(sampleset.first, test.binCount)

    return [end - start, sampleset, test]

def interpretSolution(sample, binCount):
    print('The order the bins are removed in is: ')
    for j in range(binCount):
        for i in range(binCount):
            if (('x('+str(i)+','+str(j)+')') in sample.sample) and sample.sample['x('+str(i)+','+str(j)+')'] == 1:
                print(str(j)+':'+str(i))

def parseSequences(text):
    parts = text.split('-')
    return [[int(x) for x in part.split(',')] for part in parts]

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Solve the stacking Problem using a Quantum Annealer or Simulated Annealing')
    
    parser.add_argument('-s', type=str, action='store', dest='seqs', 
            metavar='Sequences. Entries are separated by commas. Sequences are\
 separated by -.Labels are numbers', required = True)
    parser.add_argument('-m', type=str, action='store', dest='method', metavar='Method to use. Either SA or QA.', required = True)
    parser.add_argument('-nr', type=int, action='store', dest='num_reads', metavar='Number of samples to generate.', required = True)
    parser.add_argument('-db', type=int, action='store', dest='dec_bound', metavar='Boundary for decision problem', default=1)

    args = parser.parse_args(sys.argv[1:])
    sequences = parseSequences(args.seqs)
    
    if args.method == 'SA':
        solveSimAnneal(sequences, args.num_reads, args.dec_bound)
    elif args.method == 'QA':
        solveDWave(sequences, args.num_reads, args.dec_bound)
    else:
        print('Method (-m) must be either SA or QA!')
