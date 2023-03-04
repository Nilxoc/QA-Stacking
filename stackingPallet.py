import dimod
import math
from dwave.system import EmbeddingComposite, DWaveSampler
from dwave.embedding import (target_to_source, unembed_sampleset, embed_bqm, 
                             chain_to_quadratic, EmbeddedStructure)
from dwave.system.warnings import WarningAction, WarningHandler
import time
import argparse
import sys

from neal.sampler import SimulatedAnnealingSampler
from qaUtils import saveSampleset
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

"""
    This class was adapted from code made available by D-Wave Systems Inc.
    under the APACHE-2.0 License
"""

class ScalingEmbeddingComposite(EmbeddingComposite):
    def sample(self, bqm, chain_strength=None,
               chain_break_method=None,
               chain_break_fraction=True,
               embedding_parameters=None,
               return_embedding=None,
               warnings=None,
               lin_range=None,
               quad_range=None,
               **parameters):
        """Sample from the provided binary quadratic model.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            chain_strength (float/mapping/callable, optional):
                Sets the coupling strength between qubits representing variables 
                that form a :term:`chain`. Mappings should specify the required 
                chain strength for each variable. Callables should accept the BQM 
                and embedding and return a float or mapping. By default, 
                `chain_strength` is calculated with
                :func:`~dwave.embedding.chain_strength.uniform_torque_compensation`.

            chain_break_method (function/list, optional):
                Method or methods used to resolve chain breaks. If multiple
                methods are given, the results are concatenated and a new field
                called "chain_break_method" specifying the index of the method
                is appended to the sample set.
                See :func:`~dwave.embedding.unembed_sampleset` and
                :mod:`dwave.embedding.chain_breaks`.

            chain_break_fraction (bool, optional, default=True):
                Add a `chain_break_fraction` field to the unembedded response with
                the fraction of chains broken before unembedding.

            embedding_parameters (dict, optional):
                If provided, parameters are passed to the embedding method as
                keyword arguments. Overrides any `embedding_parameters` passed
                to the constructor.

            return_embedding (bool, optional):
                If True, the embedding, chain strength, chain break method and
                embedding parameters are added to :attr:`dimod.SampleSet.info`
                of the returned sample set. The default behaviour is defined
                by :attr:`return_embedding_default`, which itself defaults to
                False.

            warnings (:class:`~dwave.system.warnings.WarningAction`, optional):
                Defines what warning action to take, if any. See
                :mod:`~dwave.system.warnings`. The default behaviour is defined
                by :attr:`warnings_default`, which itself defaults to
                :class:`~dwave.system.warnings.IGNORE`

            **parameters:
                Parameters for the sampling method, specified by the child
                sampler.

        Returns:
            :obj:`dimod.SampleSet`

        Examples:
            See the example in :class:`EmbeddingComposite`.

        """
        if return_embedding is None:
            return_embedding = self.return_embedding_default

        # solve the problem on the child system
        child = self.child

        # apply the embedding to the given problem to map it to the child sampler
        __, target_edgelist, target_adjacency = self.target_structure

        # add self-loops to edgelist to handle singleton variables
        source_edgelist = list(bqm.quadratic) + [(v, v) for v in bqm.linear]

        # get the embedding
        if embedding_parameters is None:
            embedding_parameters = self.embedding_parameters
        else:
            # we want the parameters provided to the constructor, updated with
            # the ones provided to the sample method. To avoid the extra copy
            # we do an update, avoiding the keys that would overwrite the
            # sample-level embedding parameters
            embedding_parameters.update((key, val)
                                        for key, val in self.embedding_parameters
                                        if key not in embedding_parameters)

        embedding = self.find_embedding(source_edgelist, target_edgelist,
                                        **embedding_parameters)

        if bqm and not embedding:
            raise ValueError("no embedding found")

        if not hasattr(embedding, 'embed_bqm'):
            embedding = EmbeddedStructure(target_edgelist, embedding)
        
        old_bqm = bqm.copy()
        bqm.change_vartype(dimod.SPIN)

        bqm_embedded = embedding.embed_bqm(bqm, chain_strength=chain_strength,
                                           smear_vartype=dimod.SPIN)
        if lin_range:
            bqm_embedded.normalize(lin_range, quad_range)

        if warnings is None:
            warnings = self.warnings_default
        elif 'warnings' in child.parameters:
            parameters.update(warnings=warnings)

        warninghandler = WarningHandler(warnings)

        warninghandler.chain_strength(bqm, embedding.chain_strength, embedding)
        warninghandler.chain_length(embedding)

        if 'initial_state' in parameters:
            # if initial_state was provided in terms of the source BQM, we want
            # to modify it to now provide the initial state for the target BQM.
            # we do this by spreading the initial state values over the
            # chains
            state = parameters['initial_state']
            parameters['initial_state'] = {u: state[v]
                                           for v, chain in embedding.items()
                                           for u in chain}

        if self.scale_aware and 'ignored_interactions' in child.parameters:

            ignored = []
            for chain in embedding.values():
                # just use 0 as a null value because we don't actually need
                # the biases, just the interactions
                ignored.extend(chain_to_quadratic(chain, target_adjacency, 0))

            parameters['ignored_interactions'] = ignored

        response = child.sample(bqm_embedded, auto_scale=False, **parameters)

        bqm = old_bqm
        response = response.change_vartype(bqm.vartype)

        #Energies are recalculated during unembed, so scaling back is unneccessary
        def async_unembed(response):
            # unembed the sampleset aysnchronously.

            warninghandler.chain_break(response, embedding)

            sampleset = unembed_sampleset(response, embedding, source_bqm=bqm,
                                          chain_break_method=chain_break_method,
                                          chain_break_fraction=chain_break_fraction,
                                          return_embedding=return_embedding)

            if return_embedding:
                sampleset.info['embedding_context'].update(
                    embedding_parameters=embedding_parameters,
                    chain_strength=embedding.chain_strength)

            if chain_break_fraction and len(sampleset):
                warninghandler.issue("All samples have broken chains",
                                     func=lambda: (sampleset.record.chain_break_fraction.all(), None))

            if warninghandler.action is WarningAction.SAVE:
                # we're done with the warning handler so we can just pass the list
                # off, if later we want to pass in a handler or similar we should
                # do a copy
                sampleset.info.setdefault('warnings', []).extend(warninghandler.saved)

            return sampleset

        return dimod.SampleSet.from_future(response, async_unembed)

class PalletQUBOGenerator:
    def constructSequenceGraph(this):
        """!
          \brief Constructs the sequence Graph for the problem

          The sequence graph contains a directed edge between labels iff
          the label at the start of the edge must be opened before the label
          at the end of the edge can be closed.
        """
        this.sequenceGraph = set() 

        for sequence in this.sequences:
            earlierLabels = set()
            for label in sequence:
                for eLabel in earlierLabels:
                    if eLabel != label:
                        this.sequenceGraph.add((eLabel,label))
                earlierLabels.add(label)
        
        #Convert the sequenceGraph to list for conistent ordering
        this.sequenceGraph = [edge for edge in this.sequenceGraph] 

    def __init__(this, sequences, autoGenerate=True, penaltyMul=50):
        """!
          Constructs a generator for pallet-solution bqms
        
          \param sequences List of sequences to stack from. The sequences are ordered lists of labels.
          \param autoGenerate Whether to immediately generate the full bqm during construction
        """
        this.sequences = sequences

        labels = set()
        for sequence in this.sequences:
            for label in sequence:
                labels.add(label)
        this.numLabels = len(labels) #Number of different labels
 
        this.bqm = dimod.BinaryQuadraticModel(vartype='BINARY')

        this.auxSize = math.floor(math.log(this.numLabels,2))+1#Number of auxiliary variables to hold a number
       
        this.penaltyFactor = pow(2,this.auxSize)*penaltyMul #Penalty larger than maximum possible p

        if autoGenerate:
            this.generateBQM()
    
    def varName(this, i, j):
        """!
          \brief Returns the BQM-Variable name for the plan variable with the given indices
          
          \param i Index i
          \param j Index j
        """
        return 'x(' + str(i) + ',' + str(j) +')'

    def permutationConstraint(this):
        """! 
        \brief Models the constraint which ensures that each position
        is used by exactly one label and each label uses exactly one
        position"""
        for k in range(0, this.numLabels):
            for i in range(0, this.numLabels):
                iName = this.varName(k, i)
                invIName = this.varName(i,k)

                this.bqm.add_variable(iName, -this.penaltyFactor)
                this.bqm.add_variable(invIName, -this.penaltyFactor)
                for j in range(i+1, this.numLabels):
                    jName = this.varName(k,j)
                    invJName = this.varName(j,k)

                    this.bqm.add_interaction(iName, jName, 2*this.penaltyFactor)
                    this.bqm.add_interaction(invIName, invJName,2*this.penaltyFactor)

        this.bqm.offset = 2*this.penaltyFactor*this.numLabels
    
    def modelOr(this, left, right, auxName):
        """!
          \brief Models the boolean expression auxName = left OR right in the BQM
          
          \param left One of the variables of the expression
          \param right One of the variables of the expression
          \param auxName Name of the auxiliary variable which holds the result of the expression
        """
        this.bqm.add_variable(left, this.penaltyFactor)
        this.bqm.add_variable(right, this.penaltyFactor)
        this.bqm.add_variable(auxName, this.penaltyFactor)
        this.bqm.add_interaction(left,right,this.penaltyFactor)
        this.bqm.add_interaction(left, auxName, -2*this.penaltyFactor)
        this.bqm.add_interaction(right, auxName, -2*this.penaltyFactor)

    def yName(this, j, c):
        """!
          \brief Returns name for auxiliary variable holding value of Y(j,c)
        
          \param j Value of j
          \param c Value of c
        """
        return 'Y(' + str(j) + ',' + str(c) + ')'

    def y(this,j,c):
        """! 
            \brief Model Y(j,c) for the given j and c.
        
            \param j Value of j
            \param c Value of c
        """
        test  = 0
        #Y(j,c) = Y(j,c+1) OR (verodert alle Konjunktionen von i, i')
        j2 = c+1
        conjunctions = []
        for edge in this.sequenceGraph:
            left = this.varName(edge[1],j)
            right = this.varName(edge[0],j2)
            auxName = left + 'and' + right
            #AND Bedingung
            if not (auxName in this.bqm.variables):
                this.bqm.add_interaction(left, right, this.penaltyFactor)
                this.bqm.add_interaction(left, auxName, -2*this.penaltyFactor)
                this.bqm.add_interaction(right, auxName, -2*this.penaltyFactor)
                this.bqm.add_variable(auxName, 3*this.penaltyFactor)

            conjunctions.append(auxName)
            test += 1

        while(len(conjunctions) > 1):
            pair = []
            left = conjunctions.pop(0)
            right = conjunctions.pop(0)

            auxName = '(' + left +')or('+right + ')'
            test += 1
            if len(conjunctions) == 0 and c==(this.numLabels-1):
                auxName = this.yName(j,c)

            if not (auxName in this.bqm.variables):
                this.modelOr(left, right, auxName)
                
            conjunctions.append(auxName)
             
        #The expression can be modeled recursively 
        #since it grows longer with smaller c but the edges don't change
        if c < this.numLabels-2:
            left = conjunctions[0]
            right = this.yName(j, c+1)
            auxName = this.yName(j,c)
            test += 1

            this.modelOr(left,right,auxName)
        else:
            this.bqm.relabel_variables({conjunctions[0]:this.yName(j,c)})
        

    def yjc(this):
        """!
          \brief Generates all relevant expressions for Y(j,c)
        """
        #Reversed to avoid having to combine variables 
        #created by recursion
        #(relabeling with an exisiting name is not permitted)
        for c in reversed(range(0, this.numLabels-1)):
            for j in range(0, c+1):
                this.y(j,c)

    def squareAux(this, auxName, factor=1):
        """! 
          \brief Add expression to represent the square of an auxiliary variable,
        which is a natural number represented by multiple qubits
        in binary notation.

        \param auxName Name of the number to square
        \param factor Factor to multiply the squared variable by
        """
        auxName+='_'
        for i in range(0, this.auxSize):
            this.bqm.add_variable(auxName+str(i), (pow(2, i)**2)*factor)
            for j in range(i+1, this.auxSize):
                this.bqm.add_interaction(auxName+str(i), auxName+str(j), pow(2,i+j+1)*factor)

   
    def inequalityConstraints(this):
        """! Models the necessary inequalities to set w to the correct value"""
        for c in range(0, this.numLabels-1):
            for j in range(0, c+1): 
                this.bqm.add_variable(this.yName(j,c), this.penaltyFactor)
                for j2 in range(j+1, c+1):
                    this.bqm.add_interaction(this.yName(j,c), this.yName(j2,c), 2*this.penaltyFactor)

                for i in range(0, this.auxSize):
                    this.bqm.add_interaction(this.yName(j,c),'s'+str(c)+'_'+str(i), this.penaltyFactor*2*pow(2,i))
                    this.bqm.add_interaction(this.yName(j,c),'w_'+str(i), -this.penaltyFactor*2*pow(2,i))

            for i in range(0, this.auxSize):
                for j in range(0, this.auxSize):
                    this.bqm.add_interaction('s'+str(c)+'_'+str(i), 'w_'+str(j), -this.penaltyFactor*2*pow(2,i)*pow(2,j))

            this.squareAux('w', this.penaltyFactor)
            this.squareAux('s'+str(c), this.penaltyFactor)


    def generateBQM(this):
        """!
          \brief Performs all neccessary steps to fully model the problem
        """
        this.constructSequenceGraph()
        this.permutationConstraint()
        this.yjc()
        this.inequalityConstraints()
 
        for i in range(0, this.auxSize):
            this.bqm.add_variable('w_'+str(i), pow(2,i))
    
    def breakDownVariables(this):
        """!
          \brief Prints information about variable usage to console
        """
        remaining = len(this.bqm)
        plan = this.numLabels**2
        remaining -= plan
        print('Number of plan variables:', plan)
        numbers = this.auxSize*this.numLabels
        remaining -= numbers
        print('Number of variables that model numbers:', numbers)
        print('Number of variables that model boolean expressions:', remaining)
    
    def getMaxBias(this):
        """!
          \brief Returns the biggest coupler bias
        """
        maxKey = ''
        maxBias = 0
        for key1, key2 in this.bqm.iter_interactions():
            bias =  abs(this.bqm.get_quadratic(key1, key2))
            if bias > maxBias:
                    maxKey = key1+','+key2 
                    maxBias = bias

        for key in this.bqm.iter_variables():
            bias = abs(this.bqm.get_linear(key))
            if bias > maxBias:
                maxKey = key
                maxBias = bias

        print('Max bias is',maxBias,'at',maxKey)

        return maxBias

    def interpretSample(this, sample):
        """!
          Interprets the solution described by the given sample

          \param sample The sample to examine 
        """
        if sample.energy > (pow(2,this.auxSize)-1):
            print('WARNING: There appear to be violated constraints in the given sample,\
making the solution invalid!')

        print('The pallets are opened in this order:')
        for j in range(0, this.numLabels):
            for i in range(0, this.numLabels):
                if sample.sample[this.varName(i,j)] == 1:
                    print( str(j+1)+'.', i)
        print('The number of stacking places required is (according to the sample)', sample.energy+1)


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

def solveDWave(sequences, num_reads, penaltyMul=50, prefix="data/pallet/QA-", **args):
    """! 
    \brief Approximate a solutions of the Stacking Problem with the given sequences
    using a DWave Quantum Annealer

    \param sequences The sequences of the problem instance
    \param num_reads Number of samples to generate
    \param penaltyMul Value to mutiply the minimum possible penalty for violation of constraints by
    \param **args Additional keyword arguments are forwarded to DwaveSampler.sample()
    """

    test = PalletQUBOGenerator(sequences, penaltyMul = penaltyMul)
    print("Generated bqm")
    print("Number of Variables: ", len(test.bqm))
    #ic(test.bqm)
    #ic(test.penaltyFactor)
   
    sampler = EmbeddingComposite(DWaveSampler(solver='Advantage_system6.1'))
    sampleset = sampler.sample(test.bqm, num_reads=num_reads, return_embedding=True,warnings='save', **args)#PARAMETERS HERE
    sampleset.info['bqm'] = test.bqm
    sampleset.info['sequences'] = sequences
    sampleset.info['penaltyFactor'] = test.penaltyFactor
    sampleset.info['solverId'] = sampler.child.solver.id

    #ic(embeddingStats(sampleset.info['embedding_context']['embedding']))
    saveSampleset(sampleset, prefix)

    print('Lowest energy:', sampleset.first.energy)
    test.interpretSample(sampleset.first)
    test.breakDownVariables()
    return sampleset

def solveSimAnneal(sequences,num_reads, penaltyMul=50, **args):
    """! 

    \brief Approximate a solution of the Stacking Problem with the given sequences
        using Simulated Annealing with a QUBO-Formulation of the Energy Function

    \param sequences The sequences of the problem instance
    \param num_reads Number of samples to generate
    \param penaltyMul Value to mutiply the minimum possible penalty for violation of constraints by
    \param **args Additional keyword arguments are forwarded to SimulatedAnnealingSampler.sample()
    """

    test = PalletQUBOGenerator(sequences, penaltyMul = penaltyMul)
    print("Generated bqm")
    print("Number of variables: ", len(test.bqm))

    sampler = SimulatedAnnealingSampler()
    start = time.time()
    sampleset = sampler.sample(test.bqm, num_reads=num_reads, **args)
    end = time.time()
    sampleset.info['bqm'] = test.bqm
    sampleset.info['sequences'] = sequences
    sampleset.info['penaltyFactor'] = test.penaltyFactor
    saveSampleset(sampleset, "data/pallet/SA-")

    print('Lowest energy:', sampleset.first.energy)
    test.interpretSample(sampleset.first)
    test.breakDownVariables()

    return [end - start, sampleset, test]


def parseSequences(text):
    parts = text.split('-')
    return [[int(x) for x in part.split(',')] for part in parts]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve the stacking Problem using a Quantum Annealer or Simulated Annealing')
    
    requiredNamed = parser.add_argument_group('required arguments')

    requiredNamed.add_argument('-s', type=str, action='store', dest='seqs', 
            metavar='Sequences. Entries are separated by commas. Sequences are\
 separated by -.Labels are numbers', required = True)
    requiredNamed.add_argument('-m', type=str, action='store', dest='method', metavar='Method to use. Either SA or QA.', required = True)
    requiredNamed.add_argument('-nr', type=int, action='store', dest='num_reads', metavar='Number of samples to generate.', required = True)

    parser.add_argument('-p', type=int, action='store', dest='penalty', metavar='Factor to multiply lowest possible penalty A by', default = 50)

    args = parser.parse_args(sys.argv[1:])
    sequences = parseSequences(args.seqs)
    print("Solving instance " + str(sequences))
    
    if args.method == 'SA':
        solveSimAnneal(sequences, args.num_reads, args.penalty)
    elif args.method == 'QA':
        solveDWave(sequences, args.num_reads, args.penalty)
    else:
        print('Method (-m) must be either SA or QA!') 
