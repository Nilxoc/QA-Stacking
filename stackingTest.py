from stacking import StackingQUBOGenerator
import dimod

def generateZeroSample(bqm):
    sample = {}
    for var in bqm.linear:
        sample[var] = 0
    return sample

passed = 0
failed = 0

print("=====TEST 1: PERMUTATION=====")
testGen = StackingQUBOGenerator([[0,1],[1,0]])
testGen.permutationConstraint()

print("    Case 1: Fulfilled(Diagonal)")
testSample = generateZeroSample(testGen.bqm)

testSample['x(0,0)'] = 1
testSample['x(1,1)'] = 1
testSample['x(2,2)'] = 1
testSample['x(3,3)'] = 1

energy =  testGen.bqm.energy(testSample)
if(energy == 0):
    print("        Permutation Case 1 passed!")
    passed += 1
else:
    print("        Permutation Case 1 FAILED!")
    failed += 1

print("    Case 2: Violated(Bin moved twice)")
testSample = generateZeroSample(testGen.bqm)

testSample['x(0,0)'] = 1
testSample['x(2,2)'] = 1
testSample['x(3,3)'] = 1
testSample['x(2,1)'] = 1

energy = testGen.bqm.energy(testSample)
if(energy > 0):
    print("        Permutation Case 2 passed!")
    passed += 1
else: 
    print("        Permutation Case 2 FAILED!")
    failed += 1

print("    Case 3: Violated(Two bins moved at the same time)")
testSample = generateZeroSample(testGen.bqm)

testSample['x(0,0)'] = 1
testSample['x(1,1)'] = 1
testSample['x(2,1)'] = 1
testSample['x(3,3)'] = 1

energy = testGen.bqm.energy(testSample)
if(energy > 0):
    print("        Permutation Case 3 passed!")
    passed += 1
else: 
    print("        Permutation Case 3 FAILED!")
    failed += 1

print("    Case 4: Fulfilled(Mixed)")
testSample = generateZeroSample(testGen.bqm)

testSample['x(3,0)'] = 1
testSample['x(1,1)'] = 1
testSample['x(2,2)'] = 1
testSample['x(0,3)'] = 1

energy = testGen.bqm.energy(testSample)
if(energy == 0):
    print("        Permutation Case 4 passed!")
    passed += 1
else: 
    print("        Permutation Case 4 FAILED!")
    failed += 1

print("\n=====TEST 2: SEQUENCE ORDER=====")
testGen = StackingQUBOGenerator([[0,1,2],[2,1,0]])
testGen.sequenceOrder()
testGen.generateLinears()

print("    Case 1: Fulfilled(Separate Sequences)")
testSample = generateZeroSample(testGen.bqm)

testSample['x(0,0)'] = 1
testSample['x(1,1)'] = 1
testSample['x(2,2)'] = 1
testSample['x(3,3)'] = 1
testSample['x(4,4)'] = 1
testSample['x(5,5)'] = 1

energy = testGen.bqm.energy(testSample)
if(energy == 0):
    print("        Sequence Order Case 1 passed!")
    passed += 1
else:
    print("        Sequence Order Case 1 FAILED!")
    failed += 1

print("    Case 2: Fulfilled(Mixed Sequences)")
testSample = generateZeroSample(testGen.bqm)

testSample['x(0,0)'] = 1
testSample['x(3,1)'] = 1
testSample['x(4,2)'] = 1
testSample['x(1,3)'] = 1
testSample['x(2,4)'] = 1
testSample['x(5,5)'] = 1

energy = testGen.bqm.energy(testSample)
if(energy == 0):
    print("        Sequence Order Case 2 passed!")
    passed += 1
else:
    print("        Sequence Order Case 2 FAILED!")
    failed += 1

print("    Case 3: Violated")
testSample = generateZeroSample(testGen.bqm)

testSample['x(0,0)'] = 1
testSample['x(4,1)'] = 1
testSample['x(1,2)'] = 1
testSample['x(3,3)'] = 1
testSample['x(2,4)'] = 1
testSample['x(5,5)'] = 1

energy = testGen.bqm.energy(testSample)
if(energy > 0):
    print("        Sequence Order Case 3 passed!")
    passed += 1
else:
    print("        Sequence Order Case 3 FAILED!")
    failed += 1

print("\n=====TEST 3: f(t,c)=====")
testGen = StackingQUBOGenerator([[0,1],[1,0]])
testGen.ftcConstraint()
testGen.countStackingPlacesConstraint()
testGen.generateLinears()

def generateValidFSample(testGen):
    #Step 1: Plan
    testSample = generateZeroSample(testGen.bqm)
    testSample['x(0,0)'] = 1
    testSample['x(2,1)'] = 1
    testSample['x(1,2)'] = 1
    testSample['x(3,3)'] = 1
    #Step 2: OR-Expressions
    #Unfortunately this part depends on the way the or constraints are 
    #constructed and might break if it changes
    testSample['x(0,0)orx(3,0)'] = 1
    testSample['x(0,0)orx(3,0)orx(0,1)orx(3,1)'] = 1
    testSample['x(0,0)orx(3,0)orx(0,1)orx(3,1)orx(0,2)orx(3,2)'] = 1

    testSample['x(1,1)orx(2,1)'] = 1
    testSample['x(1,3)orx(2,3)orx(1,2)orx(2,2)orx(1,1)orx(2,1)'] = 1
    testSample['x(1,0)orx(2,0)orx(1,1)orx(2,1)'] = 1
    testSample['x(1,0)orx(2,0)orx(1,1)orx(2,1)orx(1,2)orx(2,2)'] = 1

    testSample['x(1,2)orx(2,2)'] = 1
    testSample['x(1,3)orx(2,3)orx(1,2)orx(2,2)'] = 1
    testSample['x(1,3)orx(2,3)orx(1,2)orx(2,2)orx(1,1)orx(2,1)'] = 1
    testSample['x(1,0)orx(2,0)orx(1,1)orx(2,1)orx(1,2)orx(2,2)'] = 1

    testSample['x(0,3)orx(3,3)'] = 1
    testSample['x(0,3)orx(3,3)orx(0,2)orx(3,2)'] = 1
    testSample['x(0,3)orx(3,3)orx(0,2)orx(3,2)orx(0,1)orx(3,1)'] = 1

    #Step 3: F Values
    testSample['f(0,0)'] = 1
    testSample['f(0,1)'] = 1
    testSample['f(0,2)'] = 1

    testSample['f(1,1)'] = 1

    #Step 4: Slack variables and p
    testSample['p_1'] = 1

    testSample['s0_0'] = 1
    testSample['s2_0'] = 1
    
    return testSample

print("    Case 1: Fulfilled")
testSample = generateValidFSample(testGen)

energy = testGen.bqm.energy(testSample)
if(energy == 0):
    print("        f(t,c) Case 1 passed!")
    passed += 1
else:
    print("        f(t,c) Case 1 FAILED!")
    failed += 1

print("    Case 2: Violated(P too low)")
testSample = generateValidFSample(testGen)

testSample['p_1'] = 0
testSample['p_0'] = 1

testSample['s0_0'] = 0
testSample['s2_0'] = 0

energy = testGen.bqm.energy(testSample)
if(energy > 0):
    print("        f(t,c) Case 2 passed!")
    passed += 1
else:
    print("        f(t,c) Case 2 FAILED!")
    failed += 1

print("    Case 3: Violated(Bad OR Statement)")
testSample = generateValidFSample(testGen)

testSample['x(0,0)orx(3,0)orx(0,1)orx(3,1)orx(0,2)orx(3,2)'] = 0
testSample['f(0,0)'] = 0 #Consequence of change above

energy = testGen.bqm.energy(testSample)
if(energy > 0):
    print("        f(t,c) Case 3 passed!")
    passed += 1
else:
    print("        f(t,c) Case 3 FAILED!")
    failed += 1

print("    Case 4: Violated(Bad AND Statement)")
testSample = generateValidFSample(testGen)

testSample['f(0,2)'] = 0

#Since the sum of f(t,2) is now 0 slack needs to increase
testSample['s2_0'] = 0
testSample['s2_1'] = 1

energy = testGen.bqm.energy(testSample)
if(energy > 0):
    print("        f(t,c) Case 4 passed!")
    passed += 1
else:
    print("        f(t,c) Case 4 FAILED!")
    failed += 1

print("\n=====TEST 4: Full Test=====")
testGen = StackingQUBOGenerator([[0,1],[1,0]])
testGen.generateBQM()

print("    Case 1: Full Test")
testSample = generateValidFSample(testGen)

energy = testGen.bqm.energy(testSample)
if(energy == 2):
    print("        Full Test Case 1 passed!")
    passed += 1
else:
    print("        Full Test Case 1 FAILED!")
    failed += 1

print("\nPassed " + str(passed) + " tests\nFailed " + str(failed) + " tests")
