import time
import math
import random

N = 128

# Precompute forbidden triples as clauses
# Each clause is a tuple (a, b, c) meaning NOT(all three included)
# We'll represent the problem as: maximize number of included elements
# subject to: for each forbidden triple (a,b,c), not all three are included

forbidden_triples = []

def precompute_triples():
    """Find all forbidden triples where lcm(a,b)==lcm(a,c)==lcm(b,c)"""
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            L_ab = math.lcm(a, b)
            for c in range(b + 1, N + 1):
                if L_ab % c != 0:
                    continue
                if math.lcm(a, c) == L_ab and math.lcm(b, c) == L_ab:
                    forbidden_triples.append((a - 1, b - 1, c - 1))

precompute_triples()

def popcount(x):
    return bin(x).count('1')

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_size = 0
    best_set = 0  # Bitmask
    
    # WalkSAT-style local search for Max-SAT
    # Variables: x[0..127], x[i]=1 means element i+1 is included
    # Clauses: for each forbidden triple (a,b,c), clause is (NOT x[a] OR NOT x[b] OR NOT x[c])
    # Objective: maximize sum(x[i])
    
    def walk_sat(seed_offset=0):
        nonlocal best_size, best_set
        
        rng = random.Random(seed_offset)
        
        # Initialize with greedy
        # Try to include as many as possible
        assignment = [0] * N
        included = set()
        
        # Greedy initialization
        for i in range(N):
            # Check if adding i violates any clause with already included elements
            can_add = True
            for (a, b, c) in forbidden_triples:
                if i in (a, b, c):
                    other1, other2 = (a, b, c)[[a, b, c].index(i)+1], (a, b, c)[[a, b, c].index(i)+2]
                    if other1 in included and other2 in included:
                        can_add = False
                        break
            if can_add:
                assignment[i] = 1
                included.add(i)
        
        current_size = len(included)
        if current_size > best_size:
            best_size = current_size
            best_set = 0
            for i in included:
                best_set |= (1 << i)
        
        # Local search
        num_restarts = 50
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random restart with perturbation
            if restart == 0:
                pass  # Use current assignment
            else:
                # Flip some random variables
                for _ in range(rng.randint(5, 20)):
                    i = rng.randint(0, N - 1)
                    assignment[i] = 1 - assignment[i]
                    if assignment[i]:
                        included.add(i)
                    else:
                        included.discard(i)
                current_size = len(included)
            
            # WalkSAT iterations
            max_iters = 2000
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break
                
                # Check if we have a valid solution (no violated clauses)
                violated = []
                for (a, b, c) in forbidden_triples:
                    if assignment[a] and assignment[b] and assignment[c]:
                        violated.append((a, b, c))
                
                if not violated:
                    # Valid solution
                    sz = sum(assignment)
                    if sz > best_size:
                        best_size = sz
                        best_set = 0
                        for i in range(N):
                            if assignment[i]:
                                best_set |= (1 << i)
                    break
                
                # Pick a random violated clause
                clause = rng.choice(violated)
                a, b, c = clause
                
                # Compute score for flipping each variable in the clause
                best_var = None
                best_score = -1
                
                for var in (a, b, c):
                    # Score = (number of satisfied clauses that would become unsatisfied) 
                    #       - (number of unsatisfied clauses that would become satisfied)
                    # Simplified: prefer flipping variables that appear in fewer clauses
                    # and have lower degree
                    score = 0
                    
                    # Count how many other variables in this clause are 1
                    others_1 = 0
                    for v in (a, b, c):
                        if v != var and assignment[v]:
                            others_1 += 1
                    
                    # If we flip var from 1 to 0, we satisfy this clause
                    # But we might break other clauses where var=1 was needed
                    # For Max-SAT, we want to maximize satisfied clauses
                    # Flipping a 1 to 0: we satisfy all clauses containing var that were unsatisfied
                    # But we break all clauses containing var that were satisfied (where var was the only 1)
                    
                    # Heuristic: prefer flipping variables with lower "degree" in violated clauses
                    # Count how many violated clauses contain this variable
                    var_violated_count = 0
                    for v_clause in violated:
                        if var in v_clause:
                            var_violated_count += 1
                    
                    # Also consider the current value
                    if assignment[var]:
                        # Flipping to 0: good if this variable is in many violated clauses
                        score = var_violated_count
                    else:
                        # Flipping to 1: bad if this variable is in many violated clauses
                        score = -var_violated_count
                    
                    # Add randomness
                    score += rng.random() * 0.5
                    
                    if score > best_score:
                        best_score = score
                        best_var = var
                
                if best_var is not None:
                    assignment[best_var] = 1 - assignment[best_var]
                    if assignment[best_var]:
                        included.add(best_var)
                    else:
                        included.discard(best_var)
            
            # After each restart, check current solution
            violated_final = []
            for (a, b, c) in forbidden_triples:
                if assignment[a] and assignment[b] and assignment[c]:
                    violated_final.append((a, b, c))
            
            if not violated_final:
                sz = sum(assignment)
                if sz > best_size:
                    best_size = sz
                    best_set = 0
                    for i in range(N):
                        if assignment[i]:
                            best_set |= (1 << i)
        
        return best_size, best_set
    
    # Run WalkSAT with multiple seeds
    for seed in range(20):
        if time.time() > deadline - 5:
            break
        walk_sat(seed)
    
    # Also try simple greedy with different orders
    def greedy(order):
        S = 0
        included = set()
        for x in order:
            can_add = True
            for (a, b, c) in forbidden_triples:
                if x in (a, b, c):
                    idx = [a, b, c].index(x)
                    others = [a, b, c][:idx] + [a, b, c][idx+1:]
                    if others[0] in included and others[1] in included:
                        can_add = False
                        break
            if can_add:
                S |= (1 << x)
                included.add(x)
        return S
    
    # Natural order
    S = greedy(range(N))
    if popcount(S) > best_size:
        best_size = popcount(S)
        best_set = S
    
    # Reverse order
    S = greedy(range(N-1, -1, -1))
    if popcount(S) > best_size:
        best_size = popcount(S)
        best_set = S
    
    # Random orders
    for _ in range(50):
        if time.time() > deadline - 3:
            break
        order = list(range(N))
        random.shuffle(order)
        S = greedy(order)
        if popcount(S) > best_size:
            best_size = popcount(S)
            best_set = S
    
    # Convert best_set bitmask to list
    result = []
    for i in range(N):
        if (best_set >> i) & 1:
            result.append(i + 1)
    
    if not result:
        result = [1]
        
    return result
