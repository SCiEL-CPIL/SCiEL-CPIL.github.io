"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    def time_left():
        return deadline - time.time()

    def solve_n(n, time_budget):
        if n < 4:
            return None

        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}

        # Generate K4s and K5s
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))

        # Clauses:
        # Type 1 (Red K4 forbidden): For each K4, the 4 triples cannot all be Red (True).
        # Clause: NOT(t1 AND t2 AND t3 AND t4) <=> (NOT t1 OR NOT t2 OR NOT t3 OR NOT t4)
        # In SAT terms with x=True=Red: Clause is satisfied if at least one x is False.
        # Type 2 (Blue K5 forbidden): For each K5, the 10 triples cannot all be Blue (False).
        # Clause: NOT(NOT t1 AND ... AND NOT t10) <=> (t1 OR ... OR t10)
        # In SAT terms: Clause is satisfied if at least one x is True.

        # We will store clauses as lists of (var_idx, sign) where sign=1 means var must be True, -1 means var must be False.
        # Actually, for local search, it's easier to just store the variable indices and the required polarity.
        # Let's represent a clause as a list of ints. Positive int means "var must be True", negative means "var must be False".
        
        clauses = []
        # Map var -> list of (clause_idx, sign_in_clause)
        var_to_clauses = [[] for _ in range(num_triples)]

        # Add K4 clauses
        for k4 in k4s:
            ts = []
            for tri in combinations(k4, 3):
                ts.append(triple_idx[tri])
            # Clause: At least one is Blue (False). So for each var, it must be False to satisfy the clause alone?
            # No, the clause is (NOT t1 OR NOT t2 ...).
            # A literal is satisfied if its value matches the polarity.
            # Literal NOT t1 is satisfied if t1 is False.
            # So we add literals (-t1, -t2, -t3, -t4).
            clause_idx = len(clauses)
            clauses.append([-t for t in ts])
            for t in ts:
                var_to_clauses[t].append((clause_idx, -1)) # sign -1 means "must be False"

        # Add K5 clauses
        for k5 in k5s:
            ts = []
            for tri in combinations(k5, 3):
                ts.append(triple_idx[tri])
            # Clause: At least one is Red (True).
            # So we add literals (t1, t2, ..., t10).
            clause_idx = len(clauses)
            clauses.append([t for t in ts])
            for t in ts:
                var_to_clauses[t].append((clause_idx, 1)) # sign 1 means "must be True"

        num_clauses = len(clauses)
        
        # State: colors (0=False/Blue, 1=True/Red)
        # We will track unsatisfied clauses.
        
        best_result = None
        
        # Try multiple restarts
        max_restarts = 50
        for r in range(max_restarts):
            if time_left() < 2.0:
                break
            
            # Random initialization
            colors = [random.randint(0, 1) for _ in range(num_triples)]
            
            # Initialize unsatisfied set
            # A clause is unsatisfied if all its literals are false.
            # Literal (var, sign) is false if (var is True and sign is -1) OR (var is False and sign is 1).
            # i.e., colors[var] != (sign == 1)
            
            unsatisfied = set()
            for c_idx, clause in enumerate(clauses):
                sat = False
                for lit in clause:
                    v = abs(lit)
                    s = 1 if lit > 0 else -1
                    # Literal is satisfied if colors[v] == (1 if s==1 else 0)
                    if colors[v] == (1 if s == 1 else 0):
                        sat = True
                        break
                if not sat:
                    unsatisfied.add(c_idx)
            
            if len(unsatisfied) == 0:
                return colors

            # Local Search
            # WalkSAT style: 
            # 1. Pick a random unsatisfied clause.
            # 2. Pick a variable in that clause to flip.
            #    - With prob p, pick a variable that minimizes the increase in unsatisfied clauses (or maximizes decrease).
            #    - With prob 1-p, pick a random variable in the clause.
            
            # Optimization: Precompute for each variable, which clauses it affects?
            # We have var_to_clauses.
            
            # To evaluate the delta of flipping a variable v:
            # For each clause c in var_to_clauses[v]:
            #   Check if c is currently unsatisfied.
            #   If it is, check if flipping v satisfies it (i.e., is v the only unsatisfied literal? No, if flipping v makes the literal true, the clause becomes satisfied ONLY IF all other literals were already true? No.
            #   A clause becomes satisfied if at least one literal is true.
            #   If c was unsatisfied, all literals were false.
            #   If we flip v such that its literal becomes true, the clause becomes satisfied.
            #   So if c is unsatisfied and v is in c, flipping v (to the correct polarity) satisfies c.
            #   
            #   If c was satisfied, it has at least one true literal.
            #   If we flip v, and v's literal was true, it becomes false.
            #   The clause becomes unsatisfied ONLY IF v was the ONLY true literal in c.
            
            # So, delta = (# unsatisfied clauses containing v that become satisfied) - (# satisfied clauses containing v that become unsatisfied)
            
            # We need to know for each satisfied clause containing v, how many true literals it has.
            # Let's maintain `true_lit_count[c]` for each clause?
            # That's expensive to update.
            
            # Alternative: Just calculate the delta on the fly for the candidate variables in the selected clause.
            # The clause size is small (4 for K4, 10 for K5).
            # So for the selected clause, we have at most 10 candidates.
            # For each candidate v, we check all clauses in var_to_clauses[v].
            # Number of clauses per variable:
            # Each triple is in C(n-3, 1) K4s and C(n-3, 2) K5s.
            # For n=34: C(31,1)=31 K4s, C(31,2)=465 K5s. Total ~500 clauses per var.
            # 10 candidates * 500 clauses = 5000 checks per step.
            # This is acceptable in Python if optimized? 5000 * 10000 steps = 50M ops. Might be slow.
            
            # Let's try a simpler heuristic first: Random Walk with minimal improvement.
            # Or just use a standard local search loop with a time check.
            
            T = 1.0
            cooling = 0.999
            min_T = 0.01
            max_steps = 5000
            
            for step in range(max_steps):
                if time_left() < 2.0:
                    break
                
                if not unsatisfied:
                    return colors
                
                # Pick random unsatisfied clause
                c_idx = random.choice(list(unsatisfied))
                clause = clauses[c_idx]
                
                # Evaluate candidates
                best_delta = float('inf')
                best_var = -1
                best_new_color = -1
                
                # With some probability, just pick a random var in the clause to escape local minima
                if random.random() < 0.1:
                    v = random.choice([abs(lit) for lit in clause])
                    new_color = 1 - colors[v]
                    # Calculate delta for this specific v
                    delta = 0
                    for c2_idx, sign in var_to_clauses[v]:
                        # Check if c2 is currently unsatisfied
                        is_unsat = c2_idx in unsatisfied
                        # Check if flipping v satisfies or unsatisfies c2
                        # Current state of literal v in c2:
                        # Literal is (v, sign).
                        # Current truth value of literal: 
                        # If sign==1, literal is true if colors[v]==1.
                        # If sign==-1, literal is true if colors[v]==0.
                        curr_lit_true = (colors[v] == (1 if sign == 1 else 0))
                        # New truth value:
                        new_lit_true = (new_color == (1 if sign == 1 else 0))
                        
                        if is_unsat:
                            # Clause is currently unsatisfied.
                            # If new_lit_true, it becomes satisfied (delta -1)
                            # If curr_lit_true was False (which it is, since unsat), and new is True, delta -1.
                            if new_lit_true:
                                delta -= 1
                        else:
                            # Clause is currently satisfied.
                            # It becomes unsatisfied only if v was the ONLY true literal.
                            # We don't track count of true literals.
                            # Approximation: Assume it becomes unsatisfied if v was true and we make it false?
                            # This is risky.
                            # Let's just use a simpler metric: 
                            # If we flip a var in an unsatisfied clause, we *might* satisfy it.
                            # If we flip a var in a satisfied clause, we *might* break it.
                            
                            # For speed, let's just use a random walk if delta calc is too complex/slow.
                            pass
                    
                    # Apply random flip
                    old_color = colors[v]
                    colors[v] = new_color
                    # Update unsatisfied set
                    # This is O(500) per flip.
                    for c2_idx, sign in var_to_clauses[v]:
                        # Check satisfaction of c2
                        # This is O(clause_size) ~ 10.
                        # Total 5000 ops per flip.
                        sat = False
                        for lit in clauses[c2_idx]:
                            vv = abs(lit)
                            ss = 1 if lit > 0 else -1
                            if colors[vv] == (1 if ss == 1 else 0):
                                sat = True
                                break
                        if sat:
                            if c2_idx in unsatisfied:
                                unsatisfied.remove(c2_idx)
                        else:
                            if c2_idx not in unsatisfied:
                                unsatisfied.add(c2_idx)
                    
                    continue

                # Greedy step: Find var in clause that minimizes delta
                # To make this fast, we'll just pick the var that is in the FEWEST other unsatisfied clauses?
                # Or just random?
                # Let's try picking a random var from the clause, but biased towards those that appear in many unsatisfied clauses?
                # No, let's just do a simple local search:
                # Pick a var in the clause.
                # If flipping it reduces unsatisfied count, do it.
                # Else, flip it with prob T?
                
                v = random.choice([abs(lit) for lit in clause])
                new_color = 1 - colors[v]
                
                # Calculate exact delta
                delta = 0
                for c2_idx, sign in var_to_clauses[v]:
                    is_unsat = c2_idx in unsatisfied
                    curr_lit_true = (colors[v] == (1 if sign == 1 else 0))
                    new_lit_true = (new_color == (1 if sign == 1 else 0))
                    
                    if is_unsat:
                        if new_lit_true:
                            delta -= 1
                    else:
                        # Check if v was the only true literal
                        # We need to check if any OTHER literal in c2 is true.
                        # If no other literal is true, and curr_lit_true, then breaking it adds 1 to delta.
                        if curr_lit_true:
                            other_true = False
                            for lit in clauses[c2_idx]:
                                if abs(lit) == v:
                                    continue
                                vv = abs(lit)
                                ss = 1 if lit > 0 else -1
                                if colors[vv] == (1 if ss == 1 else 0):
                                    other_true = True
                                    break
                            if not other_true:
                                delta += 1
                
                # Acceptance
                if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < math.exp(-delta / T)):
                    old_color = colors[v]
                    colors[v] = new_color
                    # Update unsatisfied set
                    for c2_idx, sign in var_to_clauses[v]:
                        sat = False
                        for lit in clauses[c2_idx]:
                            vv = abs(lit)
                            ss = 1 if lit > 0 else -1
                            if colors[vv] == (1 if ss == 1 else 0):
                                sat = True
                                break
                        if sat:
                            if c2_idx in unsatisfied:
                                unsatisfied.remove(c2_idx)
                        else:
                            if c2_idx not in unsatisfied:
                                unsatisfied.add(c2_idx)
                
                T *= cooling
                if T < min_T:
                    T = min_T

        return None

    # Try to find the largest n
    # Start from a known small n and go up.
    # n=10 is easy. n=34 is hard.
    # We will try n=34 first? If we have time.
    # But if n=34 is impossible to find quickly, we waste time.
    # Let's try n=10, 15, 20, 25, 30, 34.
    
    best_n = 9
    best_colors = [0 if (a+b+c) % 3 == 0 else 1 for a,b,c in combinations(range(9), 3)]
    
    targets = [10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 34]
    
    for n in targets:
        if time_left() < 5.0:
            break
        
        # Allocate time based on difficulty
        # Small n: 2s. Large n: 20s.
        if n <= 15:
            budget = 2.0
        elif n <= 25:
            budget = 5.0
        else:
            budget = 10.0
            
        if time_left() < budget + 2.0:
            continue
            
        result = solve_n(n, budget)
        if result is not None:
            best_n = n
            best_colors = result
            # If we found a valid one, we can try to go higher immediately?
            # The loop continues to the next target.
            
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
