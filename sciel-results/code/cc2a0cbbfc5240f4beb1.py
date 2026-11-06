"""
Planted 3-SAT solver using a robust WalkSAT with bitmask-accelerated state tracking,
structural initialization, and a final greedy hill-climbing polish.
"""
import time
import random
import sys

DEADLINE_FRACTION = 0.95

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    t_start = time.time()
    deadline = t_start + 330 * DEADLINE_FRACTION

    # --- Preprocessing ---
    # Convert to 0-based variable indices internally for faster access
    # Input: 1-based signed literals (1..n, -1..-n)
    # Internal: 0-based variables (0..n-1), literals stored as (var, sign) where sign is True for +, False for -
    
    clause_vars = []
    pos_count = [0] * n
    neg_count = [0] * n
    
    for c in clauses:
        c_vars = []
        for l in c:
            if l > 0:
                v = l - 1
                sign = True
                pos_count[v] += 1
            else:
                v = -l - 1
                sign = False
                neg_count[v] += 1
            c_vars.append((v, sign))
        clause_vars.append(c_vars)
        
    # Build variable-to-clause incidence lists
    # var_clauses[v] = list of (clause_idx, sign)
    var_clauses = [[] for _ in range(n)]
    for ci, c_vars in enumerate(clause_vars):
        for v, sign in c_vars:
            var_clauses[v].append((ci, sign))

    # --- State Management ---
    # assignment: list of booleans (True/False)
    # support: list of ints, support[ci] is the number of satisfied literals in clause ci
    
    def make_support(assignment):
        support = [0] * m
        for ci, c_vars in enumerate(clause_vars):
            s = 0
            for v, sign in c_vars:
                if assignment[v] == sign:
                    s += 1
            support[ci] = s
        return support

    def apply_flip(v, assignment, support, cur_unsat):
        """
        Flip variable v. Returns new cur_unsat.
        """
        old_val = assignment[v]
        new_val = not old_val
        assignment[v] = new_val
        
        delta_unsat = 0
        for ci, sign in var_clauses[v]:
            old_support = support[ci]
            if new_val == sign:
                new_support = old_support + 1
            else:
                new_support = old_support - 1
            
            support[ci] = new_support
            
            if old_support == 0 and new_support == 1:
                delta_unsat -= 1
            elif old_support == 1 and new_support == 0:
                delta_unsat += 1
        
        return cur_unsat + delta_unsat

    # --- Initialization Strategies ---
    rng = random.Random(42)
    
    def init_structural():
        """Initialize based on literal frequency bias."""
        assignment = []
        for v in range(n):
            if pos_count[v] > neg_count[v]:
                # Bias towards True if +v appears more
                p = 0.7
            elif neg_count[v] > pos_count[v]:
                # Bias towards False if -v appears more
                p = 0.3
            else:
                p = 0.5
            assignment.append(rng.random() < p)
        return assignment

    def init_random():
        return [rng.random() < 0.5 for _ in range(n)]

    # --- Search Strategies ---
    
    def walk_sat(assignment, support, cur_unsat, max_steps, noise_prob=0.25):
        """
        Standard WalkSAT.
        If noise, pick random var from random unsat clause.
        If greedy, pick var that maximizes gain (satisfies more unsat, breaks fewer sat).
        For speed, we use a hybrid: 
        1. Pick a random unsatisfied clause.
        2. With probability `noise_prob`, pick a random variable in it.
        3. Otherwise, pick the variable in it that has the best "gain" (local check within the clause is not enough, 
           but checking global gain is expensive. 
           A common fast approximation: pick the variable that appears in the most unsatisfied clauses? 
           Or just random. 
           Let's stick to random for speed, but maybe add a simple heuristic: 
           pick the variable that currently satisfies the fewest clauses? 
           Actually, standard WalkSAT random is often sufficient. 
           Let's add a "greedy" option that calculates gain for the 3 vars in the clause only.
           Gain = (unsat clauses satisfied) - (sat clauses broken).
           This is O(occurrences of the 3 vars).
        """
        for _ in range(max_steps):
            if time.time() > deadline:
                break
            if cur_unsat == 0:
                return assignment, cur_unsat, True
            
            # Pick random unsatisfied clause
            # Optimization: Maintain a list of unsat clauses? 
            # For now, random sampling. If m is small, it's fine.
            # To be faster, we could keep a list, but updating it is tricky.
            # Let's try random sampling first.
            while True:
                ci = rng.randint(0, m - 1)
                if support[ci] == 0:
                    break
            
            c_vars = clause_vars[ci]
            
            if rng.random() < noise_prob:
                # Random flip
                v, sign = c_vars[rng.randint(0, 2)]
            else:
                # Greedy flip among the 3 literals in the clause
                best_v = -1
                best_gain = -100
                
                for v, sign in c_vars:
                    gain = 0
                    # Check impact on all clauses containing v
                    for c_idx, s in var_clauses[v]:
                        if assignment[v] == s:
                            # Currently satisfied by v
                            # Flipping will make it unsatisfied if support becomes 0
                            if support[c_idx] == 1:
                                gain -= 1
                        else:
                            # Currently unsatisfied by v
                            # Flipping will satisfy it if support becomes 1
                            if support[c_idx] == 0:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                
                v = best_v

            cur_unsat = apply_flip(v, assignment, support, cur_unsat)

        return assignment, cur_unsat, (cur_unsat == 0)

    def greedy_hill_climb(assignment, support, cur_unsat, max_time=2.0):
        """
        Greedy hill climbing to polish the solution.
        """
        t_start = time.time()
        improved = True
        while improved and (time.time() - t_start) < max_time and time.time() < deadline:
            improved = False
            if cur_unsat == 0:
                return assignment, cur_unsat, True
            
            best_gain = 0
            best_v = -1
            
            for v in range(n):
                gain = 0
                for ci, sign in var_clauses[v]:
                    if assignment[v] == sign:
                        if support[ci] == 1:
                            gain -= 1
                    else:
                        if support[ci] == 0:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            
            if best_v != -1:
                cur_unsat = apply_flip(best_v, assignment, support, cur_unsat)
                improved = True
        
        return assignment, cur_unsat, (cur_unsat == 0)

    # --- Main Loop ---
    
    best_assignment = None
    best_unsat = m + 1
    
    # Phase 1: Structural Init + WalkSAT
    # Phase 2: Random Init + WalkSAT
    
    # We will run multiple restarts.
    # Each restart:
    # 1. Init (alternating structural/random)
    # 2. WalkSAT with varying noise
    # 3. If not solved, try Greedy Hill Climb on the result
    
    max_restarts = 200
    steps_per_walk = 2000
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Select initialization
        if r % 2 == 0:
            assignment = init_structural()
        else:
            assignment = init_random()
        
        support = make_support(assignment)
        cur_unsat = sum(1 for s in support if s == 0)
        
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
        
        if cur_unsat == 0:
            return assignment
        
        # Vary noise: start low, increase slightly? Or randomize?
        # Let's use a fixed noise of 0.25 for WalkSAT, which is standard.
        # Maybe vary it: noise = 0.1 + 0.2 * (r % 5) / 4.0 ?
        noise = 0.25
        
        assignment, cur_unsat, solved = walk_sat(assignment, support, cur_unsat, steps_per_walk, noise)
        
        if solved:
            return assignment
        
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
        
        # Try to polish with greedy hill climbing
        if time.time() < deadline - 5:
            assignment, cur_unsat, solved = greedy_hill_climb(assignment, support, cur_unsat)
            if solved:
                return assignment
            
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]
        
        # Perturb for next restart if we are not making progress?
        # The restart loop handles this by re-initializing.
        
        # If we have spent a lot of time, reduce steps?
        # No, keep it simple.
        
    # Final Polish on the best assignment found
    if best_assignment is not None and time.time() < deadline - 2:
        assignment = best_assignment[:]
        support = make_support(assignment)
        cur_unsat = sum(1 for s in support if s == 0)
        
        if cur_unsat > 0:
            # One last long greedy climb
            assignment, cur_unsat, solved = greedy_hill_climb(assignment, support, cur_unsat, max_time=deadline - time.time() - 1)
            if solved:
                return assignment
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]
        
        if best_unsat == 0:
            return best_assignment

    if best_assignment is not None:
        return best_assignment
    
    # Fallback
    return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
