import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s buffer
    
    # Target n. We aim for the maximum known valid n, which is 34.
    # If we can't find 34, we might try 33, etc.
    # But the prompt says "Maximize n". The seed is 9.
    # A valid 34 is a huge jump.
    
    n = 34
    
    # Generate all triples
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    
    # Map triple to index
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Generate all 4-sets (for Red K4)
    four_sets = list(combinations(range(n), 4))
    # For each 4-set, list the indices of the 4 triples
    red_clauses = []
    for s in four_sets:
        idxs = [triple_to_idx[(s[0], s[1], s[2])], 
                triple_to_idx[(s[0], s[1], s[3])],
                triple_to_idx[(s[0], s[2], s[3])],
                triple_to_idx[(s[1], s[2], s[3])]]
        red_clauses.append(idxs)
        
    # Generate all 5-sets (for Blue K5)
    five_sets = list(combinations(range(n), 5))
    blue_clauses = []
    for s in five_sets:
        idxs = [triple_to_idx[(a, b, c)] for a, b, c in combinations(s, 3)]
        blue_clauses.append(idxs)

    # Total clauses
    all_clauses = red_clauses + blue_clauses
    # Clause type: 0 for Red (needs at least one 1), 1 for Blue (needs at least one 0)
    clause_types = [0] * len(red_clauses) + [1] * len(blue_clauses)
    
    # Current coloring: 0 is Red, 1 is Blue
    # We want to satisfy:
    # Red clause (type 0): At least one var is 1.
    # Blue clause (type 1): At least one var is 0.
    
    def is_satisfied(clause_idx, colors):
        ctype = clause_types[clause_idx]
        idxs = all_clauses[clause_idx]
        if ctype == 0: # Red K4 forbidden -> not all red -> at least one blue (1)
            return any(colors[i] == 1 for i in idxs)
        else: # Blue K5 forbidden -> not all blue -> at least one red (0)
            return any(colors[i] == 0 for i in idxs)
            
    def count_violations(colors):
        v = 0
        for i in range(len(all_clauses)):
            if not is_satisfied(i, colors):
                v += 1
        return v

    # WalkSAT
    best_n = 9
    best_colors = [0 if (a+b+c)%3==0 else 1 for a,b,c in combinations(range(9), 3)]
    
    # Try to solve for n=34
    random.seed(42)
    colors = [random.randint(0, 1) for _ in range(num_triples)]
    
    # Initial check
    if count_violations(colors) == 0:
        return {'n': n, 'colors': colors}
        
    # WalkSAT loop
    num_clauses = len(all_clauses)
    # Precompute clause-to-variable mapping for faster access? 
    # all_clauses already has variable indices.
    
    p = 0.5 # Probability to pick random var in clause
    max_iterations = 1000000
    
    for it in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
            
        # Find an unsatisfied clause
        # Optimization: Instead of scanning all, we can maintain a list of unsatisfied clauses.
        # But for simplicity and given the size, scanning is O(C). C ~ 324k.
        # 100k iterations * 324k checks = 3e10 operations. Too slow in Python.
        
        # We need a faster violation check or a smarter clause selection.
        # Let's maintain a set of unsatisfied clause indices.
        
        pass # The simple loop above is too slow.
        
    # Let's use a different strategy: Local Search with efficient violation counting.
    # Or, just use a known construction if available.
    
    # Since I don't have the hardcoded 34-point coloring memorized, I will implement a 
    # faster local search using bitsets or numpy if possible, or a simpler heuristic.
    
    # Actually, for n=34, a simple hill climbing on the number of violations 
    # with random restarts might work if the violation check is fast.
    # But checking 324k clauses is expensive.
    
    # Alternative: Use the fact that the problem is small enough for a SAT solver 
    # if we had one. We don't.
    
    # Let's try a greedy construction?
    # Or, return a smaller n that is easy to validate?
    # The seed is 9. n=10, 11, ... are easy to find with local search.
    # But we want to maximize.
    
    # Let's try to find n=34 with a more efficient WalkSAT implementation.
    # We can maintain the count of "witnesses" for each clause.
    # For a Red clause (needs 1), count how many 1s are in it. If 0, it's violated.
    # For a Blue clause (needs 0), count how many 0s are in it. If 0, it's violated.
    
    # Initialize counts
    # red_clause_counts[i] = number of 1s in red_clauses[i]
    # blue_clause_counts[i] = number of 0s in blue_clauses[i]
    
    red_counts = [0] * len(red_clauses)
    blue_counts = [0] * len(blue_clauses)
    
    for i, idxs in enumerate(red_clauses):
        for v in idxs:
            if colors[v] == 1:
                red_counts[i] += 1
                
    for i, idxs in enumerate(blue_clauses):
        for v in idxs:
            if colors[v] == 0:
                blue_counts[i] += 1
                
    # Number of violations
    viol_red = sum(1 for c in red_counts if c == 0)
    viol_blue = sum(1 for c in blue_counts if c == 0)
    total_viol = viol_red + viol_blue
    
    if total_viol == 0:
        return {'n': n, 'colors': colors}
        
    # WalkSAT with counts
    # When flipping a variable v from 0 to 1:
    # - For each Red clause containing v: red_counts[i] += 1
    # - For each Blue clause containing v: blue_counts[i] -= 1 (since 0 becomes 1)
    # We need to know which clauses contain each variable.
    
    var_to_red_clauses = [[] for _ in range(num_triples)]
    var_to_blue_clauses = [[] for _ in range(num_triples)]
    
    for i, idxs in enumerate(red_clauses):
        for v in idxs:
            var_to_red_clauses[v].append(i)
            
    for i, idxs in enumerate(blue_clauses):
        for v in idxs:
            var_to_blue_clauses[v].append(i)

    # Now WalkSAT
    best_total_viol = total_viol
    best_colors = colors.copy()
    
    for it in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
            
        if total_viol == 0:
            return {'n': n, 'colors': colors}
            
        if total_viol < best_total_viol:
            best_total_viol = total_viol
            best_colors = colors.copy()
            
        # Pick a random violated clause
        # We need a list of violated clause indices to pick from.
        # Maintaining this list dynamically is complex. 
        # Given the time, let's just pick a random clause and check if it's violated.
        # If not, try again (up to a few times).
        
        # Better: Precompute a list of all clauses.
        # Since we have counts, we can pick a random clause index.
        # If it's violated, we proceed.
        
        clause_idx = random.randint(0, num_clauses - 1)
        
        # Check if violated
        if clause_idx < len(red_clauses):
            if red_counts[clause_idx] > 0:
                continue
            idxs = red_clauses[clause_idx]
            ctype = 0
        else:
            b_idx = clause_idx - len(red_clauses)
            if blue_counts[b_idx] > 0:
                continue
            idxs = blue_clauses[b_idx]
            ctype = 1
            
        # Pick a variable to flip
        if random.random() < p:
            v = random.choice(idxs)
        else:
            # Pick variable that minimizes increase in violations
            best_v = -1
            best_delta = float('inf')
            for v in idxs:
                # Calculate delta if we flip v
                # Flip v from colors[v] to 1-colors[v]
                # For Red clauses containing v:
                # If v becomes 1, red_counts increases.
                # If v becomes 0, red_counts decreases.
                # We only care about clauses that are currently violated (count 0) 
                # or might become violated.
                
                delta = 0
                old_val = colors[v]
                new_val = 1 - old_val
                
                # Update red counts
                if new_val == 1:
                    # v becomes 1
                    for rc in var_to_red_clauses[v]:
                        if red_counts[rc] == 0:
                            delta -= 1 # Was violated, now satisfied
                        # If red_counts[rc] > 0, no change in violation status
                else:
                    # v becomes 0
                    for rc in var_to_red_clauses[v]:
                        if red_counts[rc] == 1:
                            delta += 1 # Was satisfied, now violated
                        # If red_counts[rc] > 1, no change
                        
                # Update blue counts
                if new_val == 0:
                    # v becomes 0
                    for bc in var_to_blue_clauses[v]:
                        if blue_counts[bc] == 0:
                            delta -= 1 # Was violated, now satisfied
                else:
                    # v becomes 1
                    for bc in var_to_blue_clauses[v]:
                        if blue_counts[bc] == 1:
                            delta += 1 # Was satisfied, now violated
                            
                if delta < best_delta:
                    best_delta = delta
                    best_v = v
                    
            v = best_v
            
        # Flip v
        old_val = colors[v]
        colors[v] = 1 - old_val
        new_val = colors[v]
        
        # Update counts and violation status
        # Red clauses
        if new_val == 1:
            for rc in var_to_red_clauses[v]:
                if red_counts[rc] == 0:
                    total_viol -= 1
                red_counts[rc] += 1
        else:
            for rc in var_to_red_clauses[v]:
                if red_counts[rc] == 1:
                    total_viol += 1
                red_counts[rc] -= 1
                
        # Blue clauses
        if new_val == 0:
            for bc in var_to_blue_clauses[v]:
                if blue_counts[bc] == 0:
                    total_viol -= 1
                blue_counts[bc] += 1
        else:
            for bc in var_to_blue_clauses[v]:
                if blue_counts[bc] == 1:
                    total_viol += 1
                blue_counts[bc] -= 1

    # If we didn't find 0 violations for n=34, try n=33?
    # For now, return the best found. If best_total_viol > 0, it's invalid.
    # We must return a VALID coloring.
    # If we failed to find a valid one for n=34, we should fall back to a smaller n.
    
    # To ensure validity, we can try n=10, 11, etc. with the same logic, 
    # or just use the seed if 34 fails.
    # But 34 is likely solvable.
    
    if total_viol == 0:
        return {'n': n, 'colors': colors}
        
    # Fallback: Return seed if 34 failed?
    # Or try n=33?
    # Let's just return the seed for safety if 34 fails, 
    # but the code above is expected to solve 34.
    
    # To be safe, let's add a quick check for n=10 which is trivial.
    # But the prompt asks to maximize.
    
    # If 34 fails, we return the seed.
    return {'n': 9, 'colors': [0 if (a+b+c)%3==0 else 1 for a,b,c in combinations(range(9), 3)]}
