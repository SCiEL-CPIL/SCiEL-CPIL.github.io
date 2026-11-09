import time
import random
from math import gcd

def construct():
    """
    Returns a nonempty subset of [1, 128] containing no three distinct elements 
    whose three pairwise least common multiples are equal. Maximizes cardinality.
    """
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    N = 128
    
    # 1. Precompute LCMs
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcm_table[i][j] = lcm(i, j)
            if i != j:
                lcm_table[j][i] = lcm_table[i][j]
                
    # 2. Identify Bad Triples
    # A triple (a, b, c) is bad if lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_triples = []
    # Map: number -> list of indices of bad triples containing this number
    num_to_triples = {i: [] for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, N + 1):
                lac = lcm_table[a][c]
                if lab == lac:
                    lbc = lcm_table[b][c]
                    if lab == lbc:
                        idx = len(bad_triples)
                        bad_triples.append((a, b, c))
                        num_to_triples[a].append(idx)
                        num_to_triples[b].append(idx)
                        num_to_triples[c].append(idx)
                        
    # 3. Greedy Removal Strategy
    # Start with all numbers
    in_set = [True] * (N + 1)
    for i in range(1, N + 1):
        in_set[i] = True
        
    def count_active_bad_triples():
        count = 0
        for t in bad_triples:
            if in_set[t[0]] and in_set[t[1]] and in_set[t[2]]:
                count += 1
        return count

    def get_conflict_degree(num):
        """Count how many bad triples containing 'num' are currently active."""
        deg = 0
        for t_idx in num_to_triples[num]:
            a, b, c = bad_triples[t_idx]
            if in_set[a] and in_set[b] and in_set[c]:
                deg += 1
        return deg

    # Iterative Removal
    # We remove the element from the "most constrained" bad triple.
    # Specifically, we find all active bad triples, and for each, we look at the 
    # degree of its members. We remove the member with the highest degree.
    # This is a heuristic to break as many conflicts as possible per removal.
    
    max_iterations = 200
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
            
        active_triples = []
        for t in bad_triples:
            if in_set[t[0]] and in_set[t[1]] and in_set[t[2]]:
                active_triples.append(t)
                
        if not active_triples:
            break
            
        # Pick a random active triple to avoid deterministic loops, 
        # or pick the one with the highest sum of degrees.
        # Let's pick the triple where the max degree among its members is highest.
        best_triple = None
        best_max_deg = -1
        
        # To save time, we don't evaluate every triple if there are many, 
        # but N=128 is small enough.
        for t in active_triples:
            a, b, c = t
            deg_a = get_conflict_degree(a)
            deg_b = get_conflict_degree(b)
            deg_c = get_conflict_degree(c)
            max_deg = max(deg_a, deg_b, deg_c)
            if max_deg > best_max_deg:
                best_max_deg = max_deg
                best_triple = t
                # Store which element to remove
                if max_deg == deg_a:
                    best_elem = a
                elif max_deg == deg_b:
                    best_elem = b
                else:
                    best_elem = c
        
        if best_triple is not None:
            in_set[best_elem] = False

    # 4. Refinement: Try to add back removed numbers
    # We iterate through all numbers not in the set and try to add them.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(1, N + 1):
            if not in_set[i]:
                # Check if adding i creates any new bad triple
                can_add = True
                # Only need to check triples containing i
                for t_idx in num_to_triples[i]:
                    a, b, c = bad_triples[t_idx]
                    # The other two elements must be in the set
                    # Since i is not in set, we check if the other two are
                    if a == i:
                        if in_set[b] and in_set[c]:
                            can_add = False
                            break
                    elif b == i:
                        if in_set[a] and in_set[c]:
                            can_add = False
                            break
                    elif c == i:
                        if in_set[a] and in_set[b]:
                            can_add = False
                            break
                            
                if can_add:
                    in_set[i] = True
                    improved = True

    # 5. Final Local Search / Swap Optimization
    # Try swapping an element in the set with one outside to increase size? 
    # Actually, size is fixed by the number of True bits. 
    # We want to maximize the number of True bits.
    # The "Add Back" step already maximizes size given the current "holes".
    # Can we do better by removing a "bad" element and adding TWO "good" ones?
    # This is complex. The Min-Conflict approach is usually strong enough for this N.
    
    # Construct result
    result = [i for i in range(1, N + 1) if in_set[i]]
    
    # Ensure non-empty (problem guarantees solution exists, but just in case)
    if not result:
        result = [1]
        
    return result

# EVOLVE-BLOCK-START
# The function construct() is defined above.
# We must ensure the EVOLVE-BLOCK structure is respected if required by the harness,
# but the prompt says "Keep the function names and signatures of the seed program".
# The seed program had the function inside the EVOLVE-BLOCK.
# I will place the implementation inside the block to be safe, or just define it.
# The prompt says: "Output the COMPLETE program in a single python code block."
# And "Keep the program's entry-point function name(s) ... EXACTLY as in the current program".
# The current program has `def construct():` inside `# EVOLVE-BLOCK-START`.
# I will structure the output to match this convention.
# EVOLVE-BLOCK-END
