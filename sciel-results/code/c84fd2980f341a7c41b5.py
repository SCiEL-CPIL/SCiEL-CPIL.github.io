"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations
from collections import Counter, defaultdict

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    n = 128
    
    # Precompute lcm for all pairs
    lcm_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples: (a,b,c) with a<b<c where lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = set()
    
    for a in range(1, n+1):
        if time.time() > deadline - 5:
            break
        for b in range(a+1, n+1):
            Lab = lcm_table[a][b]
            for c in range(b+1, n+1):
                if time.time() > deadline - 5:
                    break
                if lcm_table[a][c] == Lab and lcm_table[b][c] == Lab:
                    bad_triples.add((a,b,c))
    
    if time.time() > deadline - 3:
        return [1, 2, 4, 8, 16, 32, 64, 128]
    
    if not bad_triples:
        return list(range(1, n+1))
    
    # Build adjacency for fast validity checks
    # For each element x, store the list of pairs (u, v) such that (x, u, v) is a bad triple
    # Note: In the triple (a,b,c), the "pairs" for a are (b,c), for b are (a,c), for c are (a,b)
    forbidden_pairs = defaultdict(list)
    for t in bad_triples:
        a, b, c = t
        forbidden_pairs[a].append((b, c))
        forbidden_pairs[b].append((a, c))
        forbidden_pairs[c].append((a, b))
    
    # Initial Greedy Solution
    elem_count = Counter()
    for t in bad_triples:
        for x in t:
            elem_count[x] += 1
    
    removed = set()
    uncovered = set(bad_triples)
    elem_triples = defaultdict(set)
    for t in bad_triples:
        for x in t:
            elem_triples[x].add(t)
    
    iteration = 0
    while uncovered and iteration < 200:
        if time.time() > deadline - 2:
            break
        iteration += 1
        best_x = None
        best_size = 0
        for x in elem_triples:
            if x in removed:
                continue
            size = len(elem_triples[x] & uncovered)
            if size > best_size:
                best_size = size
                best_x = x
        if best_x is None or best_size == 0:
            break
        removed.add(best_x)
        uncovered -= elem_triples[best_x]
    
    result_set = set(range(1, n+1) - removed)
    
    # Simulated Annealing
    # State: result_set
    # Objective: Maximize len(result_set)
    
    def is_valid_add(x):
        """Check if x can be added to result_set without creating a bad triple."""
        # x is not in result_set
        # Check all forbidden pairs for x
        for u, v in forbidden_pairs[x]:
            if u in result_set and v in result_set:
                return False
        return True

    def try_add(x):
        """Attempt to add x. Returns True if added."""
        if x in result_set:
            return False
        if is_valid_add(x):
            result_set.add(x)
            return True
        return False

    # Phase 1: Greedy Add-Back
    # Repeatedly try to add any removed element until no more can be added
    progress = True
    while progress:
        progress = False
        for x in list(removed):
            if try_add(x):
                removed.discard(x)
                progress = True
                if time.time() > deadline - 10:
                    break
        if time.time() > deadline - 10:
            break

    best_size = len(result_set)
    
    # Phase 2: Simulated Annealing
    # We will try moves:
    # 1. Add a random removed element (if valid) -> Gain 1
    # 2. Remove a random element from result_set -> Loss 1 (to enable future adds)
    # 3. Swap: Remove u from result_set, Add v from removed (if valid after removal) -> Net 0 (but might enable more)
    
    # To make SA effective, we need to explore.
    # Let's run multiple restarts or a long SA chain.
    
    current_set = set(result_set)
    current_removed = set(removed)
    
    def check_validity_in_set(s, x):
        """Check if x can be added to set s."""
        for u, v in forbidden_pairs[x]:
            if u in s and v in s:
                return False
        return True

    def add_to_set(s, x):
        s.add(x)
    
    def remove_from_set(s, x):
        s.discard(x)

    # We will maintain a list of removed elements for quick random selection
    removed_list = list(current_removed)
    
    T = 1.0
    cooling = 0.9999
    min_T = 0.01
    
    best_set = set(current_set)
    best_size = len(best_set)
    
    steps = 0
    max_steps = 500000 # Adjust based on time
    
    while T > min_T and time.time() < deadline - 2:
        steps += 1
        if steps % 1000 == 0 and time.time() > deadline - 2:
            break
            
        # Choose move type
        r = random.random()
        if r < 0.4:
            # Try to add a random removed element
            if not removed_list:
                continue
            x = random.choice(removed_list)
            if check_validity_in_set(current_set, x):
                # Accept always (gain 1)
                current_set.add(x)
                current_removed.discard(x)
                removed_list.remove(x)
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
        elif r < 0.7:
            # Remove a random element from current_set
            if not current_set:
                continue
            x = random.choice(list(current_set))
            current_set.discard(x)
            current_removed.add(x)
            removed_list.append(x)
            # Accept with probability exp(-1/T) since size decreases by 1
            if random.random() < (2.71828 ** (-1.0 / T)):
                pass # Keep state
            else:
                # Revert
                current_set.add(x)
                current_removed.discard(x)
                removed_list.remove(x)
        else:
            # Swap: Remove u from current, Add v from removed
            if not current_set or not removed_list:
                continue
            u = random.choice(list(current_set))
            v = random.choice(removed_list)
            
            # Check if v can be added if u is removed
            # Temporarily remove u
            current_set.discard(u)
            current_removed.add(u)
            removed_list.append(u)
            
            if check_validity_in_set(current_set, v):
                # Swap is valid (size stays same)
                current_set.add(v)
                current_removed.discard(v)
                removed_list.remove(v)
                
                # Accept with probability 1 (neutral move) or explore
                # To avoid getting stuck, we might accept neutral moves with high prob
                # But we need to ensure we don't oscillate.
                # Let's accept neutral moves always to explore
            else:
                # Revert swap
                current_set.add(u)
                current_removed.discard(u)
                removed_list.remove(u)
                
                # Reject swap
                # (No change to set)
            
            # Note: In the "Swap" branch, if valid, we changed the set.
            # If invalid, we reverted.
            # We should also update best_set if size increased? No, size is same.
            # But maybe this swap enables future adds? Yes, that's the point.
            
        T *= cooling
        
    # Final Greedy Add-Back on the best found set
    # We should apply the final polish to best_set, not necessarily current_set if it got worse
    # But usually the end of SA is good. Let's polish current_set and compare.
    
    # Actually, let's just use the best_set found during SA and polish it.
    final_set = set(best_set)
    final_removed = set(range(1, n+1) - final_set)
    
    # Greedy add-back on final_set
    progress = True
    while progress:
        progress = False
        for x in list(final_removed):
            if check_validity_in_set(final_set, x):
                final_set.add(x)
                final_removed.discard(x)
                progress = True
                if time.time() > deadline - 1:
                    break
        if time.time() > deadline - 1:
            break
            
    result = sorted(final_set)
    
    if not result:
        result = [1]
        
    return result
# EVOLVE-BLOCK-END
