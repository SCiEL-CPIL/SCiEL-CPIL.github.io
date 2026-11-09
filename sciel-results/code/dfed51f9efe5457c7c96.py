import time
from math import gcd, lcm
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0
    
    max_n = 512
    
    def get_divisors(n):
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return set(divs)
    
    # Precompute compatibility
    # compatible[i][j] = True if lcm(i, j) <= max_n
    compatible = np.zeros((max_n+1, max_n+1), dtype=bool)
    for i in range(1, max_n+1):
        for j in range(i, max_n+1):
            if lcm(i, j) <= max_n:
                compatible[i][j] = True
                compatible[j][i] = True
    
    def check_set(S):
        """Check if all pairs in S have lcm <= max_n."""
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if not compatible[S[i]][S[j]]:
                    return False
        return True
    
    def greedy_from_seed(seed_set):
        """Greedy expansion from a seed set."""
        S = set(seed_set)
        # Try to add elements
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Try to add any compatible element
            for x in range(1, max_n+1):
                if x in S:
                    continue
                if all(compatible[x][y] for y in S):
                    S.add(x)
                    changed = True
        return S
    
    # Start with divisors of 504 (which has 25 divisors)
    S_504 = get_divisors(504)
    best_S = set(S_504)
    best_size = len(S_504)
    
    # Greedy expansion from 504 divisors
    S_expanded = greedy_from_seed(S_504)
    if len(S_expanded) > best_size:
        best_S = S_expanded
        best_size = len(best_S)
    
    # Try greedy from other seeds
    for m in [360, 420, 480, 504, 336, 240, 288, 432]:
        if time.time() > deadline:
            break
        S_seed = get_divisors(m)
        S_exp = greedy_from_seed(S_seed)
        if len(S_exp) > best_size:
            best_S = S_exp
            best_size = len(best_S)
    
    # Local search: try 1-swap improvements
    # If removing x and adding y (where x not in S, y in S) gives same size,
    # then try adding another element.
    # More directly: try to find a set of size best_size+1 via local search.
    
    # Branch and bound on the compatibility graph
    # Build adjacency for the graph where edges connect compatible pairs
    # We want the maximum clique.
    
    # Use a simple B&B with bitsets for speed
    # Vertex ordering: by degree (descending)
    
    # First, restrict to a candidate set: numbers that are compatible with at least one element in best_S
    # Actually, for B&B, we can work on the full graph but it's 512 vertices.
    # Let's use a reduced candidate pool: numbers that are compatible with most of best_S
    
    # For B&B, use a heuristic: Bron-Kerbosch with pivot
    
    # Convert best_S to a set for fast lookup
    best_set = set(best_S)
    
    # Try to improve via local search: 
    # 1. Try adding any single element
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, max_n+1):
            if x in best_set:
                continue
            if all(compatible[x][y] for y in best_set):
                best_set.add(x)
                improved = True
                best_size = len(best_set)
    
    # 2. Try 1-for-1 swaps to enable further additions
    # For each y in best_set, for each x not in best_set:
    #   If (best_set - {y} + {x}) is a valid set, and allows adding more elements, do it.
    
    # Let's try a more systematic local search
    # Start from current best
    local_S = set(best_set)
    local_size = len(local_S)
    
    # Try to find improvements via swaps
    swap_improved = True
    while swap_improved and time.time() < deadline:
        swap_improved = False
        # For each element in local_S, try swapping with each element not in local_S
        for y in list(local_S):
            if time.time() > deadline:
                break
            for x in range(1, max_n+1):
                if x in local_S:
                    continue
                # Check if (local_S - {y} + {x}) is valid
                test_set = set(local_S)
                test_set.discard(y)
                test_set.add(x)
                valid = True
                for a in test_set:
                    if not all(compatible[a][b] for b in test_set if b != a):
                        valid = False
                        break
                if not valid:
                    continue
                # Check if we can add more elements
                expanded = set(test_set)
                for z in range(1, max_n+1):
                    if z in expanded:
                        continue
                    if all(compatible[z][w] for w in expanded):
                        expanded.add(z)
                if len(expanded) > local_size:
                    local_S = expanded
                    local_size = len(local_S)
                    swap_improved = True
                    if local_size > best_size:
                        best_set = set(local_S)
                        best_size = local_size
                    break  # restart swap search with new local_S
    
    if len(best_set) > best_size:
        best_size = len(best_set)
    
    # Try a few random restarts with random subsets
    import random
    random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline:
            break
        # Create a random seed
        # Pick a random number, take its divisors as seed
        r = random.randint(1, max_n)
        S_seed = get_divisors(r)
        if len(S_seed) < 10:
            continue
        S_exp = greedy_from_seed(S_seed)
        if len(S_exp) > best_size:
            best_set = S_exp
            best_size = len(best_set)
            # Local search on this
            local_S = set(best_set)
            local_size = len(local_S)
            swap_improved = True
            while swap_improved and time.time() < deadline:
                swap_improved = False
                for y in list(local_S):
                    if time.time() > deadline:
                        break
                    for x in range(1, max_n+1):
                        if x in local_S:
                            continue
                        test_set = set(local_S)
                        test_set.discard(y)
                        test_set.add(x)
                        valid = True
                        for a in test_set:
                            if not all(compatible[a][b] for b in test_set if b != a):
                                valid = False
                                break
                        if not valid:
                            continue
                        expanded = set(test_set)
                        for z in range(1, max_n+1):
                            if z in expanded:
                                continue
                            if all(compatible[z][w] for w in expanded):
                                expanded.add(z)
                        if len(expanded) > local_size:
                            local_S = expanded
                            local_size = len(local_S)
                            if local_size > best_size:
                                best_set = set(local_S)
                                best_size = local_size
                            swap_improved = True
                            break
                    if swap_improved:
                        break
    
    return list(best_set)
