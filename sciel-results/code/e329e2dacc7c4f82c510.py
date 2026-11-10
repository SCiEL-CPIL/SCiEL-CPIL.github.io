"""Combined program: hypergraph MaxSAT-style local search for large integer sets avoiding equal-gcd triples."""

# EVOLVE-BLOCK-START
import math
import random
import time
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 90.0

def _bad_triple(a, b, c):
    """Return True if the triple (a,b,c) has all three pairwise gcds equal."""
    g_ab = math.gcd(a, b)
    g_ac = math.gcd(a, c)
    g_bc = math.gcd(b, c)
    return g_ab == g_ac == g_bc

def construct():
    """Construct a large subset of [1,128] with no three distinct elements
    whose three pairwise gcds are all equal."""
    start = time.time()
    deadline = _DEADLINE - 2.0
    
    N = 128
    all_numbers = list(range(1, N + 1))
    
    # Precompute GCD table
    g = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g[i][j] = math.gcd(i, j)
            g[j][i] = g[i][j]
    
    # Precompute all bad triples
    bad_triples = []
    for a, b, c in combinations(all_numbers, 3):
        if _bad_triple(a, b, c):
            bad_triples.append((a, b, c))
    
    def count_violations(S):
        """Count violations and return conflict counts per element."""
        S_list = list(S)
        conflict_count = {x: 0 for x in S}
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                gab = g[a][b]
                for k in range(j + 1, n):
                    c = S_list[k]
                    if gab == g[a][c] == g[b][c]:
                        conflict_count[a] += 1
                        conflict_count[b] += 1
                        conflict_count[c] += 1
        return conflict_count
    
    def local_search_remove(initial_set):
        """Remove elements to eliminate all violations."""
        S = set(initial_set)
        while time.time() < deadline:
            conflict_count = count_violations(S)
            max_conf = max(conflict_count.values()) if conflict_count else 0
            if max_conf == 0:
                break
            # Remove element with max conflict, random tie-break
            candidates = [x for x in S if conflict_count[x] == max_conf]
            v = random.choice(candidates)
            S.discard(v)
        return S
    
    def swap_improve(S):
        """Try to improve S by swapping elements."""
        S = set(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list = list(S)
            # Try removing each element and adding others
            for rem in S_list:
                S_trial = S - {rem}
                # Try to add elements not in S_trial
                candidates = [x for x in all_numbers if x not in S_trial]
                # Sort by how "easy" they are to add (fewer conflicts)
                def ease(x):
                    S_trial_list = list(S_trial)
                    count = 0
                    for i in range(len(S_trial_list)):
                        a = S_trial_list[i]
                        gax = g[a][x]
                        for j in range(i + 1, len(S_trial_list)):
                            b = S_trial_list[j]
                            if g[a][b] == gax == g[b][x]:
                                count += 1
                    return count
                
                candidates.sort(key=ease)
                added = 0
                for x in candidates:
                    if time.time() > deadline:
                        break
                    # Check if x can be added
                    S_trial_list = list(S_trial)
                    valid = True
                    for i in range(len(S_trial_list)):
                        a = S_trial_list[i]
                        gax = g[a][x]
                        for j in range(i + 1, len(S_trial_list)):
                            b = S_trial_list[j]
                            if g[a][b] == gax == g[b][x]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        S_trial.add(x)
                        added += 1
                        if len(S_trial) > len(S):
                            S = S_trial
                            improved = True
                            break
                if improved:
                    break
        return S
    
    def greedy_add(order):
        """Greedy add in given order."""
        S = set()
        S_list = []
        for x in order:
            if time.time() > deadline:
                break
            valid = True
            for i in range(len(S_list)):
                a = S_list[i]
                gax = g[a][x]
                for j in range(i + 1, len(S_list)):
                    b = S_list[j]
                    if g[a][b] == gax == g[b][x]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S.add(x)
                S_list.append(x)
        return S
    
    best_set = set()
    
    # Strategy 1: Local search from full set
    random.seed(42)
    S1 = local_search_remove(all_numbers)
    if len(S1) > len(best_set):
        best_set = S1
    
    # Strategy 2: Local search from structured subsets
    if time.time() < deadline:
        # Even numbers
        S2 = local_search_remove([x for x in all_numbers if x % 2 == 0])
        if len(S2) > len(best_set):
            best_set = S2
    
    if time.time() < deadline:
        # Numbers divisible by 2 or 3
        S3 = local_search_remove([x for x in all_numbers if x % 2 == 0 or x % 3 == 0])
        if len(S3) > len(best_set):
            best_set = S3
    
    # Strategy 3: Greedy add with various orders
    for seed in range(20):
        if time.time() > deadline:
            break
        random.seed(seed + 100)
        order = all_numbers[:]
        random.shuffle(order)
        Sg = greedy_add(order)
        if len(Sg) > len(best_set):
            best_set = Sg
    
    # Strategy 4: Swap improvement on best set
    if time.time() < deadline:
        S_swapped = swap_improve(best_set)
        if len(S_swapped) > len(best_set):
            best_set = S_swapped
    
    # Strategy 5: More local search iterations from different starts
    for seed in range(10):
        if time.time() > deadline:
            break
        random.seed(seed + 200)
        # Start with random subset
        random_subset = random.sample(all_numbers, 80)
        Sr = local_search_remove(random_subset)
        if len(Sr) > len(best_set):
            best_set = Sr
    
    # Final swap improvement
    if time.time() < deadline:
        S_final = swap_improve(best_set)
        if len(S_final) > len(best_set):
            best_set = S_final
    
    if not best_set:
        return [1]
    
    # Verify
    S_list = list(best_set)
    for a, b, c in combinations(S_list, 3):
        if _bad_triple(a, b, c):
            # Should not happen, but return a safe fallback
            return [1]
    
    return S_list

# EVOLVE-BLOCK-END
