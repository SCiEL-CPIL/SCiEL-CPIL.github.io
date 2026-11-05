import time
from math import gcd
from itertools import combinations

def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are equal.
    
    Approach: Minimum hitting set on bad triples.
    1. Enumerate all bad triples (a,b,c) with gcd(a,b)=gcd(a,c)=gcd(b,c).
    2. Find minimum set of elements to remove so every bad triple is hit.
    3. Use branch-and-bound with good heuristics (most-constrained-first).
    """
    DEADLINE = time.time() + 80.0  # safety margin before 90s kill
    
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute gcd matrix
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_mat[i][j] = gcd(i, j)
            gcd_mat[j][i] = gcd_mat[i][j]
    
    # Precompute all bad triples
    bad_triples = []
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            for k in range(j + 1, N + 1):
                if gcd_mat[i][j] == gcd_mat[i][k] == gcd_mat[j][k]:
                    bad_triples.append((i, j, k))
    
    # For each element, list of bad triple indices it belongs to
    elem_to_triples = [[] for _ in range(N + 1)]
    for ti, (a, b, c) in enumerate(bad_triples):
        elem_to_triples[a].append(ti)
        elem_to_triples[b].append(ti)
        elem_to_triples[c].append(ti)
    
    # Count how many bad triples each element is in
    triple_count = [len(elem_to_triples[x]) for x in range(N + 1)]
    
    # Branch and bound to find minimum hitting set
    # State: which triples are still "active" (not yet hit)
    # We'll use a recursive approach with bitmasks for triples if feasible,
    # but with potentially many triples, we use a set-based approach.
    
    # Convert to a more efficient representation
    # For each triple, store as a frozenset of elements
    # We'll track active triples as a set of indices
    
    num_triples = len(bad_triples)
    
    # For branch and bound, we need to efficiently check if a triple is hit
    # and find the most constrained remaining triple.
    
    # Precompute for each element, the list of triples (as indices)
    # Already have elem_to_triples
    
    best_removed = set()  # best hitting set found so far
    best_size = N  # start with removing all (worst case)
    
    # We'll do a DFS: at each step, pick an active triple, branch on which of its 3 elements to remove.
    # Use pruning: if current removed count >= best_size, prune.
    # Heuristic: pick the active triple with the element that has the most remaining active triples.
    
    import sys
    sys.setrecursionlimit(10000)
    
    # For efficiency, represent active triples as a Python set of indices
    # But with potentially thousands of triples, this might be slow.
    # Let's check how many bad triples there are.
    
    # Actually, let's use a different approach: iterative greedy + local search,
    # but with a smarter greedy that considers the global structure.
    
    # Alternative: Since we're removing few elements, let's use a
    # "conflict-driven" approach:
    # 1. Start with S = all elements
    # 2. Find all bad triples in S
    # 3. For each element, count how many bad triples it's in
    # 4. Remove the element with the highest count
    # 5. Repeat until no bad triples
    # 6. Then try to re-add removed elements (local search)
    
    def find_bad_triples_in(S):
        """Find all bad triples within set S."""
        S_list = sorted(S)
        triples = []
        n = len(S_list)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = S_list[i], S_list[j], S_list[k]
                    if gcd_mat[a][b] == gcd_mat[a][c] == gcd_mat[b][c]:
                        triples.append((a, b, c))
        return triples
    
    def greedy_delete():
        """Greedy: repeatedly remove the element in the most bad triples."""
        S = set(range(1, N + 1))
        removed = []
        
        while True:
            if time.time() > DEADLINE:
                break
            triples = find_bad_triples_in(S)
            if not triples:
                break
            # Count how many bad triples each element is in
            count = {}
            for (a, b, c) in triples:
                count[a] = count.get(a, 0) + 1
                count[b] = count.get(b, 0) + 1
                count[c] = count.get(c, 0) + 1
            # Remove the element with the highest count
            max_count = max(count.values())
            candidates = [x for x, c in count.items() if c == max_count]
            # Tie-break: remove the one with highest triple_count globally
            candidates.sort(key=lambda x: -triple_count[x])
            to_remove = candidates[0]
            S.discard(to_remove)
            removed.append(to_remove)
        
        return S, removed
    
    def local_search(S, removed, max_rounds=50):
        """Try to re-add removed elements and swap to improve."""
        S = set(S)
        removed = list(removed)
        
        for round_num in range(max_rounds):
            if time.time() > DEADLINE:
                break
            improved = False
            
            # Try adding each removed element
            for x in removed[:]:
                if time.time() > DEADLINE:
                    break
                if x in S:
                    continue
                # Check if x can be added
                S_list = sorted(S)
                can_add = True
                n = len(S_list)
                for i in range(n):
                    for j in range(i + 1, n):
                        u, v = S_list[i], S_list[j]
                        if gcd_mat[x][u] == gcd_mat[x][v] == gcd_mat[u][v]:
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    S.add(x)
                    removed.remove(x)
                    improved = True
            
            # Try swaps: remove one from S, add one from removed
            if not improved:
                for x in list(S):
                    if time.time() > DEADLINE:
                        break
                    temp_S = S - {x}
                    for y in removed:
                        if time.time() > DEADLINE:
                            break
                        if y in S:
                            continue
                        # Check if y can be added to temp_S
                        S_list = sorted(temp_S)
                        can_add = True
                        n = len(S_list)
                        for i in range(n):
                            for j in range(i + 1, n):
                                u, v = S_list[i], S_list[j]
                                if gcd_mat[y][u] == gcd_mat[y][v] == gcd_mat[u][v]:
                                    can_add = False
                                    break
                            if not can_add:
                                break
                        if can_add:
                            S = temp_S
                            S.add(y)
                            removed.remove(y)
                            removed.append(x)
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return S, removed
    
    # Phase 1: Greedy deletion
    S, removed = greedy_delete()
    best_S = set(S)
    best_size = len(S)
    
    # Phase 2: Local search
    S, removed = local_search(S, removed, max_rounds=100)
    if len(S) > best_size:
        best_size = len(S)
        best_S = set(S)
    
    # Phase 3: Multiple randomized greedy starts
    import random
    random.seed(42)
    
    for trial in range(20):
        if time.time() > DEADLINE:
            break
        
        # Randomized greedy: when there's a tie, pick randomly
        S = set(range(1, N + 1))
        removed = []
        
        while True:
            if time.time() > DEADLINE:
                break
            triples = find_bad_triples_in(S)
            if not triples:
                break
            count = {}
            for (a, b, c) in triples:
                count[a] = count.get(a, 0) + 1
                count[b] = count.get(b, 0) + 1
                count[c] = count.get(c, 0) + 1
            max_count = max(count.values())
            candidates = [x for x, c in count.items() if c == max_count]
            # Random tie-break
            to_remove = random.choice(candidates)
            S.discard(to_remove)
            removed.append(to_remove)
        
        # Local search
        S, removed = local_search(S, removed, max_rounds=50)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    # Phase 4: More aggressive local search on best
    S, removed = local_search(best_S, [x for x in range(1, N + 1) if x not in best_S], max_rounds=200)
    if len(S) > best_size:
        best_size = len(S)
        best_S = set(S)
    
    if not best_S:
        best_S = {1}
    
    return sorted(best_S)
