import time
import random
import itertools
from collections import deque

# EVOLVE-BLOCK-START

def _build_graph(n, p, seed):
    """Build adjacency sets from the same PRNG stream as the seed program."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch_max_cliques(adj, min_size=2):
    """Enumerate all maximal cliques of size >= min_size using Bron-Kerbosch with pivot."""
    n = len(adj)
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= min_size:
                cliques.append(frozenset(R))
            return
        # Choose pivot u from P union X to reduce recursion
        U = P | X
        if not U:
            return
        # Pick pivot with max degree in P (heuristic)
        pivot = max(U, key=lambda u: len(P & adj[u]))
        candidates = P - adj[pivot]
        for v in list(candidates):
            R.add(v)
            P_new = P & adj[v]
            X_new = X & adj[v]
            bk(R, P_new, X_new)
            R.remove(v)
            P.discard(v)
            X.add(v)
    
    bk(set(), set(range(n)), set())
    return cliques

def _solve_instance(adj, time_limit=40.0):
    """Solve the minimum transversal problem for maximal cliques using Branch and Bound with improvements."""
    n = len(adj)
    cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
    
    if not cliques:
        return []
    
    num_cliques = len(cliques)
    
    # Precompute vertex -> clique indices
    vertex_cliques = [[] for _ in range(n)]
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v].append(i)
    
    # Bitmask representation
    # clique_mask[v] has bit i set if v is in clique i
    clique_mask = [0] * n
    for v in range(n):
        mask = 0
        for i in vertex_cliques[v]:
            mask |= (1 << i)
        clique_mask[v] = mask
        
    # Sort vertices by coverage (number of cliques) descending to help B&B ordering
    vertex_freq = [len(vertex_cliques[v]) for v in range(n)]
    
    # Precompute for each clique: list of vertices in it, sorted by frequency descending
    clique_vertices = []
    for i, clique in enumerate(cliques):
        verts = sorted(clique, key=lambda v: vertex_freq[v], reverse=True)
        clique_vertices.append(verts)
        
    best_solution = None
    best_size = num_cliques + 1 # Upper bound
    
    # Initial greedy solution to set a good upper bound - try multiple strategies
    def greedy_solve():
        S = set()
        uncovered = (1 << num_cliques) - 1
        while uncovered:
            best_v = -1
            best_count = -1
            for v in range(n):
                if v in S:
                    continue
                count = bin(clique_mask[v] & uncovered).count('1')
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v == -1 or best_count == 0:
                break
            S.add(best_v)
            uncovered &= ~clique_mask[best_v]
        return S if not uncovered else None
    
    # Try greedy
    S_greedy = greedy_solve()
    if S_greedy is not None:
        best_size = len(S_greedy)
        best_solution = set(S_greedy)
    
    # B&B Search with improved clique selection
    start_time = time.time()
    
    # Precompute: for each clique, the set of vertices that can cover it
    # and track which vertices are "active" (not yet decided)
    
    def count_bits(x):
        return bin(x).count('1')
    
    def lower_bound(uncovered_mask, current_size):
        """Estimate minimum additional vertices needed."""
        # Simple lower bound: each vertex can cover at most max_cover vertices
        if not uncovered_mask:
            return 0
        max_cover = 0
        for v in range(n):
            c = count_bits(clique_mask[v] & uncovered_mask)
            if c > max_cover:
                max_cover = c
        if max_cover == 0:
            return 1000  # Shouldn't happen
        return (count_bits(uncovered_mask) + max_cover - 1) // max_cover
    
    def dfs(current_size, uncovered_mask, path):
        nonlocal best_size, best_solution
        
        if time.time() - start_time > time_limit:
            return False # Timeout
        
        if uncovered_mask == 0:
            if current_size < best_size:
                best_size = current_size
                best_solution = set(path)
            return True # Success
            
        # Pruning with lower bound
        if current_size + lower_bound(uncovered_mask, current_size) >= best_size:
            return False
            
        if current_size >= best_size:
            return False
            
        # Pick the clique with fewest candidate vertices (most constrained)
        # Try a few strategies: lowest set bit, or most constrained
        # Use most constrained: find clique with minimum number of vertices that can cover it
        best_clique_idx = -1
        best_num_candidates = 1000
        # Sample a few cliques to find the most constrained one
        candidates_to_check = []
        um = uncovered_mask
        while um:
            bit = um & -um
            idx = bit.bit_length() - 1
            candidates_to_check.append(idx)
            um ^= bit
            if len(candidates_to_check) >= 10:  # Limit to 10 for speed
                break
        
        for ci in candidates_to_check:
            verts = clique_vertices[ci]
            # Count how many of these vertices can still be chosen (they're all available since we haven't excluded any)
            # In our formulation, all vertices are always available
            num_cand = len(verts)
            if num_cand < best_num_candidates:
                best_num_candidates = num_cand
                best_clique_idx = ci
        
        if best_clique_idx == -1:
            # Fallback to lowest set bit
            target_clique_idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
        else:
            target_clique_idx = best_clique_idx
        
        # Get vertices in this clique
        verts = clique_vertices[target_clique_idx]
        
        # Try each vertex in the clique
        for v in verts:
            new_uncovered = uncovered_mask & ~clique_mask[v]
            
            path.append(v)
            dfs(current_size + 1, new_uncovered, path)
            path.pop()
            
            # If we found a solution with size <= current_size, we can prune
            if best_size <= current_size:
                return False
                
    dfs(0, (1 << num_cliques) - 1, [])
    
    # Local search polish: try to remove vertices from the solution
    if best_solution is not None and time.time() - start_time < time_limit - 2:
        polished = set(best_solution)
        improved = True
        while improved:
            improved = False
            for v in list(polished):
                # Try removing v
                candidate = polished - {v}
                # Check if candidate still covers all cliques
                ok = True
                for i, clique in enumerate(cliques):
                    if not (clique & candidate):
                        ok = False
                        break
                if ok:
                    polished = candidate
                    improved = True
                    best_size = len(polished)
                    best_solution = set(polished)
    
    if best_solution is not None:
        return sorted(best_solution)
    else:
        # Fallback to greedy if B&B timed out without finding anything better
        if S_greedy is not None:
            return sorted(S_greedy)
        # Last resort: return all vertices
        return list(range(n))

def construct():
    deadline = time.time() + 145  # 5s safety margin
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    # Allocate more time to harder instances (larger n tends to be harder)
    time_weights = [0.25, 0.35, 0.40]
    
    for idx, (n, p, seed) in enumerate(instances):
        remaining_time = deadline - time.time() - 3  # 3s safety margin
        if remaining_time < 5:
            # Fallback: simple greedy
            adj = _build_graph(n, p, seed)
            cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
            if not cliques:
                results.append([])
                continue
            # Simple greedy
            S = set()
            is_covered = [False] * len(cliques)
            vertex_cliques = [[] for _ in range(n)]
            for i, c in enumerate(cliques):
                for v in c:
                    vertex_cliques[v].append(i)
                    
            remaining = list(range(len(cliques)))
            while remaining:
                best_v = -1
                best_count = -1
                for v in range(n):
                    if v in S: continue
                    cnt = sum(1 for ci in vertex_cliques[v] if not is_covered[ci])
                    if cnt > best_count:
                        best_count = cnt
                        best_v = v
                if best_v == -1: break
                S.add(best_v)
                for ci in vertex_cliques[best_v]:
                    if not is_covered[ci]:
                        is_covered[ci] = True
                        remaining.remove(ci)
            results.append(sorted(S))
            continue
            
        time_budget = remaining_time * time_weights[idx]
        adj = _build_graph(n, p, seed)
        sol = _solve_instance(adj, time_limit=time_budget)
        results.append(sol)
        
        # Log progress for debugging (commented out)
        # print(f"Instance {idx}: n={n}, p={p}, seed={seed}, solution size={len(sol)}, time used={time.time()-start_time}")
        
    return results

# EVOLVE-BLOCK-END
