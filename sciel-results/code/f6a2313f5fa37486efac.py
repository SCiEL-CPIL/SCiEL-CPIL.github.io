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
        U = P | X
        if not U:
            return
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
    """Solve the minimum transversal problem for maximal cliques."""
    n = len(adj)
    cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
    
    if not cliques:
        return []
    
    num_cliques = len(cliques)
    
    # Precompute vertex -> clique indices mapping
    vertex_cliques = [[] for _ in range(n)]
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v].append(i)
    
    # Precompute bitmask for each vertex (which cliques it covers)
    vertex_mask = [0] * n
    for v in range(n):
        for ci in vertex_cliques[v]:
            vertex_mask[v] |= (1 << ci)
    
    all_cliques_mask = (1 << num_cliques) - 1
    
    # Greedy initialization
    S = set()
    covered_mask = 0
    
    while covered_mask != all_cliques_mask:
        best_v = -1
        best_gain = -1
        for v in range(n):
            if v in S:
                continue
            gain = vertex_mask[v] & (~covered_mask)
            gain_count = bin(gain).count('1')
            if gain_count > best_gain:
                best_gain = gain_count
                best_v = v
        if best_v == -1:
            break
        S.add(best_v)
        covered_mask |= vertex_mask[best_v]
    
    best_solution = list(S)
    best_size = len(best_solution)
    
    # Local search: try to remove vertices
    current_S = set(best_solution)
    current_cover = [0] * num_cliques
    for v in current_S:
        for ci in vertex_cliques[v]:
            current_cover[ci] += 1
    
    def is_valid(cover):
        return min(cover) >= 1
    
    # Greedy removal
    improved = True
    while improved:
        improved = False
        for v in list(current_S):
            new_cover = current_cover[:]
            for ci in vertex_cliques[v]:
                new_cover[ci] -= 1
            if is_valid(new_cover):
                current_S.remove(v)
                current_cover = new_cover
                improved = True
                break
    
    if len(current_S) < best_size:
        best_size = len(current_S)
        best_solution = list(current_S)
    
    # Exact search: iterative deepening
    start_time = time.time()
    
    # Try to find a solution of size k for k from 1 to best_size
    for k in range(1, best_size + 1):
        if time.time() - start_time > time_limit:
            break
        
        found = [False]
        
        def dfs(exact_k, depth, remaining_mask, chosen, chosen_set):
            if found[0]:
                return
            if time.time() - start_time > time_limit:
                found[0] = True  # Stop
                return
            
            if depth == exact_k:
                if remaining_mask == 0:
                    found[0] = True
                    return
                return
            
            if remaining_mask == 0:
                return
            
            # Lower bound: need at least ceil(popcount(remaining_mask) / max_cover) more vertices
            # Simple bound: if remaining_mask has bits and no depth left, fail
            if depth == exact_k:
                return
            
            # Pick a clique to cover (one with fewest candidates)
            # Find a set bit in remaining_mask
            # Try each vertex that covers this clique
            # Pick the clique with minimum candidates for better pruning
            
            # Find which cliques are still uncovered
            uncovered = remaining_mask
            # Pick the uncovered clique with fewest candidate vertices
            best_clique = -1
            best_cand_count = float('inf')
            best_cand_list = []
            
            while uncovered:
                lsb = uncovered & (-uncovered)
                ci = lsb.bit_length() - 1
                uncovered ^= lsb
                
                # Count vertices that cover this clique and are not yet chosen
                cand_count = 0
                cand_list = []
                for v in range(n):
                    if v in chosen_set:
                        continue
                    if vertex_mask[v] & lsb:
                        cand_count += 1
                        cand_list.append(v)
                
                if cand_count < best_cand_count:
                    best_cand_count = cand_count
                    best_cand_list = cand_list
                    if cand_count == 0:
                        break
            
            if best_cand_count == 0:
                return  # Dead end
            
            # Try each candidate
            for v in best_cand_list:
                if found[0]:
                    break
                chosen_set.add(v)
                new_remaining = remaining_mask & (~vertex_mask[v])
                dfs(exact_k, depth + 1, new_remaining, chosen + [v], chosen_set)
                chosen_set.remove(v)
        
        dfs(k, 0, all_cliques_mask, [], set())
        
        if found[0]:
            # We found a solution of size k (or less, but we're trying k)
            # Actually, found[0] could be set due to timeout, so check
            if time.time() - start_time <= time_limit:
                best_size = k
                break
    
    # Final local search polish
    current_S = set(best_solution)
    current_cover = [0] * num_cliques
    for v in current_S:
        for ci in vertex_cliques[v]:
            current_cover[ci] += 1
    
    # Try swaps
    for _ in range(100):
        if time.time() - start_time > time_limit:
            break
        improved = False
        for v in list(current_S):
            for u in range(n):
                if u in current_S:
                    continue
                # Try swap v -> u
                new_cover = current_cover[:]
                for ci in vertex_cliques[v]:
                    new_cover[ci] -= 1
                for ci in vertex_cliques[u]:
                    new_cover[ci] += 1
                if is_valid(new_cover):
                    current_S.remove(v)
                    current_S.add(u)
                    current_cover = new_cover
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    if len(current_S) < best_size:
        best_size = len(current_S)
        best_solution = list(current_S)
    
    return sorted(best_solution)

def construct():
    deadline = time.time() + 145  # 5s safety margin
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    time_per_instance = (deadline - time.time()) / len(instances)
    
    for n, p, seed in instances:
        if time.time() > deadline - 5:
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
            
        adj = _build_graph(n, p, seed)
        sol = _solve_instance(adj, time_limit=time_per_instance - 1)
        results.append(sol)
        
    return results

# EVOLVE-BLOCK-END
