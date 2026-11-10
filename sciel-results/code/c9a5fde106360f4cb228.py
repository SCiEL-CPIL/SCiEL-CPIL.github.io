"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from functools import lru_cache

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _precompute_cliques(n, adj):
    """Precompute all maximal cliques and independent sets as bitmasks."""
    cliques = [0] * (1 << n)
    independents = [0] * (1 << n)
    
    for mask in range(1, 1 << n):
        # Check if mask is a clique
        is_clique = True
        is_indep = True
        vertices = [i for i in range(n) if mask & (1 << i)]
        for u in vertices:
            for v in vertices:
                if u != v:
                    if v not in adj[u]:
                        is_clique = False
                    else:
                        is_indep = False
        cliques[mask] = is_clique
        independents[mask] = is_indep
    
    return cliques, independents

def _greedy_partition(n, adj, cliques, independents):
    parts = []
    part_masks = []
    for v in range(n):
        best_part = None
        best_gain = -1
        for i, pmask in enumerate(part_masks):
            trial_mask = pmask | (1 << v)
            if cliques[trial_mask] or independents[trial_mask]:
                gain = bin(trial_mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_part = i
        if best_part is not None:
            part_masks[best_part] |= (1 << v)
        else:
            part_masks.append(1 << v)
    
    # Convert masks back to lists
    result = []
    for pmask in part_masks:
        part = [i for i in range(n) if pmask & (1 << i)]
        if part:
            result.append(part)
    return result

def _solve_with_k_parts(n, adj, cliques, independents, k, deadline, start_time):
    """Try to find a cochromatic partition with exactly k parts using backtracking."""
    if time.time() - start_time > deadline:
        return None
    
    # Order vertices by degree (higher degree first) for better pruning
    vertex_order = sorted(range(n), key=lambda v: -len(adj[v]))
    
    parts = [0] * k  # bitmask for each part
    seen_part_masks = set()  # For symmetry breaking
    
    def backtrack(idx):
        if time.time() - start_time > deadline:
            return False
        
        if idx == n:
            # All vertices assigned; check all parts are non-empty
            return all(p != 0 for p in parts)
        
        v = vertex_order[idx]
        v_bit = 1 << v
        
        # Symmetry breaking: only try part 0 or parts with different mask patterns
        tried_patterns = set()
        for i in range(k):
            # Check if this part's mask pattern has been tried
            if parts[i] in tried_patterns:
                continue
            tried_patterns.add(parts[i])
            
            trial_mask = parts[i] | v_bit
            if cliques[trial_mask] or independents[trial_mask]:
                parts[i] = trial_mask
                if backtrack(idx + 1):
                    return True
                parts[i] = trial_mask ^ v_bit
        
        return False
    
    if backtrack(0):
        result = []
        for pmask in parts:
            if pmask != 0:
                part = [i for i in range(n) if pmask & (1 << i)]
                result.append(part)
        return result
    return None

def _better_greedy(n, adj, cliques, independents):
    """Improved greedy that tries multiple vertex orderings."""
    best_parts = None
    best_count = n + 1
    
    # Try several orderings
    orderings = [
        list(range(n)),
        sorted(range(n), key=lambda v: len(adj[v])),  # by degree
        sorted(range(n), key=lambda v: -len(adj[v])),  # by degree desc
        sorted(range(n), key=lambda v: (len(adj[v]), v)),  # degree then index
        sorted(range(n), key=lambda v: (-len(adj[v]), v)),  # -degree then index
    ]
    
    for ordering in orderings:
        parts = []
        part_masks = []
        for v in ordering:
            best_part = None
            best_gain = -1
            for i, pmask in enumerate(part_masks):
                trial_mask = pmask | (1 << v)
                if cliques[trial_mask] or independents[trial_mask]:
                    gain = bin(trial_mask).count('1')
                    if gain > best_gain:
                        best_gain = gain
                        best_part = i
            if best_part is not None:
                part_masks[best_part] |= (1 << v)
            else:
                part_masks.append(1 << v)
        
        count = len(part_masks)
        if count < best_count:
            best_count = count
            best_parts = [ [i for i in range(n) if pmask & (1 << i)] for pmask in part_masks if pmask != 0 ]
    
    return best_parts

def construct():
    n = P['n']
    adj = _build_adj(n, P['edges'])
    
    start_time = time.time()
    deadline = 80.0  # Safety margin from 90s limit
    
    # Precompute clique/independent set checks
    cliques, independents = _precompute_cliques(n, adj)
    
    # Get initial solution from greedy
    best_parts = _better_greedy(n, adj, cliques, independents)
    best_count = len(best_parts)
    
    # Try to improve with exact search for fewer parts
    for target in range(1, best_count):
        if time.time() - start_time > deadline:
            break
        found = _solve_with_k_parts(n, adj, cliques, independents, target, deadline, start_time)
        if found is not None:
            best_parts = found
            best_count = target
            break
        # If we couldn't find a solution with 'target' parts, try target+1 as well
        # in case the search was cut short by deadline
        if time.time() - start_time > deadline - 5:
            break
    
    # Ensure proper types: list of lists of Python ints
    final_result = []
    for part in best_parts:
        final_result.append([int(v) for v in part])
    
    return final_result
# EVOLVE-BLOCK-END
