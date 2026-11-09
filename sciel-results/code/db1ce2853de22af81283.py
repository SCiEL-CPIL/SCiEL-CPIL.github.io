import numpy as np
import time
from itertools import combinations

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Generate all possible 2x2x2 Strassen block positions
    all_blocks = []
    for i0 in range(n - 1):  # i0 = 0, 1
        for j0 in range(m - 1):  # j0 = 0, 1, 2
            for k0 in range(p - 1):  # k0 = 0, 1, 2, 3
                all_blocks.append((i0, j0, k0))
    
    # For each block, compute which (i,j,k) entries it covers
    block_coverage = {}
    for block in all_blocks:
        i0, j0, k0 = block
        entries = set()
        for i in range(i0, i0 + 2):
            for j in range(j0, j0 + 2):
                for k in range(k0, k0 + 2):
                    entries.add((i, j, k))
        block_coverage[block] = entries
    
    # Find all maximal sets of disjoint blocks
    # A set of blocks is disjoint if no two blocks share any (i,j,k) entry
    def is_disjoint(block_set):
        covered = set()
        for b in block_set:
            for entry in block_coverage[b]:
                if entry in covered:
                    return False
                covered.add(entry)
        return True
    
    # Find all maximal disjoint sets by trying all subsets
    # Since there are only 24 blocks, we can be smart about this
    # First, find all pairs that are disjoint
    disjoint_pairs = set()
    for i, b1 in enumerate(all_blocks):
        for j, b2 in enumerate(all_blocks):
            if i >= j:
                continue
            if is_disjoint([b1, b2]):
                disjoint_pairs.add((i, j))
    
    # Find all maximal disjoint sets using backtracking
    # But to keep it fast, let's just try a few promising configurations
    
    # The 4-block configuration we know works:
    known_good = [(0, 0, 0), (0, 0, 2), (0, 2, 0), (0, 2, 2)]
    
    # Try to find better configurations with more blocks
    # Maximum possible blocks: since each block covers 8 entries and we have 60 total,
    # at most 7 blocks (56 entries), but disjointness is the constraint
    
    # Let's do a BFS/DFS to find all maximal disjoint sets
    # But limit the search to keep it fast
    
    def find_maximal_disjoint_sets(blocks_list, current_set, covered_entries, results):
        if time.time() > deadline - 5:
            return
        # Check if current set is maximal (no more blocks can be added)
        is_maximal = True
        for b in blocks_list:
            if b in current_set:
                continue
            if is_disjoint(current_set | {b}):
                is_maximal = False
                break
        
        if is_maximal:
            results.append(frozenset(current_set))
            return
        
        # Try adding each remaining block
        for b in blocks_list:
            if b in current_set:
                continue
            if is_disjoint(current_set | {b}):
                new_covered = covered_entries | block_coverage[b]
                find_maximal_disjoint_sets(blocks_list, current_set | {b}, new_covered, results)
                if time.time() > deadline - 5:
                    return
    
    # To keep search manageable, let's just try a few specific configurations
    # Configuration 1: The known 4-block set
    configs = [
        [(0, 0, 0), (0, 0, 2), (0, 2, 0), (0, 2, 2)],
        [(1, 0, 0), (1, 0, 2), (1, 2, 0), (1, 2, 2)],
        [(0, 0, 0), (0, 0, 2), (0, 2, 0), (0, 2, 2), (1, 0, 0)],  # May not be disjoint
    ]
    
    # Filter to only valid (disjoint) configurations
    valid_configs = []
    for config in configs:
        if is_disjoint(config):
            valid_configs.append(config)
    
    # Also try to find 5-block or larger configurations
    # Let's do a limited search
    found_configs = []
    found_configs.append(frozenset(known_good))
    
    # Try adding one more block to known_good
    for b in all_blocks:
        if b in known_good:
            continue
        test_config = known_good + [b]
        if is_disjoint(test_config):
            found_configs.append(frozenset(test_config))
    
    # Deduplicate
    unique_configs = []
    seen = set()
    for fc in found_configs:
        if fc not in seen:
            seen.add(fc)
            unique_configs.append(list(fc))
    
    # Evaluate each configuration
    def evaluate_config(blocks):
        """Compute the rank for a given set of Strassen blocks."""
        covered = set()
        for i0, j0, k0 in blocks:
            for i in range(i0, i0 + 2):
                for j in range(j0, j0 + 2):
                    for k in range(k0, k0 + 2):
                        covered.add((i, j, k))
        
        # Number of schoolbook terms needed
        schoolbook_count = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        schoolbook_count += 1
        
        total_rank = len(blocks) * 7 + schoolbook_count
        return total_rank, covered
    
    best_rank = rank_schoolbook
    best_blocks = []
    
    for config in unique_configs:
        rank, covered = evaluate_config(config)
        if rank < best_rank:
            best_rank = rank
            best_blocks = config
    
    # Also try the schoolbook-only baseline
    if best_rank > rank_schoolbook:
        best_rank = rank_schoolbook
        best_blocks = []
    
    # Build the decomposition
    if not best_blocks:
        # Pure schoolbook
        rank = rank_schoolbook
        U = np.zeros((size_u, rank))
        V = np.zeros((size_v, rank))
        W = np.zeros((size_w, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, c] = 1
                    V[v, c] = 1
                    W[w, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build terms list
    terms = []
    
    def add_strassen_block(i0, j0, k0):
        def u_idx(li, lj):
            return (i0 + li) * m + (j0 + lj)
        def v_idx(lj, lk):
            return (j0 + lj) * p + (k0 + lk)
        def w_idx(lk, li):
            return (k0 + lk) * n + (i0 + li)
        
        # Strassen terms for 2x2
        terms.append(({u_idx(0,0): 1, u_idx(1,1): 1}, 
                      {v_idx(0,0): 1, v_idx(1,1): 1}, 
                      {w_idx(0,0): 1, w_idx(1,1): 1}))
        terms.append(({u_idx(1,0): 1, u_idx(1,1): 1}, 
                      {v_idx(0,0): 1}, 
                      {w_idx(1,0): 1, w_idx(1,1): -1}))
        terms.append(({u_idx(0,0): 1}, 
                      {v_idx(1,0): 1, v_idx(1,1): -1}, 
                      {w_idx(1,0): 1, w_idx(1,1): 1}))
        terms.append(({u_idx(1,1): 1}, 
                      {v_idx(0,1): 1, v_idx(1,1): 1}, 
                      {w_idx(0,0): 1, w_idx(1,1): 1}))
        terms.append(({u_idx(0,0): 1, u_idx(0,1): 1}, 
                      {v_idx(1,1): 1}, 
                      {w_idx(0,0): -1, w_idx(0,1): 1}))
        terms.append(({u_idx(1,0): 1, u_idx(0,0): -1}, 
                      {v_idx(0,0): 1, v_idx(0,1): 1}, 
                      {w_idx(0,1): 1}))
        terms.append(({u_idx(0,1): 1, u_idx(1,1): -1}, 
                      {v_idx(1,0): 1, v_idx(1,1): 1}, 
                      {w_idx(0,0): -1}))
    
    for i0, j0, k0 in best_blocks:
        add_strassen_block(i0, j0, k0)
    
    # Track covered entries
    covered = set()
    for i0, j0, k0 in best_blocks:
        for i in range(i0, i0 + 2):
            for j in range(j0, j0 + 2):
                for k in range(k0, k0 + 2):
                    covered.add((i, j, k))
    
    # Add schoolbook terms for uncovered entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
    
    rank = len(terms)
    
    # Build U, V, W matrices
    U = np.zeros((size_u, rank))
    V = np.zeros((size_v, rank))
    W = np.zeros((size_w, rank))
    
    for c, (u_data, v_data, w_data) in enumerate(terms):
        for idx, val in u_data.items():
            U[idx, c] += val
        for idx, val in v_data.items():
            V[idx, c] += val
        for idx, val in w_data.items():
            W[idx, c] += val
    
    # Verify
    recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    target = np.zeros((size_u, size_v, size_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target[u, v, w] = 1.0
    
    if not np.allclose(recon, target, atol=1e-9):
        # Fallback to schoolbook
        rank = rank_schoolbook
        U = np.zeros((size_u, rank))
        V = np.zeros((size_v, rank))
        W = np.zeros((size_w, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, c] = 1
                    V[v, c] = 1
                    W[w, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
