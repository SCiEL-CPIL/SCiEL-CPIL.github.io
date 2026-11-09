import time
import math
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = 55.0
    
    N = 128
    full_mask = (1 << (N + 1)) - 2  # bits 1..N set
    
    # Precompute LCMs
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    # Find all bad triples
    bad_triple_masks = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm(a, b)
            for c in range(b + 1, N + 1):
                if lab == lcm(a, c) == lcm(b, c):
                    mask = (1 << a) | (1 << b) | (1 << c)
                    bad_triple_masks.append(mask)
    
    # If no bad triples, return full set
    if not bad_triple_masks:
        return [i for i in range(1, N + 1) if full_mask & (1 << i)]
    
    def is_valid(mask):
        for t in bad_triple_masks:
            if (mask & t) == t:
                return False
        return True

    # Construct Weighted Conflict Graph
    # Adjacency matrix W where W[u][v] = number of bad triples containing u and v
    W = np.zeros((N + 1, N + 1), dtype=np.float64)
    for t in bad_triple_masks:
        # Extract vertices from mask
        verts = []
        x = t
        while x:
            lsb = x & -x
            idx = lsb.bit_length() - 1
            verts.append(idx)
            x ^= lsb
        # Add weights for pairs
        W[verts[0]][verts[1]] += 1
        W[verts[1]][verts[0]] += 1
        W[verts[0]][verts[2]] += 1
        W[verts[2]][verts[0]] += 1
        W[verts[1]][verts[2]] += 1
        W[verts[2]][verts[1]] += 1
    
    # Spectral Heuristic:
    # We want to find a large independent set.
    # Approximation: Max-Cut or similar.
    # Let's use the principal eigenvector of the Laplacian or Adjacency.
    # High eigenvector components often indicate "core" nodes in a community.
    # For Independent Set, we might want to remove nodes with high weighted degree first.
    # Or, we can use the eigenvector to guide a greedy selection.
    
    # Compute principal eigenvector of W
    try:
        # Use numpy's eigh for symmetric matrix
        # We care about the largest eigenvalue's eigenvector
        # Note: W is symmetric
        # For efficiency, we can use power iteration or just eigh if N is small (129 is small)
        vals, vecs = np.linalg.eigh(W)
        # Principal eigenvector corresponds to largest eigenvalue
        # eigh returns eigenvalues in ascending order
        idx_max = np.argmax(vals)
        v_max = vecs[:, idx_max]
    except Exception:
        # Fallback to weighted degree
        v_max = np.sum(W, axis=1)
        
    # Normalize v_max to [0, 1] for use as priority
    # We want to REMOVE nodes with high "conflict".
    # So high v_max value -> remove early?
    # Let's check correlation. Usually, nodes in dense clusters have high eigenvector centrality.
    # We want an independent set, so we should avoid dense clusters.
    # So, we should REMOVE nodes with high centrality first.
    
    def greedy_remove_spectral(full_mask, randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
            
        mask = full_mask
        # Precompute list of indices sorted by centrality (descending)
        # We will iterate through this list, but only remove if the node is still in mask
        # And we should check if removing it is "worth it" (reduces conflicts significantly)
        
        # Simple approach:
        # 1. Sort nodes by centrality (high to low)
        # 2. Iterate. If node is in mask, and removing it eliminates at least one active triple, remove it.
        #    Actually, just removing high centrality nodes greedily is a good heuristic for Max Independent Set on dense graphs.
        
        # To allow randomness, we can add noise to the scores
        scores = v_max.copy()
        if randomize:
            noise = np.random.normal(0, 0.1, size=N + 1)
            scores += noise
            
        # Sort indices 1..N by score descending
        order = list(range(1, N + 1))
        order.sort(key=lambda i: -scores[i])
        
        # We need to be careful: just removing top K might not be optimal.
        # Standard Greedy:
        # While there are active triples:
        #   Find the node in the mask that is in the most active triples (or highest centrality among those)
        #   Remove it.
        
        # Let's stick to the "remove if it helps" logic but guided by spectral order.
        # Actually, a better spectral heuristic for IS is:
        # Start with full set.
        # While invalid:
        #   Compute "conflict count" for each node in mask (number of bad triples fully in mask containing node)
        #   Pick node with max (conflict_count * alpha + centrality * beta)
        #   Remove it.
        
        # Let's implement this dynamic scoring.
        
        while True:
            if time.time() - start_time > deadline - 5:
                break
                
            active_triples = [t for t in bad_triple_masks if (mask & t) == t]
            if not active_triples:
                break
            
            counts = [0] * (N + 1)
            for t in active_triples:
                x = t
                while x:
                    lsb = x & -x
                    idx = lsb.bit_length() - 1
                    counts[idx] += 1
                    x ^= lsb
            
            # Combine dynamic count with static spectral score
            # Normalize counts?
            max_count = max(counts)
            if max_count == 0:
                break
                
            best_node = -1
            best_score = -1
            
            for i in range(1, N + 1):
                if mask & (1 << i):
                    # Heuristic: Prioritize removing nodes with high conflict count.
                    # Use spectral score to break ties or slightly weight the decision.
                    # score = counts[i] * 10 + v_max[i]
                    # If randomize, add noise
                    s = counts[i] * 10.0 + v_max[i]
                    if randomize:
                        s += random.gauss(0, 0.5)
                    if s > best_score:
                        best_score = s
                        best_node = i
                        
            if best_node == -1:
                break
                
            mask &= ~(1 << best_node)
            
        return mask

    def add_back(mask):
        improved = True
        while improved:
            improved = False
            for i in range(1, N + 1):
                if not (mask & (1 << i)):
                    new_mask = mask | (1 << i)
                    if is_valid(new_mask):
                        mask = new_mask
                        improved = True
        return mask
    
    def swap_improve(mask):
        improved = True
        while improved:
            improved = False
            in_mask = [i for i in range(1, N + 1) if mask & (1 << i)]
            not_in_mask = [i for i in range(1, N + 1) if not (mask & (1 << i))]
            
            if not not_in_mask:
                break
                
            for i in range(len(not_in_mask)):
                v = not_in_mask[i]
                for j in range(i + 1, len(not_in_mask)):
                    w = not_in_mask[j]
                    
                    M_prime = mask | (1 << v) | (1 << w)
                    blocking_triples = [t for t in bad_triple_masks if (M_prime & t) == t]
                    
                    if not blocking_triples:
                        if is_valid(M_prime):
                            mask = M_prime
                            improved = True
                            break
                        # If invalid, need to remove u
                    
                    common_intersection = None
                    for t in blocking_triples:
                        if common_intersection is None:
                            common_intersection = t
                        else:
                            common_intersection &= t
                    
                    if common_intersection is None:
                        continue
                        
                    candidate_u = None
                    x = common_intersection
                    while x:
                        lsb = x & -x
                        idx = lsb.bit_length() - 1
                        if mask & (1 << idx):
                            candidate_u = idx
                            break
                        x ^= lsb
                    
                    if candidate_u is not None:
                        mask = (mask & ~(1 << candidate_u)) | (1 << v) | (1 << w)
                        improved = True
                        break
            
            if improved:
                continue
                
        return mask
    
    best_mask = 0
    best_size = 0
    
    # Deterministic run
    m = greedy_remove_spectral(full_mask, randomize=False)
    m = add_back(m)
    m = swap_improve(m)
    size = m.bit_count()
    if size > best_size:
        best_size = size
        best_mask = m
        
    # Randomized runs
    # Use different seeds to perturb the spectral heuristic
    for i in range(100):
        if time.time() - start_time > deadline - 5:
            break
        m = greedy_remove_spectral(full_mask, randomize=True, seed=42 + i)
        m = add_back(m)
        m = swap_improve(m)
        size = m.bit_count()
        if size > best_size:
            best_size = size
            best_mask = m
            
    result = [i for i in range(1, N + 1) if best_mask & (1 << i)]
    return result
