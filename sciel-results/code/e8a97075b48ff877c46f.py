import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    D = 11
    
    # Generate supports (4-subsets)
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports)  # 330
    
    # Generate all 5280 weight-4 vectors
    all_vectors = []
    vec_support = []
    for s_idx, s in enumerate(supports):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            all_vectors.append(v)
            vec_support.append(s_idx)
    
    all_vectors_np = np.array(all_vectors, dtype=np.int8)
    n_total = len(all_vectors_np)  # 5280
    
    # Build adjacency bitmasks
    # adj[i] is an integer with bit j set if i and j conflict (dot > 2)
    adj = [0] * n_total
    
    # Build support adjacency
    tri_to_tet = defaultdict(list)
    for i, s in enumerate(supports):
        for tri in itertools.combinations(s, 3):
            tri_to_tet[tri].append(i)
            
    support_pairs = set()
    for tri, tets in tri_to_tet.items():
        for u in range(len(tets)):
            for v in range(u + 1, len(tets)):
                i, j = tets[u], tets[v]
                support_pairs.add((min(i, j), max(i, j)))
                
    # Compute conflicts
    for s1, s2 in support_pairs:
        base1 = s1 * 16
        base2 = s2 * 16
        v1 = all_vectors_np[base1:base1+16]
        v2 = all_vectors_np[base2:base2+16]
        d = v1 @ v2.T
        indices = np.argwhere(d > 2)
        for i, j in indices:
            idx1 = base1 + i
            idx2 = base2 + j
            adj[idx1] |= (1 << idx2)
            adj[idx2] |= (1 << idx1)
            
    # Add the 22 weight-1 vectors (norm squared 4)
    # These are compatible with all weight-4 vectors and each other.
    # We will just add them to the final count.
    extra_vectors = []
    for i in range(D):
        v_pos = [0]*D
        v_neg = [0]*D
        v_pos[i] = 2
        v_neg[i] = -2
        extra_vectors.append(v_pos)
        extra_vectors.append(v_neg)
        
    extra_count = len(extra_vectors)
    
    # MIS Search on the 5280 weight-4 vectors
    rng = np.random.default_rng(42)
    
    best_selected_mask = 0
    best_k = 0
    
    # Greedy + Local Search
    max_restarts = 200
    
    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
            
        # Greedy MIS
        available = (1 << n_total) - 1
        selected = 0
        # To speed up min-degree selection, we might just pick random from available
        # or use a simple heuristic.
        # Given the graph structure, random greedy is often decent.
        
        # Better: Greedy with min degree approximation
        # For simplicity and speed in Python, let's use a randomized greedy
        # that picks a random node from the available set.
        # To improve, we can bias towards nodes with fewer neighbors in available set.
        
        while available:
            # Get list of available indices
            # This is slow if done naively.
            # Let's just pick a random bit.
            # Or, better, maintain a list.
            # But bitset operations are fast.
            
            # Pick random available node
            # Method: get integer value of available, pick random bit position
            # This is O(log N) if we can isolate bits.
            # Simpler: convert to list of indices? No, 5280 items.
            
            # Let's use a simple greedy: pick the node with min degree in available.
            # Calculating degree for all available nodes is O(N^2) or O(E).
            # With bitsets, degree of u in available is popcount(adj[u] & available).
            
            # To avoid O(N) scan for min degree every step, we can just pick random.
            # Random greedy is fast.
            
            # Get a random index from available
            # Trick: take available, count bits, pick random rank, find k-th set bit.
            # Or simpler: just pick a random number 0..N-1 and check if set?
            # No, that's inefficient.
            
            # Let's stick to a faster heuristic:
            # Pick a random node from available.
            avail_list = [i for i in range(n_total) if (available >> i) & 1]
            if not avail_list:
                break
            u = avail_list[rng.integers(len(avail_list))]
            
            selected |= (1 << u)
            available &= ~(1 << u)
            available &= ~adj[u]
            
        k = bin(selected).count('1')
        
        # Local Search: 1-swap
        improved = True
        while improved:
            improved = False
            elapsed = time.time() - start_time
            if elapsed > deadline - 5:
                break
                
            # Get list of selected nodes
            sel_list = [i for i in range(n_total) if (selected >> i) & 1]
            
            for u in sel_list:
                # Try to remove u and add compatible neighbors
                # Neighbors of u that are not in selected
                candidates = adj[u] & ~selected
                # Check each candidate v
                # We need v to be compatible with selected \ {u}
                # i.e., (adj[v] & (selected ^ (1<<u))) == 0
                
                # Iterate through bits in candidates
                c = candidates
                while c:
                    # Get lowest set bit
                    lsb = c & -c
                    v = lsb.bit_length() - 1
                    c ^= lsb
                    
                    # Check compatibility
                    if (adj[v] & (selected ^ (1 << u))) == 0:
                        # Swap u for v
                        selected ^= (1 << u)
                        selected |= (1 << v)
                        improved = True
                        # We can break and restart the local search loop
                        # Or continue checking other swaps
                        # Breaking allows us to re-evaluate the new state
                        break
                if improved:
                    break
                    
        k = bin(selected).count('1')
        if k > best_k:
            best_k = k
            best_selected_mask = selected
            
    # Construct final result
    # Weight-4 vectors from mask
    res_w4 = []
    for i in range(n_total):
        if (best_selected_mask >> i) & 1:
            res_w4.append(all_vectors_np[i])
            
    # Add weight-1 vectors
    res_all = np.array(res_w4 + extra_vectors, dtype=np.int64)
    
    return res_all

