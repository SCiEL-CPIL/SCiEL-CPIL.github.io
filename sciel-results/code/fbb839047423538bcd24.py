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
    
    # Build support adjacency: supports sharing exactly 3 elements
    tri_to_tet = defaultdict(list)
    for i, s in enumerate(supports):
        for tri in itertools.combinations(s, 3):
            tri_to_tet[tri].append(i)
    
    support_adj = [[] for _ in range(n_supports)]
    for tri, tets in tri_to_tet.items():
        for u in range(len(tets)):
            for v in range(u + 1, len(tets)):
                i, j = tets[u], tets[v]
                support_adj[i].append(j)
                support_adj[j].append(i)
    
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
    
    # Build vector-level conflict graph (adjacency list)
    # Two vectors conflict if their dot product > 2
    # We only need to check pairs of vectors whose supports share >= 3 elements
    # (if supports share <= 2 elements, dot product is at most 2 automatically)
    
    # For efficiency, precompute dot products between vectors in adjacent supports
    # and build the conflict adjacency list
    
    # First, let's build a mapping from support index to vector indices
    # Vectors for support s are at indices s*16 to s*16+15
    
    # Build conflict adjacency using support structure
    # For each pair of supports that share exactly 3 elements, check all 16x16 pairs
    
    # Precompute the dot product matrices for adjacent support pairs
    # and extract conflict edges
    
    # To save memory and time, we'll build the adjacency list directly
    # by iterating over support pairs that share 3 elements
    
    # Create a set of support pairs that share 3 elements
    support_pairs = set()
    for tri, tets in tri_to_tet.items():
        for u in range(len(tets)):
            for v in range(u + 1, len(tets)):
                i, j = tets[u], tets[v]
                support_pairs.add((min(i, j), max(i, j)))
    
    # Build vector-level adjacency list
    # adj[i] = list of j such that dot(i, j) > 2
    adj = [[] for _ in range(n_total)]
    
    for s1, s2 in support_pairs:
        base1 = s1 * 16
        base2 = s2 * 16
        
        # Get vectors for these two supports
        v1 = all_vectors_np[base1:base1+16]
        v2 = all_vectors_np[base2:base2+16]
        
        # Compute 16x16 dot product matrix
        d = v1 @ v2.T
        
        # Find conflicts: d > 2
        indices = np.argwhere(d > 2)
        for i, j in indices:
            idx1 = base1 + i
            idx2 = base2 + j
            adj[idx1].append(idx2)
            adj[idx2].append(idx1)
    
    # Convert adjacency list to sorted lists for faster access
    for i in range(n_total):
        adj[i].sort()
    
    # Initial degrees
    initial_deg = np.array([len(adj[i]) for i in range(n_total)], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_k = 0
    best_selected = None
    
    # Strategy: Multiple approaches
    # 1. Greedy minimum-degree MIS (fast, many restarts)
    # 2. Local search / hill climbing on the greedy solution
    
    max_greedy_restarts = 300
    
    # Phase 1: Greedy Minimum Degree MIS with random restarts
    for restart in range(max_greedy_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        avail = np.ones(n_total, dtype=bool)
        deg = initial_deg.copy()
        selected = np.zeros(n_total, dtype=bool)
        
        # Greedy Minimum Degree MIS
        # We maintain a list of available nodes and pick the one with minimum degree
        # To make this efficient, we use a simple approach:
        # At each step, find the available node with minimum degree, select it,
        # and remove it and its neighbors from available.
        
        # For speed, we'll use a simpler greedy: pick random available node
        # with minimum degree among a random subset, or just min degree overall.
        
        # Actually, let's use a more efficient approach:
        # Maintain a list of available indices. At each step, find min degree,
        # select that node, update degrees of neighbors.
        
        # To avoid O(N) scan each time, we can use a priority queue,
        # but for 5280 nodes, O(N) scan is fine if we do it in NumPy.
        
        while avail.any():
            # Find available nodes
            avail_indices = np.where(avail)[0]
            if len(avail_indices) == 0:
                break
            
            # Get their degrees
            degs = deg[avail_indices]
            min_deg = np.min(degs)
            
            # Find all nodes with min degree
            min_rel = np.where(degs == min_deg)[0]
            
            # Pick one randomly
            pick_rel = min_rel[rng.integers(len(min_rel))]
            u = avail_indices[pick_rel]
            
            # Select u
            selected[u] = True
            avail[u] = False
            
            # Remove neighbors
            for v in adj[u]:
                if avail[v]:
                    avail[v] = False
                    # Update degrees of v's neighbors
                    for w in adj[v]:
                        if avail[w]:
                            deg[w] -= 1
        
        k = np.sum(selected)
        
        if k > best_k:
            best_k = k
            best_selected = selected.copy()
        
        # Phase 2: Local search to improve the solution
        # Try to add nodes that are compatible with all selected nodes
        # and not already selected
        improved = True
        while improved:
            improved = False
            elapsed = time.time() - start_time
            if elapsed > deadline - 5:
                break
            
            # For each unselected node, check if it can be added
            # To speed up, only check nodes whose support has adjacent selected nodes
            # (if no adjacent selected nodes, it can definitely be added)
            
            # Build a set of selected nodes for fast lookup
            # Actually, let's just iterate through all unselected nodes
            
            for i in range(n_total):
                if selected[i]:
                    continue
                
                # Check if i is compatible with all selected nodes
                # i conflicts with j if j in adj[i] and selected[j]
                # So we need: no j in adj[i] with selected[j] = True
                
                ok = True
                for j in adj[i]:
                    if selected[j]:
                        ok = False
                        break
                
                if ok:
                    selected[i] = True
                    improved = True
        
        k = np.sum(selected)
        if k > best_k:
            best_k = k
            best_selected = selected.copy()
    
    # Phase 3: Try a few more sophisticated local search moves
    # Try swapping: remove one selected node, add two compatible ones
    # This is expensive, so only do it if we have time
    
    if best_selected is not None and time.time() - start_time < deadline - 20:
        # Try swap moves
        for _ in range(50):
            elapsed = time.time() - start_time
            if elapsed > deadline - 5:
                break
            
            selected = best_selected.copy()
            k = np.sum(selected)
            
            # Try removing a random selected node and adding compatible ones
            sel_indices = np.where(selected)[0]
            if len(sel_indices) == 0:
                break
            
            # Try a few random removals
            for _ in range(20):
                elapsed = time.time() - start_time
                if elapsed > deadline - 5:
                    break
                
                rem = sel_indices[rng.integers(len(sel_indices))]
                selected[rem] = False
                
                # Try to add back compatible nodes
                improved = True
                while improved:
                    improved = False
                    for i in range(n_total):
                        if selected[i]:
                            continue
                        ok = True
                        for j in adj[i]:
                            if selected[j]:
                                ok = False
                                break
                        if ok:
                            selected[i] = True
                            improved = True
                
                k_new = np.sum(selected)
                if k_new > k:
                    k = k_new
                    if k > best_k:
                        best_k = k
                        best_selected = selected.copy()
                    break  # Restart swap search with better solution
    
    if best_selected is None:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    result = all_vectors_np[best_selected]
    return result.astype(np.int64)
