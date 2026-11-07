import time
import numpy as np
from itertools import product

def sign_edges(graph, budget):
    deadline = time.time() + 85.0  # safety margin under 90s
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    
    # Parse the prism structure: m = n/2
    m = n // 2
    
    # Identify edge types by position in the edge list
    # For a prism graph with vertices 0..m-1 (top) and m..2m-1 (bottom):
    # - Top cycle edges: (i, i+1) for i=0..m-2, and (m-1, 0)
    # - Bottom cycle edges: (m+i, m+i+1) for i=0..m-2, and (m+m-1, m)
    # - Matching edges: (i, m+i) for i=0..m-1
    
    top_edges = []  # indices of top cycle edges in the edge list
    bottom_edges = []  # indices of bottom cycle edges
    matching_edges = []  # indices of matching edges
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_edges.append(idx)
        elif u >= m and v >= m:
            bottom_edges.append(idx)
        else:
            matching_edges.append(idx)
    
    # Sort each list to get consistent ordering
    top_edges.sort()
    bottom_edges.sort()
    matching_edges.sort()
    
    def build_signed(signs):
        S = np.zeros((n, n), dtype=np.float64)
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            S[u, v] = s
            S[v, u] = s
        return S
    
    def objective(signs):
        S = build_signed(signs)
        ev = np.linalg.eigvalsh(S)
        return float(np.max(np.abs(ev)))
    
    def check_connected(signs):
        """Check if the lift is connected using BFS on the 2n-vertex lift."""
        # Build adjacency for the lift
        lift_adj = [[] for _ in range(2 * n)]
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            if s == 1:
                # +1: joins equal sides
                lift_adj[u * 2].append(v * 2)
                lift_adj[v * 2].append(u * 2)
                lift_adj[u * 2 + 1].append(v * 2 + 1)
                lift_adj[v * 2 + 1].append(u * 2 + 1)
            else:
                # -1: joins opposite sides
                lift_adj[u * 2].append(v * 2 + 1)
                lift_adj[v * 2 + 1].append(u * 2)
                lift_adj[u * 2 + 1].append(v * 2)
                lift_adj[v * 2].append(u * 2 + 1)
        
        # BFS from (0, 0)
        visited = [False] * (2 * n)
        queue = [0]
        visited[0] = True
        count = 1
        while queue:
            node = queue.pop(0)
            for neighbor in lift_adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    count += 1
                    queue.append(neighbor)
        return count == 2 * n
    
    best_signs = None
    best_obj = float('inf')
    
    # Strategy: exploit the algebraic structure
    # The spectrum of the signed prism depends on:
    # 1. The sign patterns on top and bottom cycles (2^m each, but only the product matters for the "twist")
    # 2. The sign pattern on matching edges (2^m)
    # 3. The products T, B, C
    
    # For small m (4,5,7,9), we can afford to search over reduced parameter spaces.
    # Key insight: for a given cycle, only the product of signs and the "shape" of the sign sequence matter.
    # But for circulant structure, the eigenvalues depend on the Fourier transform of the sign sequence.
    
    # Let's use a hybrid approach:
    # 1. For m <= 5: exhaustive search over all 2^(3m) sign patterns (max 32768 for m=5)
    # 2. For m > 5: structured search + local optimization
    
    def exhaustive_search(m):
        """Exhaustive search for small m."""
        nonlocal best_signs, best_obj
        total_edges = 3 * m
        # Map from (top_signs, bottom_signs, matching_signs) to edge index signs
        for top_bits in range(1 << m):
            if time.time() > deadline - 5.0:
                return
            top_signs = []
            for i in range(m):
                top_signs.append(1 if (top_bits >> i) & 1 else -1)
            
            for bottom_bits in range(1 << m):
                if time.time() > deadline - 5.0:
                    return
                bottom_signs = []
                for i in range(m):
                    bottom_signs.append(1 if (bottom_bits >> i) & 1 else -1)
                
                for match_bits in range(1 << m):
                    match_signs = []
                    for i in range(m):
                        match_signs.append(1 if (match_bits >> i) & 1 else -1)
                    
                    # Build the full sign array
                    signs = np.zeros(total_edges, dtype=np.int8)
                    for i, idx in enumerate(top_edges):
                        signs[idx] = top_signs[i]
                    for i, idx in enumerate(bottom_edges):
                        signs[idx] = bottom_signs[i]
                    for i, idx in enumerate(matching_edges):
                        signs[idx] = match_signs[i]
                    
                    # Check connectivity
                    if not check_connected(signs):
                        continue
                    
                    obj = objective(signs)
                    if obj < best_obj - 1e-12:
                        best_obj = obj
                        best_signs = signs.copy()
    
    def structured_search(m):
        """Structured search exploiting circulant structure for larger m."""
        nonlocal best_signs, best_obj
        
        # The key parameters are:
        # - Top cycle sign pattern (2^m possibilities)
        # - Bottom cycle sign pattern (2^m possibilities)  
        # - Matching sign pattern (2^m possibilities)
        # - But the spectrum depends on the Fourier transforms of these sequences
        
        # For efficiency, we can fix the matching signs to be all +1 (or all -1) initially,
        # then optimize the cycle signs, then refine.
        
        # Phase 1: Try all combinations of top and bottom cycle signs with fixed matching
        for match_pattern in [0, (1 << m) - 1]:  # all +1 or all -1
            match_signs = []
            for i in range(m):
                match_signs.append(1 if (match_pattern >> i) & 1 else -1)
            
            for top_bits in range(1 << m):
                if time.time() > deadline - 5.0:
                    return
                top_signs = []
                for i in range(m):
                    top_signs.append(1 if (top_bits >> i) & 1 else -1)
                
                for bottom_bits in range(1 << m):
                    if time.time() > deadline - 5.0:
                        return
                    bottom_signs = []
                    for i in range(m):
                        bottom_signs.append(1 if (bottom_bits >> i) & 1 else -1)
                    
                    signs = np.zeros(3 * m, dtype=np.int8)
                    for i, idx in enumerate(top_edges):
                        signs[idx] = top_signs[i]
                    for i, idx in enumerate(bottom_edges):
                        signs[idx] = bottom_signs[i]
                    for i, idx in enumerate(matching_edges):
                        signs[idx] = match_signs[i]
                    
                    if not check_connected(signs):
                        continue
                    
                    obj = objective(signs)
                    if obj < best_obj - 1e-12:
                        best_obj = obj
                        best_signs = signs.copy()
        
        # Phase 2: Local optimization from best found
        if best_signs is not None:
            _local_optimize(best_signs.copy())
    
    def _local_optimize(start_signs):
        """Local optimization via greedy single-flip and pair-flip."""
        nonlocal best_signs, best_obj
        cur_signs = start_signs.copy()
        cur_obj = objective(cur_signs)
        
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_signs = cur_signs.copy()
        
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            
            # Try single flips
            best_flip = -1
            best_flip_obj = cur_obj
            for e in range(num_edges):
                old = cur_signs[e]
                cur_signs[e] = -old
                if check_connected(cur_signs):
                    o = objective(cur_signs)
                    if o < best_flip_obj - 1e-12:
                        best_flip_obj = o
                        best_flip = e
                cur_signs[e] = old
            
            if best_flip >= 0:
                cur_signs[best_flip] = -cur_signs[best_flip]
                cur_obj = best_flip_obj
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best_signs = cur_signs.copy()
                improved = True
            else:
                # Try pair flips
                for _ in range(100):
                    if time.time() > deadline - 3.0:
                        break
                    e1 = np.random.randint(num_edges)
                    e2 = np.random.randint(num_edges)
                    if e1 == e2:
                        continue
                    cur_signs[e1] = -cur_signs[e1]
                    cur_signs[e2] = -cur_signs[e2]
                    if check_connected(cur_signs):
                        o = objective(cur_signs)
                        if o < cur_obj - 1e-12:
                            cur_obj = o
                            if cur_obj < best_obj:
                                best_obj = cur_obj
                                best_signs = cur_signs.copy()
                            improved = True
                            break
                        else:
                            cur_signs[e1] = -cur_signs[e1]
                            cur_signs[e2] = -cur_signs[e2]
                    else:
                        cur_signs[e1] = -cur_signs[e1]
                        cur_signs[e2] = -cur_signs[e2]
    
    # Choose strategy based on m
    if m <= 5:
        exhaustive_search(m)
    else:
        structured_search(m)
    
    # Final: if no valid solution found, use a simple heuristic
    if best_signs is None:
        # Try all +1 signs
        signs = np.ones(num_edges, dtype=np.int8)
        if check_connected(signs):
            best_signs = signs
        else:
            # Try all -1 signs
            signs = -np.ones(num_edges, dtype=np.int8)
            if check_connected(signs):
                best_signs = signs
            else:
                # Random search as fallback
                rng = np.random.default_rng(3)
                for _ in range(1000):
                    if time.time() > deadline - 2.0:
                        break
                    signs = rng.choice([-1, 1], size=num_edges)
                    if check_connected(signs):
                        best_signs = signs
                        break
    
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=np.int8)
    
    return best_signs.astype(np.int8)
