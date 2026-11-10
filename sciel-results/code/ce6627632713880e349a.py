"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 115  # safety margin before 150s kill
    n = 40
    
    # Build adjacency as bitmasks for fast operations
    adj_mask = [0] * n
    
    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
    
    def neighbors_mask(u):
        return adj_mask[u]
    
    def degree(u):
        return bin(adj_mask[u]).count('1')
    
    def total_edges():
        return sum(degree(u) for u in range(n)) // 2
    
    def has_c6_fast(u, v):
        """Check if adding edge (u,v) creates a C6 using bitmasks."""
        nu = adj_mask[u] & ~(1 << v)  # neighbors of u except v
        nv = adj_mask[v] & ~(1 << u)  # neighbors of v except u
        
        a = nu
        while a:
            la = a & (-a)
            a ^= la
            ai = la.bit_length() - 1
            
            nb = adj_mask[ai] & ~(1 << u) & ~(1 << v) & ~(1 << ai)
            b = nb
            while b:
                lb = b & (-b)
                b ^= lb
                bi = lb.bit_length() - 1
                
                nc = adj_mask[bi] & ~(1 << u) & ~(1 << v) & ~(1 << ai) & ~(1 << bi)
                c = nc
                while c:
                    lc = c & (-c)
                    c ^= lc
                    ci = lc.bit_length() - 1
                    
                    nd = adj_mask[ci] & nv & ~(1 << u) & ~(1 << ai) & ~(1 << bi) & ~(1 << ci)
                    if nd:
                        return True
        
        return False
    
    def spectral_gap_score(u, v):
        """Compute a heuristic score based on spectral gap for adding edge (u,v).
        
        We approximate the effect on spectral gap by looking at:
        1. Current degrees of u and v (lower degree = more "room" to add)
        2. Number of common neighbors (more common neighbors = more clustering = potentially worse for C6 avoidance)
        3. Distance-based heuristic: vertices farther apart in the graph are better candidates
        
        This is a fast surrogate that avoids full eigendecomposition.
        """
        deg_u = degree(u)
        deg_v = degree(v)
        
        # Common neighbors
        common = adj_mask[u] & adj_mask[v]
        common_count = bin(common).count('1')
        
        # Score: prefer edges between low-degree vertices with few common neighbors
        # The spectral gap tends to increase when we connect sparse parts of the graph
        base_score = (n - deg_u - deg_v) * 2.0 - common_count * 3.0
        
        # Bonus for connecting different "clusters" (heuristic: different K5 blocks)
        block_u = u // 5
        block_v = v // 5
        if block_u != block_v:
            base_score += 5.0
        
        return base_score
    
    def compute_spectral_gap_adj_matrix():
        """Compute the actual spectral gap from the current adjacency matrix."""
        A = np.zeros((n, n))
        for u in range(n):
            for v in range(u + 1, n):
                if (1 << v) & adj_mask[u]:
                    A[u, v] = 1.0
                    A[v, u] = 1.0
        eigvals = np.linalg.eigvalsh(A)
        eigvals = eigvals[::-1]  # sort descending
        if len(eigvals) < 2:
            return 0.0
        return eigvals[0] - eigvals[1]
    
    # Start with 8 disjoint K5s
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                add_edge(5 * k + a, 5 * k + b)
    
    # Track best state
    best_adj_mask = list(adj_mask)
    best_count = total_edges()
    
    # All possible pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Spectral-guided greedy addition
    for iteration in range(150):
        if time.time() > deadline - 5:
            break
        
        # Get all missing edges and score them
        missing_edges = [(u, v) for (u, v) in all_pairs if not ((1 << v) & adj_mask[u])]
        
        if not missing_edges:
            break
        
        # Score each missing edge using the spectral gap heuristic
        scored_edges = []
        for (u, v) in missing_edges:
            score = spectral_gap_score(u, v)
            scored_edges.append((score, u, v))
        
        # Sort by score descending (highest score first)
        scored_edges.sort(key=lambda x: -x[0])
        
        added_any = False
        for (score, u, v) in scored_edges:
            if time.time() > deadline - 3:
                break
            if (1 << v) & adj_mask[u]:
                continue
            if not has_c6_fast(u, v):
                add_edge(u, v)
                added_any = True
        
        current_count = total_edges()
        if current_count > best_count:
            best_count = current_count
            best_adj_mask = list(adj_mask)
        
        if not added_any:
            # Try perturbation: remove a random edge and try to re-add different ones
            if time.time() > deadline - 3:
                break
            all_current_edges = [(u, v) for u in range(n) for v in range(u+1, n) if (1 << v) & adj_mask[u]]
            if all_current_edges:
                ru, rv = random.choice(all_current_edges)
                remove_edge(ru, rv)
                
                # Re-score and try to add edges
                missing_edges = [(u, v) for (u, v) in all_pairs if not ((1 << v) & adj_mask[u])]
                scored_edges = []
                for (u, v) in missing_edges:
                    score = spectral_gap_score(u, v)
                    scored_edges.append((score, u, v))
                scored_edges.sort(key=lambda x: -x[0])
                
                for (score, u, v) in scored_edges:
                    if time.time() > deadline - 3:
                        break
                    if (1 << v) & adj_mask[u]:
                        continue
                    if not has_c6_fast(u, v):
                        add_edge(u, v)
                
                current_count = total_edges()
                if current_count > best_count:
                    best_count = current_count
                    best_adj_mask = list(adj_mask)
            else:
                break
    
    # Final polish: try to add any remaining edges in random order
    if time.time() < deadline - 2:
        random.shuffle(all_pairs)
        for (u, v) in all_pairs:
            if time.time() > deadline - 1:
                break
            if (1 << v) & adj_mask[u]:
                continue
            if not has_c6_fast(u, v):
                add_edge(u, v)
        
        current_count = total_edges()
        if current_count > best_count:
            best_count = current_count
            best_adj_mask = list(adj_mask)
    
    # Build final edge list from the best state
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (1 << v) & best_adj_mask[u]:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
