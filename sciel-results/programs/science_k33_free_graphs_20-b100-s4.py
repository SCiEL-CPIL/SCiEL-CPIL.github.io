import time
import numpy as np
from itertools import combinations

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 125.0  # Safety margin
    
    # Precompute all triples
    triples = list(combinations(range(n), 3))
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_index = {}
    for idx, (i, j) in enumerate(all_pairs):
        pair_index[(i, j)] = idx
    
    def get_neighbors(adj_masks, v):
        """Get list of neighbors of vertex v."""
        mask = adj_masks[v]
        nbrs = []
        for x in range(n):
            if x != v and (mask >> x) & 1:
                nbrs.append(x)
        return nbrs
    
    def can_add(adj_masks, u, v):
        """
        Check if adding edge (u,v) creates a K3,3.
        
        A K3,3 violation involving new edge (u,v) means:
        - There exist vertices a,b such that v is a common neighbor of a,b,
          and |N(u) ∩ N(a) ∩ N(b)| >= 2 (so after adding (u,v), v becomes
          a 3rd common neighbor of {u,a,b}... wait, no.
        
        Actually: after adding (u,v), a K3,3 exists iff there are 3 vertices 
        {x,y,z} with 3 common neighbors. The new edge (u,v) can only create
        new K3,3s where u and v are in different parts of the bipartition.
        
        Case A: u is in Part A, v is in Part B.
        Part A = {u, a, b}, Part B = {v, c, d}
        Need: v ∈ N(a) ∩ N(b) (i.e., a and b both adjacent to v)
        And: |N(u) ∩ N(a) ∩ N(b)| >= 2 before adding (u,v)
        Because after adding, v joins N(u), making v a common neighbor of {u,a,b}
        if v ∈ N(a) and v ∈ N(b).
        So total common neighbors of {u,a,b} = old_count + 1 (for v)
        We need this < 3, i.e., old_count + 1 < 3, i.e., old_count <= 1.
        Violation if old_count >= 2 AND v ∈ N(a) AND v ∈ N(b).
        
        Case B: v is in Part A, u is in Part B.
        Part A = {v, a, b}, Part B = {u, c, d}
        Need: u ∈ N(a) ∩ N(b)
        And: |N(v) ∩ N(a) ∩ N(b)| >= 2 before adding (u,v)
        Violation if old_count >= 2 AND u ∈ N(a) AND u ∈ N(b).
        """
        # Case A: check pairs (a,b) where v ∈ N(a) ∩ N(b), a,b ≠ u
        nbrs_v = get_neighbors(adj_masks, v)
        for i in range(len(nbrs_v)):
            a = nbrs_v[i]
            if a == u:
                continue
            for j in range(i + 1, len(nbrs_v)):
                b = nbrs_v[j]
                if b == u:
                    continue
                # v is in N(a) and N(b) by construction
                common = adj_masks[u] & adj_masks[a] & adj_masks[b]
                if common.bit_count() >= 2:
                    return False
        
        # Case B: check pairs (a,b) where u ∈ N(a) ∩ N(b), a,b ≠ v
        nbrs_u = get_neighbors(adj_masks, u)
        for i in range(len(nbrs_u)):
            a = nbrs_u[i]
            if a == v:
                continue
            for j in range(i + 1, len(nbrs_u)):
                b = nbrs_u[j]
                if b == v:
                    continue
                # u is in N(a) and N(b) by construction
                common = adj_masks[v] & adj_masks[a] & adj_masks[b]
                if common.bit_count() >= 2:
                    return False
        
        return True
    
    def count_violations(adj_masks):
        """Count number of K3,3 violations (triples with >= 3 common neighbors)."""
        count = 0
        for i, j, k in triples:
            common = adj_masks[i] & adj_masks[j] & adj_masks[k]
            if common.bit_count() >= 3:
                count += 1
        return count
    
    def is_valid(adj_masks):
        """Check if graph is K3,3-free."""
        for i, j, k in triples:
            common = adj_masks[i] & adj_masks[j] & adj_masks[k]
            if common.bit_count() >= 3:
                return False
        return True
    
    def clean_graph(adj_masks, rng):
        """Remove edges until graph is K3,3-free. Returns number of edges removed."""
        removed = 0
        max_iters = 1000
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            # Find a violating triple
            found = False
            for i, j, k in triples:
                common = adj_masks[i] & adj_masks[j] & adj_masks[k]
                if common.bit_count() >= 3:
                    # Get common neighbors
                    nbrs = []
                    for x in range(n):
                        if (common >> x) & 1:
                            nbrs.append(x)
                    # Remove a random edge from the K3,3
                    # The K3,3 has edges from {i,j,k} to nbrs
                    # Pick a random edge to remove
                    part_a = [i, j, k]
                    for _try in range(30):
                        a = part_a[rng.integers(0, 3)]
                        b = nbrs[rng.integers(0, len(nbrs))]
                        if (adj_masks[a] >> b) & 1:
                            adj_masks[a] &= ~(1 << b)
                            adj_masks[b] &= ~(1 << a)
                            removed += 1
                            break
                    else:
                        # Fallback
                        a = i
                        b = nbrs[0]
                        adj_masks[a] &= ~(1 << b)
                        adj_masks[b] &= ~(1 << a)
                        removed += 1
                    found = True
                    break
            if not found:
                break
        return removed
    
    def greedy_add(adj_masks, rng, deadline_check=True):
        """Greedily add edges in random order."""
        added = 0
        missing = [(i, j) for i in range(n) for j in range(i+1, n) 
                   if not ((adj_masks[i] >> j) & 1)]
        rng.shuffle(missing)
        
        for u, v in missing:
            if deadline_check and time.time() > deadline - 3:
                break
            if can_add(adj_masks, u, v):
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                added += 1
        return added
    
    def local_search(adj_masks, rng, max_rounds=30):
        """
        Local search: try 1-for-1 and 1-for-2 edge swaps.
        Returns number of improvements made.
        """
        improvements = 0
        
        for rnd in range(max_rounds):
            if time.time() > deadline - 5:
                break
            
            round_improved = False
            edges = [(i, j) for i in range(n) for j in range(i+1, n) 
                    if (adj_masks[i] >> j) & 1]
            rng.shuffle(edges)
            
            for u, v in edges:
                if time.time() > deadline - 5:
                    break
                
                # Remove edge (u,v)
                adj_masks[u] &= ~(1 << v)
                adj_masks[v] &= ~(1 << u)
                
                # Try to add 1 or 2 edges
                missing = [(i, j) for i in range(n) for j in range(i+1, n)
                          if not ((adj_masks[i] >> j) & 1)]
                rng.shuffle(missing)
                
                added_edges = []
                for a, b in missing:
                    if len(added_edges) >= 2:
                        break
                    if can_add(adj_masks, a, b):
                        adj_masks[a] |= (1 << b)
                        adj_masks[b] |= (1 << a)
                        added_edges.append((a, b))
                
                if len(added_edges) > 1:
                    # Net gain
                    improvements += len(added_edges) - 1
                    round_improved = True
                elif len(added_edges) == 1:
                    # Same count, keep the new edge (might help later)
                    round_improved = True
                    improvements += 0
                else:
                    # Restore original edge
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
            
            if not round_improved:
                break
        
        return improvements
    
    def edge_count(adj_masks):
        return sum((adj_masks[i] >> j) & 1 for i in range(n) for j in range(i+1, n))
    
    best_adj = [0] * n
    best_count = 0
    
    rng = np.random.default_rng(42)
    
    num_restarts = 500
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Random initialization
        p = 0.3 + 0.4 * rng.random()
        adj_masks = [0] * n
        
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < p:
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
        
        # Clean
        clean_graph(adj_masks, rng)
        
        # Greedy add
        greedy_add(adj_masks, rng)
        
        # Local search
        local_search(adj_masks, rng, max_rounds=20)
        
        # Second greedy pass
        greedy_add(adj_masks, rng)
        
        # Final local search
        local_search(adj_masks, rng, max_rounds=10)
        
        cur_count = edge_count(adj_masks)
        
        if cur_count > best_count:
            best_count = cur_count
            best_adj = adj_masks[:]
            
            if best_count >= 58:
                if time.time() > deadline - 30:
                    break
    
    # Extract edges from best_adj
    best_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                best_edges.append([i, j])
    
    return best_edges
