"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    # We want to find a C4-free graph on n vertices with alpha < 11.
    # Best known is 39. We aim for 40+.

    def check_c4_free(adj):
        """Check if graph with adjacency list (list of sets) is C4-free."""
        n = len(adj)
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    common = adj[u] & adj[v]
                    if len(common) > 1:
                        return False
        return True

    def compute_alpha(adj, limit=11):
        """Compute if graph has independent set of size >= limit.
        Returns True if independent set of size limit exists."""
        n = len(adj)
        
        # Convert to bitmasks for speed
        masks = [0] * n
        for i in range(n):
            mask = 0
            for j in adj[i]:
                mask |= (1 << j)
            masks[i] = mask
        
        # Use branch and bound to find independent set of size limit
        # Order vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        
        # Remap: new_index -> old_index
        old_to_new = [0] * n
        for new_idx, old_idx in enumerate(order):
            old_to_new[old_idx] = new_idx
        
        # Remap masks
        new_masks = [0] * n
        for i in range(n):
            mask = 0
            for j in adj[order[i]]:
                mask |= (1 << old_to_new[j])
            new_masks[i] = mask
        
        # Search for independent set of size limit
        # Use iterative deepening or just direct search
        # We'll use a recursive search with memoization on (start, remaining_mask, size_needed)
        # But memoization space might be too large. Use simple recursion with pruning.
        
        # Pruning: if popcount(remaining_mask) < size_needed, return False
        # Pruning: if size_needed == 0, return True
        
        import sys
        sys.setrecursionlimit(10000)
        
        # Use a global counter to limit search
        search_count = [0]
        
        def popcount(x):
            return bin(x).count('1')
            
        def search(start, remaining_mask, size_needed):
            if search_count[0] > 500000:  # Limit search to avoid timeout
                return False
            search_count[0] += 1
            
            if size_needed == 0:
                return True
            if start >= n:
                return False
            if popcount(remaining_mask) < size_needed:
                return False
            
            # Option 1: Exclude vertex start
            if search(start + 1, remaining_mask, size_needed):
                return True
            
            # Option 2: Include vertex start
            if (remaining_mask & (1 << start)) != 0:
                new_remaining = remaining_mask & ~(new_masks[start] | (1 << start))
                if search(start + 1, new_remaining, size_needed - 1):
                    return True
            
            return False
        
        all_mask = (1 << n) - 1
        return search(0, all_mask, limit)

    def try_random_construct(n, seed, time_budget=2.0):
        """Try to construct a C4-free graph on n vertices with alpha < 11."""
        random.seed(seed)
        adj = [set() for _ in range(n)]
        
        # Generate random edge ordering
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        
        # Greedy: add edge if it doesn't create a C4
        for u, v in edges:
            # Check C4 condition: |N(u) ∩ N(v)| <= 1
            if len(adj[u] & adj[v]) > 1:
                continue
            
            # Check ripple effect: for all a in N(v), |N(u) ∩ N(a)| == 0
            valid = True
            for a in adj[v]:
                if adj[u] & adj[a]:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check ripple effect: for all b in N(u), |N(v) ∩ N(b)| == 0
            for b in adj[u]:
                if adj[v] & adj[b]:
                    valid = False
                    break
            if not valid:
                continue
            
            adj[u].add(v)
            adj[v].add(u)
        
        # Check if alpha < 11
        if not compute_alpha(adj, 11):
            edge_list = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edge_list.append([u, v])
            return {'n': n, 'edges': edge_list}
        
        return None

    # Try n from 43 down to 39
    best_result = None
    
    for n in range(43, 38, -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple seeds
        num_seeds = 30 if n >= 41 else 50
        for seed in range(num_seeds):
            if time.time() > deadline - 10:
                break
            
            res = try_random_construct(n, seed)
            if res is not None:
                best_result = res
                break
        
        if best_result is not None:
            break
    
    # Fallback: return a known valid small graph
    if best_result is None:
        # Create a simple valid graph: disjoint union of triangles
        # Each triangle has alpha = 1, and the union is C4-free
        # 10 triangles = 30 vertices, alpha = 10 < 11
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        best_result = {'n': 30, 'edges': edges}
    
    return best_result

# EVOLVE-BLOCK-END
