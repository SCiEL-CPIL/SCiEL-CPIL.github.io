import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin

    def compute_alpha(adj_mask, n):
        """Compute maximum independent set size using Branch and Bound with good heuristics."""
        best = [0]
        
        # Precompute adjacency in the complement for faster candidate filtering
        # In an independent set, no two vertices can be adjacent in G
        # So candidates for next vertex must be non-adjacent to chosen
        
        def dfs(candidates, size):
            if size > best[0]:
                best[0] = size
            if candidates == 0:
                return
            # Upper bound: current size + number of remaining candidates
            if size + candidates.bit_count() <= best[0]:
                return
            
            # Pick a vertex to branch on
            # Heuristic: Pick vertex with max degree in the candidate subgraph
            # This tends to prune more effectively
            c = candidates
            max_deg = -1
            chosen = -1
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                # Degree of v within candidates
                deg = (adj_mask[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    chosen = v
            
            if chosen == -1:
                return
                
            # Branch 1: Include 'chosen'
            # Remove chosen and all its neighbors from candidates
            new_candidates = candidates & ~(adj_mask[chosen] | (1 << chosen))
            dfs(new_candidates, size + 1)
            
            # Branch 2: Exclude 'chosen'
            new_candidates = candidates & ~(1 << chosen)
            dfs(new_candidates, size)

        all_candidates = (1 << n) - 1
        dfs(all_candidates, 0)
        return best[0]

    def try_find_graph(n, time_limit):
        """
        Try to find a C4-free graph on n vertices with alpha < 11.
        Uses a greedy random construction.
        """
        if time.time() > deadline:
            return None
            
        adj_mask = [0] * n
        
        for i in range(n):
            # Try to connect i to some previous vertices
            # We want to maximize degree to keep alpha low.
            # But we must respect C4-free.
            
            # Candidates: previous vertices j < i
            # We can choose a subset of candidates to connect to i.
            
            # Let's try to pick as many as possible.
            # Greedy: iterate j in random order.
            # If we can connect i to j without violating C4, do it.
            
            # The check for adding edge (i, j):
            # For all v in N(j):
            #    Check if N(i) & N(v) is non-empty. If yes, FAIL.
            # This ensures that adding (i, j) doesn't create a C4.
            
            candidates = list(range(i))
            random.shuffle(candidates)
            
            for j in candidates:
                if time.time() > deadline:
                    return None
                
                # Check if we can add (i, j)
                can_add = True
                # For all v in N(j):
                v_mask = adj_mask[j]
                while v_mask:
                    lsb = v_mask & (-v_mask)
                    v = lsb.bit_length() - 1
                    v_mask ^= lsb
                    
                    # Check if N(i) & N(v) is non-empty
                    if adj_mask[i] & adj_mask[v]:
                        can_add = False
                        break
                
                if can_add:
                    # Add edge (i, j)
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        
        # After construction, check if alpha < 11
        if time.time() > deadline:
            return None
            
        alpha = compute_alpha(adj_mask, n)
        if alpha < 11:
            return adj_mask
        else:
            return None

    # Fallback: 10 disjoint triangles (n=30, alpha=10)
    fallback_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        fallback_edges += [[a, b], [a, c], [b, c]]
    
    best_n = 30
    best_edges = fallback_edges
    
    # Try to find a larger graph
    # Start from high n and go down
    # Use multiple random seeds for diversity
    
    for n in range(43, 30, -1):
        if time.time() > deadline - 5:
            break
            
        # Try multiple random constructions with different seeds
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
                
            random.seed(attempt * 1000 + n)
            adj_mask = try_find_graph(n, 10)
            if adj_mask:
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (adj_mask[u] >> v) & 1:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
                
    return {'n': best_n, 'edges': best_edges}
