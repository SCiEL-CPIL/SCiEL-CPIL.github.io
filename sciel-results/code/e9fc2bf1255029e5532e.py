import random
import itertools
import time

def construct():
    """
    Constructs a graph with no regular induced subgraph of size >= 6.
    Uses incremental construction with backtracking for n=21.
    Falls back to n=20 if 21 fails.
    """
    
    def check_subsets_with_v(adj, v, n, max_k=10):
        """
        Checks if adding vertex v (with current adj) creates any regular induced subgraph
        of size k in [6, max_k] that includes v.
        Returns True if valid (no violations), False otherwise.
        """
        # Get previous vertices
        prev_verts = list(range(v))
        if len(prev_verts) < 5:
            return True
        
        # We need to check subsets S of size k such that v in S.
        # S = {v} U T, where T is a subset of prev_verts of size k-1.
        # We check k from 6 to min(max_k, v+1).
        
        for k in range(6, min(max_k, v + 1) + 1):
            t_size = k - 1
            if t_size > len(prev_verts):
                continue
                
            # Iterate over all combinations of t_size vertices from prev_verts
            for T in itertools.combinations(prev_verts, t_size):
                # Build mask for S = {v} U T
                mask = (1 << v)
                for u in T:
                    mask |= (1 << u)
                
                # Check if induced subgraph on S is regular
                # Degrees of vertices in S
                degs = []
                for u in T:
                    # Degree of u in S is count of neighbors of u in S
                    # adj[u] & mask
                    deg = (adj[u] & mask).bit_count()
                    degs.append(deg)
                
                # Degree of v in S
                deg_v = (adj[v] & mask).bit_count()
                degs.append(deg_v)
                
                # Check if all degrees are equal
                if len(set(degs)) == 1:
                    return False
        
        return True

    def try_build(n, time_limit=110):
        """
        Attempts to build a valid graph on n vertices.
        """
        start_time = time.time()
        adj = [0] * n
        
        # We build vertex by vertex.
        # For simplicity, we use a random restart strategy for the edge choices.
        # If we get stuck, we restart.
        
        for restart in range(1000):
            if time.time() - start_time > time_limit:
                break
                
            # Reset
            adj = [0] * n
            
            success = True
            for v in range(n):
                if v == 0:
                    continue
                
                # Try random edge configurations to previous vertices
                found_valid = False
                # Limit tries per vertex to avoid infinite loops
                for attempt in range(100):
                    # Generate random edges to previous vertices
                    for u in range(v):
                        if random.random() < 0.5:
                            adj[v] |= (1 << u)
                            adj[u] |= (1 << v)
                        else:
                            adj[v] &= ~(1 << u)
                            adj[u] &= ~(1 << v)
                    
                    # Check validity for this v
                    if check_subsets_with_v(adj, v, n, max_k=10):
                        found_valid = True
                        break
                    else:
                        # Revert edges to previous vertices for this v?
                        # We just overwrite in next attempt, so no need to revert explicitly if we re-randomize.
                        # But we need to clear adj[v] first?
                        # Actually, the loop above sets adj[v] and adj[u].
                        # If we fail, we just try again.
                        # However, adj[u] might have been modified.
                        # We need to be careful.
                        # Let's clear adj[v] bits for u < v before re-trying.
                        for u in range(v):
                            adj[v] &= ~(1 << u)
                            adj[u] &= ~(1 << v)
                
                if not found_valid:
                    success = False
                    break
            
            if success:
                # Final full check just in case (max_k was limited to 10)
                # For n=21, we should check up to n.
                # But 10 is a good heuristic. If it passes 10, it's likely valid.
                # Let's do a full check for the final candidate.
                if full_check(n, adj):
                    return adj
                else:
                    # It failed the full check, so this graph is invalid.
                    # Continue to next restart.
                    pass
        
        return None

    def full_check(n, adj):
        """
        Exhaustive check for any regular induced subgraph of size >= 6.
        """
        for k in range(6, n + 1):
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                
                first_deg = -1
                is_reg = True
                for v in comb:
                    deg = (adj[v] & mask).bit_count()
                    if first_deg == -1:
                        first_deg = deg
                    elif deg != first_deg:
                        is_reg = False
                        break
                
                if is_reg:
                    return False
        return True

    # Try n=22 first? Might be too hard.
    # Try n=21.
    adj_21 = try_build(21, time_limit=100)
    if adj_21 is not None:
        edges = []
        for i in range(21):
            for j in range(i + 1, 21):
                if (adj_21[i] & (1 << j)) != 0:
                    edges.append([i, j])
        return {'n': 21, 'edges': edges}

    # Try n=20
    adj_20 = try_build(20, time_limit=15)
    if adj_20 is not None:
        edges = []
        for i in range(20):
            for j in range(i + 1, 20):
                if (adj_20[i] & (1 << j)) != 0:
                    edges.append([i, j])
        return {'n': 20, 'edges': edges}

    # Fallback
    return {'n': 5, 'edges': []}
