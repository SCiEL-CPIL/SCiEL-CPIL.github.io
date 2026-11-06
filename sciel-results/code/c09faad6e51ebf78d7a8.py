import random
import itertools
import time

def construct():
    max_n = 22
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer for safety
    
    def find_regular_subgraph(adj, n):
        """
        Finds any regular induced subgraph of size >= 6.
        Returns (mask, vertices_list) or (None, None) if none found.
        """
        # Check sizes from 6 to n
        for k in range(6, n + 1):
            # Iterate combinations
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                
                # Check regularity
                # Get degrees of all vertices in mask
                # Optimization: if k is small, this is fast.
                # We can break early if degrees differ.
                
                verts = list(combo)
                d0 = (adj[verts[0]] & mask).bit_count()
                is_reg = True
                for v in verts[1:]:
                    dv = (adj[v] & mask).bit_count()
                    if dv != d0:
                        is_reg = False
                        break
                if is_reg:
                    return mask, verts
        return None, None

    def try_local_search(n, max_iters=5000):
        """
        Attempt to find a valid graph on n vertices using local search.
        Starts with a random graph and flips edges to remove regular subgraphs.
        """
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        for it in range(max_iters):
            mask, verts = find_regular_subgraph(adj, n)
            if mask is None:
                return adj
            
            # Found a violation. Flip an edge to break it.
            # Strategy: Flip a random edge within the violating subgraph.
            if len(verts) > 1:
                u, v = random.sample(verts, 2)
                # Flip edge (u, v)
                if adj[u] & (1 << v):
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            else:
                # Should not happen for k>=6
                break
        
        return None

    # Try to find a valid graph for n from max_n down to 6
    for n in range(max_n, 5, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few restarts for this n
        found = False
        max_restarts = 20
        for r in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            
            # Adjust iterations based on remaining time
            remaining = time_limit - (time.time() - start_time)
            if remaining < 5:
                break
            
            # Heuristic: smaller n is easier, so allow more iterations?
            # Or just fixed iterations per restart.
            adj = try_local_search(n, max_iters=2000)
            
            if adj is not None:
                # Verify just in case
                mask, _ = find_regular_subgraph(adj, n)
                if mask is None:
                    # Construct edge list
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
        
        # If we didn't find one for this n, try smaller n
        # But we must ensure we return *some* valid graph.
        # The loop continues to n-1.
        
    # Fallback: return a small valid graph (e.g., n=6, empty graph? No, empty graph on 6 is regular (deg 0).
    # We need a graph with NO regular induced subgraph of size >= 6.
    # If n < 6, it's trivially valid.
    # If n=6, we need a non-regular graph on 6 vertices.
    # Example: Path P6? Induced subgraphs of size 6 is just P6. Is P6 regular? No.
    # So any non-regular graph on 6 vertices is valid for n=6.
    # Let's construct a simple valid graph on n=6.
    # Vertices 0..5. Edges: (0,1), (1,2), (2,3), (3,4). (0,5) no.
    # Degrees: 1, 2, 2, 2, 1, 0. Not regular.
    # So this is a valid graph for n=6.
    
    # But we want to maximize n. If local search failed for 22, 21, ..., we should return the best found.
    # The loop above returns immediately if found.
    # If it finishes without returning, it means no valid graph was found for n > 5 within time.
    # We should return a valid graph for a small n to be safe, though the task implies we should find large ones.
    # However, the evaluator scores based on n. If we fail to find large n, we get low score.
    # Let's ensure we at least return n=6 valid graph.
    
    # Simple valid graph for n=6
    n = 6
    adj = [0] * n
    # Add edges to make it non-regular
    adj[0] |= (1 << 1)
    adj[1] |= (1 << 0) | (1 << 2)
    adj[2] |= (1 << 1) | (1 << 3)
    adj[3] |= (1 << 2) | (1 << 4)
    adj[4] |= (1 << 3)
    # Degrees: 1, 2, 2, 2, 1, 0.
    # Check all subsets of size >= 6. Only one subset: all 6.
    # It is not regular.
    # So this is valid.
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return {'n': n, 'edges': edges}
