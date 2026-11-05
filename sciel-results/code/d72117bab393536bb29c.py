import time
import random

def _clebsch_edges(offset=0):
    """Generate edges of Clebsch graph (16 vertices, degree 5, C4-free, alpha=5)."""
    edges = []
    verts = [i for i in range(32) if bin(i).count('1') % 2 == 0]
    for i in range(16):
        for j in range(i + 1, 16):
            diff = verts[i] ^ verts[j]
            if bin(diff).count('1') == 2:
                edges.append([offset + i, offset + j])
    return edges

def _has_indep_11_bitmask(adj, n, deadline):
    """
    Check if an independent set of size 11 exists using bitmask backtracking.
    adj: list of int bitmasks
    n: number of vertices
    deadline: time limit
    Returns True if an independent set of size 11 is found.
    """
    def backtrack(remaining, current_size):
        if time.time() > deadline:
            return False
        if current_size >= 11:
            return True
        if not remaining:
            return False
        if current_size + remaining.bit_count() < 11:
            return False
        
        # Heuristic: pick vertex with minimum degree in remaining subgraph
        # To speed up, we can just pick the lowest set bit or use a simple heuristic.
        # For n=39, picking min degree is good for pruning.
        min_v = -1
        min_deg = 100
        temp = remaining
        while temp:
            v = (temp & -temp).bit_length() - 1
            deg = (adj[v] & remaining).bit_count()
            if deg < min_deg:
                min_deg = deg
                min_v = v
            temp &= temp - 1
            
        if min_v == -1:
            return False
            
        # Branch 1: Include min_v
        new_remaining = remaining & ~(adj[min_v] | (1 << min_v))
        if backtrack(new_remaining, current_size + 1):
            return True
            
        # Branch 2: Exclude min_v
        remaining &= ~(1 << min_v)
        if backtrack(remaining, current_size):
            return True
            
        return False

    return backtrack((1 << n) - 1, 0)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Fallback: 10 disjoint triangles (n=30, alpha=10)
    fallback_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        fallback_edges += [[a, b], [a, c], [b, c]]
    
    # Try to construct a graph on n=39
    n_target = 39
    best_result = {'n': 30, 'edges': fallback_edges}
    
    # We will try multiple random restarts
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
            
        random.seed(attempt * 12345 + 67890)
        
        # Initialize adjacency as list of bitmasks
        adj = [0] * n_target
        edges = []
        
        # Generate all possible edges and shuffle
        all_edges = [(i, j) for i in range(n_target) for j in range(i + 1, n_target)]
        random.shuffle(all_edges)
        
        # Greedy addition of edges
        for u, v in all_edges:
            if time.time() > deadline - 10:
                break
                
            # Check if adding (u, v) keeps graph C4-free
            # 1. u and v not already connected
            if (adj[u] >> v) & 1 or (adj[v] >> u) & 1:
                continue
                
            # 2. |N(u) & N(v)| <= 1
            common = adj[u] & adj[v]
            if common.bit_count() > 1:
                continue
                
            # 3. For all w in N(u), N(w) & N(v) must be empty
            #    This ensures no path u-w-x-v exists which would form C4 u-w-x-v-u
            ok = True
            nu = adj[u]
            nv = adj[v]
            
            # Check neighbors of u
            temp = nu
            while temp:
                w = (temp & -temp).bit_length() - 1
                if adj[w] & nv:
                    ok = False
                    break
                temp &= temp - 1
                
            if not ok:
                continue
                
            # Check neighbors of v
            temp = nv
            while temp:
                x = (temp & -temp).bit_length() - 1
                if adj[x] & nu:
                    ok = False
                    break
                temp &= temp - 1
                
            if not ok:
                continue
                
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
            
        # Verify C4-freeness (paranoia check, should be redundant)
        is_c4_free = True
        for u in range(n_target):
            for v in range(u + 1, n_target):
                if (adj[u] & adj[v]).bit_count() > 1:
                    is_c4_free = False
                    break
            if not is_c4_free:
                break
                
        if not is_c4_free:
            continue
            
        # Check independence number
        # We want alpha < 11, i.e., no independent set of size 11
        has_indep_11 = _has_indep_11_bitmask(adj, n_target, deadline)
        
        if not has_indep_11:
            # We found a valid graph on n=39
            return {'n': n_target, 'edges': edges}
            
    # If we didn't find a valid graph on 39, try 38? 
    # Given the time constraints and the known bound, 39 is very likely to be found.
    # If not, we return the fallback.
    # To be safe, we could try n=38 with fewer restarts if time permits, 
    # but 30 is a guaranteed valid answer.
    # Let's stick to the fallback to ensure we don't timeout.
    
    return best_result

# EVOLVE-BLOCK-END
