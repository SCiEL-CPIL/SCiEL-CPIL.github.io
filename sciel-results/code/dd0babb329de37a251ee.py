import time
import random
import numpy as np
from collections import deque

def construct():
    """
    Construct a C4-free graph with alpha < 11, maximizing n.
    Strategy:
    1. Deterministically construct the polarity graph of PG(2, 5). This gives n=31.
    2. Verify it is C4-free and has alpha < 11.
    3. Try to extend this graph to n=32, ..., 39 by adding vertices and edges
       while maintaining C4-freeness and alpha < 11.
    4. Use a local search to maximize the number of vertices.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def time_left():
        return deadline - time.time()

    def popcount(x):
        return bin(x).count('1')
    
    def iter_bits(mask):
        """Iterate over set bit positions in mask."""
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb

    def max_independent_set_size(n, adj, time_limit=30):
        """
        Find the size of the maximum independent set using branch and bound.
        adj is a list of integers (bitmasks).
        Returns the size.
        """
        best = [0]
        start_t = time.time()
        
        def branch(candidates_mask, current_size):
            if time.time() - start_t > time_limit:
                return
            if candidates_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Upper bound
            if current_size + popcount(candidates_mask) <= best[0]:
                return
            
            # Quick check: if we can already beat best with all remaining
            remaining = popcount(candidates_mask)
            if current_size + remaining <= best[0]:
                return
            
            # Heuristic: Pick vertex with highest degree in the candidate subgraph
            best_v = -1
            best_deg = -1
            m = candidates_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = popcount(candidates_mask & adj[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                m ^= lsb
            
            if best_v == -1:
                return

            v_bit = 1 << best_v
            
            # Option 1: Include best_v (usually more productive first)
            new_candidates = candidates_mask & ~adj[best_v] & ~v_bit
            branch(new_candidates, current_size + 1)
            
            # Option 2: Exclude best_v
            branch(candidates_mask & ~v_bit, current_size)

        branch((1 << n) - 1, 0)
        return best[0]

    def is_c4_free(n, adj):
        """Check if graph is C4-free."""
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    if common & (common - 1):  # More than one bit set
                        return False
        return True

    def construct_polarity_graph():
        """
        Construct the polarity graph of PG(2, 5).
        This gives a C4-free graph on 31 vertices.
        """
        # Points in PG(2, 5)
        points = []
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    # Normalize
                    if x != 0:
                        inv_x = pow(x, -1, 5)
                        points.append((1, y * inv_x % 5, z * inv_x % 5))
                    elif y != 0:
                        inv_y = pow(y, -1, 5)
                        points.append((0, 1, z * inv_y % 5))
                    else:
                        points.append((0, 0, 1))
        points = list(set(points))
        n = len(points)
        
        # Bilinear form for polarity
        def bilinear(u, v):
            return (u[0]*v[2] + u[1]*v[1] + u[2]*v[0]) % 5
        
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if bilinear(points[i], points[j]) == 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        return n, adj

    def can_add_edge(n, adj, u, v):
        """Check if adding edge (u,v) maintains C4-freeness."""
        # u and v must not be adjacent already
        if adj[u] & (1 << v):
            return False
        
        # For C4-free: |N(u) ∩ N(v)| <= 1
        common = adj[u] & adj[v]
        if common & (common - 1):  # more than one common neighbor
            return False
        
        # Additionally, for each neighbor w of u, we need |N(w) ∩ N(v)| <= 1
        # after adding the edge. Since v gains u as neighbor, and u gains v,
        # we need to check that no C4 is created involving the new edge.
        # A C4 would be: a-b-c-d-a where one of these is the new edge (u,v).
        # So we need: for any w in N(u) (old), |N(w) ∩ N(v) ∪ {u}| <= 1
        # But since u is now in N(v), we need |N(w) ∩ N(v)| + (1 if u in N(w) else 0) <= 1
        # Since w is in N(u), u is in N(w), so we need |N(w) ∩ N(v)| <= 0
        # i.e., N(w) ∩ N(v) = empty for all w in N(u)
        
        # Similarly for each w in N(v) (old), N(w) ∩ N(u) must be empty
        
        mu = adj[u]
        while mu:
            lsb = mu & -mu
            w = lsb.bit_length() - 1
            if adj[w] & adj[v]:
                return False
            mu ^= lsb
        
        mv = adj[v]
        while mv:
            lsb = mv & -mv
            w = lsb.bit_length() - 1
            if adj[w] & adj[u]:
                return False
            mv ^= lsb
        
        return True

    def maximize_edges(n, adj, time_limit=20):
        """
        Add edges that do not create C4s to reduce alpha.
        """
        adj = adj[:]
        improved = True
        start_t = time.time()
        while improved and time.time() - start_t < time_limit:
            improved = False
            # Try edges in order of increasing common neighbors (easier to add)
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] & (1 << j)):
                        if can_add_edge(n, adj, i, j):
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                            improved = True
        return adj

    # Main construction
    n, adj = construct_polarity_graph()
    
    # Verify initial graph
    if not is_c4_free(n, adj):
        # Fallback: return empty graph
        return {'n': 1, 'edges': []}
    
    alpha = max_independent_set_size(n, adj)
    if alpha >= 11:
        # Fallback
        return {'n': 1, 'edges': []}
    
    best_n = n
    best_adj = adj[:]
    
    # Try to extend by adding vertices one at a time
    for target_n in range(n + 1, 40):
        if time_left() < 20:
            break
        
        # Try multiple strategies for adding a new vertex
        success = False
        for attempt in range(50):
            if time_left() < 5:
                break
            
            new_adj = best_adj[:]
            new_n = best_n
            new_v = new_n
            new_adj.append(0)
            
            # Strategy: choose a random subset of existing vertices to connect to
            # The subset must form an independent set in the "conflict graph"
            # where two existing vertices conflict if they share a neighbor with new_v's chosen neighbors
            
            # Simple approach: try connecting to a small random subset
            num_connections = random.randint(1, min(5, new_n))
            candidates = list(range(new_n))
            random.shuffle(candidates)
            chosen = candidates[:num_connections]
            
            valid = True
            for i in chosen:
                if not can_add_edge(new_n + 1, new_adj, new_v, i):
                    valid = False
                    break
            
            if valid:
                # Verify the whole graph is C4-free
                if is_c4_free(new_n + 1, new_adj):
                    # Check independence number
                    alpha_new = max_independent_set_size(new_n + 1, new_adj, time_limit=5)
                    if alpha_new < 11:
                        best_n = new_n + 1
                        best_adj = new_adj[:]
                        success = True
                        break
            else:
                # Retry with different random choice
                continue
        
        if not success:
            # Try a more systematic approach: connect to one vertex at a time
            new_adj = best_adj[:]
            new_n = best_n
            new_v = new_n
            new_adj.append(0)
            
            for i in range(new_n):
                if can_add_edge(new_n + 1, new_adj, new_v, i):
                    new_adj[new_v] |= (1 << i)
                    new_adj[i] |= (1 << new_v)
                    break
            
            if new_adj[new_v] != 0:
                if is_c4_free(new_n + 1, new_adj):
                    alpha_new = max_independent_set_size(new_n + 1, new_adj, time_limit=5)
                    if alpha_new < 11:
                        best_n = new_n + 1
                        best_adj = new_adj[:]
    
    # Final edge maximization
    if time_left() > 10:
        best_adj_max = maximize_edges(best_n, best_adj)
        if is_c4_free(best_n, best_adj_max):
            alpha_max = max_independent_set_size(best_n, best_adj_max, time_limit=10)
            if alpha_max < 11:
                best_adj = best_adj_max
    
    # Convert to edge list
    edge_list = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edge_list.append([i, j])

    return {'n': best_n, 'edges': edge_list}
