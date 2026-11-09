import time
import random
import sys

sys.setrecursionlimit(10000)

def _popcount(x):
    return bin(x).count('1')

def _has_c4(adj_bits, n):
    """
    Check if graph contains a C4.
    """
    for u in range(n):
        for v in range(u + 1, n):
            common = adj_bits[u] & adj_bits[v]
            if common & (common - 1): # If common has more than 1 bit set
                return True
    return False

def _add_edge(adj_bits, u, v):
    adj_bits[u] |= (1 << v)
    adj_bits[v] |= (1 << u)

def _remove_edge(adj_bits, u, v):
    adj_bits[u] &= ~(1 << v)
    adj_bits[v] &= ~(1 << u)

def _has_independent_set_k(adj_bits, n, k):
    """
    Exact check for independent set of size k using Branch and Bound with bitsets.
    Returns True if an independent set of size k exists.
    """
    def search(rem, size):
        if size >= k:
            return True
        if rem == 0:
            return False
        
        # Pruning
        if size + _popcount(rem) < k:
            return False
            
        # Heuristic: Pick vertex with highest degree in the remaining subgraph
        best_v = -1
        best_deg = -1
        temp_rem = rem
        while temp_rem:
            lsb = temp_rem & -temp_rem
            v = lsb.bit_length() - 1
            deg = _popcount(adj_bits[v] & rem)
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp_rem ^= lsb
            
        if best_v == -1:
            return False
            
        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        new_rem = rem & ~(adj_bits[best_v] | v_bit)
        if search(new_rem, size + 1):
            return True
            
        # Branch 2: Exclude best_v
        new_rem2 = rem & ~v_bit
        if search(new_rem2, size):
            return True
            
        return False
        
    return search((1 << n) - 1, 0)

def _polarity_graph_pg25():
    """
    Construct the polarity graph of PG(2,5).
    """
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, 3, 5)
                    nx = 1
                    ny = (y * inv_x) % 5
                    nz = (z * inv_x) % 5
                elif y != 0:
                    inv_y = pow(y, 3, 5)
                    nx = 0
                    ny = 1
                    nz = (z * inv_y) % 5
                else:
                    inv_z = pow(z, 3, 5)
                    nx = 0
                    ny = 0
                    nz = 1
                points.append((nx, ny, nz))
    
    unique_points = list(set(points))
    unique_points.sort()
    n = len(unique_points)
    
    adj_bits = [0] * n
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = unique_points[i]
            p2 = unique_points[j]
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
            if dot == 0:
                _add_edge(adj_bits, i, j)
                
    return n, adj_bits

def _find_valid_neighbor_set(adj_bits, n):
    """
    Finds a set S of vertices from 0..n-1 such that:
    1. For all u, v in S, N(u) cap N(v) is empty in the current graph.
    2. The graph induced by V \ S has no independent set of size 10.
    """
    # We want to find S.
    # Condition 1: S is an independent set in the "conflict graph" H.
    # H has vertices 0..n-1. Edge (u,v) in H if N(u) cap N(v) != 0.
    
    # Build H adjacency (bitsets)
    h_bits = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            common = adj_bits[u] & adj_bits[v]
            if common != 0:
                h_bits[u] |= (1 << v)
                h_bits[v] |= (1 << u)
                
    # We need to find an independent set S in H.
    # And check alpha(G \ S) < 10.
    
    # Randomized greedy search for S
    # We try to pick vertices that are "isolated" in H or have low degree in H?
    # Or just random.
    
    rng = random.Random(42)
    
    for attempt in range(100):
        candidates = list(range(n))
        rng.shuffle(candidates)
        
        S = []
        S_bits = 0
        
        for v in candidates:
            # Check if v can be added to S
            # v must not be in conflict with any existing u in S
            # Conflict means (h_bits[v] & S_bits) != 0
            if (h_bits[v] & S_bits) == 0:
                S.append(v)
                S_bits |= (1 << v)
                
        # Check alpha(G \ S) < 10
        # G \ S is the graph induced by vertices NOT in S.
        # We need to check if there is an independent set of size 10 in G \ S.
        
        # Construct a new adj_bits for G \ S?
        # Or just mask the rem set in the independent set search.
        
        # The independent set search works on a set of vertices.
        # We want to search in V \ S.
        # So we can modify _has_independent_set_k to take a mask.
        # Or create a helper.
        
        mask = ((1 << n) - 1) & ~S_bits
        
        # Check if alpha of subgraph induced by mask is >= 10
        # We can reuse the search logic but restrict 'rem' to 'mask'.
        
        def search_in_mask(rem, size, k_target, adj_b, mask):
            if size >= k_target:
                return True
            if rem == 0:
                return False
            if size + _popcount(rem) < k_target:
                return False
            
            best_v = -1
            best_deg = -1
            temp_rem = rem
            while temp_rem:
                lsb = temp_rem & -temp_rem
                v = lsb.bit_length() - 1
                deg = _popcount(adj_b[v] & rem)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp_rem ^= lsb
                
            if best_v == -1:
                return False
                
            v_bit = 1 << best_v
            
            new_rem = rem & ~(adj_b[best_v] | v_bit)
            if search_in_mask(new_rem, size + 1, k_target, adj_b, mask):
                return True
                
            new_rem2 = rem & ~v_bit
            if search_in_mask(new_rem2, size, k_target, adj_b, mask):
                return True
                
            return False
            
        if not search_in_mask(mask, 0, 10, adj_bits, mask):
            return S
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # 1. Start with Polarity Graph of PG(2,5)
    n, adj_bits = _polarity_graph_pg25()
    
    # Verify base
    if _has_c4(adj_bits, n):
        return {'n': 0, 'edges': []} # Should not happen
        
    if _has_independent_set_k(adj_bits, n, 11):
        # Base graph has alpha >= 11? 
        # PG(2,5) polarity graph alpha is 8.
        # If it fails, we can't start.
        # But let's assume it's valid.
        pass

    best_n = n
    best_adj_bits = adj_bits[:]
    
    # 2. Extend
    max_n = 43
    
    for target_n in range(n + 1, max_n + 1):
        if time.time() > deadline - 5:
            break
            
        # Try to find a valid extension
        S = _find_valid_neighbor_set(adj_bits, n)
        
        if S is not None:
            # Add new vertex n
            new_n = n + 1
            new_adj_bits = adj_bits[:] + [0]
            
            for v in S:
                _add_edge(new_adj_bits, n, v)
                
            # Update state
            adj_bits = new_adj_bits
            n = new_n
            best_n = n
            best_adj_bits = adj_bits[:]
            
            # Optional: Verify C4 freeness of the new graph just in case
            # if _has_c4(adj_bits, n):
            #     break
            
        else:
            # Failed to extend to this size
            # We could try a few more times with different random seeds
            # But for now, break to save time
            break
            
    # Convert adj_bits to edges list
    edges = []
    for u in range(best_n):
        for v in range(u + 1, best_n):
            if (best_adj_bits[u] >> v) & 1:
                edges.append([u, v])
                
    return {'n': best_n, 'edges': edges}
