import time
import random
from itertools import combinations

def _has_triangle_circulant(n, S):
    S_mod = set(s % n for s in S)
    N = set()
    for s in S_mod:
        if s != 0:
            N.add(s)
            N.add((n - s) % n)
    N_list = list(N)
    for a in N_list:
        for b in N_list:
            c = (a + b) % n
            if c in N:
                return True
    return False

def _max_independent_set_size_bitset(adj, n, k):
    full_mask = (1 << n) - 1
    non_neighbors = [0] * n
    for i in range(n):
        non_neighbors[i] = full_mask & ~adj[i] & ~(1 << i)
    
    found = [False]
    
    def dfs(candidates_mask, current_size):
        if found[0]:
            return
        if current_size == k:
            found[0] = True
            return
        if candidates_mask == 0:
            return
        if bin(candidates_mask).count('1') < k - current_size:
            return
        
        mask = candidates_mask
        best_v = None
        best_count = -1
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            nn_count = bin(candidates_mask & non_neighbors[v]).count('1')
            if nn_count > best_count:
                best_count = nn_count
                best_v = v
            mask ^= lsb
        
        if best_v is None:
            return
        
        new_cands = candidates_mask & non_neighbors[best_v] & ~(1 << best_v)
        dfs(new_cands, current_size + 1)
        
        if found[0]:
            return
        
        dfs(candidates_mask & ~(1 << best_v), current_size)
    
    dfs(full_mask, 0)
    return found[0]

def _circulant_to_bitset(n, S):
    adj = [0] * n
    S_mod = set(s % n for s in S)
    N = set()
    for s in S_mod:
        if s != 0:
            N.add(s)
            N.add((n - s) % n)
            
    for i in range(n):
        for s in N:
            j = (i + s) % n
            if j > i:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _bitset_to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def _random_greedy(n, p, deadline):
    """
    Random greedy construction:
    1. Start with empty graph.
    2. Iterate over all pairs (u, v).
    3. With probability p, add edge (u, v) if it doesn't create a triangle.
    4. After adding all edges, check independence number.
    """
    adj = [0] * n
    # Precompute for triangle check: for each u, v, check if common neighbor exists.
    # This is O(n^3) naive. 
    # Better: maintain adj as bitsets.
    # When considering edge (u, v), triangle exists if adj[u] & adj[v] != 0.
    
    for u in range(n):
        for v in range(u + 1, n):
            if time.time() > deadline:
                break
            if random.random() < p:
                # Check triangle: common neighbors of u and v
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        if time.time() > deadline:
            break
    
    if not _max_independent_set_size_bitset(adj, n, 10):
        return adj
    return None

def _local_search(adj, n, deadline, max_iter=200):
    """
    Local search to reduce independence number.
    Try flipping edges to break independent sets.
    """
    # This is complex to implement correctly and efficiently within time limits.
    # Given the constraints, we rely on the circulant search which is more systematic.
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    found_graph = None
    
    # Try n=40 first
    for target_n in [40, 39, 38]:
        if time.time() > deadline:
            break
            
        n = target_n
        candidates = list(range(1, n // 2 + 1))
        
        if n == 40:
            sizes_to_try = [2, 3, 4, 5, 6, 7, 8]
        elif n == 39:
            sizes_to_try = [3, 4, 5, 6, 7]
        else:
            sizes_to_try = [3, 4, 5, 6]
            
        for size in sizes_to_try:
            if time.time() > deadline:
                break
                
            for S in combinations(candidates, size):
                if time.time() > deadline:
                    break
                
                S_set = set(S)
                has_tri = False
                for a in S_set:
                    for b in S_set:
                        if (a + b) % n in S_set and (a + b) % n != 0:
                            has_tri = True
                            break
                        if (a - b) % n in S_set and (a - b) % n != 0:
                            has_tri = True
                            break
                    if has_tri:
                        break
                
                if has_tri:
                    continue
                
                if _has_triangle_circulant(n, S):
                    continue
                
                adj = _circulant_to_bitset(n, S)
                if not _max_independent_set_size_bitset(adj, n, 10):
                    found_graph = {'n': n, 'edges': _bitset_to_edges(adj, n)}
                    break
            
            if found_graph:
                break
        
        if found_graph:
            return found_graph

    # If circulant search fails, try random greedy for n=40 and n=39
    for target_n in [40, 39]:
        if time.time() > deadline:
            break
        
        n = target_n
        # Try different probabilities
        for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            if time.time() > deadline:
                break
            
            adj = _random_greedy(n, p, deadline)
            if adj:
                found_graph = {'n': n, 'edges': _bitset_to_edges(adj, n)}
                break
        
        if found_graph:
            return found_graph

    # Final fallback: n=9 empty graph
    return {'n': 9, 'edges': []}

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, edges={len(res['edges'])}")
