import time
import random

def _has_independent_set_10(adj_mask, n):
    """
    Checks if there is an independent set of size 10.
    adj_mask[i] is a bitmask of neighbors of i.
    Uses branch and bound with minimum-degree-first branching.
    """
    
    def dfs(available_mask, size):
        if size >= 10:
            return True
        if available_mask == 0:
            return False
        
        count = bin(available_mask).count('1')
        if count < 10 - size:
            return False
        
        # Find vertex with minimum degree in the available subgraph
        best_v = -1
        best_deg = n + 1
        temp = available_mask
        while temp:
            v_bit = temp & -temp
            v = v_bit.bit_length() - 1
            deg_in_avail = bin(adj_mask[v] & available_mask).count('1')
            if deg_in_avail < best_deg:
                best_deg = deg_in_avail
                best_v = v
                if best_deg == 0:
                    break
            temp &= temp - 1
        
        if best_v == -1:
            return False
        
        v_bit = 1 << best_v
        
        # Include v: remove v and all its neighbors from available
        new_avail = available_mask & ~(adj_mask[best_v] | v_bit)
        if dfs(new_avail, size + 1):
            return True
            
        # Exclude v: just remove v from available
        new_avail = available_mask & ~v_bit
        if dfs(new_avail, size):
            return True
            
        return False

    return dfs((1 << n) - 1, 0)

def _find_independent_set(adj_mask, n, target_size=10):
    """
    Finds an independent set of size target_size if one exists.
    Returns the set as a list of vertex indices, or None.
    """
    result = []
    
    def dfs(available_mask, size):
        if size >= target_size:
            return True
        if available_mask == 0:
            return False
        
        count = bin(available_mask).count('1')
        if count < target_size - size:
            return False
        
        # Find vertex with minimum degree in available subgraph
        best_v = -1
        best_deg = n + 1
        temp = available_mask
        while temp:
            v_bit = temp & -temp
            v = v_bit.bit_length() - 1
            deg_in_avail = bin(adj_mask[v] & available_mask).count('1')
            if deg_in_avail < best_deg:
                best_deg = deg_in_avail
                best_v = v
                if best_deg == 0:
                    break
            temp &= temp - 1
        
        if best_v == -1:
            return False
        
        v_bit = 1 << best_v
        
        # Include v
        result.append(best_v)
        new_avail = available_mask & ~(adj_mask[best_v] | v_bit)
        if dfs(new_avail, size + 1):
            return True
        result.pop()
        
        # Exclude v
        new_avail = available_mask & ~v_bit
        if dfs(new_avail, size):
            return True
            
        return False

    if dfs((1 << n) - 1, 0):
        return result[:]
    return None

def _random_maximal_triangle_free(n, seed=None):
    """
    Generate a random maximal triangle-free graph by adding edges in random order.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    
    adj = [0] * n
    edges_to_add = []
    for i in range(n):
        for j in range(i + 1, n):
            edges_to_add.append((i, j))
    rng.shuffle(edges_to_add)
    
    for u, v in edges_to_add:
        common = adj[u] & adj[v]
        if common == 0:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    return adj

def _edge_addition_search(adj, n, deadline, max_rounds=30):
    """
    Iteratively find large independent sets and add edges within them
    (that don't create triangles) to break them.
    """
    current_adj = adj[:]
    current_has_10 = _has_independent_set_10(current_adj, n)
    
    if not current_has_10:
        return current_adj, True
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Find a large independent set
        indep_set = _find_independent_set(current_adj, n, target_size=10)
        if indep_set is None:
            indep_set = _find_independent_set(current_adj, n, target_size=9)
        
        if indep_set is None:
            if not _has_independent_set_10(current_adj, n):
                return current_adj, True
            continue
        
        # Find candidate edges to add within the independent set
        candidates = []
        for idx in range(len(indep_set)):
            for jdx in range(idx + 1, len(indep_set)):
                u = indep_set[idx]
                v = indep_set[jdx]
                if current_adj[u] & (1 << v):
                    continue
                common = current_adj[u] & current_adj[v]
                if common != 0:
                    continue
                candidates.append((u, v))
        
        if not candidates:
            # Try random edge additions
            for _ in range(15):
                if time.time() > deadline:
                    break
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                if current_adj[i] & (1 << j):
                    continue
                common = current_adj[i] & current_adj[j]
                if common != 0:
                    continue
                
                new_adj = current_adj[:]
                new_adj[i] |= (1 << j)
                new_adj[j] |= (1 << i)
                
                if not _has_independent_set_10(new_adj, n):
                    current_adj = new_adj
                    return current_adj, True
            continue
        
        # Try candidate edge additions
        random.shuffle(candidates)
        for u, v in candidates[:8]:
            if time.time() > deadline:
                break
            
            new_adj = current_adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
            if not _has_independent_set_10(new_adj, n):
                current_adj = new_adj
                return current_adj, True
    
    return current_adj, not _has_independent_set_10(current_adj, n)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit
    
    best_n = 0
    best_edges = []
    
    # Try from largest to smallest
    for target_n in [40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20]:
        if time.time() > deadline:
            break
        
        found = False
        attempts = 0
        max_attempts = 25  # Limit attempts to stay within time
        
        while attempts < max_attempts and time.time() < deadline and not found:
            attempts += 1
            
            # Generate random maximal triangle-free graph
            adj = _random_maximal_triangle_free(target_n, seed=attempts * 7919 + target_n * 13)
            
            # Check if it already works
            if not _has_independent_set_10(adj, target_n):
                best_n = target_n
                best_edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if adj[i] & (1 << j):
                            best_edges.append([i, j])
                found = True
                break
            
            # Try edge addition search to reduce independence number
            if time.time() < deadline - 8:
                adj2, success = _edge_addition_search(adj, target_n, deadline, max_rounds=20)
                if success:
                    best_n = target_n
                    best_edges = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if adj2[i] & (1 << j):
                                best_edges.append([i, j])
                    found = True
                    break
        
        if found:
            break
    
    # Fallback: ensure we return a valid graph
    if best_n == 0:
        for small_n in range(20, 10, -1):
            if time.time() > deadline:
                break
            for s in range(30):
                adj = _random_maximal_triangle_free(small_n, seed=s * 997 + small_n)
                if not _has_independent_set_10(adj, small_n):
                    best_n = small_n
                    best_edges = []
                    for i in range(small_n):
                        for j in range(i + 1, small_n):
                            if adj[i] & (1 << j):
                                best_edges.append([i, j])
                    break
            if best_n > 0:
                break
    
    # Ultimate fallback: complete graph K_n is triangle-free only if n<=2
    # Actually K_2 is triangle-free and has alpha=1 < 10
    if best_n == 0:
        best_n = 2
        best_edges = [[0, 1]]
    
    return {'n': best_n, 'edges': best_edges}
