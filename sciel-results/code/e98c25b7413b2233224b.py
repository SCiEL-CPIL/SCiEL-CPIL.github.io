import time
import random
import numpy as np

def _build_adj_list(n, edges):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _has_k4(adj, n):
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                common = adj[u] & adj[v]
                m = common
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    m ^= lsb
                    if common & adj[w]:
                        return True
    return False

def _has_independent_6(adj, n):
    full_mask = (1 << n) - 1
    comp_adj = [0] * n
    for i in range(n):
        comp_adj[i] = full_mask ^ adj[i] ^ (1 << i)
    
    def dfs(verts, depth):
        if depth == 6:
            return True
        if not verts:
            return False
        v = verts
        while v:
            lsb = v & -v
            u = lsb.bit_length() - 1
            v ^= lsb
            new_verts = v & comp_adj[u]
            if dfs(new_verts, depth + 1):
                return True
    return dfs(full_mask, 0)

def _is_valid_graph(n, edges):
    adj = _build_adj_list(n, edges)
    if _has_k4(adj, n):
        return False
    if _has_independent_6(adj, n):
        return False
    return True

def _edges_from_adj(adj, n):
    edges = []
    for u in range(n):
        m = adj[u]
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            if u < v:
                edges.append([u, v])
    return edges

def _count_k4_np(A):
    # A is n x n 0/1 matrix
    n = A.shape[0]
    count = 0
    # For each edge (u,v), count common neighbors w, then for each w, count common neighbors of u,v,w
    # This is O(n^4) worst case but vectorized inner loops help.
    # Actually, let's use a slightly optimized loop for the specific K4 check.
    # Since n <= 40, we can afford a decently optimized Python loop with bitsets or numpy slicing.
    # Let's stick to a fast numpy check for the objective function.
    
    # K4 count = sum_{u<v<w<x} A[u,v]A[u,w]A[u,x]A[v,w]A[v,x]A[w,x]
    # We can iterate over edges (u,v) and look at common neighbors.
    edges = np.argwhere((A == 1) & (np.triu(np.ones_like(A), k=1) == 1))
    for u, v in edges:
        common = (A[u] & A[v])
        # Number of triangles in the common neighborhood
        # Triangles in subgraph induced by common neighbors
        sub = A[np.ix_(common.astype(bool), common.astype(bool))]
        # Count triangles in sub
        # This is getting complicated. Let's use a simpler penalty: just check existence?
        # No, SA needs a scalar. 
        # Let's just count K4s using a bitset approach in Python for speed during SA?
        # Or just use a simpler proxy: sum of degrees? No.
        
        # Let's go back to bitsets for the objective function, it's likely faster for n=40 in Python than numpy overhead for small matrices.
        pass
    return count

def _count_k4_bitset(adj):
    n = len(adj)
    count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                common = adj[u] & adj[v]
                # Count edges in the subgraph induced by 'common'
                # For each w in common, count edges from w to other nodes in common
                m = common
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    m ^= lsb
                    # Count neighbors of w in common
                    count += (common & adj[w]).bit_count()
    return count // 6 # Each K4 is counted 6 times (once for each of the 6 edges in the K4? No, for each edge in K4, the other 2 are in common. There are 6 edges. Each K4 has 6 edges. For each edge, we count the other 2 vertices. So each K4 is counted 6 times.)

def _count_ind6_bitset(adj):
    n = len(adj)
    full_mask = (1 << n) - 1
    comp_adj = [full_mask ^ adj[i] ^ (1 << i) for i in range(n)]
    
    count = 0
    # Count independent sets of size 6 = Cliques of size 6 in complement
    # Iterate over edges in complement?
    # Let's just do a simple DFS count with pruning, but that might be slow.
    # For n=40, counting all independent 6-sets is hard.
    # Maybe just check existence for penalty? No, SA needs gradient.
    # Let's use a heuristic: sum of degrees in complement?
    # Actually, for n=40, finding if there IS an independent 6-set is fast (DFS).
    # Counting them is slow.
    # Let's use a penalty based on the *number* of independent sets of size 6 found by a limited search?
    # Or just use 1 if exists, 0 if not? That's a step function, bad for SA.
    
    # Alternative: Use the number of "almost" independent sets?
    # Let's try counting independent sets of size 4 or 5 as a proxy?
    # Or better: Use the density of the complement.
    
    # Let's stick to a robust but slightly slower bitset count for K4, and for Ind6, 
    # we can count the number of independent sets of size 6 using a recursive count with memoization? 
    # No, n=40 is too big for memoization on subsets.
    
    # Let's use a different objective: 
    # Penalty = 100 * (num K4s) + 1 * (num Independent 6-sets)
    # If we can't count Ind6s fast, we can just check existence and use a random sample?
    
    # Let's try counting Ind6s using the same edge-iteration logic as K4 but on complement.
    # An independent set of size 6 in G is a K6 in G_bar.
    # So we just need _count_k4_bitset on the complement, but for size 6.
    
    def count_cliques_size_k(comp_adj, k):
        # Count cliques of size k
        # For k=6, this is heavy.
        # Let's just count K4s in complement (Ind4s in G) and K5s?
        # Actually, let's just use a simple heuristic for the objective:
        # Score = - (10 * num_K4s + num_Ind6s)
        # If we can't count Ind6s fast, we can approximate.
        
        # Let's just count K4s in G and check if Ind6 exists.
        # If Ind6 exists, penalty += 1.
        # This is a step function but might work with SA if we add noise.
        pass
    return 0

def _objective(adj, n):
    k4_count = _count_k4_bitset(adj)
    # Check for independent 6
    if _has_independent_6(adj, n):
        ind6_penalty = 1
    else:
        ind6_penalty = 0
    return 100 * k4_count + ind6_penalty

def _simulated_annealing(n, time_limit, seed=None):
    if seed is not None:
        random.seed(seed)
    
    start = time.time()
    deadline = start + time_limit
    
    # Initialize with a random graph
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < 0.5:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    current_obj = _objective(adj, n)
    best_adj = adj[:]
    best_obj = current_obj
    
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline and T > 0.01:
        # Pick a random edge to flip
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        
        # Flip edge
        if (adj[u] >> v) & 1:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        new_obj = _objective(adj, n)
        delta = new_obj - current_obj
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_adj = adj[:]
                if best_obj == 0:
                    break
        else:
            # Revert
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        T *= cooling
        
    if best_obj == 0:
        return _edges_from_adj(best_adj, n)
    return None

def _paley_graph(p):
    squares = {x * x % p for x in range(1, p)}
    edges = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                edges.append([i, j])
    return {'n': p, 'edges': edges}

def _clebsch_graph():
    n = 16
    vertices = []
    for i in range(16):
        bits = bin(i).count('1')
        if bits % 2 == 0:
            vertices.append(i)
    idx_map = {v: i for i, v in enumerate(vertices)}
    edges = []
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            diff = bin(vertices[i] ^ vertices[j]).count('1')
            if diff in (1, 3):
                edges.append([i, j])
    return {'n': n, 'edges': edges}

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    candidates = [
        ('paley37', _paley_graph(37)),
        ('paley29', _paley_graph(29)),
        ('paley17', _paley_graph(17)),
        ('clebsch', _clebsch_graph()),
    ]
    
    best_graph = None
    best_n = 0
    
    for name, g in candidates:
        if time.time() > deadline:
            break
        if _is_valid_graph(g['n'], g['edges']):
            if g['n'] > best_n:
                best_graph = g
                best_n = g['n']
                
    # Try Simulated Annealing for larger n
    for target_n in [35, 34, 33, 32, 30]:
        if time.time() > deadline:
            break
        if target_n <= best_n:
            continue
            
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        time_per_target = min(remaining / 2, 20)
        
        found = False
        for attempt in range(3):
            if time.time() > deadline:
                break
            edges = _simulated_annealing(target_n, time_per_target / 3, seed=100 + attempt)
            if edges is not None:
                g = {'n': target_n, 'edges': edges}
                if _is_valid_graph(target_n, edges):
                    if target_n > best_n:
                        best_graph = g
                        best_n = target_n
                    found = True
                    break
                    
        if found:
            break
            
    if best_graph is None:
        best_graph = _clebsch_graph()
        
    return best_graph
