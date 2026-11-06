import time
import random
from itertools import combinations

def generate_paley(q):
    """Generate the Paley graph on q vertices."""
    if q % 4 != 1:
        return None
    squares = set()
    for i in range(1, q):
        squares.add((i * i) % q)
    
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in squares:
                edges.append([i, j])
    return {'n': q, 'edges': edges}

def build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def get_bitmasks(adj, n):
    """Convert adjacency sets to bitmasks for fast intersection checks."""
    masks = [0] * n
    for u in range(n):
        mask = 0
        for v in adj[u]:
            mask |= (1 << v)
        masks[u] = mask
    return masks

def has_k4(adj, n):
    """Check if graph contains a K4 using bitmasks."""
    masks = get_bitmasks(adj, n)
    for u in range(n):
        mu = masks[u]
        # Get neighbors of u as list
        neighbors = [v for v in range(n) if v != u and (mu >> v) & 1]
        for i in range(len(neighbors)):
            v = neighbors[i]
            for j in range(i + 1, len(neighbors)):
                w = neighbors[j]
                # Check if v-w is an edge
                if (masks[v] >> w) & 1:
                    # Triangle u,v,w exists. Check for 4th vertex x in common neighbors
                    common = masks[v] & masks[w] & mu
                    # Remove u, v, w from consideration
                    common &= ~(1 << u) & ~(1 << v) & ~(1 << w)
                    if common:
                        return True
    return False

def has_independent_set(adj, n, k):
    """Check if graph contains an independent set of size k using backtracking with bitmasks."""
    if n < k:
        return False
    masks = get_bitmasks(adj, n)
    
    def backtrack(start, current_size, available_mask):
        if current_size == k:
            return True
        if start >= n:
            return False
        # Pruning: not enough vertices left
        remaining = bin(available_mask).count('1')
        if remaining < k - current_size:
            return False
        
        for v in range(start, n):
            if not ((available_mask >> v) & 1):
                continue
            # Check if v is independent from all in current set
            # v must not be adjacent to any vertex in current set
            # Since we track available_mask as vertices that can still be added,
            # if v is in available_mask, it's already known to be non-adjacent to all in current set
            new_mask = available_mask & ~(1 << v)
            if backtrack(v + 1, current_size + 1, new_mask):
                return True
            available_mask = new_mask
        return False
    
    # Initial available mask: all vertices
    initial_mask = (1 << n) - 1
    return backtrack(0, 0, initial_mask)

def find_independent_sets_size_k(adj, n, k, max_count=5000):
    """Find independent sets of size k, up to max_count."""
    if n < k:
        return []
    masks = get_bitmasks(adj, n)
    sets = []
    
    def backtrack(start, current_set, available_mask):
        if len(sets) >= max_count:
            return
        if len(current_set) == k:
            sets.append(tuple(current_set))
            return
        if start >= n:
            return
        remaining = bin(available_mask).count('1')
        if remaining < k - len(current_set):
            return
        
        for v in range(start, n):
            if not ((available_mask >> v) & 1):
                continue
            current_set.append(v)
            new_mask = available_mask & ~(1 << v)
            # For independence: v must not be adjacent to any in current_set
            # Since available_mask already excludes adjacent vertices, this is guaranteed
            backtrack(v + 1, current_set, new_mask)
            current_set.pop()
            if len(sets) >= max_count:
                return
            available_mask = new_mask
    
    initial_mask = (1 << n) - 1
    backtrack(0, [], initial_mask)
    return sets

def try_add_vertex(adj, n, deadline):
    """
    Tries to find a valid set of neighbors N for a new vertex v_new.
    Returns the set N if found, else None.
    Constraints on N:
    1. N is triangle-free in G.
    2. Non-neighbors (V \\ N) do not contain an independent set of size 5.
       Equivalent: N intersects every independent set of size 5 in G.
    """
    masks = get_bitmasks(adj, n)
    
    # Find independent sets of size 5 (limited count for performance)
    ind_sets_5 = find_independent_sets_size_k(adj, n, 5, max_count=5000)
    
    if not ind_sets_5:
        # No independent sets of size 5, so constraint 2 is always satisfied.
        # We just need a triangle-free set N.
        N = set()
        remaining = list(range(n))
        random.shuffle(remaining)
        for v in remaining:
            neigh_in_N = adj[v].intersection(N)
            triangle_formed = False
            for u in neigh_in_N:
                for w in neigh_in_N:
                    if u < w and w in adj[u]:
                        triangle_formed = True
                        break
                if triangle_formed:
                    break
            if not triangle_formed:
                N.add(v)
        return N

    # Convert ind_sets_5 to bitmasks for fast checking
    ind_set_masks = []
    for s in ind_sets_5:
        m = 0
        for v in s:
            m |= (1 << v)
        ind_set_masks.append(m)
    
    # Backtracking search for N
    # N must hit all sets in ind_set_masks
    # N must be triangle-free
    
    def is_triangle_free_with_v(N_mask, v):
        """Check if adding v to N (represented by N_mask) keeps it triangle-free."""
        # v forms a triangle with u, w in N if u-w is edge AND v-u is edge AND v-w is edge
        # i.e., there exist u, w in N such that u-w is edge and both are neighbors of v
        neigh_v_in_N = masks[v] & N_mask
        # For each pair of neighbors of v in N, check if they're connected
        # Iterate over neighbors of v in N
        bits = []
        temp = neigh_v_in_N
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            bits.append(idx)
            temp ^= lsb
        
        for i in range(len(bits)):
            u = bits[i]
            # Check if u is connected to any other vertex in bits
            for j in range(i + 1, len(bits)):
                w = bits[j]
                if (masks[u] >> w) & 1:
                    return False
        return True
    
    def backtrack(N_mask, active_indices):
        if time.time() > deadline:
            return None
        if not active_indices:
            return N_mask
        
        # Pick the most constrained active set (smallest)
        # Use a heuristic: pick random for diversity
        idx = random.choice(active_indices)
        target_mask = ind_set_masks[idx]
        
        # Try each vertex in target_mask
        candidates = []
        temp = target_mask
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            candidates.append(v)
            temp ^= lsb
        random.shuffle(candidates)
        
        for v in candidates:
            # Check if adding v keeps N triangle-free
            if not is_triangle_free_with_v(N_mask, v):
                continue
            
            new_N_mask = N_mask | (1 << v)
            
            # Update active indices: remove any set that contains v
            new_active = []
            for ai in active_indices:
                if not (ind_set_masks[ai] >> v) & 1:
                    new_active.append(ai)
            
            result = backtrack(new_N_mask, new_active)
            if result is not None:
                return result
                
        return None
    
    # Try multiple restarts
    for attempt in range(30):
        if time.time() > deadline:
            break
        all_indices = list(range(len(ind_set_masks)))
        res = backtrack(0, all_indices)
        if res is not None:
            # Convert bitmask back to set
            N = set()
            temp = res
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                N.add(v)
                temp ^= lsb
            return N
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    g17 = generate_paley(17)
    if not g17:
        return {'n': 1, 'edges': []}
        
    adj = build_adjacency(17, g17['edges'])
    n = 17
    
    if has_k4(adj, n) or has_independent_set(adj, n, 6):
        return {'n': 1, 'edges': []}
        
    while n < 35 and time.time() < deadline - 10:
        new_N = try_add_vertex(adj, n, deadline)
        
        if new_N is not None:
            adj.append(set(new_N))
            for u in new_N:
                adj[u].add(n)
            n += 1
        else:
            # Stuck - try a few more attempts with different randomization
            stuck = True
            for _ in range(5):
                if time.time() > deadline - 10:
                    break
                new_N = try_add_vertex(adj, n, deadline)
                if new_N is not None:
                    adj.append(set(new_N))
                    for u in new_N:
                        adj[u].add(n)
                    n += 1
                    stuck = False
                    break
            if stuck:
                break

    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
