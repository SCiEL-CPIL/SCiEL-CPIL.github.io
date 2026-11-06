import time
import random
import sys
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

def has_k4(adj, n):
    """Check if graph contains a K4."""
    for u in range(n):
        neighbors = list(adj[u])
        for i in range(len(neighbors)):
            v = neighbors[i]
            for j in range(i + 1, len(neighbors)):
                w = neighbors[j]
                if w in adj[v]:
                    # u, v, w form a triangle
                    common_vw = adj[v].intersection(adj[w])
                    for x in common_vw:
                        if x in adj[u] and x != v and x != w:
                            return True
    return False

def get_independent_sets_size_k_bitmask(adj, n, k):
    """
    Return a list of independent sets of size k, represented as bitmasks.
    Uses backtracking.
    """
    if n < k:
        return []
    
    sets = []
    # Pre-compute adjacency masks for fast intersection checks?
    # adj_mask[i] is an integer where bit j is 1 if (i,j) is an edge.
    adj_mask = [0] * n
    for u in range(n):
        for v in adj[u]:
            adj_mask[u] |= (1 << v)
    
    def backtrack(start, current_mask, current_size):
        if current_size == k:
            sets.append(current_mask)
            return
        
        # Pruning: if not enough vertices left
        if n - start < k - current_size:
            return
            
        for v in range(start, n):
            # Check if v is independent with current_mask
            # v is independent with current_mask if no vertex in current_mask is adjacent to v
            # adj_mask[v] has bits for neighbors of v.
            # If adj_mask[v] & current_mask != 0, then v is connected to someone in current_mask.
            if adj_mask[v] & current_mask:
                continue
                
            new_mask = current_mask | (1 << v)
            backtrack(v + 1, new_mask, current_size + 1)
            
    backtrack(0, 0, 0)
    return sets

def try_add_vertex(adj, n, deadline, ind_sets_5_masks):
    """
    Tries to find a valid set of neighbors N for a new vertex v_new.
    Returns the set N if found, else None.
    Constraints on N:
    1. N is triangle-free in G.
    2. N intersects every independent set of size 5 in G.
       (ind_sets_5_masks is a list of bitmasks representing these sets)
    """
    if not ind_sets_5_masks:
        # No independent sets of size 5, so constraint 2 is always satisfied.
        # We just need a triangle-free set N.
        # Greedy maximal triangle-free set.
        N = set()
        remaining = list(range(n))
        random.shuffle(remaining)
        for v in remaining:
            # Check if adding v to N creates a triangle
            neigh_in_N = adj[v].intersection(N)
            triangle_formed = False
            # Check if any edge exists within neigh_in_N
            for u in neigh_in_N:
                if adj[u].intersection(neigh_in_N):
                    triangle_formed = True
                    break
            if not triangle_formed:
                N.add(v)
        return N

    # Backtracking search for N
    # State: 
    #   N_mask: bitmask of vertices in N
    #   active_sets: list of bitmasks of independent sets not yet hit by N
    
    # Pre-compute for each vertex v, the mask of independent sets it hits?
    # hit_mask[v] is a bitmask over the index of ind_sets?
    # Or simpler: just filter active_sets.
    
    # To speed up, we can order vertices.
    # Heuristic: vertices that appear in more independent sets are more "useful".
    vertex_counts = [0] * n
    for mask in ind_sets_5_masks:
        # Count bits in mask? No, we need to know which vertices are in which sets.
        # This is expensive to pre-compute.
        pass
    
    # Let's just use a randomized DFS.
    # We need to select a subset N of V such that:
    # 1. N is triangle-free.
    # 2. For all mask in ind_sets_5_masks: (N_mask & mask) != 0.
    
    # We can formulate this as a SAT problem, but let's do backtracking.
    # We will decide for each vertex whether to include it in N or not.
    # However, we don't need to decide for ALL vertices. We just need to find ANY valid N.
    # A valid N doesn't have to be maximal.
    
    # Strategy: 
    # Pick a random independent set I from ind_sets_5_masks.
    # We MUST pick at least one vertex from I to include in N.
    # So, we can branch on the vertices of I.
    
    def backtrack(N_mask, active_sets_indices):
        if not active_sets_indices:
            return N_mask
            
        if time.time() > deadline:
            return None
            
        # Pick a set to satisfy
        # Heuristic: pick the set with the fewest remaining candidates?
        # We don't have that info easily.
        # Just pick the first one.
        target_set_idx = active_sets_indices[0]
        target_mask = ind_sets_5_masks[target_set_idx]
        
        # Get vertices in target_mask
        # We need to iterate through bits of target_mask
        candidates = []
        m = target_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            v = lsb.bit_length() - 1
            candidates.append(v)
            m ^= lsb
            
        random.shuffle(candidates)
        
        for v in candidates:
            # Check if adding v to N is valid (triangle-free constraint)
            # v must not form a triangle with any edge in N.
            # i.e., for any u, w in N, if u-w is edge, v cannot be connected to both.
            # Equivalent: N \ (N \ N) ... 
            # Let's check: Does N contain an edge (u,w) such that v is connected to both u and w?
            # This is equivalent to: (adj[v] & N_mask) contains an edge in the subgraph induced by N?
            # No, it's: Let N_v = N_mask & adj_mask[v].
            # If there is an edge between any two vertices in N_v, then adding v creates a triangle.
            
            N_v_mask = N_mask & adj_mask[v]
            # Check if N_v_mask contains an edge.
            # We can check this by iterating through bits of N_v_mask and checking adjacency.
            # Or pre-compute edge masks?
            
            triangle_free = True
            mv = N_v_mask
            while mv:
                lsb = mv & -mv
                u = lsb.bit_length() - 1
                # Check if u is connected to any other vertex in N_v_mask
                # adj_mask[u] & N_v_mask & ~lsb
                if adj_mask[u] & (N_v_mask ^ lsb):
                    triangle_free = False
                    break
                mv ^= lsb
                
            if not triangle_free:
                continue
                
            new_N_mask = N_mask | (1 << v)
            
            # Update active sets: remove any set that contains v
            new_active_indices = []
            for idx in active_sets_indices:
                if not (ind_sets_5_masks[idx] & (1 << v)):
                    new_active_indices.append(idx)
            
            result = backtrack(new_N_mask, new_active_indices)
            if result is not None:
                return result
                
        return None

    # Try multiple restarts
    for _ in range(50):
        res = backtrack(0, list(range(len(ind_sets_5_masks))))
        if res is not None:
            # Convert mask back to set
            N = set()
            m = res
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                N.add(v)
                m ^= lsb
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
    
    if has_k4(adj, n) or (lambda a, n: len(get_independent_sets_size_k_bitmask(a, n, 6)) > 0)(adj, n):
        return {'n': 1, 'edges': []}
        
    while n < 35 and time.time() < deadline:
        # Check time
        if time.time() > deadline - 10:
            break
            
        # Get independent sets of size 5
        # This can be slow if n is large.
        # If n > 25, maybe skip full enumeration and use heuristic?
        # But let's try.
        
        ind_sets_5_masks = get_independent_sets_size_k_bitmask(adj, n, 5)
        
        # Optimization: if there are too many, we might timeout.
        # C(30, 5) is 142506. C(35, 5) is 324632.
        # Backtracking prunes heavily.
        
        new_N = try_add_vertex(adj, n, deadline, ind_sets_5_masks)
        
        if new_N is not None:
            # Add new vertex n with neighbors new_N
            adj.append(set(new_N))
            for u in new_N:
                adj[u].add(n)
            n += 1
        else:
            # Stuck
            break

    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
