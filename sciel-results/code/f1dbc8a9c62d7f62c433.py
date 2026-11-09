import time
import random
import sys
import math

DEADLINE = 330.0
SAFETY_MARGIN = 5.0

def _paley_graph(q):
    """Construct Paley graph on q vertices. Returns list of bitmasks."""
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return q, adj

def _cyclic_double_cover(q):
    """
    Construct a cyclic double cover using group action on Z_q × {0,1}.
    Vertex (i, p) is adjacent to (j, p') if:
      1. (j - i) mod q is a quadratic residue mod q
      2. p XOR p' = 1 (parity must flip)
    
    This creates a bipartite-like structure with algebraic symmetry.
    """
    n = 2 * q
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * n
    
    # For each pair (i, j) with i < j, check if (j-i) mod q is a residue
    for i in range(q):
        for j in range(i + 1, q):
            diff = (j - i) % q
            if diff in residues:
                # Connect (i,0)-(j,1) and (i,1)-(j,0)
                v_i0 = i * 2
                v_i1 = i * 2 + 1
                v_j0 = j * 2
                v_j1 = j * 2 + 1
                
                adj[v_i0] |= (1 << v_j1)
                adj[v_j1] |= (1 << v_i0)
                
                adj[v_i1] |= (1 << v_j0)
                adj[v_j0] |= (1 << v_i1)
    
    # Handle wrap-around: (q-1, p) to (0, p')
    # diff = (0 - (q-1)) % q = 1
    # If 1 is a residue, add edges
    if 1 in residues:
        v_qm1_0 = (q - 1) * 2
        v_qm1_1 = (q - 1) * 2 + 1
        v_0_0 = 0
        v_0_1 = 1
        
        adj[v_qm1_0] |= (1 << v_0_1)
        adj[v_0_1] |= (1 << v_qm1_0)
        
        adj[v_qm1_1] |= (1 << v_0_0)
        adj[v_0_0] |= (1 << v_qm1_1)
    
    return n, adj

def _find_violations_fast(adj, n, limit=50):
    """Find up to 'limit' K5 or I5 violations efficiently."""
    violations = []
    
    # Check K5
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                common = adj[i] & adj[j]
                mask = common
                while mask and len(violations) < limit:
                    lsb = mask & -mask
                    k = lsb.bit_length() - 1
                    mask ^= lsb
                    if k <= j:
                        continue
                    C = adj[k] & common
                    if C & (C - 1):
                        mask_c = C
                        while mask_c and len(violations) < limit:
                            lsb_c = mask_c & -mask_c
                            u = lsb_c.bit_length() - 1
                            mask_c ^= lsb_c
                            if u <= k:
                                continue
                            if (adj[u] & C):
                                candidates = adj[u] & C
                                if candidates:
                                    lsb_v = candidates & -candidates
                                    v = lsb_v.bit_length() - 1
                                    violations.append(frozenset([i, j, k, u, v]))
    
    # Check I5 (K5 in complement)
    comp_adj = [0] * n
    full_mask = (1 << n) - 1
    for i in range(n):
        comp_adj[i] = full_mask ^ (1 << i) ^ adj[i]
        
    for i in range(n):
        for j in range(i + 1, n):
            if (comp_adj[i] >> j) & 1:
                common = comp_adj[i] & comp_adj[j]
                mask = common
                while mask and len(violations) < limit:
                    lsb = mask & -mask
                    k = lsb.bit_length() - 1
                    mask ^= lsb
                    if k <= j:
                        continue
                    C = comp_adj[k] & common
                    if C & (C - 1):
                        mask_c = C
                        while mask_c and len(violations) < limit:
                            lsb_c = mask_c & -mask_c
                            u = lsb_c.bit_length() - 1
                            mask_c ^= lsb_c
                            if u <= k:
                                continue
                            if (comp_adj[u] & C):
                                candidates = comp_adj[u] & C
                                if candidates:
                                    lsb_v = candidates & -candidates
                                    v = lsb_v.bit_length() - 1
                                    violations.append(frozenset([i, j, k, u, v]))
    
    return violations

def _is_55_free(adj, n):
    """Check if graph is (5,5)-free using bitsets."""
    if n < 5:
        return True
    
    # Check K5
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                common = adj[i] & adj[j]
                mask = common
                while mask:
                    lsb = mask & -mask
                    k = lsb.bit_length() - 1
                    mask ^= lsb
                    if k <= j:
                        continue
                    C = adj[k] & common
                    if C & (C - 1):
                        mask_c = C
                        while mask_c:
                            lsb_c = mask_c & -mask_c
                            u = lsb_c.bit_length() - 1
                            mask_c ^= lsb_c
                            if (adj[u] & C):
                                return False
    
    # Check I5 (K5 in complement)
    comp_adj = [0] * n
    full_mask = (1 << n) - 1
    for i in range(n):
        comp_adj[i] = full_mask ^ (1 << i) ^ adj[i]
        
    for i in range(n):
        for j in range(i + 1, n):
            if (comp_adj[i] >> j) & 1:
                common = comp_adj[i] & comp_adj[j]
                mask = common
                while mask:
                    lsb = mask & -mask
                    k = lsb.bit_length() - 1
                    mask ^= lsb
                    if k <= j:
                        continue
                    C = comp_adj[k] & common
                    if C & (C - 1):
                        mask_c = C
                        while mask_c:
                            lsb_c = mask_c & -mask_c
                            u = lsb_c.bit_length() - 1
                            mask_c ^= lsb_c
                            if (comp_adj[u] & C):
                                return False
    return True

def _simulated_annealing_prune(adj, n, time_start):
    """
    Use simulated annealing to find a (5,5)-free subgraph with maximum vertices.
    State: set of active vertices (bitmask)
    Move: add or remove a vertex
    Cost: number of violations (K5 or I5)
    """
    deadline = time_start + (DEADLINE - SAFETY_MARGIN)
    active_mask = (1 << n) - 1  # Start with all vertices
    
    # Precompute violation count for initial state
    violations = _find_violations_fast(adj, n, limit=100)
    current_cost = len(violations)
    
    best_mask = active_mask
    best_cost = current_cost
    best_n = n
    
    T = 1.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Pick a random move: add or remove a vertex
        if random.random() < 0.5 and active_mask != (1 << n) - 1:
            # Remove a random active vertex
            inactive = ((1 << n) - 1) ^ active_mask
            if inactive:
                lsb = inactive & -inactive
                v = lsb.bit_length() - 1
                new_mask = active_mask & ~(1 << v)
        else:
            # Add a random inactive vertex
            inactive = ((1 << n) - 1) ^ active_mask
            if inactive:
                lsb = inactive & -inactive
                v = lsb.bit_length() - 1
                new_mask = active_mask | (1 << v)
            else:
                continue
        
        # Compute new cost
        new_violations = _find_violations_fast(adj, n, limit=100)
        new_cost = len(new_violations)
        
        # Accept if better, or with probability exp(-delta/T)
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            active_mask = new_mask
            current_cost = new_cost
            
            # Track best
            active_count = bin(active_mask).count('1')
            if active_count > best_n or (active_count == best_n and current_cost < best_cost):
                best_n = active_count
                best_mask = active_mask
                best_cost = current_cost
        
        T *= cooling_rate
    
    return best_n, best_mask

def _remove_vertices(adj, n, to_remove):
    """Remove a set of vertices from the graph."""
    if not to_remove:
        return n, adj
        
    to_remove_set = set(to_remove)
    new_n = n - len(to_remove_set)
    
    index_map = {}
    new_idx = 0
    for i in range(n):
        if i not in to_remove_set:
            index_map[i] = new_idx
            new_idx += 1
            
    new_adj = [0] * new_n
    for i in range(n):
        if i in to_remove_set:
            continue
        ni = index_map[i]
        mask = adj[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            mask ^= lsb
            if j in to_remove_set or i == j:
                continue
            nj = index_map[j]
            new_adj[ni] |= (1 << nj)
            
    return new_n, new_adj

def _to_edges(adj, n):
    """Convert bitset adjacency to list of (i, j) edges."""
    edges = []
    for i in range(n):
        mask = adj[i] & ~((1 << (i + 1)) - 1)
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            edges.append((i, j))
            mask ^= lsb
    return edges

def construct():
    """Construct a (5,5)-free graph. Returns (n, edges)."""
    time_start = time.time()
    
    # Strategy 1: Start with Paley graph on 37 vertices
    q, adj_paley = _paley_graph(37)
    best_n = q
    best_adj = adj_paley[:]
    
    # Verify seed
    if not _is_55_free(adj_paley, q):
        raise ValueError("Seed graph is not (5,5)-free")
        
    random.seed(42)
    
    # Strategy 2: Cyclic double cover with simulated annealing
    n_dc, adj_dc = _cyclic_double_cover(37)
    
    # Use simulated annealing to find a (5,5)-free subgraph
    n_sa, mask_sa = _simulated_annealing_prune(adj_dc, n_dc, time_start)
    
    if n_sa > best_n:
        # Extract the subgraph
        active_vertices = []
        mask = mask_sa
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            active_vertices.append(v)
            mask ^= lsb
        
        # Build new adjacency for active vertices
        index_map = {v: i for i, v in enumerate(active_vertices)}
        new_n = len(active_vertices)
        new_adj = [0] * new_n
        
        for old_i in active_vertices:
            ni = index_map[old_i]
            mask_i = adj_dc[old_i]
            while mask_i:
                lsb = mask_i & -mask_i
                old_j = lsb.bit_length() - 1
                mask_i ^= lsb
                if old_j in index_map:
                    nj = index_map[old_j]
                    new_adj[ni] |= (1 << nj)
        
        if _is_55_free(new_adj, new_n) and new_n > best_n:
            best_n = new_n
            best_adj = new_adj[:]
            print(f"Cyclic double cover pruned to {best_n} vertices", file=sys.stderr)
    
    # Strategy 3: Try to extend the best graph found so far
    current_adj = best_adj[:]
    current_n = best_n
    
    while time.time() - time_start < DEADLINE - SAFETY_MARGIN:
        # Try to add a vertex with random neighborhood
        success = False
        for attempt in range(100):
            if time.time() - time_start > DEADLINE - SAFETY_MARGIN:
                break
                
            # Random neighborhood
            S_mask = 0
            for i in range(current_n):
                if random.random() < 0.5:
                    S_mask |= (1 << i)
            
            # Add new vertex
            new_n = current_n + 1
            new_adj = current_adj[:] + [0]
            
            for i in range(current_n):
                if (S_mask >> i) & 1:
                    new_adj[i] |= (1 << new_n)
                    new_adj[new_n] |= (1 << i)
            
            if _is_55_free(new_adj, new_n):
                current_adj = new_adj
                current_n = new_n
                success = True
                
                if current_n > best_n:
                    best_n = current_n
                    best_adj = current_adj[:]
                    print(f"Extended to {current_n} vertices", file=sys.stderr)
                break
        
        if not success:
            # Try random edge flips to unblock
            improved = False
            for _ in range(50):
                if time.time() - time_start > DEADLINE - SAFETY_MARGIN:
                    break
                    
                i = random.randint(0, current_n - 1)
                j = random.randint(0, current_n - 1)
                if i == j:
                    continue
                
                bit_i = 1 << j
                bit_j = 1 << i
                current_adj[i] ^= bit_i
                current_adj[j] ^= bit_j
                
                if _is_55_free(current_adj, current_n):
                    improved = True
                    break
                else:
                    current_adj[i] ^= bit_i
                    current_adj[j] ^= bit_j
            
            if not improved:
                break
    
    edges = _to_edges(best_adj, best_n)
    return best_n, edges

if __name__ == "__main__":
    n, edges = construct()
    print(f"Constructed graph with {n} vertices and {len(edges)} edges")
