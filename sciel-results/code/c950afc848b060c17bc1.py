import random
import itertools
import time

# Precompute Q3 properties
q3_edges = []
q3_adj = [0]*8
for i in range(8):
    adj = 0
    for j in range(8):
        if i != j and bin(i ^ j).count('1') == 1:
            adj |= (1 << j)
    q3_adj[i] = adj

def contains_q3_fast(subset, adj_bits):
    """
    Fast check if induced subgraph on 8 vertices contains Q3.
    adj_bits[v] is a bitmask of neighbors of vertex v.
    subset is a tuple/list of 8 vertex ids.
    """
    # Map local indices 0-7 to global vertices
    # Build local adjacency bitmasks
    local_adj = [0]*8
    for i in range(8):
        vi = subset[i]
        for j in range(8):
            if i != j and (adj_bits[vi] >> subset[j]) & 1:
                local_adj[i] |= (1 << j)
    
    # Check if local_adj contains Q3 as subgraph
    # Q3 vertices 0-7, edges where popcount(i^j)==1
    # Use backtracking with pruning
    mapping = [-1]*8
    
    # Degree check: all must have degree >= 3 in local graph
    for i in range(8):
        if bin(local_adj[i]).count('1') < 3:
            return False
    
    # Edge count check
    total_deg = sum(bin(local_adj[i]).count('1') for i in range(8))
    if total_deg < 24:  # 12 edges * 2
        return False
    
    # Backtracking
    def backtrack(q3_v):
        if q3_v == 8:
            return True
        # Try each unmapped local vertex
        used_mask = 0
        for u in range(q3_v):
            used_mask |= (1 << mapping[u])
        
        for s_idx in range(8):
            if (used_mask >> s_idx) & 1:
                continue
            # Check compatibility with already mapped neighbors
            valid = True
            for u in range(q3_v):
                if (q3_adj[u] >> q3_v) & 1:
                    if not ((local_adj[mapping[u]] >> s_idx) & 1):
                        valid = False
                        break
            if valid:
                mapping[q3_v] = s_idx
                if backtrack(q3_v + 1):
                    return True
                mapping[q3_v] = -1
        return False
    
    for first in range(8):
        mapping[0] = first
        if backtrack(1):
            return True
        mapping[0] = -1
    
    return False

def find_q3_subset(adj_bits, n):
    """Find any 8-vertex subset containing Q3. Returns subset or None."""
    # Random sampling first
    for _ in range(200):
        subset = random.sample(range(n), 8)
        if contains_q3_fast(subset, adj_bits):
            return subset
    
    # Full scan with early termination
    for subset in itertools.combinations(range(n), 8):
        if contains_q3_fast(subset, adj_bits):
            return list(subset)
    
    return None

def construct():
    n = 16
    total_edges = n * (n-1) // 2  # 120
    
    # Start with a moderately dense random graph
    adj_bits = [0]*n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.7:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    
    start_time = time.time()
    time_limit = 115
    
    # Phase 1: Remove edges to destroy all Q3s
    while time.time() - start_time < time_limit - 20:
        q3_sub = find_q3_subset(adj_bits, n)
        if q3_sub is None:
            break
        
        # Find all edges within this Q3 subset
        edges_in_q3 = []
        for i in range(8):
            for j in range(i+1, 8):
                if (adj_bits[q3_sub[i]] >> q3_sub[j]) & 1:
                    edges_in_q3.append((q3_sub[i], q3_sub[j]))
        
        if not edges_in_q3:
            break
        
        # Remove a random edge
        u, v = random.choice(edges_in_q3)
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)
    
    # Phase 2: Local search to add edges
    # Simulated annealing style
    current_edges = sum(bin(adj_bits[i]).count('1') for i in range(n)) // 2
    
    # Get all missing edges
    missing = []
    for i in range(n):
        for j in range(i+1, n):
            if not ((adj_bits[i] >> j) & 1):
                missing.append((i, j))
    
    random.shuffle(missing)
    
    best_edges = current_edges
    best_adj_bits = adj_bits[:]
    
    # Try adding edges one by one, checking for Q3
    for u, v in missing:
        if time.time() - start_time > time_limit - 10:
            break
        
        # Temporarily add edge
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
        
        # Check if this creates a Q3
        # Only need to check subsets containing both u and v
        creates_q3 = False
        other_vertices = [x for x in range(n) if x != u and x != v]
        
        for rest in itertools.combinations(other_vertices, 6):
            subset = list(rest) + [u, v]
            if contains_q3_fast(subset, adj_bits):
                creates_q3 = True
                break
        
        if creates_q3:
            # Revert
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
        else:
            current_edges += 1
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj_bits = adj_bits[:]
    
    # Phase 3: Local search - try swapping edges to improve
    # Try to find an edge to remove and two edges to add
    if time.time() - start_time < time_limit - 5:
        for iteration in range(50):
            if time.time() - start_time > time_limit - 3:
                break
            
            # Get current edge list
            current_edge_set = set()
            for i in range(n):
                for j in range(i+1, n):
                    if (adj_bits[i] >> j) & 1:
                        current_edge_set.add((i, j))
            
            # Try random swaps: remove one edge, add one edge
            if not current_edge_set:
                break
            
            u1, v1 = random.choice(list(current_edge_set))
            
            # Find a missing edge to add
            missing_now = []
            for i in range(n):
                for j in range(i+1, n):
                    if not ((adj_bits[i] >> j) & 1) and (i,j) != (u1,v1) and (j,i) != (u1,v1):
                        missing_now.append((i, j))
            
            if not missing_now:
                break
            
            u2, v2 = random.choice(missing_now)
            
            # Try the swap
            adj_bits[u1] &= ~(1 << v1)
            adj_bits[v1] &= ~(1 << u1)
            adj_bits[u2] |= (1 << v2)
            adj_bits[v2] |= (1 << u2)
            
            # Check for Q3
            has_q3 = find_q3_subset(adj_bits, n) is not None
            
            if has_q3:
                # Revert
                adj_bits[u1] |= (1 << v1)
                adj_bits[v1] |= (1 << u1)
                adj_bits[u2] &= ~(1 << v2)
                adj_bits[v2] &= ~(1 << u2)
            else:
                # Keep swap, count edges
                new_edges = sum(bin(adj_bits[i]).count('1') for i in range(n)) // 2
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj_bits = adj_bits[:]
    
    # Use the best solution found
    adj_bits = best_adj_bits
    
    # Final verification and cleanup
    if time.time() - start_time < time_limit - 1:
        while time.time() - start_time < time_limit - 0.5:
            q3_sub = find_q3_subset(adj_bits, n)
            if q3_sub is None:
                break
            edges_in_q3 = []
            for i in range(8):
                for j in range(i+1, 8):
                    if (adj_bits[q3_sub[i]] >> q3_sub[j]) & 1:
                        edges_in_q3.append((q3_sub[i], q3_sub[j]))
            if not edges_in_q3:
                break
            u, v = random.choice(edges_in_q3)
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
    
    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (adj_bits[i] >> j) & 1:
                edges.append([i, j])
    
    return edges

# EVOLVE-BLOCK-END
