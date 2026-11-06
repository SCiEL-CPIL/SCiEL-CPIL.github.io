import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline with 10s safety margin for 150s kill
    
    d = 7
    n = 1 << d  # 128 vertices
    total_edges = n * d // 2  # 448 edges
    
    # Generate all edges
    edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
    
    # Edge index lookup
    edge_index = {e: i for i, e in enumerate(edges)}
    
    # Build conflict graph: two edges conflict if they are in the same C4 face
    # A C4 face is defined by two distinct bit positions i < j and a vertex x
    # where the face vertices are x, x^(1<<i), x^(1<<j), x^(1<<i)^(1<<j)
    # The four edges of this face are:
    # (x, x^(1<<i)), (x^(1<<j), x^(1<<i)^(1<<j)), (x, x^(1<<j)), (x^(1<<i), x^(1<<i)^(1<<j))
    
    # For each edge, find all other edges that conflict with it (share a C4)
    # Two edges conflict if they are in at least one common C4 face
    
    # Precompute conflicts
    # For each edge (u,v), the dimension is the bit that differs
    def edge_dim(e):
        u, v = e
        diff = u ^ v
        return diff.bit_length() - 1  # 0-indexed bit position
    
    # For each edge, find conflicting edges
    # An edge (u,v) with dim i conflicts with edge (w,z) with dim j (j != i)
    # if there exists a C4 containing both.
    # Specifically, if edge1 = (u, u^(1<<i)) and edge2 = (w, w^(1<<j)),
    # they are in a common C4 if:
    # - The C4 is spanned by dimensions i and j
    # - The C4 vertices are {a, a^(1<<i), a^(1<<j), a^(1<<i)^(1<<j)} for some a
    # - Both edges must be among the 4 edges of this C4
    
    # More efficiently: for edge (u, v) with dim i, and another edge (w, z) with dim j:
    # They conflict if u is in {w, w^(1<<j), z^(1<<i), z^(1<<i)^(1<<j)} appropriately...
    
    # Let's use a different approach: for each pair of dimensions (i, j), 
    # enumerate all C4s spanned by i and j, and mark conflicts.
    
    # Number of C4s: for each pair (i,j), there are 2^(d-2) C4s
    # Total C4s: C(7,2) * 2^5 = 21 * 32 = 672
    
    # Build adjacency for conflict graph
    # We'll represent the conflict graph as a list of sets
    conflicts = [set() for _ in range(total_edges)]
    
    # For each pair of dimensions i < j, enumerate all C4s
    for i in range(d):
        for j in range(i + 1, d):
            # A C4 spanned by dims i and j is determined by fixing all other bits
            # The free bits are all except i and j
            mask_i = 1 << i
            mask_j = 1 << j
            
            # Iterate over all combinations of the other bits
            # The "base" vertex has bits i and j set to 0, and other bits vary
            other_bits = [k for k in range(d) if k != i and k != j]
            
            for mask_val in range(1 << len(other_bits)):
                # Construct base vertex with bits i,j = 0
                base = 0
                for idx, bit in enumerate(other_bits):
                    if mask_val & (1 << idx):
                        base |= (1 << bit)
                
                # The four vertices of the C4
                v0 = base
                v1 = base | mask_i
                v2 = base | mask_j
                v3 = base | mask_i | mask_j
                
                # The four edges of this C4
                e0 = (min(v0, v1), max(v0, v1))  # dim i
                e1 = (min(v2, v3), max(v2, v3))  # dim i
                e2 = (min(v0, v2), max(v0, v2))  # dim j
                e3 = (min(v1, v3), max(v1, v3))  # dim j
                
                # Get indices
                idx0 = edge_index[e0]
                idx1 = edge_index[e1]
                idx2 = edge_index[e2]
                idx3 = edge_index[e3]
                
                # All pairs of these 4 edges conflict
                for a in range(4):
                    for b in range(a + 1, 4):
                        idxs = [idx0, idx1, idx2, idx3]
                        conflicts[idxs[a]].add(idxs[b])
                        conflicts[idxs[b]].add(idxs[a])
    
    # Now we have the conflict graph. We want maximum independent set.
    # Use greedy + local search heuristic.
    
    best_size = 0
    best_set = set()
    
    # Try multiple random orderings
    rng = random.Random(42)
    
    def greedy_is(order):
        """Greedy independent set: process edges in given order, add if no conflict with already added."""
        selected = set()
        for idx in order:
            if not (selected & conflicts[idx]):
                selected.add(idx)
        return selected
    
    def local_search(initial_set):
        """Local search: try to improve by swapping edges."""
        current = set(initial_set)
        improved = True
        while improved:
            improved = False
            # Try adding an unselected edge
            for idx in range(total_edges):
                if idx not in current:
                    if not (current & conflicts[idx]):
                        current.add(idx)
                        improved = True
                        break
            # Try removing one edge and adding two
            if not improved:
                # Try 1-for-2 swaps
                for rem in list(current):
                    current.discard(rem)
                    # Try to add back or add new
                    best_add = None
                    for idx in range(total_edges):
                        if idx not in current:
                            if not (current & conflicts[idx]):
                                current.add(idx)
                                # Now try to add one more
                                for idx2 in range(total_edges):
                                    if idx2 not in current:
                                        if not (current & conflicts[idx2]):
                                            current.add(idx2)
                                            if len(current) > best_size if False else True:
                                                best_add = (idx, idx2)
                                            current.discard(idx2)
                                            break
                                current.discard(idx)
                                break
                    if best_add is None:
                        current.add(rem)
                    else:
                        improved = True
                        break
                else:
                    # No improvement found
                    pass
        return current
    
    # Simple local search: just try to add more edges
    def simple_local_search(initial_set):
        current = set(initial_set)
        improved = True
        while improved:
            improved = False
            for idx in range(total_edges):
                if idx not in current and not (current & conflicts[idx]):
                    current.add(idx)
                    improved = True
        return current
    
    # Run multiple random orderings
    num_trials = 0
    while time.time() < deadline - 5:
        order = list(range(total_edges))
        rng.shuffle(order)
        
        # Greedy
        is_set = greedy_is(order)
        
        # Local search to fill
        is_set = simple_local_search(is_set)
        
        if len(is_set) > best_size:
            best_size = len(is_set)
            best_set = set(is_set)
        
        num_trials += 1
    
    # Convert back to edge list
    result = [edges[i] for i in sorted(best_set)]
    return result
