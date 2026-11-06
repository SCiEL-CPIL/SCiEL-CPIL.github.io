import time
import random
import numpy as np

def construct():
    n = 40
    start_time = time.time()
    deadline = 148.0  # Safety margin

    # Adjacency matrix as a numpy array of uint8 for fast access
    # adj[u, v] = 1 if edge exists
    adj = np.zeros((n, n), dtype=np.uint8)
    
    # List of all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Helper to check if adding edge (u, v) creates a C6
    # A new C6 must use (u, v). So we look for a path u-x-y-z-v of length 4.
    # x in N(u), z in N(v), y in N(x) ∩ N(z).
    # u, x, y, z, v must be distinct.
    def would_create_c6(u, v):
        # Get neighbors of u and v
        # We iterate x in N(u) and z in N(v)
        # To speed up, we can use bitsets or just loops since n=40 is small.
        # However, Python loops are slow. Let's try to optimize the inner loop.
        
        # Neighbors of u
        nu = np.where(adj[u] == 1)[0]
        nv = np.where(adj[v] == 1)[0]
        
        for x in nu:
            if x == v: continue
            # Neighbors of x
            nx = np.where(adj[x] == 1)[0]
            for z in nv:
                if z == u or z == x: continue
                # Check if x and z have a common neighbor y
                # y in N(x) ∩ N(z), y != u, v, x, z
                # Intersection of nx and nz
                # nz = np.where(adj[z] == 1)[0]
                # common = np.intersect1d(nx, nz)
                # This is too slow in Python loops.
                
                # Alternative: Precompute common neighbors?
                # Or use bitsets.
                # Let's stick to a slightly optimized loop for now, or precompute neighbor lists.
                
                # Actually, let's use a different strategy for the check.
                # Count paths of length 2 between x and z.
                # If count > 0, there is a y.
                # But we need y distinct from u, v, x, z.
                # If x and z are adjacent, y can't be x or z anyway.
                # If x and z are not adjacent, y is just any common neighbor.
                
                # Let's use bitsets for speed.
                # Represent adjacency rows as integers (40 bits).
                pass
        
        # The numpy approach inside the loop might be slower than optimized Python lists if not careful.
        # Let's use Python lists for neighbors, updated incrementally.
        return False # Placeholder, will replace with list-based check

    # Better approach: Maintain neighbor lists as Python sets or lists for fast iteration.
    # But updating them on every move is O(n).
    # Given n=40, O(n) is small. The C6 check is the bottleneck.
    
    # Let's use a bitset representation for the C6 check.
    # adj_bits[u] is an integer where bit v is set if (u,v) is an edge.
    adj_bits = [0] * n
    
    def update_bit(u, v, val):
        if val:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        else:
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
    
    def check_c6(u, v):
        # Check if adding (u,v) creates a C6.
        # Path: u-x-y-z-v
        # x in N(u), z in N(v), y in N(x) ∩ N(z)
        # y != u, v, x, z
        
        nu = adj_bits[u]
        nv = adj_bits[v]
        
        # Iterate x in N(u)
        # We can iterate over set bits
        x_mask = nu & ~(1 << v) # x != v
        while x_mask:
            # Get lowest set bit
            lsb = x_mask & -x_mask
            x = lsb.bit_length() - 1
            x_mask ^= lsb
            
            # Iterate z in N(v)
            z_mask = nv & ~(1 << u) & ~(1 << x) # z != u, z != x
            while z_mask:
                lsb_z = z_mask & -z_mask
                z = lsb_z.bit_length() - 1
                z_mask ^= lsb_z
                
                # Check common neighbors of x and z
                # y in N(x) ∩ N(z)
                # y != u, v, x, z
                common = adj_bits[x] & adj_bits[z]
                # Mask out u, v, x, z
                mask = ~(1 << u) & ~(1 << v) & ~(1 << x) & ~(1 << z)
                if common & mask:
                    return True
        return False

    # Initial state: Random sparse graph to allow exploration
    # Or start with 8 K5s as a good baseline?
    # The prompt suggests a different strategy. SA from random is different.
    # But starting from a good graph is usually better for SA.
    # Let's start with the 8 K5s structure but randomize the vertex labels?
    # No, let's start with a random graph of moderate density, e.g., p=0.3
    # Edges ~ 0.3 * 780 = 234. This is likely to have many C6s.
    # We need to start with a valid graph or repair it.
    # Repairing is hard.
    # Let's start with the 8 K5s (80 edges, valid) and then run SA.
    # This is structurally different from "greedy add" because we allow removals and swaps.
    
    def init_graph():
        nonlocal adj_bits
        adj_bits = [0] * n
        edges = []
        # 8 disjoint K5s
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    u = base + a
                    v = base + b
                    update_bit(u, v, 1)
                    edges.append([u, v])
        return edges

    edges = init_graph()
    current_count = len(edges)
    best_edges = edges[:]
    best_count = current_count
    
    # SA Parameters
    T_start = 10.0
    T_end = 0.1
    alpha = 0.999 # Cooling rate
    T = T_start
    
    rng = random.Random(42)
    
    # Precompute all non-edges and edges for move generation?
    # Or just pick random u, v.
    
    while time.time() < deadline:
        # Generate a random move
        # 1. Add a random non-edge
        # 2. Remove a random edge
        # 3. Swap: remove edge e1, add edge e2
        
        move_type = rng.random()
        
        # Pick random u, v
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        if move_type < 0.5:
            # Try to add (u, v)
            if (adj_bits[u] >> v) & 1:
                continue # Already an edge
            if check_c6(u, v):
                delta = -1 # Reject, or accept with prob exp(-1/T)?
                # In SA, we usually define energy.
                # Objective: Maximize edges.
                # Energy = -count.
                # If we reject, no change.
                # If we accept a bad move (creates C6), we must fix it?
                # No, the state must always be valid (C6-free).
                # So we can only add if it doesn't create a C6.
                # This restricts the search space to valid graphs.
                # Then "removal" is always an improvement (lower energy? No, lower count).
                # Wait, if we only allow valid moves, then adding an edge is always an improvement in count.
                # So we should always add if possible?
                # Then we are back to greedy.
                # To escape local optima, we need to allow *temporary* invalid states?
                # Or allow *removal* of edges to enable *different* additions.
                # If we are at a local maximum (no more edges can be added), we must remove an edge.
                # Removing an edge decreases count. This is a "worsening" move.
                # We accept it with probability exp(-delta/T).
                
                # So:
                # If we can add (u,v) without C6: Accept (delta = -1 in energy, i.e., count +1).
                # If we cannot add: Try to remove an edge?
                
                # Let's restructure:
                # If (u,v) is not an edge:
                #   If check_c6 is False: Add it. (Always accept, count increases)
                #   If check_c6 is True: 
                #      Maybe we should remove a random edge instead?
                #      Or accept the "removal" of a different edge to open up space?
                
                # Actually, if we can't add (u,v), we do nothing for this move?
                # That leads to stagnation if we hit a local max.
                # So, if we hit a local max (no addable edges), we must force a removal.
                
                # How to detect local max?
                # We can track if we've tried many adds and failed.
                # Or, simply: if we pick a non-edge that can't be added, we instead pick a random edge to remove?
                
                # Let's try: 
                # If (u,v) is non-edge:
                #   If valid to add: Add.
                #   Else: Remove a random existing edge (u2, v2).
                # This ensures we move out of local optima.
                
                # But removing a random edge might break a lot of potential.
                # Maybe remove an edge that is "critical"?
                # For now, random removal.
                
                # Pick a random existing edge to remove
                # We need a list of current edges for this.
                # Maintaining a list of edges is O(1) update.
                
                # Let's maintain a list of edges.
                pass
        else:
            # Explicit removal
            # Pick a random edge from current list
            pass

    # This implementation needs a list of edges to pick from.
    # Let's rewrite with a proper edge list.
    
    return best_edges

# Corrected Implementation
def construct():
    n = 40
    start_time = time.time()
    deadline = 148.0

    adj_bits = [0] * n
    
    def update_bit(u, v, val):
        if val:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        else:
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
            
    def check_c6(u, v):
        nu = adj_bits[u]
        nv = adj_bits[v]
        
        x_mask = nu & ~(1 << v)
        while x_mask:
            lsb = x_mask & -x_mask
            x = lsb.bit_length() - 1
            x_mask ^= lsb
            
            z_mask = nv & ~(1 << u) & ~(1 << x)
            while z_mask:
                lsb_z = z_mask & -z_mask
                z = lsb_z.bit_length() - 1
                z_mask ^= lsb_z
                
                common = adj_bits[x] & adj_bits[z]
                mask = ~(1 << u) & ~(1 << v) & ~(1 << x) & ~(1 << z)
                if common & mask:
                    return True
        return False

    # Initialize with 8 K5s
    edges = []
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u = base + a
                v = base + b
                update_bit(u, v, 1)
                edges.append((u, v))
                
    current_count = len(edges)
    best_edges = [e[:] for e in edges] # Store as lists
    best_count = current_count
    
    T = 10.0
    alpha = 0.9995
    rng = random.Random(42)
    
    # Edge set for O(1) lookup
    edge_set = set(edges)
    
    while time.time() < deadline:
        # Pick a random move
        # 70% chance to try add, 30% chance to remove
        
        if rng.random() < 0.7:
            # Try add
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v: continue
            if u > v: u, v = v, u
            
            if (u, v) in edge_set:
                continue
                
            if not check_c6(u, v):
                # Add
                update_bit(u, v, 1)
                edges.append((u, v))
                edge_set.add((u, v))
                current_count += 1
                
                if current_count > best_count:
                    best_count = current_count
                    best_edges = [list(e) for e in edges]
            else:
                # Cannot add. Do nothing or trigger removal?
                # To escape local optima, if we fail to add, we might force a removal.
                # But that's aggressive. Let's just let the removal probability handle it.
                pass
        else:
            # Remove
            if not edges: continue
            idx = rng.randint(0, len(edges) - 1)
            u, v = edges[idx]
            
            # Calculate delta
            # Removing an edge decreases count by 1.
            # Energy change = +1 (worse).
            # Prob = exp(-1/T)
            
            # But wait, if we are in a local maximum, we MUST remove to progress.
            # If we are not, removing is bad.
            # SA will handle this.
            
            # Acceptance probability
            if rng.random() < np.exp(-1.0 / T):
                # Accept removal
                update_bit(u, v, 0)
                edges.pop(idx)
                edge_set.remove((u, v))
                current_count -= 1
                
        T *= alpha
        if T < 0.01:
            T = 0.01

    return best_edges
