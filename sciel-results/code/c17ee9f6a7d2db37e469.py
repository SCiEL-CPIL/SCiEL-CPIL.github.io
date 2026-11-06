"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
import time
import sys

N = 64
NUM_POINTS = N * N

# EVOLVE-BLOCK-START

def construct():
    """
    Find a large set of grid points in [0, N)^2 with no three points forming an isosceles triangle.
    Uses bitset-accelerated greedy and local search on a precomputed conflict graph.
    """
    deadline = time.time() + 290  # Leave margin before 300s limit
    
    # Pre-compute all points
    points = np.array([(x, y) for y in range(N) for x in range(N)], dtype=np.int32)
    xs = points[:, 0]
    ys = points[:, 1]
    
    # Pre-compute squared distances between all pairs
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    D2 = dx * dx + dy * dy  # (NUM_POINTS, NUM_POINTS)
    
    # Build conflict graph using bitsets (Python integers)
    # conflict[u] is an integer with bit v set if u and v conflict
    conflict = [0] * NUM_POINTS
    
    # Helper to get bitset from list of indices
    def list_to_bitset(lst):
        b = 0
        for idx in lst:
            b |= (1 << idx)
        return b

    # Precompute distance groups for each point
    # dist_groups[u][d] = list of points v such that D2[u,v] == d
    dist_groups = [None] * NUM_POINTS
    
    for u in range(NUM_POINTS):
        row = D2[u, :]
        # Exclude self
        row[u] = -1 
        unique_d, inverse = np.unique(row, return_inverse=True)
        groups = {}
        for d_idx, d_val in enumerate(unique_d):
            if d_val == -1:
                continue
            pts_at_d = np.where(inverse == d_idx)[0]
            groups[int(d_val)] = pts_at_d
        dist_groups[u] = groups

    # Build conflict graph
    # Case 1: Vertex at u. u, v, w form isosceles if d(u,v) == d(u,w).
    # So v and w conflict.
    for u in range(NUM_POINTS):
        for d, pts in dist_groups[u].items():
            if len(pts) < 2:
                continue
            bitset = list_to_bitset(pts)
            for v in pts:
                # v conflicts with all other points in pts
                # conflict[v] |= bitset ^ (1 << v)
                conflict[v] |= (bitset ^ (1 << v))

    # Case 2: Vertex at v. u, v, w form isosceles if d(u,v) == d(v,w).
    # So u and w conflict.
    for u in range(NUM_POINTS):
        for d, pts in dist_groups[u].items():
            for v in pts:
                # v is at distance d from u
                # We need w such that d(v,w) == d
                # w is in dist_groups[v][d]
                if d in dist_groups[v]:
                    ws = dist_groups[v][d]
                    for w in ws:
                        if w == u:
                            continue
                        # u and w conflict
                        conflict[u] |= (1 << w)
                        conflict[w] |= (1 << u)

    # Local Search with Bitsets
    all_bits = (1 << NUM_POINTS) - 1
    
    best_size = 0
    best_bits = 0
    
    random.seed(42)
    np.random.seed(42)
    
    # Greedy Initialization
    def greedy_init():
        available = all_bits
        current_bits = 0
        # List of indices for random selection
        # We can just pick random bits from available
        while available:
            # Pick a random set bit
            # Efficient way: generate random number, find set bit
            # Or just iterate? No, N=4096, list is fine.
            # Let's use a list of available indices for O(1) removal? 
            # Bitset removal is fast.
            
            # Simple random pick from available
            # Convert available to list? No, too slow.
            # Use random index and check if bit is set?
            # Better: maintain a list of available indices.
            
            pass
        
        # Let's implement a simpler greedy:
        # Order points by random permutation
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        
        current_bits = 0
        for p in order:
            if (1 << p) & available:
                # Check if p can be added
                # p can be added if conflict[p] & current_bits == 0
                if (conflict[p] & current_bits) == 0:
                    current_bits |= (1 << p)
                    available &= ~(1 << p)
                    # Remove neighbors of p from available
                    available &= ~conflict[p]
        return current_bits

    # Run multiple greedy inits
    for _ in range(50):
        if time.time() > deadline - 30:
            break
        bits = greedy_init()
        size = bits.bit_count()
        if size > best_size:
            best_size = size
            best_bits = bits

    # Local Search
    def local_search(bits):
        current_bits = bits
        current_size = current_bits.bit_count()
        T = 1.0
        cooling = 0.9995
        min_T = 0.01
        
        while time.time() < deadline - 10 and T > min_T:
            # Pick a move
            move = random.random()
            
            if move < 0.4:
                # Add a point
                # Find unselected points
                unselected = all_bits & ~current_bits
                if unselected:
                    # Pick random unselected
                    # Generate random index
                    p = random.randint(0, NUM_POINTS - 1)
                    if (1 << p) & unselected:
                        if (conflict[p] & current_bits) == 0:
                            current_bits |= (1 << p)
                            current_size += 1
                            if current_size > best_size:
                                best_size = current_size
                                best_bits = current_bits
            elif move < 0.7:
                # Remove a point
                if current_bits:
                    p = random.randint(0, NUM_POINTS - 1)
                    if (1 << p) & current_bits:
                        current_bits &= ~(1 << p)
                        current_size -= 1
            else:
                # Remove 1, Add 2
                if current_bits:
                    p = random.randint(0, NUM_POINTS - 1)
                    if (1 << p) & current_bits:
                        # Tentatively remove p
                        new_bits = current_bits & ~(1 << p)
                        new_size = current_size - 1
                        
                        # Try to add two points
                        added = 0
                        for _ in range(2):
                            q = random.randint(0, NUM_POINTS - 1)
                            if (1 << q) & ~new_bits:
                                if (conflict[q] & new_bits) == 0:
                                    new_bits |= (1 << q)
                                    new_size += 1
                                    added += 1
                        
                        if added > 0:
                            # Accept if size increased or same (with prob)
                            if new_size > current_size or (new_size == current_size and random.random() < 0.5):
                                current_bits = new_bits
                                current_size = new_size
                                if current_size > best_size:
                                    best_size = current_size
                                    best_bits = current_bits
                        # If added == 0, revert (do nothing)
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return current_bits, current_size

    # Run local search from best solution
    # And from a few random restarts
    for _ in range(20):
        if time.time() > deadline - 20:
            break
        # Start from best or random
        if random.random() < 0.5:
            start_bits = best_bits
        else:
            start_bits = greedy_init()
        
        bits, size = local_search(start_bits)
        if size > best_size:
            best_size = size
            best_bits = bits

    # Convert best_bits to list of points
    result = []
    for i in range(NUM_POINTS):
        if (best_bits >> i) & 1:
            result.append((int(points[i, 0]), int(points[i, 1])))
    
    return result

# EVOLVE-BLOCK-END
