# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        < 0.5 -> route the demand clockwise
        >= 0.5 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one entry per demand.
    """
    n = 10
    demands = [
        (0, 5, 3),
        (1, 7, 2),
        (2, 9, 4),
        (3, 6, 1),
        (0, 4, 5),
        (2, 8, 2),
        (5, 9, 3),
        (1, 4, 2),
        (6, 0, 4),
        (3, 8, 1),
        (7, 2, 3),
        (4, 9, 2),
        (8, 1, 5),
        (0, 6, 1),
        (5, 2, 2),
    ]
    D = len(demands)
    
    # Precompute for each demand: which edges are used if clockwise, which if counter-clockwise
    # Clockwise from s to t uses edges s, s+1, ..., t-1 (mod n)
    # Counter-clockwise uses the rest.
    
    def edges_cw(s, t):
        edges = []
        if s == t:
            return list(range(n))
        e = s
        while e != t:
            edges.append(e)
            e = (e + 1) % n
        return edges
    
    def edges_ccw(s, t):
        cw_edges = set(edges_cw(s, t))
        return [e for e in range(n) if e not in cw_edges]
    
    # Precompute edge sets for both directions for each demand
    # Represent as boolean masks or lists of indices
    cw_edge_sets = []
    ccw_edge_sets = []
    weights = np.zeros(D)
    
    for i, (s, t, w) in enumerate(demands):
        cw_edge_sets.append(edges_cw(s, t))
        ccw_edge_sets.append(edges_ccw(s, t))
        weights[i] = w

    # We will enumerate all 2^D possibilities.
    # We use Gray code to ensure only one bit changes at a time,
    # allowing us to update the load vector incrementally.
    
    # Initial state: all demands routed Clockwise (bit 0)
    # Load vector
    loads = np.zeros(n)
    # Routing bits: 0 for CW, 1 for CCW
    routing_bits = np.zeros(D, dtype=np.int8)
    
    # Initialize loads with all CW
    for i in range(D):
        w = weights[i]
        for e in cw_edge_sets[i]:
            loads[e] += w
            
    best_max_load = np.max(loads)
    best_routing = routing_bits.copy()
    
    # Total iterations: 2^D
    total_iterations = 1 << D
    
    # Gray code iteration
    # g = i XOR (i >> 1)
    # To find which bit changed between g(i) and g(i+1), we can just look at the index i.
    # The bit that flips in Gray code step i is the position of the least significant set bit in i.
    # i.e., bit_index = (i & -i).bit_length() - 1
    
    for i in range(1, total_iterations):
        # Determine which demand direction to flip
        # The bit that changes in Gray code from i-1 to i is the trailing zero count of i?
        # Let's verify:
        # i=1 (1): g(0)=0, g(1)=1. Flip bit 0.
        # i=2 (10): g(1)=1, g(2)=3. Flip bit 1.
        # i=3 (11): g(2)=3, g(3)=2. Flip bit 0.
        # i=4 (100): g(3)=2, g(4)=6. Flip bit 2.
        # The bit index is the position of the lowest set bit in i.
        
        # Find lowest set bit index
        # i & -i isolates the lowest set bit.
        # bit_length() gives position + 1.
        bit_idx = (i & -i).bit_length() - 1
        
        # Flip the routing for demand bit_idx
        old_dir = routing_bits[bit_idx]
        new_dir = 1 - old_dir
        routing_bits[bit_idx] = new_dir
        
        w = weights[bit_idx]
        
        if new_dir == 1:
            # Switching from CW to CCW
            # Remove CW contribution, Add CCW contribution
            for e in cw_edge_sets[bit_idx]:
                loads[e] -= w
            for e in ccw_edge_sets[bit_idx]:
                loads[e] += w
        else:
            # Switching from CCW to CW
            # Remove CCW contribution, Add CW contribution
            for e in ccw_edge_sets[bit_idx]:
                loads[e] -= w
            for e in cw_edge_sets[bit_idx]:
                loads[e] += w
        
        current_max_load = np.max(loads)
        
        if current_max_load < best_max_load:
            best_max_load = current_max_load
            best_routing = routing_bits.copy()
            
    # Construct the final routing array
    # routing[i] >= 0.5 means CCW (bit 1)
    # routing[i] < 0.5 means CW (bit 0)
    final_routing = best_routing.astype(float)
    
    return final_routing

# EVOLVE-BLOCK-END
