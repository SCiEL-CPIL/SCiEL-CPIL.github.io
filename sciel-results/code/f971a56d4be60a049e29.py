import random
import time
import math

def _neighbours(point):
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _grow(sequence, dimension, rng):
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _calc_delta_e_pivot(sequence, chain, where, i, new_seg):
    # i is the index of the first moved residue.
    # new_seg is the list of new positions for chain[i:]
    # We need to compare cross-boundary contacts.
    # Old contacts: pairs (u, v) where u < i, v >= i, dist(u,v)=1
    # New contacts: pairs (u, v) where u < i, v >= i, dist(u, new_v)=1
    
    # Note: Internal contacts within v>=i are preserved.
    # Internal contacts within u<i are preserved.
    
    old_cross = 0
    new_cross = 0
    
    # Helper to check if two positions are neighbors
    def is_neighbor(p1, p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2]) == 1

    # Iterate over fixed part u < i
    for u_idx in range(i):
        u_pos = chain[u_idx]
        if sequence[u_idx] != 'H': continue
        
        # Check old contacts with v >= i
        for v_idx in range(i, len(chain)):
            v_pos = chain[v_idx]
            if sequence[v_idx] != 'H': continue
            if is_neighbor(u_pos, v_pos):
                old_cross += 1
                
        # Check new contacts with v >= i
        for k, v_new_pos in enumerate(new_seg):
            v_idx = i + k
            if sequence[v_idx] != 'H': continue
            if is_neighbor(u_pos, v_new_pos):
                new_cross += 1
                
    return -(new_cross - old_cross) # Energy is -contacts

def fold(sequence, dimension):
    N = len(sequence)
    if N == 0: return []
    
    rng = random.Random(12345 + N)
    
    # 1. Initialize
    chain = _grow(sequence, dimension, rng)
    if chain is None:
        # Fallback: straight line (rare for H/P but possible)
        chain = [tuple([i] + [0]*(dimension-1)) for i in range(N)]
        
    occupied = set(chain)
    where = {p: i for i, p in enumerate(chain)}
    current_e = _energy(sequence, chain)
    best_e = current_e
    best_chain = list(chain)
    
    start_time = time.time()
    deadline = start_time + 280 # Safety margin
    
    T = 1.0
    T_min = 1e-4
    
    # Move generation: Pivot
    while time.time() < deadline:
        # Choose pivot index i (1 to N-2)
        i = rng.randint(1, N-2)
        
        # Choose rotation
        # Axis is vector from chain[i-1] to chain[i]
        p_prev = chain[i-1]
        p_curr = chain[i]
        axis = (p_curr[0]-p_prev[0], p_curr[1]-p_prev[1], p_curr[2]-p_prev[2])
        
        # Generate rotation matrix or just try 3 possible 90 deg rotations?
        # In cubic lattice, a 90 deg rotation around a unit axis maps lattice to lattice.
        # We can just try all 3 possible orthogonal directions for the new first step?
        # No, the segment is rigid.
        # Easier: Try all 3 possible neighbors of p_prev that are NOT p_curr?
        # No, that's just moving the head.
        # Pivot: The bond (i-1, i) is fixed. The rest rotates.
        # There are 3 possible positions for the new p_curr (neighbors of p_prev, excluding p_curr_old? No, p_curr_old is one of them).
        # Actually, there are 5 neighbors of p_prev. One is p_curr_old. The other 4 are potential new p_curr positions.
        # But the rotation must be rigid.
        # If we pick a new p_curr, the rest of the segment follows by rotating around the axis.
        
        # Let's just pick a random neighbor of p_prev, call it new_p_curr.
        # If new_p_curr == p_curr, skip.
        # Then we need to rotate the rest of the segment.
        # This requires calculating the rotation.
        
        # Alternative: Use "Crankshaft" or "Local" moves which are easier?
        # Let's stick to a simpler "Random Walk" style move if Pivots are too complex to code correctly quickly.
        # But Pivots are essential for long chains.
        
        # Let's implement a simple "Step" move for the last residue? No, that's just growth.
        # How about "Slide"?
        
        # Let's try a different approach: **Simulated Annealing with Local Moves (Residue Shuffling)**
        # This is risky for validity.
        
        # Let's go back to **Pivot** but implement the rotation explicitly.
        # Axis is unit vector A.
        # We rotate vector V (relative to p_prev) by 90, 180, 270 degrees around A.
        # Since A is unit, and lattice is cubic, the rotation maps integer coords to integer coords.
        
        # Get all possible rotations of the segment.
        # This is computationally heavy to generate all 3 rotations for every move.
        
        # Let's use a different move: **Crankshaft** (length 3).
        # Pick i, j=i+2.
        # Rotate segment i+1 around axis i-j? No.
        # Standard Crankshaft: Pick 3 residues i, i+1, i+2.
        # Rotate i+1 to a neighbor of i that is not i+2? No, i+1 must stay connected to i and i+2.
        # This is a "Pivot" of length 1?
        
        # Let's use **Pivot** with pre-computed rotation logic.
        # For a unit axis (dx, dy, dz), the rotation matrices for 90 deg are simple.
        # But we have 3 axes.
        
        # Given time constraints, let's use a **Greedy + Local Search** hybrid.
        # 1. Grow.
        # 2. Local Search: Try to move each residue to a neighbor if it reduces energy and doesn't collide.
        # This is "Steepest Descent" with random restarts.
        
        # Let's implement **Local Search** (Steepest Descent) with **Random Restarts**.
        # It's simpler and often effective for HP.
        
        # Reset time check
        if time.time() > deadline: break
        
        # Try to improve current chain via local moves
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(N):
                p = chain[idx]
                for q in _neighbours(p):
                    if q in occupied:
                        continue
                    # Check if moving idx to q is valid
                    # It must not collide with anything else.
                    # Since q is not in occupied, it's free.
                    # But we must ensure connectivity?
                    # If we move an internal residue, we break the chain.
                    # So we can ONLY move the last residue? Or use Pivots.
                    
                    # OK, Local Search on internal residues is invalid for SAWs unless we use Pivots/Crankshafts.
                    
                    # So we MUST use Pivots.
                    
        # Let's implement a **Simplified Pivot**:
        # Pick i. Pick a random neighbor q of chain[i-1] (q != chain[i]).
        # This defines the new direction for the bond (i-1, i).
        # We then rotate the segment i..N-1 to follow this new direction.
        # This is a valid move if the new segment doesn't collide.
        
        # To rotate the segment, we can just "grow" it again from chain[i-1] in the direction of q,
        # but we must maintain the relative geometry? No, that changes the shape.
        # A Pivot is a RIGID rotation.
        
        # Let's calculate the rigid rotation.
        # Axis A = chain[i] - chain[i-1].
        # We want to rotate the vector B = chain[i+1] - chain[i] (and so on) around A.
        # There are 3 possible 90-degree rotations.
        
        # Given the complexity, I will use a **Monte Carlo** approach with **Pivots**
        # where I explicitly compute the new positions for the segment.
        
        # Optimization: Only try pivots for i in [1, N-2].
        
        # Let's write the rotation logic.
        # Axis (dx, dy, dz).
        # If dx=1, dy=0, dz=0 (X-axis):
        #   (x, y, z) -> (x, -z, y) or (x, z, -y) for 90 deg.
        # We can generalize.
        
        # This is getting too complex to code blind.
        
        # **Alternative Strategy**: Use the existing `_grow` but with **Backtracking** or **Multiple Random Restarts** with **Higher Diversity**.
        # The seed uses 200 restarts.
        # If we increase restarts to 2000, and make the growth slightly more randomized (lower greediness), we might find better solutions.
        # But 2000 restarts * 26 sequences * 9s? No, 2000 restarts must fit in 9s.
        # 9s / 2000 = 4.5ms per restart.
        # Growth is $O(N)$. 136 residues. Very fast.
        # So we can do many more restarts.
        
        # Let's try **10000 restarts** with a **Randomized Greedy** that has a temperature parameter.
        # Or just **Random Walk** with bias towards H-H contacts.
        
        # Let's combine:
        # 1. Run many fast greedy restarts.
        # 2. Take the top 10 best and run a short **Local Search** (Pivots) on them.
        
        # This is a "Multi-Start" strategy.
        
    # Implementation of Multi-Start + Local Search
    pass
