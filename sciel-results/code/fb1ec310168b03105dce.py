# EVOLVE-BLOCK-START
import time
import random
import math

_NEIGHBORS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]

def _get_deadline():
    """Return the global deadline (330s wall clock with safety margin)."""
    if not hasattr(_get_deadline, '_start'):
        _get_deadline._start = time.time()
    return _get_deadline._start + 325.0

def _time_left():
    """Return seconds remaining before deadline."""
    return max(0.0, _get_deadline() - time.time())

def _grow(sequence, rng):
    """
    Greedy chain growth with contact map optimization.
    Returns (chain, occupied_dict) or None if growth fails.
    """
    n = len(sequence)
    if n <= 1:
        return [(0, 0, 0)], {(0, 0, 0): 0}
    
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        best_score = -100
        best_sites = []
        
        for site in _NEIGHBORS:
            # Calculate absolute position
            sx = last[0] + site[0]
            sy = last[1] + site[1]
            sz = last[2] + site[2]
            abs_site = (sx, sy, sz)
            
            if abs_site in occupied:
                continue
            
            # Score: number of H residues in chain that would be contacted
            score = 0
            if sequence[i] == 'H':
                # Check all previous H residues for contact
                # Optimization: Only check recent ones? No, for short chains check all.
                # For longer chains, we can limit range or use spatial index.
                # Given n<=136, checking all previous is O(N^2) total which is ~18k ops per step * 136 steps = 2.5M ops. Acceptable.
                for j in range(i):
                    if i - j <= 1:
                        continue
                    if sequence[j] != 'H':
                        continue
                    cj = chain[j]
                    if abs(sx - cj[0]) + abs(sy - cj[1]) + abs(sz - cj[2]) == 1:
                        score += 1
            
            # Add some randomness to tie-breaking to encourage diversity
            score += rng.random() * 0.1
            
            if score > best_score:
                best_score = score
                best_sites = [abs_site]
            elif score == best_score:
                best_sites.append(abs_site)
        
        if not best_sites:
            return None
        
        site = rng.choice(best_sites)
        chain.append(site)
        occupied[site] = i
    
    return chain, occupied

def _compute_energy(sequence, chain):
    """Compute energy = -number of H-H contacts (non-consecutive neighbors)."""
    n = len(chain)
    pos_to_idx = {chain[i]: i for i in range(n)}
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y, z = chain[i]
        for nb in _NEIGHBORS:
            j = pos_to_idx.get((x+nb[0], y+nb[1], z+nb[2]))
            if j is not None and j > i + 1:
                contacts += 1
    return -contacts

def _pivot_move(chain, pivot_idx, occupied, rng):
    """
    Perform a pivot move: rotate the tail of the chain (pivot_idx+1 to end)
    around the bond (pivot_idx-1, pivot_idx).
    Returns (new_chain, new_occupied) or None if invalid.
    """
    n = len(chain)
    if pivot_idx <= 0 or pivot_idx >= n - 1:
        return None
    
    # The pivot axis is defined by vector v = chain[pivot_idx] - chain[pivot_idx-1]
    # We need to rotate the sub-chain chain[pivot_idx+1:] around the point chain[pivot_idx]
    # such that the first bond of the sub-chain (chain[pivot_idx+1] - chain[pivot_idx])
    # is rotated to one of the other 5 directions relative to the pivot bond.
    
    p_prev = chain[pivot_idx - 1]
    p_pivot = chain[pivot_idx]
    p_next = chain[pivot_idx + 1]
    
    # Vector from pivot to prev
    v_prev = (p_prev[0] - p_pivot[0], p_prev[1] - p_pivot[1], p_prev[2] - p_pivot[2])
    # Vector from pivot to next
    v_next = (p_next[0] - p_pivot[0], p_next[1] - p_pivot[1], p_next[2] - p_pivot[2])
    
    # Find all possible new directions for v_next
    # They must be unit vectors, not equal to v_prev, and not equal to v_next (to avoid no-op)
    # Actually, any unit vector except v_prev is a valid new direction for the bond.
    # But we must ensure the rest of the chain doesn't collide.
    
    possible_dirs = []
    for d in _NEIGHBORS:
        if d != v_prev: # Cannot fold back
            if d != v_next: # Avoid no-op
                possible_dirs.append(d)
    
    if not possible_dirs:
        return None
    
    new_dir = rng.choice(possible_dirs)
    
    # We need to rotate the entire tail (from pivot_idx+1 onwards) by the rotation that maps v_next to new_dir.
    # This is complex to do directly with matrices for every step.
    # Simpler approach: Generate the new tail by walking from p_pivot in direction new_dir,
    # but we need to preserve the relative shape of the tail?
    # No, a pivot move rotates the *entire* tail rigidly.
    
    # Let's find the rotation matrix R such that R * v_next = new_dir.
    # There are multiple such rotations. We can pick one arbitrarily or try all?
    # Standard pivot move: Rotate around the axis defined by cross product?
    # Actually, for HP lattice, we can just enumerate the 5 valid new positions for chain[pivot_idx+1]
    # and then check if the rest of the chain, transformed by the same rotation, is valid.
    
    # To make this efficient and simple, let's use a specific rotation logic.
    # If v_next is (1,0,0) and new_dir is (0,1,0), rotation is around z-axis.
    
    # Helper to apply rotation defined by mapping v_next to new_dir
    # We can define the rotation by finding an orthonormal basis.
    
    # For simplicity and speed in Python, we might just try the 5 possible new positions for the first step of the tail
    # and then recursively check? No, that's not a rigid rotation.
    
    # Let's implement a specific rotation.
    # Cross product v_next x new_dir gives axis.
    # Dot product gives angle.
    
    # Alternative: Pre-compute all 6x6 rotation permutations? No.
    
    # Let's use a simpler heuristic:
    # If we pick a new position for chain[pivot_idx+1], say p_new_next,
    # we can define the rotation axis as cross(v_next, new_dir).
    # If v_next and new_dir are perpendicular (90 deg), axis is cross product.
    # If they are 180 deg apart, axis can be any perpendicular vector.
    
    # Let's just pick one of the 5 valid neighbors for p_new_next.
    p_new_next = (p_pivot[0] + new_dir[0], p_pivot[1] + new_dir[1], p_pivot[2] + new_dir[2])
    
    # Now, we need to transform the rest of the chain.
    # The rotation is the unique rotation (or one of them) that maps v_next to new_dir and keeps the lattice integer.
    # Actually, on the cubic lattice, any rotation that maps one unit vector to another is an integer rotation.
    
    # Let's find the rotation matrix.
    # We can use the fact that R maps v_next to new_dir.
    # We can also map a perpendicular vector.
    
    # To avoid complex matrix math, let's just try to "walk" the tail?
    # No, the tail shape must be preserved.
    
    # Let's use a library-free rotation implementation for 90/180 degree turns on cubic lattice.
    # Since all bonds are axis-aligned, the rotation will map axes to axes.
    
    # If v_next is (1,0,0) and new_dir is (0,1,0):
    # x->y, y->-x, z->z (or similar)
    
    # Let's define a function to get the rotation matrix.
    def get_rotation(v_from, v_to):
        # v_from and v_to are unit vectors
        # If they are the same, identity
        # If they are opposite, 180 deg rotation around any perpendicular axis
        # If perpendicular, 90 deg rotation around cross product
        
        # Simple case: Perpendicular
        cross = (
            v_from[1]*v_to[2] - v_from[2]*v_to[1],
            v_from[2]*v_to[0] - v_from[0]*v_to[2],
            v_from[0]*v_to[1] - v_from[1]*v_to[0]
        )
        
        if cross == (0,0,0):
            # Parallel or Anti-parallel
            if v_from == v_to:
                return [[1,0,0],[0,1,0],[0,0,1]]
            else:
                # 180 deg rotation. Pick an axis perpendicular to v_from.
                # If v_from is (1,0,0), axis can be (0,1,0). Rotation: x->-x, y->-y, z->z
                # General 180 deg rotation around axis a: R = 2*a*a^T - I
                # Let's just handle the 6 cases explicitly for speed and correctness
                pass
        
        # This is getting complicated. Let's use a simpler strategy:
        # Instead of a general pivot, let's use "Slithering" or "Pivot" with pre-computed rotations.
        
        # Actually, for the scope of this improvement, let's stick to the "Move single residue" local search but fix it properly,
        # or use a simpler "Tug of War" or "End Move" if Pivot is too hard to implement robustly in Python quickly.
        
        # However, Pivot is much better. Let's implement a specific case.
        # If v_next is (1,0,0):
        #   to (0,1,0): x->y, y->-x, z->z  (Matrix: [[0,0,0],[1,0,0],[0,1,0]]? No. (x,y,z)->(y,-x,z)?
        #   Let's check: (1,0,0) -> (0,-1,0)? No.
        #   Rotation around Z by 90 deg: (x,y,z) -> (-y,x,z). (1,0,0) -> (0,1,0). Correct.
        #   Matrix: [[0,-1,0],[1,0,0],[0,0,1]]
        
        # Let's pre-compute the 6x6 rotation mappings for all pairs of unit vectors.
        # There are 6 * 5 = 30 mappings.
        
        return None # Placeholder

    # Given the complexity, let's revert to a robust Local Search that fixes the current bugs,
    # and add Simulated Annealing to it. The current local search is broken because it doesn't check connectivity properly.
    
    # Fixed Local Search:
    # 1. Pick a random index i (1 to n-2, ends are fixed to avoid disconnection? Or handle ends carefully)
    # 2. Try to move chain[i] to a neighbor nb.
    # 3. Check if nb is free (not in occupied, except if i is being moved from there).
    # 4. Check if chain[i-1] is adjacent to nb AND chain[i+1] is adjacent to nb.
    # 5. If valid, compute delta energy.
    
    return None

def _delta_energy(sequence, chain, i, old_pos, new_pos, occupied):
    """
    Calculate change in energy when moving residue i from old_pos to new_pos.
    Assumes the move is valid (self-avoiding, connectivity preserved).
    """
    delta = 0
    # Contacts involving i
    # Remove old contacts
    if sequence[i] == 'H':
        x, y, z = old_pos
        for nb in _NEIGHBORS:
            j = occupied.get((x+nb[0], y+nb[1], z+nb[2]))
            if j is not None and abs(i - j) > 1:
                if sequence[j] == 'H':
                    delta -= 1 # Lost a contact (energy was -1, now 0, so delta is +1? No, energy = -contacts.
                    # Old energy contribution: -1. New: 0. Change: +1.
                    # Wait, delta_energy = new_energy - old_energy.
                    # If we lose a contact, new_energy > old_energy (less negative).
                    # So delta = +1.
                    delta += 1
    
    # Add new contacts
    if sequence[i] == 'H':
        x, y, z = new_pos
        for nb in _NEIGHBORS:
            j = occupied.get((x+nb[0], y+nb[1], z+nb[2]))
            if j is not None and abs(i - j) > 1:
                if sequence[j] == 'H':
                    delta -= 1 # Gained a contact. New energy is lower.
    
    return delta

def _local_search_sa(sequence, chain, occupied, time_budget, rng):
    """
    Local search with Simulated Annealing.
    Moves single residues to adjacent free sites, maintaining connectivity.
    """
    n = len(chain)
    energy = _compute_energy(sequence, chain)
    best_energy = energy
    best_chain = list(chain)
    
    deadline = time.time() + time_budget
    temp = 1.0
    cool = 0.99
    
    while time.time() < deadline:
        # Pick a random interior residue (1 to n-2) to ensure connectivity
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        
        # Find valid moves
        valid_moves = []
        for nb in _NEIGHBORS:
            new_pos = (old_pos[0] + nb[0], old_pos[1] + nb[1], old_pos[2] + nb[2])
            
            # Check if new_pos is free
            # It's free if it's not in occupied, OR if it's the old_pos (no-op, skip)
            # Since we are moving i, old_pos is currently in occupied with value i.
            # If new_pos is in occupied and occupied[new_pos] != i, it's blocked.
            if new_pos in occupied and occupied[new_pos] != i:
                continue
            
            # Check connectivity with neighbors
            # chain[i-1] must be adjacent to new_pos
            d_prev = abs(chain[i-1][0] - new_pos[0]) + abs(chain[i-1][1] - new_pos[1]) + abs(chain[i-1][2] - new_pos[2])
            if d_prev != 1:
                continue
            # chain[i+1] must be adjacent to new_pos
            d_next = abs(chain[i+1][0] - new_pos[0]) + abs(chain[i+1][1] - new_pos[1]) + abs(chain[i+1][2] - new_pos[2])
            if d_next != 1:
                continue
            
            valid_moves.append(new_pos)
        
        if not valid_moves:
            temp *= cool
            continue
        
        new_pos = rng.choice(valid_moves)
        
        # Calculate delta energy
        d_e = _delta_energy(sequence, chain, i, old_pos, new_pos, occupied)
        
        # Metropolis criterion
        if d_e < 0 or rng.random() < math.exp(-d_e / temp):
            # Accept move
            # Update occupied
            del occupied[old_pos]
            occupied[new_pos] = i
            chain[i] = new_pos
            energy += d_e
            
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
        
        temp *= cool
        if temp < 0.01:
            temp = 1.0 # Reheat occasionally
    
    return best_chain, best_energy

def _fold_single(sequence, n_restarts, time_budget, rng):
    """
    Run multiple greedy restarts with local search.
    Returns the best chain found.
    """
    n = len(sequence)
    if n <= 2:
        return [(i, 0, 0) for i in range(n)]
    
    best_chain = None
    best_energy = float('inf')
    
    start = time.time()
    deadline = start + time_budget
    
    for restart in range(n_restarts):
        if time.time() > deadline - 2:
            break
        
        # Greedy growth
        result = _grow(sequence, rng)
        if result is None:
            continue
        chain, occupied = result
        
        # Local search with remaining time
        remaining = deadline - time.time() - 1.0
        if remaining < 0.1:
            break
        
        chain, energy = _local_search_sa(sequence, chain, occupied, remaining, rng)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
    
    return best_chain if best_chain else [(i, 0, 0) for i in range(n)]

def _fallback_chain(n):
    """Simple straight-line fallback."""
    return [(i, 0, 0) for i in range(n)]

def fold(sequence, dimension):
    """
    Fold an HP sequence on the simple cubic lattice.
    Returns a list of len(sequence) lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    remaining = _time_left()
    if remaining < 1.0:
        return [list(p) for p in _fallback_chain(n)]
    
    # Allocate time for this call
    per_call_budget = min(remaining * 0.9, 30.0)
    
    # Number of restarts: more for shorter chains
    if n <= 48:
        n_restarts = max(5, int(per_call_budget / 2.0))
    elif n <= 80:
        n_restarts = max(3, int(per_call_budget / 4.0))
    else:
        n_restarts = max(2, int(per_call_budget / 6.0))
    
    rng = random.Random(42 + n * 7)
    
    best_chain = _fold_single(sequence, n_restarts, per_call_budget, rng)
    
    # Validate self-avoidance
    if len(set(best_chain)) != n:
        best_chain = _fallback_chain(n)
    
    # Validate unit steps
    for i in range(1, n):
        d = abs(best_chain[i][0] - best_chain[i-1][0]) + \
            abs(best_chain[i][1] - best_chain[i-1][1]) + \
            abs(best_chain[i][2] - best_chain[i-1][2])
        if d != 1:
            best_chain = _fallback_chain(n)
            break
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
