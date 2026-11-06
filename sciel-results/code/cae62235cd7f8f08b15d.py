"""
HP lattice protein folding on the cubic lattice.
Strategy: Simulated Annealing with Pivot, Reversal, and Crankshaft moves.
"""
import random
import sys
import time
import math

# EVOLVE-BLOCK-START

def _energy(seq, chain):
    """Compute energy of a chain."""
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i in range(n):
        if seq[i] != 'H':
            continue
        p = chain[i]
        for q in _neighbours(p):
            j = pos_to_idx.get(q)
            if j is not None and abs(i - j) > 1 and seq[j] == 'H':
                if i < j:
                    contacts += 1
    return -contacts

def _neighbours(point):
    """Yield the 6 neighbours of a 3D lattice point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _pivot_move(chain, i, rng):
    """
    Perform a pivot move: rotate the segment chain[i+1:] around the bond (chain[i], chain[i+1]).
    Returns (new_chain, success).
    """
    n = len(chain)
    if i >= n - 1:
        return None, False
    
    # The pivot is at chain[i]. The vector from chain[i] to chain[i+1] is v.
    # We rotate the rest of the chain (i+1 to end) around the axis defined by v?
    # Actually, in lattice models, a pivot move usually means choosing a new direction for the bond (i, i+1)
    # and then propagating that change? No, that's not a standard pivot.
    
    # Standard Pivot Move in 3D:
    # Pick a pivot bond (i, i+1).
    # Rotate the entire tail (i+1...n-1) around the bond axis? 
    # In cubic lattice, we can't rotate continuously. 
    # Instead, we pick a new direction for the bond (i, i+1) such that it is perpendicular to the previous one? 
    # Or we pick a new position for chain[i+1] that is a neighbor of chain[i] but not chain[i-1] (if i>0) and not occupied.
    # Then we need to check if the rest of the chain can be translated/rotated? 
    # Actually, the standard "Pivot Move" in MC for polymers is:
    # 1. Pick a pivot index i.
    # 2. Rotate the segment i+1...n-1 around the bond i-i+1.
    # In a cubic lattice, this is tricky. A common approximation is to pick a new direction for the bond (i, i+1) 
    # and then try to re-fold the tail? That's expensive.
    
    # Alternative: "Criss-cross" or "Crankshaft" moves are easier.
    # Let's use a move that swaps the direction of a segment.
    # Or, simpler: "Reversal Move".
    # Reversal Move: Pick i < j. Reverse the segment chain[i+1...j].
    # This is valid if the new connections are valid.
    
    # Let's implement Reversal Move.
    pass

def _reversal_move(chain, i, j, rng):
    """
    Reverse the segment chain[i+1...j].
    Returns (new_chain, success).
    """
    if i >= j:
        return None, False
    
    # The segment to reverse is chain[i+1] to chain[j].
    # The connection before the segment is chain[i] - chain[i+1].
    # The connection after the segment is chain[j] - chain[j+1] (if j < n-1).
    
    # After reversal:
    # chain[i] is connected to chain[j].
    # chain[j-1] is connected to chain[j-2] ... chain[i+1].
    # chain[i+1] is connected to chain[j] (wait, the sequence is reversed).
    
    # Let's trace:
    # Original: ... A(i) - B(i+1) - C(i+2) ... D(j) - E(j+1) ...
    # Reversed segment B..D becomes D..C..B.
    # New: ... A(i) - D(j) - C(j-1) ... B(i+1) - E(j+1) ...
    
    # Checks:
    # 1. A(i) and D(j) must be neighbors.
    # 2. If j < n-1, B(i+1) and E(j+1) must be neighbors.
    # 3. No self-intersections in the new configuration.
    
    n = len(chain)
    a = chain[i]
    b_new_head = chain[j]
    b_new_tail = chain[i+1]
    
    # Check 1: a and b_new_head must be neighbors
    if abs(a[0]-b_new_head[0]) + abs(a[1]-b_new_head[1]) + abs(a[2]-b_new_head[2]) != 1:
        return None, False
    
    # Check 2: b_new_tail and next must be neighbors
    if j < n - 1:
        e = chain[j+1]
        if abs(b_new_tail[0]-e[0]) + abs(b_new_tail[1]-e[1]) + abs(b_new_tail[2]-e[2]) != 1:
            return None, False
    
    # Construct new chain
    new_chain = chain[:]
    # Reverse the segment i+1 to j
    segment = chain[i+1:j+1]
    segment.reverse()
    new_chain[i+1:j+1] = segment
    
    # Check self-avoidance
    # We only need to check the moved points against the rest, and within themselves (but they were self-avoiding before, and reversal preserves relative distances? No, it's a rigid motion? No, it's a reflection/rotation? 
    # In a lattice, reversing a path is a valid geometric operation if the endpoints connect. 
    # The set of points occupied by the segment is the same. 
    # So we only need to check that the new endpoints don't collide with the rest of the chain (excluding the segment itself).
    # Actually, since the set of points is identical, internal collisions are impossible if there were none before.
    # We just need to ensure the new connections don't cause collisions with the *outside* of the segment?
    # Wait, the points are the same. The only risk is if the new chain crosses itself at the boundaries.
    # But the points are the same set. So if the old chain was self-avoiding, the new chain is self-avoiding 
    # UNLESS the new connections (A-D and B-E) cause an overlap? No, A and D are distinct points in the old chain (unless i and j are close).
    # If i and j are close, say j=i+1, segment is [B]. Reversal does nothing.
    # If j=i+2, segment is [B, C]. Reversed [C, B].
    # Old: A-B-C-E. New: A-C-B-E.
    # Points are {A,B,C,E}. Same set.
    # So self-avoidance is preserved automatically if the old chain was self-avoiding.
    # We only need to check the connectivity constraints which we already did.
    
    return new_chain, True

def _crankshaft_move(chain, i, j, rng):
    """
    Crankshaft move: Rotate the segment chain[i+1...j] around the axis defined by chain[i] and chain[i+1]? 
    No, standard crankshaft: 
    Pick i, j. 
    The segment i+1...j is rotated around the bond (i, i+1)? No, that's pivot.
    Crankshaft: 
    The segment i+1...j is moved such that the bond (i, i+1) and (j, j+1) remain fixed in orientation? 
    Actually, a common crankshaft move in 3D lattice:
    Pick i, j. 
    Try to move the segment i+1...j to a new position such that:
    1. chain[i] is still connected to new_chain[i+1]
    2. new_chain[j] is still connected to chain[j+1]
    3. The segment is rigidly translated/rotated? 
    This is complex. Let's stick to Reversal and maybe "Swap" moves or simple "Pivot" approximation.
    
    Let's implement a "Pivot" move that picks a new direction for bond (i, i+1) and then greedily re-places the tail? 
    No, that's not a single MC step.
    
    Let's use a "Random Walk" restart if stuck? 
    Or better: **Insertion/Deletion moves**? 
    No, let's use **Pivot Move** properly.
    In 3D, a pivot move rotates the tail around the pivot bond. 
    Since we are on a lattice, we can enumerate all possible rotations of the tail that are valid lattice moves.
    However, the tail is long. 
    A simpler, effective move is the **"Criss-cross"** or just **Reversal**.
    Reversal is very powerful.
    
    Let's add a **"Shift"** move: Move the entire tail (i+1...n-1) by one step in a random direction, if possible.
    """
    pass

def _shift_move(chain, i, rng):
    """
    Shift the tail chain[i+1:] by one unit in a random direction.
    """
    n = len(chain)
    if i >= n - 1:
        return None, False
    
    directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    dx, dy, dz = rng.choice(directions)
    
    new_chain = chain[:]
    for k in range(i+1, n):
        new_chain[k] = (chain[k][0] + dx, chain[k][1] + dy, chain[k][2] + dz)
    
    # Check connectivity at the cut
    if abs(chain[i][0] - new_chain[i+1][0]) + abs(chain[i][1] - new_chain[i+1][1]) + abs(chain[i][2] - new_chain[i+1][2]) != 1:
        return None, False
        
    # Check self-avoidance
    # The tail points are shifted. We need to check if any new tail point collides with the head (0...i) or other tail points.
    # Since the tail is rigidly shifted, internal collisions are preserved.
    # We only need to check collisions between the new tail and the head (0...i).
    head_set = set(chain[:i+1])
    for k in range(i+1, n):
        if new_chain[k] in head_set:
            return None, False
            
    return new_chain, True

def _sa_fold(seq, dimension, time_budget=5.0, rng=None):
    if rng is None:
        rng = random.Random()
    
    n = len(seq)
    if n < 4:
        return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    # Initialize with a random walk
    chain = [tuple([0] * dimension)]
    occupied = {chain[0]: 0}
    last_dir = None
    
    for i in range(1, n):
        valid_dirs = []
        for axis in range(dimension):
            for step in (-1, 1):
                if last_dir is not None and last_dir[0] == axis and last_dir[1] == -step:
                    continue
                q = list(chain[-1])
                q[axis] += step
                q = tuple(q)
                if q not in occupied:
                    valid_dirs.append((axis, step))
        
        if not valid_dirs:
            # Try any direction
            for axis in range(dimension):
                for step in (-1, 1):
                    q = list(chain[-1])
                    q[axis] += step
                    q = tuple(q)
                    if q not in occupied:
                        valid_dirs.append((axis, step))
        
        if not valid_dirs:
            return None
        
        axis, step = rng.choice(valid_dirs)
        last_dir = (axis, step)
        q = list(chain[-1])
        q[axis] += step
        q = tuple(q)
        chain.append(q)
        occupied[q] = i
    
    if len(chain) < n:
        return None
    
    current_energy = _energy(seq, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    T = 10.0
    cooling = 0.9999
    
    moves_count = 0
    
    while time.time() - start_time < time_budget:
        T *= cooling
        if T < 0.1:
            # Restart with higher T if stuck?
            if rng.random() < 0.01:
                T = 5.0
            
        # Choose a move
        move_type = rng.random()
        
        if move_type < 0.5:
            # Reversal
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            new_chain, success = _reversal_move(chain, i, j, rng)
        elif move_type < 0.8:
            # Shift
            i = rng.randint(0, n - 2)
            new_chain, success = _shift_move(chain, i, rng)
        else:
            # Single residue move (for local fine-tuning)
            i = rng.randint(1, n - 2)
            p = chain[i]
            axis = rng.randint(0, dimension - 1)
            step = rng.choice((-1, 1))
            q = list(p)
            q[axis] += step
            q = tuple(q)
            
            if q not in occupied:
                prev_p = chain[i - 1]
                next_p = chain[i + 1]
                if abs(prev_p[0] - q[0]) + abs(prev_p[1] - q[1]) + abs(prev_p[2] - q[2]) == 1 and \
                   abs(next_p[0] - q[0]) + abs(next_p[1] - q[1]) + abs(next_p[2] - q[2]) == 1:
                    new_chain = chain[:]
                    new_chain[i] = q
                    success = True
                else:
                    success = False
            else:
                success = False
        
        if not success:
            continue
            
        new_energy = _energy(seq, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 0.01)):
            chain = new_chain
            current_energy = new_energy
            # Update occupied set? 
            # _energy builds its own map. 
            # For the single residue move, we should update occupied for efficiency in next step?
            # But _reversal and _shift change many points. 
            # Rebuilding occupied set is O(N). 
            # Let's just let _energy handle it, or maintain it carefully.
            # For simplicity and correctness, we rely on _energy's internal map.
            # But _shift and _reversal don't update the `occupied` variable used in initialization.
            # We don't use `occupied` in the SA loop, only in init. 
            # Wait, I used `occupied` in the single residue move check `if q not in occupied`.
            # I need to maintain `occupied`.
            
            # Let's update occupied
            # This is inefficient if we do it for every move.
            # But N is small (<=136). 
            # Rebuilding occupied is O(N). 
            # Let's just rebuild it if the move is accepted? 
            # Or better, maintain it.
            
            # For now, let's just accept and update chain. 
            # The `occupied` check in single residue move is the bottleneck if not updated.
            # Let's update `occupied` after acceptance.
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                
            # Update occupied set for future single-residue checks
            # This is O(N). 
            # We can defer this? No, the check `if q not in occupied` needs it.
            # Let's just rebuild it. 
            occupied = {p: i for i, p in enumerate(chain)}
            
    return best_chain

def fold(sequence, dimension):
    """
    Fold a protein sequence on the cubic lattice.
    Returns a list of lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n < 4:
        return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    # Time budget per sequence
    # Total 240s for 26 sequences -> ~9s each.
    # We will use multiple restarts with shorter time budgets.
    time_budget = 8.0
    num_restarts = 3
    
    best_chain = None
    best_energy = float('inf')
    
    start_time = time.time()
    
    for r in range(num_restarts):
        if time.time() - start_time > time_budget:
            break
            
        seed = 12345 + r * 100 + len(sequence)
        rng = random.Random(seed)
        
        # Allocate time for this restart
        remaining = time_budget - (time.time() - start_time)
        t_alloc = remaining / (num_restarts - r) * 0.9
        
        chain = _sa_fold(sequence, dimension, time_budget=t_alloc, rng=rng)
        
        if chain is not None and len(chain) == n:
            e = _energy(sequence, chain)
            if e < best_energy:
                best_energy = e
                best_chain = chain
    
    if best_chain is None:
        # Fallback
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
