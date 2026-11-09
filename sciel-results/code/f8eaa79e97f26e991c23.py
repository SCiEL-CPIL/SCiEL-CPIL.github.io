"""HP lattice protein folding using Simulated Annealing with hybrid local and global moves."""
# EVOLVE-BLOCK-START
import random
import math
import time

_DIRS3 = [(0,0,1),(0,0,-1),(0,1,0),(0,-1,0),(1,0,0),(-1,0,0)]

def _add3(p, d):
    return (p[0]+d[0], p[1]+d[1], p[2]+d[2])

def _sub3(p, q):
    return (p[0]-q[0], p[1]-q[1], p[2]-q[2])

def _is_adj3(p, q):
    return abs(p[0]-q[0]) + abs(p[1]-q[1]) + abs(p[2]-q[2]) == 1

def _generate_spiral(n, rng, dim=3):
    """
    Generate a compact spiral-like initial conformation.
    This is better than a random SAW for HP folding as it starts compact.
    """
    if n == 1:
        return [(0,0,0)]
    if n == 2:
        return [(0,0,0), (1,0,0)]
    
    # Simple heuristic: fold in a 2D plane first, then perturb in Z
    # This creates a compact "blob"
    chain = [(0,0,0)]
    occupied = {(0,0,0)}
    x, y, z = 0, 0, 0
    # Directions: Right, Down, Left, Up (clockwise spiral)
    dirs = [(1,0,0), (0,-1,0), (-1,0,0), (0,1,0)]
    dir_idx = 0
    steps_in_dir = 1
    
    for i in range(1, n):
        placed = False
        # Try current direction
        for _ in range(4):
            dx, dy, dz = dirs[dir_idx % 4]
            nx, ny, nz = x + dx, y + dy, z + dz
            if (nx, ny, nz) not in occupied:
                x, y, z = nx, ny, nz
                chain.append((x, y, z))
                occupied.add((x, y, z))
                placed = True
                break
            
            # If blocked, turn right (change direction)
            dir_idx += 1
        
        if not placed:
            # Fallback: if stuck, try to extend in Z or restart with a different seed logic
            # For simplicity in this generator, if we get stuck in the spiral logic,
            # we fall back to a random SAW from this point or just restart the whole gen.
            # But a simple spiral usually works for N < 200.
            # Let's try a simpler "fold" strategy if spiral fails: 
            # Actually, let's just use a random SAW if spiral fails, but spiral rarely fails for N<150 if we allow "turning".
            # The above loop turns if blocked. If all 4 are blocked, we are trapped.
            # To ensure validity, we check if we are trapped.
            # If trapped, we break and use a fallback.
            pass
            
        if not placed:
             # Trapped. Use a simple straight line fallback for the rest? No, that's bad.
             # Let's just restart the generation with a different randomization if we fail.
             # But for now, let's assume the simple turning logic works.
             # If it fails, we return a straight line to be safe (valid but bad energy).
             # Better: use a random SAW generator as fallback.
             pass

    # If the spiral got stuck, we might have a shorter chain.
    # Let's implement a robust fallback: Random SAW
    if len(chain) < n:
         # Fallback to Random SAW
         chain = []
         occupied = set()
         for _ in range(10):
             chain = [(0,0,0)]
             occupied = {(0,0,0)}
             success = True
             for i in range(1, n):
                 last = chain[-1]
                 neighbors = [_add3(last, d) for d in _DIRS3]
                 rng.shuffle(neighbors)
                 placed = False
                 for site in neighbors:
                     if site not in occupied:
                         chain.append(site)
                         occupied.add(site)
                         placed = True
                         break
                 if not placed:
                     success = False
                     break
             if success:
                 return chain
         # Final fallback: straight line
         return [(i, 0, 0) for i in range(n)]
         
    return chain

def _compute_energy(sequence, chain, where):
    """Compute energy using a precomputed position-to-index map."""
    n = len(sequence)
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        p = chain[i]
        for d in _DIRS3:
            q = _add3(p, d)
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts

def _sa_fold(sequence, dimension, rng, time_limit):
    """
    Simulated Annealing with hybrid moves.
    """
    n = len(sequence)
    if n <= 2:
        return _generate_spiral(n, rng)
    
    # Initialize with spiral
    chain = _generate_spiral(n, rng)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain, where)
    
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    T = 10.0
    min_T = 0.1
    cooling_rate = 0.9999
    
    # Move probabilities
    prob_local = 0.7
    prob_pivot = 0.3
    
    steps = 0
    max_steps = 1000000 # Safety
    
    while time.time() - start_time < time_limit and steps < max_steps:
        steps += 1
        
        # Adaptive temperature
        T = max(min_T, T * cooling_rate)
        
        move_type = rng.random()
        
        if move_type < prob_local:
            # Local Move: Rotate a single residue (i) by 90 or 180 degrees around the bond (i-1, i)
            # This is fast to check.
            if n < 3:
                continue
                
            i = rng.randint(1, n-1)
            # Get bond vector
            v = _sub3(chain[i], chain[i-1])
            
            # Generate a new position for i by rotating v around the previous bond? 
            # No, standard local move: pick a neighbor of i-1 that is not i-2 and not occupied.
            # Let's try moving residue i to a new position.
            # Candidates are neighbors of chain[i-1] excluding chain[i-2] (if exists) and chain[i] (current).
            
            prev = chain[i-1]
            curr = chain[i]
            prev_prev = chain[i-2] if i > 1 else None
            
            candidates = []
            for d in _DIRS3:
                new_pos = _add3(prev, d)
                if new_pos == curr:
                    continue
                if prev_prev is not None and new_pos == prev_prev:
                    continue
                if new_pos in where:
                    continue
                candidates.append(new_pos)
            
            if not candidates:
                continue
                
            new_pos = rng.choice(candidates)
            
            # Check if this move breaks the chain? 
            # The bond (i, i+1) must still be unit step.
            # New bond is (new_pos, chain[i+1]).
            # Since new_pos is a neighbor of prev, and curr was a neighbor of prev,
            # new_pos and curr are distance 2 from prev? No.
            # prev is at (0,0,0). curr is at (1,0,0). new_pos is at (0,1,0).
            # dist(new_pos, curr) = 2. So the bond (i, i+1) would be length 2?
            # Wait, if we move i, the bond (i, i+1) changes.
            # The bond (i-1, i) is fixed to be unit step by construction (new_pos is neighbor of prev).
            # The bond (i, i+1) must be unit step.
            # So we need dist(new_pos, chain[i+1]) == 1.
            
            valid_cands = []
            for c in candidates:
                if i < n-1:
                    if _is_adj3(c, chain[i+1]):
                        valid_cands.append(c)
            
            if not valid_cands:
                continue
                
            new_pos = rng.choice(valid_cands)
            
            # Apply move
            old_pos = curr
            chain[i] = new_pos
            del where[old_pos]
            where[new_pos] = i
            
            # Calculate delta energy
            # Only contacts involving i or neighbors of i change?
            # Actually, only H-H contacts involving i or its neighbors can change.
            # But for simplicity and correctness, let's just recompute if we are fast enough?
            # N=136, recompute is O(N). 10k steps/sec is fine.
            # But we can be smarter.
            
            # Let's just recompute energy for now, it's robust.
            new_energy = _compute_energy(sequence, chain, where)
            delta_e = new_energy - current_energy
            
            if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
            else:
                # Revert
                chain[i] = old_pos
                del where[new_pos]
                where[old_pos] = i

        else:
            # Pivot Move
            if n < 4:
                continue
            pivot_idx = rng.randint(1, n-2)
            # Tail is chain[pivot_idx:]
            # Rotate tail around chain[pivot_idx-1]
            
            pivot_point = chain[pivot_idx-1]
            tail = chain[pivot_idx:]
            
            # Relative coords
            rel_tail = [_sub3(p, pivot_point) for p in tail]
            
            # Choose rotation
            # 90 deg around x, y, z
            axis = rng.randint(0, 2)
            sign = 1 if rng.random() < 0.5 else -1
            
            rotated = []
            for p in rel_tail:
                x, y, z = p
                if axis == 0: # x
                    new_p = (x, sign*z, -sign*y)
                elif axis == 1: # y
                    new_p = (-sign*z, y, sign*x)
                else: # z
                    new_p = (sign*y, -sign*x, z)
                rotated.append(_add3(pivot_point, new_p))
            
            new_chain = chain[:pivot_idx] + rotated
            
            # Check validity
            valid = True
            seen = set(chain[:pivot_idx])
            # Add tail
            for p in new_chain:
                if p in seen:
                    valid = False
                    break
                seen.add(p)
            
            if valid:
                # Check unit steps at the junction
                if pivot_idx > 0 and pivot_idx < n-1:
                    if not _is_adj3(new_chain[pivot_idx-1], new_chain[pivot_idx]):
                        valid = False
            
            if not valid:
                continue
                
            # Check unit steps in tail? 
            # Rotation preserves distances, so internal tail steps are unit.
            # Only junction needs check.
            
            new_where = {p: i for i, p in enumerate(new_chain)}
            new_energy = _compute_energy(sequence, new_chain, new_where)
            delta_e = new_energy - current_energy
            
            if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
                chain = new_chain
                where = new_where
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)

    return best_chain

def fold(sequence, dimension):
    """
    Find the best conformation using Simulated Annealing.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    # Time budget management
    total_time_budget = 20.0 # Leave some margin for 300s total, assuming multiple calls? 
    # The prompt says "240 s CPU budget for all sequences together".
    # There are 26 sequences. 240/26 ~= 9.2 seconds per sequence.
    # Let's use 8.0 seconds per sequence to be safe.
    time_per_sequence = 8.0
    
    # Multiple restarts
    num_restarts = 3
    time_per_restart = time_per_sequence / num_restarts
    
    best_chain = None
    best_energy = float('inf')
    
    for restart in range(num_restarts):
        seed = 12345 + len(sequence) * 100 + restart
        rng = random.Random(seed)
        
        chain = _sa_fold(sequence, dimension, rng, time_limit=time_per_restart)
        
        # Rebuild where for energy check
        where = {p: i for i, p in enumerate(chain)}
        energy = _compute_energy(sequence, chain, where)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    # Final verification
    if best_chain is None:
        best_chain = _generate_spiral(n, random.Random(0))
        
    seen = set()
    valid = True
    for p in best_chain:
        if p in seen:
            valid = False
            break
        seen.add(p)
    
    if valid:
        for i in range(1, n):
            diff = abs(best_chain[i][0] - best_chain[i-1][0]) + \
                   abs(best_chain[i][1] - best_chain[i-1][1]) + \
                   abs(best_chain[i][2] - best_chain[i-1][2])
            if diff != 1:
                valid = False
                break
    
    if not valid:
        best_chain = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
