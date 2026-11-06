"""
Rewritten program for HP lattice protein folding using Simulated Annealing
with pivot and kink moves.
"""
# EVOLVE-BLOCK-START
import random
import time
import math

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy(sequence, chain):
    """
    Compute energy: -number of H-H contacts (non-consecutive).
    chain: list of tuples (x, y, z)
    """
    n = len(chain)
    if n < 2:
        return 0
    
    # Map position to index for O(1) lookup
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    
    for i in range(n):
        if sequence[i] == 'H':
            p = chain[i]
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None:
                    # Check if not consecutive in sequence
                    if abs(i - j) > 1:
                        if sequence[j] == 'H':
                            contacts += 1
    return -contacts

def _random_walk(sequence, dimension, rng):
    """
    Generate a random self-avoiding walk.
    Returns None if it fails to complete (traps itself).
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for nb in _neighbours(last):
            if nb not in occupied:
                candidates.append(nb)
        
        if not candidates:
            return None # Trapped
        
        # Random choice
        next_pos = rng.choice(candidates)
        chain.append(next_pos)
        occupied[next_pos] = i
        
    return chain

def _try_pivot_move(chain, rng):
    """
    Try a pivot move: pick a pivot index i, rotate the segment [i+1:] 
    around chain[i] by 90 or 180 degrees.
    Returns (new_chain, success)
    """
    n = len(chain)
    if n < 3:
        return None, False
        
    # Choose pivot index (1 to n-2 to allow rotation of at least one segment)
    # Avoiding 0 and n-1 to keep the "root" fixed or simple
    i = rng.randint(1, n - 2)
    
    # Directions: 6 possible directions
    # We need to rotate the vector from chain[i] to chain[i+1]
    # Let v = chain[i+1] - chain[i]
    # We want to find a new vector v' such that |v'|=1, v' != v, and v' is a valid lattice step.
    # Actually, simpler: pick a new direction for the first step of the tail.
    # The tail is chain[i+1:].
    # Let's define the tail relative to chain[i].
    # tail_rel = [p - chain[i] for p in chain[i+1:]]
    # We rotate the entire tail.
    
    # Instead of rotating coordinates, we can just pick a new direction for the bond (i, i+1).
    # The rest of the chain must be shifted/rotated accordingly.
    # This is complex. A simpler "pivot" in lattice terms is often just moving the segment.
    # Let's use a simpler move: "Segment Move" or "Crankshaft"?
    # Or just "Kink" moves which are cheaper.
    # For long chains, pivots are better.
    
    # Let's implement a proper pivot:
    # 1. Pick i.
    # 2. Get vector v = chain[i+1] - chain[i].
    # 3. Generate all 6 possible unit vectors.
    # 4. Exclude v.
    # 5. Pick a random new vector v_new.
    # 6. Determine the rotation matrix R that maps v to v_new.
    #    There are multiple rotations. We can just pick one that works.
    #    Actually, it's easier to just try all 6 new directions for the first bond of the tail,
    #    and for each, check if the rest of the tail (rotated) is valid.
    #    But rotating the rest is O(N).
    
    # Alternative: "Crankshaft" move is easier?
    # Let's stick to Kink moves for simplicity and reliability in Python, 
    # but add "Segment Shift" if possible?
    # Actually, for HP folding, Kink moves are often sufficient if we have many restarts.
    # Let's try a "Local Segment Reversal" or "180 degree turn" of a short segment?
    
    # Let's implement a robust Kink move first.
    # Kink: Move residue i (2 <= i <= n-3) to a neighbor site that is not occupied 
    # and maintains connectivity to i-1 and i+1.
    # This is what the previous program did but with SA.
    
    # To make it different, let's try "Pivot" properly.
    # We rotate the tail chain[i+1:] around chain[i].
    # We need to find a rotation R such that R(chain[i+1] - chain[i]) is a valid unit vector 
    # and not equal to (chain[i+1] - chain[i]).
    
    v = tuple(chain[i+1][k] - chain[i][k] for k in range(dimension))
    
    # Possible new vectors
    possible_new_v = []
    for axis in range(dimension):
        for step in (-1, 1):
            nv = [0] * dimension
            nv[axis] = step
            if tuple(nv) != v:
                possible_new_v.append(tuple(nv))
                
    if not possible_new_v:
        return None, False
        
    new_v = rng.choice(possible_new_v)
    
    # We need a rotation matrix R that maps v to new_v.
    # In 3D, we can construct this.
    # Let's just try all 24 rotation matrices? No, too slow.
    # We can determine R by mapping the basis.
    # Simpler: Just try to apply the move by checking if the rotated tail is valid.
    # But how to rotate?
    
    # Let's use a simpler move: "Segment Move"
    # Pick a segment [i, j] and try to move it to a nearby location?
    
    # Given the complexity of Pivot in pure Python without numpy for matrices, 
    # let's use "Kink" moves but with a higher acceptance rate (SA) and 
    # also "Reversal" moves (reverse the order of a segment, which is valid for SAW).
    
    # Reversal Move:
    # Pick i, j (i < j). Reverse chain[i:j+1].
    # Check if the new positions are free.
    # This is O(N) to check.
    
    # Let's implement Reversal.
    i = rng.randint(0, n - 2)
    j = rng.randint(i + 1, n - 1)
    
    # The segment is chain[i..j]
    # The new chain will be chain[0..i-1] + reversed(chain[i..j]) + chain[j+1..]
    # We need to check if the new positions are occupied by the rest of the chain.
    
    # Occupied set excluding the segment
    occupied = set(chain)
    for k in range(i, j + 1):
        occupied.remove(chain[k])
        
    new_segment = list(reversed(chain[i:j+1]))
    
    # Check collisions
    valid = True
    for p in new_segment:
        if p in occupied:
            valid = False
            break
            
    if not valid:
        return None, False
        
    # Check connectivity at the boundaries
    # The connection between i-1 and i (which is now the last element of the reversed segment, i.e., original j)
    # And between j and j+1 (which is now the first element of the reversed segment, i.e., original i)
    
    # Wait, reversing a segment in a SAW:
    # Original: ... A - B - C - D ...
    # Reverse B..D: ... A - D - C - B ...
    # The bond A-B becomes A-D? No.
    # The sequence of residues is fixed. We are moving the *positions*.
    # If we reverse the *positions* of the segment, the sequence of residues stays the same?
    # No, the task says "list of len(sequence) integer lattice points".
    # The residue at index k is at chain[k].
    # If we reverse the *positions*, the residue at index i moves to old position j.
    # This changes the connectivity.
    # Residue i-1 is at pos[i-1]. Residue i is now at pos[j].
    # Are pos[i-1] and pos[j] adjacent?
    # Originally pos[i-1] and pos[i] were adjacent.
    # And pos[j] and pos[j+1] were adjacent.
    # So we need to check if pos[i-1] is adjacent to pos[j] (new pos of i)
    # And if pos[j+1] is adjacent to pos[i] (new pos of j)
    
    # Actually, a "Reversal" move in SAWs usually reverses the *path*, which implies the sequence order is reversed?
    # No, the sequence is fixed. We are just permuting the coordinates.
    # This is a "Segment Reversal" of coordinates.
    # Is it a valid move?
    # Yes, if the new boundaries connect.
    
    # Let's check boundaries:
    if i > 0:
        # New pos of i is old pos of j
        # Check dist(chain[i-1], chain[j]) == 1
        d = sum(abs(chain[i-1][k] - chain[j][k]) for k in range(dimension))
        if d != 1:
            return None, False
            
    if j < n - 1:
        # New pos of j is old pos of i
        # Check dist(chain[j+1], chain[i]) == 1
        d = sum(abs(chain[j+1][k] - chain[i][k]) for k in range(dimension))
        if d != 1:
            return None, False
            
    # If all checks pass, return new chain
    new_chain = chain[:i] + new_segment + chain[j+1:]
    return new_chain, True

def _try_kink_move(chain, rng):
    """
    Try a kink move: Move an interior residue to a free neighbor site.
    """
    n = len(chain)
    if n < 4:
        return None, False
        
    i = rng.randint(1, n - 2)
    current = chain[i]
    
    # Find free neighbors of current
    candidates = []
    occupied = set(chain)
    
    for nb in _neighbours(current):
        if nb in occupied:
            continue
        # Check if nb is adjacent to i-1 and i+1
        d_prev = sum(abs(nb[k] - chain[i-1][k]) for k in range(dimension))
        d_next = sum(abs(nb[k] - chain[i+1][k]) for k in range(dimension))
        
        # For a valid kink move, the new position must be adjacent to both neighbors
        # Actually, no. The new position just needs to be adjacent to i-1 and i+1 
        # IF we are moving the residue.
        # Wait, if we move residue i to nb, then:
        # Residue i-1 is at chain[i-1]. Residue i is at nb.
        # They must be adjacent.
        # Residue i is at nb. Residue i+1 is at chain[i+1].
        # They must be adjacent.
        if d_prev == 1 and d_next == 1:
            # Also ensure nb is not the same as current (it's a neighbor, so it isn't)
            # And ensure nb is not occupied (checked above)
            candidates.append(nb)
            
    if not candidates:
        return None, False
        
    new_pos = rng.choice(candidates)
    new_chain = list(chain)
    new_chain[i] = new_pos
    return new_chain, True

def _anneal(sequence, chain, dimension, rng, start_time, deadline):
    """
    Simulated Annealing loop.
    """
    n = len(sequence)
    current_chain = list(chain)
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Cooling schedule
    # T starts high, ends low.
    # We have limited time.
    # Let's run for a fixed number of steps or until time is up.
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    steps = 0
    max_steps = 100000 # Safety cap, time is the real limit
    
    while time.time() < deadline:
        if steps % 1000 == 0:
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        # Choose move type
        # Kink is cheaper. Pivot/Reversal is more powerful.
        # Let's use 80% Kink, 20% Reversal
        if rng.random() < 0.8:
            new_chain, success = _try_kink_move(current_chain, rng)
        else:
            new_chain, success = _try_reversal_move(current_chain, dimension, rng)
            
        if not success or new_chain is None:
            steps += 1
            continue
            
        new_energy = _compute_energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta_e / T)
        if delta_e < 0:
            current_chain = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(new_chain)
        else:
            # delta_e >= 0
            # prob = exp(-delta_e / T)
            # If T is very small, this is 0.
            if T > 0:
                prob = math.exp(-delta_e / T)
                if rng.random() < prob:
                    current_chain = new_chain
                    current_energy = new_energy
                    
        steps += 1
        
    return best_chain, best_energy

def _try_reversal_move(chain, dimension, rng):
    """
    Try a segment reversal move.
    """
    n = len(chain)
    if n < 3:
        return None, False
        
    # Pick segment length
    # Short segments are more likely to be valid.
    # Let's pick length 2 to min(10, n-2)
    max_len = min(10, n - 2)
    seg_len = rng.randint(2, max_len)
    
    i = rng.randint(0, n - seg_len - 1)
    j = i + seg_len
    
    # Segment is chain[i..j]
    # Reversed: chain[i..j] reversed
    
    # Check collisions
    occupied = set(chain)
    for k in range(i, j + 1):
        occupied.remove(chain[k])
        
    new_segment = list(reversed(chain[i:j+1]))
    
    valid = True
    for p in new_segment:
        if p in occupied:
            valid = False
            break
            
    if not valid:
        return None, False
        
    # Check boundaries
    if i > 0:
        # New pos of i is old pos of j
        d = sum(abs(chain[i-1][k] - chain[j][k]) for k in range(dimension))
        if d != 1:
            return None, False
            
    if j < n - 1:
        # New pos of j is old pos of i
        d = sum(abs(chain[j+1][k] - chain[i][k]) for k in range(dimension))
        if d != 1:
            return None, False
            
    new_chain = chain[:i] + new_segment + chain[j+1:]
    return new_chain, True

def fold(sequence, dimension):
    """
    Fold the sequence using Simulated Annealing.
    """
    start_time = time.time()
    # Budget: 240s for all sequences. 
    # The contract says "240 s CPU budget for all sequences together".
    # But the evaluator kills at 330s wall clock.
    # We should distribute time.
    # Since we don't know how many sequences are called, we assume one call per sequence.
    # We should use a fixed fraction of the total budget?
    # Or just a fixed time per sequence?
    # If there are 26 sequences, 240/26 ~ 9s each.
    # Let's aim for 8 seconds per sequence to be safe.
    deadline = start_time + 8.0
    
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension for _ in range(n)]
    
    rng = random.Random(42)
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Generate initial chains
    # Try a few random walks
    num_initial = 5
    for _ in range(num_initial):
        chain = _random_walk(sequence, dimension, rng)
        if chain is None:
            continue
        energy = _compute_energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            
    if best_chain is None:
        # Fallback: straight line (might not be self-avoiding if n is large? No, straight line is SAW)
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
        
    # Phase 2: Annealing
    # Run SA on the best initial chain
    # If we have time, run SA on other good chains?
    # Let's just run SA on the best one for the remaining time.
    
    if time.time() < deadline - 0.5:
        final_chain, final_energy = _anneal(sequence, best_chain, dimension, rng, start_time, deadline)
        if final_energy < best_energy:
            best_energy = final_energy
            best_chain = final_chain
            
    # Convert to list of lists
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
