"""HP lattice protein folding using simulated annealing with reversals and slithering."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline guard
_START_TIME = time.time()
_TOTAL_BUDGET = 180.0  # Safety margin from 210s limit
_DEADLINE = _START_TIME + _TOTAL_BUDGET

def _time_left():
    return _DEADLINE - time.time()

def _neighbours(point):
    """Yield the 4 neighbors of a point in 2D."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _build_indices(chain):
    """Build dictionary mapping coordinate to index."""
    return {p: i for i, p in enumerate(chain)}

def _compute_energy(sequence, chain, indices=None):
    """Compute the energy of the chain. Energy = - (number of H-H contacts)."""
    if indices is None:
        indices = _build_indices(chain)
    
    contacts = 0
    n = len(chain)
    for i in range(n):
        if sequence[i] == 'H':
            p = chain[i]
            for q in _neighbours(p):
                j = indices.get(q)
                if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                    # Avoid double counting: only count if i < j
                    if i < j:
                        contacts += 1
    return -contacts

def _greedy_grow(sequence, rng):
    """
    Generate a random self-avoiding walk using a greedy strategy with some randomness.
    Returns None if stuck.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Start at origin
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        
        for q in _neighbours(last):
            if q in occupied:
                continue
            
            # Score candidates
            score = rng.random()
            
            # Prefer moves that create H-H contacts
            if sequence[i] == 'H':
                for h_pos in _neighbours(q):
                    idx = occupied.get(h_pos)
                    if idx is not None and idx < i - 1 and sequence[idx] == 'H':
                        score += 10.0
            
            # Slight preference for moves that don't trap the chain (simple heuristic: count free neighbors)
            free_neighbors = 0
            for f in _neighbours(q):
                if f not in occupied:
                    free_neighbors += 1
            score += free_neighbors * 0.1
            
            candidates.append((score, q))
        
        if not candidates:
            return None
        
        # Pick the best candidate (highest score)
        candidates.sort(key=lambda x: x[0], reverse=True)
        next_pos = candidates[0][1]
        
        chain.append(next_pos)
        occupied[next_pos] = i
        
    return chain

def _simulated_annealing(sequence, initial_chain, time_limit):
    """
    Perform Simulated Annealing on the initial chain.
    Returns the best chain found.
    """
    n = len(sequence)
    if n < 2:
        return initial_chain
    
    # State
    chain = list(initial_chain)
    indices = _build_indices(chain)
    current_energy = _compute_energy(sequence, chain, indices)
    best_energy = current_energy
    best_chain = list(chain)
    
    # SA Parameters
    start_temp = 1.0
    end_temp = 0.001
    cooling_rate = 0.999
    
    # We will run for a fixed number of iterations or until time limit
    # Estimate iterations based on time limit
    # Each iteration involves a move and energy calc.
    # Let's aim for ~100k iterations per second if possible, but safe to just loop until time.
    
    temp = start_temp
    rng = random.Random(hash(tuple(initial_chain)) % (2**32))
    
    start_time = time.time()
    
    while temp > end_temp and (time.time() - start_time) < time_limit:
        
        # Choose move type: 70% slither, 30% reversal
        if rng.random() < 0.7:
            # Slither: move the tail
            if n < 2:
                break
            last_idx = n - 1
            last_pos = chain[last_idx]
            
            candidates = []
            for q in _neighbours(last_pos):
                if q in indices:
                    # Can't move into occupied space, unless it's the previous residue (backtrack)
                    # But slithering usually implies extending or shifting.
                    # Standard slither: remove last, add new last.
                    # The new last cannot be the previous last (chain[last_idx-1]) as that would be a backtrack which is allowed in SAW?
                    # Actually, in SAW, you can't step on any occupied point.
                    # So if q is in indices, it's invalid unless q == chain[last_idx-1] (which is occupied by last_idx-1).
                    # Wait, if we remove last_idx, then chain[last_idx-1] becomes the new last.
                    # The new position q must not be occupied by any other residue.
                    # Since we are removing last_idx, the only "occupied" points are 0..last_idx-1.
                    # So q must not be in indices[0:last_idx].
                    # If q == chain[last_idx-1], it is in indices. So we can't step back immediately.
                    # This is a standard constraint for slithering moves in SAW.
                    continue
                
                candidates.append(q)
            
            if not candidates:
                continue
            
            new_pos = rng.choice(candidates)
            
            # Check if new_pos is valid (not in 0..last_idx-1)
            # We already checked q in indices, which includes 0..last_idx.
            # Since we are removing last_idx, we need to ensure new_pos is not in 0..last_idx-1.
            # The check `if q in indices` covers all occupied points.
            # If we remove last_idx, the set of occupied points becomes indices without last_idx.
            # So if new_pos was not in indices (before removal), it is definitely not in the new set.
            
            # Calculate Delta E
            # Only the last residue moves.
            # Old contacts involving last_idx:
            old_contacts = 0
            if sequence[last_idx] == 'H':
                for q in _neighbours(last_pos):
                    j = indices.get(q)
                    if j is not None and j != last_idx - 1 and sequence[j] == 'H':
                        if last_idx < j: # Only count if i < j to match global calc
                            old_contacts += 1
                    elif j == last_idx - 1:
                        # Adjacent, no contact energy
                        pass
                    # Note: The global calc counts i<j.
                    # If last_idx is the larger index, it counts neighbors with j < last_idx.
                    # If last_idx is the smaller index, it counts neighbors with j > last_idx.
                    # Let's just recalculate energy for the whole chain for simplicity in Slither?
                    # No, that's O(N). For Slither, O(1) is possible.
                    # Let's stick to a slightly slower but safer O(N) energy check for Slither if N is small, 
                    # or implement precise delta.
                    # Given N <= 100, O(N) energy check is fast (~100 ops).
                    # Let's just recompute energy for the modified chain.
                    pass
            
            # Modify chain
            old_last = chain.pop()
            chain.append(new_pos)
            
            # Update indices
            del indices[old_last]
            indices[new_pos] = last_idx
            
            new_energy = _compute_energy(sequence, chain, indices)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / temp):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
            else:
                # Revert
                chain.pop()
                chain.append(old_last)
                del indices[new_pos]
                indices[old_last] = last_idx
                
        else:
            # Reversal: reverse a segment [i, j]
            # i is the start index, j is the end index.
            # Length must be >= 3 (i.e., j - i >= 2) to have a non-trivial reversal?
            # Reversal of length 2 (i, i+1) just swaps them? No, it reverses the direction.
            # If we reverse [i, j], the segment becomes chain[j], chain[j-1]... chain[i].
            # The connection at i-1 -> i must still be valid (unit step).
            # The connection at j -> j+1 must still be valid.
            # And the internal segment must be self-avoiding.
            
            if n < 4:
                continue
            
            # Choose random i, j
            i = rng.randint(0, n - 3)
            j = rng.randint(i + 2, n - 1)
            
            # Extract segment
            segment = chain[i:j+1]
            reversed_segment = segment[::-1]
            
            # Check connectivity at boundaries
            # Left boundary: chain[i-1] to reversed_segment[0] (which is original chain[j])
            # If i > 0:
            if i > 0:
                prev = chain[i-1]
                new_start = reversed_segment[0]
                if abs(prev[0] - new_start[0]) + abs(prev[1] - new_start[1]) != 1:
                    continue
            
            # Right boundary: reversed_segment[-1] (which is original chain[i]) to chain[j+1]
            # If j < n-1:
            if j < n - 1:
                next_res = chain[j+1]
                new_end = reversed_segment[-1]
                if abs(next_res[0] - new_end[0]) + abs(next_res[1] - new_end[1]) != 1:
                    continue
            
            # Check self-avoidance of the new full chain
            # We need to ensure that the reversed segment does not collide with:
            # 1. The rest of the chain (0..i-1 and j+1..n-1)
            # 2. Itself (internal collisions)
            
            # Create a set of occupied points excluding the segment [i, j]
            # occupied_rest = set(chain[0:i]) | set(chain[j+1:n])
            
            # For efficiency, we can check collisions on the fly.
            # But building a set for the rest is O(N).
            # Let's build the new chain and check for duplicates.
            
            new_chain = chain[:i] + reversed_segment + chain[j+1:]
            
            # Check for self-avoidance
            seen = set()
            valid = True
            for p in new_chain:
                if p in seen:
                    valid = False
                    break
                seen.add(p)
            
            if not valid:
                continue
            
            # Calculate energy
            new_indices = _build_indices(new_chain)
            new_energy = _compute_energy(sequence, new_chain, new_indices)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / temp):
                chain = new_chain
                indices = new_indices
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
        
        # Cooling
        temp *= cooling_rate
        
        # Periodic check for time
        if _time_left() < 5.0:
            break

    return best_chain

def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    Returns a list of [x, y] coordinates.
    """
    if dimension != 2:
        raise ValueError("Only 2D supported")
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    # Time allocation strategy:
    # 1. Generate initial candidates.
    # 2. Run SA on the best few candidates.
    # 3. Use remaining time for more candidates.
    
    # We have ~180s total for 10 sequences, so ~18s per sequence.
    # Let's allocate 10s for SA and 8s for generation/exploration.
    
    best_chain = None
    best_energy = float('inf')
    
    rng = random.Random(12345 + len(sequence))
    
    # Phase 1: Generate a few initial candidates and SA them
    # We will try to generate 5-10 good initial chains.
    
    candidates = []
    gen_start = time.time()
    gen_time_limit = 3.0  # 3 seconds for generation
    
    while time.time() - gen_start < gen_time_limit and _time_left() > 10.0:
        chain = _greedy_grow(sequence, rng)
        if chain is not None:
            energy = _compute_energy(sequence, chain)
            candidates.append((energy, chain))
            
    if not candidates:
        # Fallback: straight line
        return [[i, 0] for i in range(n)]
    
    # Sort candidates by energy
    candidates.sort(key=lambda x: x[0])
    
    # Phase 2: SA on the top candidates
    # Allocate remaining time minus a safety margin for SA
    sa_time_per_candidate = 2.0 # 2 seconds per SA run
    
    # How many candidates can we SA?
    # Let's take the top 3-5.
    num_sa_candidates = min(5, len(candidates))
    
    for i in range(num_sa_candidates):
        if _time_left() < 5.0:
            break
            
        _, init_chain = candidates[i]
        
        # Adjust time limit for this SA run
        # If we have 10s left and 2 candidates left, give 5s each.
        remaining_candidates = num_sa_candidates - i
        time_limit = (_time_left() - 2.0) / remaining_candidates
        time_limit = max(0.5, time_limit)
        
        sa_chain = _simulated_annealing(sequence, init_chain, time_limit)
        
        if sa_chain:
            sa_energy = _compute_energy(sequence, sa_chain)
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_chain = sa_chain
        
        # Also update with the initial chain if it's better (unlikely but possible)
        if _compute_energy(sequence, init_chain) < best_energy:
            best_energy = _compute_energy(sequence, init_chain)
            best_chain = init_chain

    # Phase 3: If we have time, generate more candidates and SA them briefly
    # This is a "scout" phase.
    if _time_left() > 10.0:
        scout_start = time.time()
        while time.time() - scout_start < (_time_left() - 5.0):
            chain = _greedy_grow(sequence, rng)
            if chain is None:
                continue
            
            energy = _compute_energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
                
            # Quick SA on this candidate if it's promising (close to best)
            if energy < best_energy + 2: # Within 2 units of best
                # Give it a short SA run
                quick_sa_chain = _simulated_annealing(sequence, chain, 1.0)
                if quick_sa_chain:
                    qe = _compute_energy(sequence, quick_sa_chain)
                    if qe < best_energy:
                        best_energy = qe
                        best_chain = quick_sa_chain

    if best_chain is None:
        # Fallback
        return [[i, 0] for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
