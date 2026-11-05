"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline guard
START_TIME = time.time()
TIME_LIMIT = 240.0  # 240 seconds total for all sequences
SAFETY_MARGIN = 5.0 # Stop 5 seconds before hard limit

def _remaining_time():
    return TIME_LIMIT - (time.time() - START_TIME) - SAFETY_MARGIN

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _manhattan_dist(p1, p2):
    return sum(abs(a - b) for a, b in zip(p1, p2))

def _calculate_energy(sequence, chain):
    """Calculate energy of a given chain. Energy = - (# H-H contacts)."""
    if not chain:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _is_valid_move(chain, start, end, new_segment):
    """
    Check if replacing chain[start:end+1] with new_segment is valid (self-avoiding).
    new_segment must have the same length as the original segment.
    The connection points chain[start-1] and chain[end+1] must remain valid.
    """
    n = len(chain)
    if start == 0 and end == n - 1:
        # Whole chain moved, just check self-avoidance of new_segment
        return len(set(new_segment)) == len(new_segment)
    
    # Check connections
    prev_point = chain[start - 1] if start > 0 else None
    next_point = chain[end + 1] if end < n - 1 else None
    
    if prev_point is not None:
        if _manhattan_dist(prev_point, new_segment[0]) != 1:
            return False
        if prev_point in new_segment:
            return False
    if next_point is not None:
        if _manhattan_dist(new_segment[-1], next_point) != 1:
            return False
        if next_point in new_segment:
            return False
            
    # Check self-avoidance within new_segment
    if len(set(new_segment)) != len(new_segment):
        return False
        
    # Check collision with rest of chain
    # Rest of chain is chain[:start] + chain[end+1:]
    # We can optimize this by checking only against the specific points, but for simplicity 
    # and given small N, set intersection is okay. 
    # Actually, we should check if any point in new_segment is in the rest of the chain.
    rest_chain = chain[:start] + chain[end+1:]
    rest_set = set(rest_chain)
    for p in new_segment:
        if p in rest_set:
            return False
            
    return True

def _generate_random_walk(sequence, dimension, rng):
    """Generate a self-avoiding random walk starting at origin."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    chain = [(0, 0, 0)]
    occupied = {chain[0]: 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for nb in _neighbours(last):
            if nb not in occupied:
                candidates.append(nb)
        
        if not candidates:
            return None # Dead end
        
        # Pick randomly
        next_pt = rng.choice(candidates)
        chain.append(next_pt)
        occupied[next_pt] = i
        
    return chain

def _generate_greedy_walk(sequence, dimension, rng):
    """Generate a self-avoiding walk preferring H contacts."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
        
    chain = [(0, 0, 0)]
    occupied = {chain[0]: 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for nb in _neighbours(last):
            if nb not in occupied:
                # Score the candidate
                score = rng.random()
                if sequence[i] == "H":
                    # Check for potential H contacts with existing H's
                    for q in _neighbours(nb):
                        if q in occupied:
                            j = occupied[q]
                            if j < i - 1 and sequence[j] == "H":
                                score += 1.0
                candidates.append((score, nb))
                
        if not candidates:
            return None
            
        # Sort by score descending, pick top few or best
        candidates.sort(key=lambda x: x[0], reverse=True)
        # Pick best or random among top 3 to add diversity
        k = min(3, len(candidates))
        best_score, next_pt = rng.choice(candidates[:k])
        
        chain.append(next_pt)
        occupied[next_pt] = i
        
    return chain

def _local_search(sequence, chain, max_iters, rng, time_check_func):
    """
    Perform local search using 2-opt moves (segment reversal) and pivot moves.
    Returns improved chain.
    """
    n = len(chain)
    current_energy = _calculate_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Precompute contact map for faster energy delta? 
    # For now, we recalculate energy for simplicity, but we can optimize if needed.
    # Given N <= 136, O(N) energy calc is fast enough if moves are cheap.
    
    for _ in range(max_iters):
        if time_check_func() <= 0:
            break
            
        # Choose a move
        # 1. 2-opt: Reverse a segment [i, j]
        # 2. Pivot: Rotate a segment around a pivot
        
        # Let's try 2-opt first as it's simpler to validate
        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, n - 1)
        
        # Try reversing segment i..j
        new_seg = list(reversed(chain[i:j+1]))
        
        if _is_valid_move(chain, i, j, new_seg):
            new_chain = chain[:i] + new_seg + chain[j+1:]
            new_energy = _calculate_energy(sequence, new_chain)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = new_chain
                chain = new_chain # Accept improvement immediately for next step
                current_energy = new_energy
                continue # Restart loop to pick new i, j
        
        # Try Pivot move: Pick a pivot point p, and a segment starting after p.
        # Rotate the segment by 90, 180, 270 degrees around the bond connecting p and p+1.
        # This is more complex. Let's stick to 2-opt and maybe simple "kicks" if 2-opt stalls.
        # Actually, 2-opt on a lattice is restrictive. Let's add a "random kick" move.
        # Pick a random index k, and a random direction, and try to move the rest of the chain? No, that's not a move.
        
        # Let's implement a simple "local flip" or "swap" if possible? 
        # No, swapping breaks connectivity.
        
        # Alternative: Randomly perturb a few steps?
        # Let's try a "segment rotation" move.
        # Pick i, j. Rotate segment i..j around the vector (chain[i-1] -> chain[i])?
        
        # Given time constraints, 2-opt is a good start. If it stalls, we rely on SA restarts.
        
    return best_chain, best_energy

def _simulated_annealing(sequence, dimension, rng, time_budget):
    """
    Run SA for a specific time budget.
    """
    start_t = time.time()
    n = len(sequence)
    
    # Generate initial conformation
    init_chain = None
    # Try greedy first
    init_chain = _generate_greedy_walk(sequence, dimension, rng)
    if init_chain is None:
        init_chain = _generate_random_walk(sequence, dimension, rng)
    if init_chain is None:
        # Fallback to straight line (unfolded)
        init_chain = [(i, 0, 0) for i in range(n)]
        
    current_chain = init_chain
    current_energy = _calculate_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # SA Parameters
    T0 = 1.0
    Tf = 0.01
    # We will adjust cooling rate based on time, but since we use time-based loop, 
    # we can just decrease T linearly with time or use a fixed number of steps.
    # Let's use a fixed number of steps estimated from time.
    # Approx steps per second? Hard to say. Let's do a time-based loop with T decay.
    
    T = T0
    
    while time.time() - start_t < time_budget:
        if _remaining_time() <= 0:
            break
            
        # Generate a move
        # 1. 2-opt
        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, n - 1)
        
        new_seg = list(reversed(current_chain[i:j+1]))
        
        if _is_valid_move(current_chain, i, j, new_seg):
            new_chain = current_chain[:i] + new_seg + current_chain[j+1:]
            new_energy = _calculate_energy(sequence, new_chain)
            
            delta_e = new_energy - current_energy
            
            # Accept if better or with probability exp(-delta_e / T)
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                current_chain = new_chain
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
        
        # Decay temperature
        # Simple linear decay with time is tricky without knowing step count.
        # Let's just decay T a bit every step.
        T *= 0.9999 # Slow decay
        
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Best conformation using Simulated Annealing and Local Search.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    # Allocate time based on sequence length
    # Longer sequences need more time, but we have a global limit.
    # Let's reserve 50% of remaining time for the longest sequence if it's the last one?
    # Simpler: Allocate proportional to n, but cap it.
    # Actually, let's just use a fixed time slice per call, ensuring we don't exceed global limit.
    # The evaluator calls fold() 26 times.
    # If we don't know which call is which, we should be conservative.
    # Let's assume the calls are sequential.
    
    # Strategy: 
    # 1. Check global remaining time.
    # 2. If very little left, return best known (straight line or previous best if cached? No cache allowed across calls easily).
    # 3. Allocate a chunk of time for this specific fold call.
    
    # To be safe, let's allocate 5 seconds per call on average? 26 * 5 = 130s. Leaves buffer.
    # But longer chains are harder.
    # Let's allocate time = min(10, max(2, 240 / 26 * 1.5)) ?
    # Let's just use a dynamic time budget: 
    # If we have > 100s left, give this call 10s.
    # If we have < 20s left, give this call 2s.
    
    rem = _remaining_time()
    if rem <= 0:
        # Out of time, return straight line
        return [[i, 0, 0] for i in range(n)]
        
    # Heuristic time allocation
    if n > 100:
        time_budget = min(rem * 0.5, 20.0)
    elif n > 50:
        time_budget = min(rem * 0.3, 10.0)
    else:
        time_budget = min(rem * 0.2, 5.0)
        
    # Ensure we don't go below a minimum
    time_budget = max(time_budget, 1.0)
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0 # Energy is negative, so 0 is "worse" than any valid folded state usually
    
    # Run multiple SA restarts if time permits
    # Estimate time per SA run
    # Let's just run SA until time_budget is exhausted.
    
    chain, energy = _simulated_annealing(sequence, dimension, rng, time_budget)
    
    if chain is None:
        # Fallback
        chain = _generate_random_walk(sequence, dimension, rng)
        if chain is None:
            chain = [(i, 0, 0) for i in range(n)]
            energy = _calculate_energy(sequence, chain)
    else:
        energy = _calculate_energy(sequence, chain)
        
    # Try a few local search polish steps if time remains
    rem = _remaining_time()
    if rem > 2.0:
        # Run local search on the best chain found
        local_time = min(rem * 0.5, 5.0)
        polished_chain, polished_energy = _local_search(sequence, chain, 1000, rng, lambda: _remaining_time())
        if polished_energy < energy:
            chain = polished_chain
            energy = polished_energy
            
    return [list(p) for p in chain]
# EVOLVE-BLOCK-END
