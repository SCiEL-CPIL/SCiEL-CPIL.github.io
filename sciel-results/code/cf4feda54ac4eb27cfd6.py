"""HP lattice protein folding combining greedy growth with local search refinement."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Track global deadline to ensure we finish within the sandbox limits
_START_TIME = time.time()
# The task says 120s CPU budget for all sequences. Evaluator kills at 210s.
# We aim to finish well before 120s to be safe, leaving margin for overhead.
DEADLINE = 100.0  # Seconds from start

def _remaining_time():
    return max(0, DEADLINE - (time.time() - _START_TIME))

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute the energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
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

def _local_search(sequence, chain, dimension, rng, time_limit):
    """
    Refine a chain using simulated annealing with local moves.
    Returns the best chain found.
    """
    if not chain:
        return chain
    
    n = len(chain)
    if n < 4:
        return chain

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Simple SA parameters
    T = 1.0
    T_min = 0.05
    cooling_rate = 0.99
    
    # Pre-compute a set of occupied positions for quick lookup
    # Note: Updating this set efficiently during moves is tricky. 
    # For small N, recomputing or using a set is fast enough.
    
    start_time = time.time()
    steps = 0
    
    while T > T_min and (time.time() - start_time) < time_limit and _remaining_time() > 5:
        # Choose a random move
        # Move types: 
        # 1. Single residue move (move a non-endpoint to an adjacent empty spot)
        # 2. Segment reversal (reverse a small segment)
        
        move_type = rng.random()
        moved = False
        old_chain = list(current_chain) # Backup for revert
        
        if move_type < 0.7:
            # Single move
            # Pick a random index between 1 and n-2
            idx = rng.randint(1, n - 2)
            old_pos = current_chain[idx]
            # Try to move to a random neighbor
            neighbors = list(_neighbours(old_pos))
            rng.shuffle(neighbors)
            for new_pos in neighbors:
                # Check if new_pos is occupied by another part of the chain
                # We need to check if new_pos is in current_chain, excluding the element at idx
                # Since n is small, linear scan is okay, but let's use a set for O(1)
                # Rebuilding set every step is O(N). 
                # For N=100, O(N) is 100 ops. 1000 steps = 100k ops. Very fast.
                occ_set = set(current_chain)
                # Remove old pos from set temporarily to check collision
                occ_set.remove(old_pos)
                if new_pos not in occ_set:
                    # Move is valid
                    current_chain[idx] = new_pos
                    moved = True
                    break
        else:
            # Reversal move
            # Pick a segment length
            seg_len = rng.randint(2, min(5, n - 2))
            start_idx = rng.randint(1, n - seg_len - 1)
            end_idx = start_idx + seg_len - 1
            
            # Perform reversal
            segment = current_chain[start_idx:end_idx+1]
            segment.reverse()
            
            # Check validity
            valid = True
            # Check boundaries
            if start_idx > 0 and current_chain[start_idx-1] in segment:
                valid = False
            if end_idx < n-1 and current_chain[end_idx+1] in segment:
                valid = False
            
            # Check internal self-avoidance
            if valid:
                # Check if any point in segment collides with the rest of the chain
                rest = set(current_chain[:start_idx] + current_chain[end_idx+1:])
                if any(p in rest for p in segment):
                    valid = False
            
            if valid:
                current_chain[start_idx:end_idx+1] = segment
                moved = True
            else:
                # Revert
                current_chain[start_idx:end_idx+1] = segment # segment is already reversed, so this restores it? 
                # Wait, segment was reversed in place. 
                # current_chain[start_idx:end_idx+1] is the reversed version.
                # To revert, we reverse it again.
                segment.reverse()
                current_chain[start_idx:end_idx+1] = segment
                # Actually, simpler: just copy from old_chain if invalid
                current_chain = list(old_chain)

        if not moved:
            T *= cooling_rate
            continue
            
        new_energy = _energy(sequence, current_chain)
        delta = new_energy - current_energy
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(current_chain)
        else:
            # Revert
            current_chain = list(old_chain)
            
        T *= cooling_rate
        steps += 1
        
    return best_chain

def fold(sequence, dimension):
    """
    Best conformation using greedy growth with optional local search refinement.
    """
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0 # Energy is negative, so 0 is "worse" than any valid fold
    
    # Phase 1: Generate diverse initial chains using greedy growth
    # The original program used 200 restarts. We can keep that.
    # We will keep the top 5 candidates for refinement.
    candidates = []
    
    # Determine time budget for growth
    # If we have plenty of time, do more growths.
    # If time is tight, do fewer.
    num_growths = 100
    
    for _ in range(num_growths):
        if _remaining_time() < 10:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        candidates.append((value, chain))
        
        # Keep only the best few to save memory/time, or just track best
        if best_chain is None or value < best_energy:
            best_energy = value
            best_chain = chain
            
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in best_chain]
    
    # Phase 2: Refine the best candidate (and maybe second best) using local search
    # Only do this if we have time left
    if _remaining_time() > 15:
        # Sort candidates by energy
        candidates.sort(key=lambda x: x[0])
        
        # Try to refine the best candidate
        # Allocate time based on remaining budget
        refine_time = min(5.0, _remaining_time() / 2.0)
        
        if refine_time > 1.0:
            refined_chain = _local_search(sequence, best_chain, dimension, rng, refine_time)
            refined_energy = _energy(sequence, refined_chain)
            
            if refined_energy < best_energy:
                best_energy = refined_energy
                best_chain = refined_chain
                
            # If time allows, try refining the second best
            if len(candidates) > 1 and _remaining_time() > 10:
                second_best_chain = candidates[1][1]
                refine_time2 = min(5.0, _remaining_time() / 2.0)
                if refine_time2 > 1.0:
                    refined_chain2 = _local_search(sequence, second_best_chain, dimension, rng, refine_time2)
                    refined_energy2 = _energy(sequence, refined_chain2)
                    if refined_energy2 < best_energy:
                        best_energy = refined_energy2
                        best_chain = refined_chain2

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
