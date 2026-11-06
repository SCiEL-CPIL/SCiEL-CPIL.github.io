"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline guard
_START_TIME = time.time()
_TOTAL_BUDGET = 240.0  # 240s CPU budget for all sequences
# We reserve a small buffer for overhead, so we aim for 235s usage
_EFFECTIVE_BUDGET = 235.0

def _neighbours(point):
    """Yield 6 neighbors on cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Calculate energy of a chain. Energy = - (number of H-H contacts)."""
    if not chain:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                # j must be a valid index, not adjacent in chain, and H
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _grow(sequence, dimension, rng):
    """Greedy growth with randomization."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            gain = 0
            if sequence[i] == 'H':
                # Count new H-H contacts if we place H here
                for q in _neighbours(site):
                    if q in occupied:
                        prev_idx = occupied[q]
                        if prev_idx < i - 1 and sequence[prev_idx] == 'H':
                            gain += 1
            
            # Heuristic: prefer sites with more free space to avoid trapping
            free_space = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free_space += 1
            
            # Score: gain is dominant, free space is secondary, random noise
            score = gain * 10.0 + free_space * 0.1 + rng.random()
            options.append((score, site))
        
        if not options:
            return None # Dead end
        
        # Pick best option
        best_score, best_site = max(options)
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain

def _local_refine(sequence, chain, dimension, rng, time_limit):
    """
    Refine chain using Simulated Annealing with local moves.
    Returns best chain and energy found.
    """
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    T = 2.0
    T_min = 0.01
    # Estimate number of steps we can afford
    # Assume ~1000 steps per second is a conservative estimate for Python with overhead
    # We will just loop until time_limit is reached
    
    step_count = 0
    while True:
        if time.time() - start_time > time_limit:
            break
        
        # Cooldown
        # Simple geometric cooling
        T = max(T_min, T * 0.999)
        
        # Choose a random internal index to move
        # Avoid ends to maintain connectivity easily
        i = rng.randint(1, n - 2)
        
        # Try moving current_chain[i] to a neighbor
        pos = current_chain[i]
        neighbors = list(_neighbours(pos))
        rng.shuffle(neighbors)
        
        moved = False
        for new_pos in neighbors:
            if new_pos in occupied:
                continue
            
            # Check if moving i to new_pos creates a collision with i-1 or i+1?
            # No, because i-1 and i+1 are fixed. The only risk is if new_pos is 
            # currently occupied by another part of the chain (checked above).
            # But we must ensure the move doesn't disconnect the chain?
            # A single residue move keeps the chain connected if the residue moves
            # to an adjacent spot. The bond lengths are preserved.
            
            # Calculate Delta E
            # Remove i from pos
            # Add i to new_pos
            
            # We need to check contacts involving i
            # Old contacts involving i:
            old_delta = 0
            if sequence[i] == 'H':
                for q in _neighbours(pos):
                    j = occupied.get(q)
                    if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                        old_delta -= 1 # We lose this contact (energy increases, so delta is positive? No, energy is -contacts. Losing contact means energy goes up. So delta = -old_contacts - (-new_contacts). Let's just recalc or be careful.)
            
            # It's safer and fast enough to just recalc energy for the whole chain?
            # N is up to 136. Recalc is O(N). 
            # But we are in a tight loop. 
            # Let's try incremental.
            
            # Energy = -Contacts.
            # Delta E = E_new - E_old
            # E_new = - (C_old - C_lost + C_gained)
            # E_old = - C_old
            # Delta E = C_old - C_lost + C_gained - C_old = C_gained - C_lost
            # So if we gain 1 contact, Delta E = 1 (Energy increases, BAD).
            # If we lose 1 contact, Delta E = -1 (Energy decreases, GOOD).
            
            # Let's count C_lost: Contacts involving i at pos that will be broken.
            # Let's count C_gained: Contacts involving i at new_pos that will be formed.
            
            c_lost = 0
            if sequence[i] == 'H':
                for q in _neighbours(pos):
                    j = occupied.get(q)
                    if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                        c_lost += 1
            
            c_gained = 0
            if sequence[i] == 'H':
                for q in _neighbours(new_pos):
                    j = occupied.get(q)
                    if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                        c_gained += 1
            
            delta_e = c_gained - c_lost
            
            # Acceptance criterion
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                # Accept move
                # Update chain
                occupied.pop(pos)
                occupied[new_pos] = i
                current_chain[i] = new_pos
                current_energy += delta_e
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                
                moved = True
                break # Only one move per iteration
        
        # If no move was made, we can still update T
        step_count += 1

    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Main entry point.
    """
    n = len(sequence)
    # Allocate time based on length
    # Longer chains are harder, but also have more potential contacts.
    # Let's give each sequence a fixed portion of the budget, 
    # but allow early termination if we hit a good score or run out of time.
    
    # Simple strategy: 
    # 1. Generate a few initial structures.
    # 2. Refine the best one(s).
    
    # Time budget for this sequence
    # We don't know how many sequences are left, but we can estimate.
    # The task says "240 s CPU budget for all sequences together".
    # We will assume we are called sequentially.
    # We will track global time usage.
    
    global _START_TIME
    elapsed = time.time() - _START_TIME
    remaining = _EFFECTIVE_BUDGET - elapsed
    
    if remaining <= 0:
        # Out of time, return a straight line
        return [[i, 0, 0] for i in range(n)]
    
    # Determine number of restarts
    # Short chains: more restarts. Long chains: fewer restarts, longer refinement.
    if n <= 20:
        n_restarts = 50
        refine_time = min(0.1, remaining / n_restarts)
    elif n <= 50:
        n_restarts = 30
        refine_time = min(0.5, remaining / n_restarts)
    elif n <= 100:
        n_restarts = 20
        refine_time = min(2.0, remaining / n_restarts)
    else:
        n_restarts = 10
        refine_time = min(5.0, remaining / n_restarts)
    
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0 # Maximize energy (minimize negative value)
    
    for r in range(n_restarts):
        if time.time() - _START_TIME > _EFFECTIVE_BUDGET:
            break
            
        # Generate initial chain
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Refine
        # Allocate remaining time for this refinement, but cap it
        time_left_for_seq = _EFFECTIVE_BUDGET - (time.time() - _START_TIME)
        if time_left_for_seq <= 0:
            break
        
        # Give this refinement a portion of the remaining time
        # If we have many restarts left, give less time to each.
        # If this is the last restart, give all remaining time.
        if r == n_restarts - 1:
            t_alloc = time_left_for_seq
        else:
            t_alloc = min(refine_time, time_left_for_seq / (n_restarts - r))
        
        refined_chain, energy = _local_refine(sequence, chain, dimension, rng, t_alloc)
        
        if best_chain is None or energy < best_energy:
            best_energy = energy
            best_chain = refined_chain
        
        # Early stopping if we hit a very good score? 
        # We don't know the best known energy, so we can't easily do this.
        # But if energy is very low (many contacts), we might stop.
        # Max possible contacts is roughly n/2 for HP chains? 
        # Let's just run out of time.

    if best_chain is None:
        # Fallback: straight line
        best_chain = [[i, 0, 0] for i in range(n)]
    else:
        # Convert tuples to lists if necessary (they are already lists from _grow, 
        # but _local_refine might return lists of tuples or lists)
        # _grow returns list of tuples.
        # _local_refine returns list of tuples (from chain which is list of tuples).
        # The interface requires list of lists.
        best_chain = [list(p) for p in best_chain]
        
    return best_chain

# EVOLVE-BLOCK-END
