"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

# Global deadline guard: must finish well before 330s wall-clock.
_TOTAL_START = time.time()
_SAFETY_MARGIN = 30.0  # seconds to leave before hard kill

def _remaining_time():
    """Return remaining seconds before our safety deadline."""
    return _TOTAL_START + 300.0 - _SAFETY_MARGIN - time.time()

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
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
            # Heuristic: Encourage compactness by penalizing distance from origin slightly?
            # Or just keep the free space bonus.
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Add a small random factor to break ties
            score = gain + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        # Pick the best option
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _try_single_move(sequence, chain, occupied, i, rng):
    """Try to move residue i to a neighbor site. Returns (new_chain, new_energy, delta) or None."""
    n = len(chain)
    old_site = chain[i]
    
    for new_site in _neighbours(old_site):
        # Check if new_site is free
        if new_site in occupied:
            # If it's occupied by i itself, it's not a move. But neighbours are distinct.
            # If occupied by j != i, skip.
            if occupied[new_site] != i:
                continue
            else:
                continue # Should not happen for neighbours
        
        # Check adjacency with prev
        if i > 0:
            prev = chain[i-1]
            if abs(new_site[0]-prev[0]) + abs(new_site[1]-prev[1]) + abs(new_site[2]-prev[2]) != 1:
                continue
        # Check adjacency with next
        if i < n - 1:
            nxt = chain[i+1]
            if abs(new_site[0]-nxt[0]) + abs(new_site[1]-nxt[1]) + abs(new_site[2]-nxt[2]) != 1:
                continue
        
        # Calculate delta energy
        delta = 0
        if sequence[i] == "H":
            # Lose contacts at old_site
            for q in _neighbours(old_site):
                j = occupied.get(q)
                if j is not None and j != i and sequence[j] == "H" and abs(i-j) > 1:
                    delta += 1
            # Gain contacts at new_site
            for q in _neighbours(new_site):
                j = occupied.get(q)
                if j is not None and j != i and sequence[j] == "H" and abs(i-j) > 1:
                    delta -= 1
        
        return new_site, delta
    return None

def _try_two_move(sequence, chain, occupied, i, rng):
    """Try to move residues i and i+1. Returns (new_i, new_ip1, delta) or None."""
    n = len(chain)
    if i + 1 >= n:
        return None
        
    old_i = chain[i]
    old_ip1 = chain[i+1]
    
    # The new position for i must be adjacent to chain[i-1] (if i>0)
    # And the new position for i+1 must be adjacent to new_i.
    # Also, new_i and new_ip1 must be adjacent.
    
    # Iterate over possible new positions for i
    # new_i must be a neighbor of old_i? No, it can be any neighbor of chain[i-1] that is free.
    # But to keep the move local, we can restrict new_i to neighbors of old_i.
    # Actually, a standard 2-residue move moves the pair to a new location.
    # Let's try moving the pair to a neighbor of the current pair's "center" or just try all neighbors of old_i.
    
    for new_i in _neighbours(old_i):
        # Check if new_i is valid for position i
        # 1. Not occupied by j != i, i+1
        # Note: old_i is occupied by i. old_ip1 is occupied by i+1.
        # If new_i is old_ip1, then i moves to i+1's spot. That's a swap? 
        # Standard moves usually disallow swapping with immediate neighbors to maintain unit steps?
        # If i moves to old_ip1, then i+1 must move to a neighbor of old_ip1.
        # Let's just check if new_i is occupied by someone else.
        if new_i in occupied:
            occ_idx = occupied[new_i]
            if occ_idx != i and occ_idx != i+1:
                continue
            # If it's i or i+1, we are effectively swapping or moving into the pair.
            # If new_i == old_i, no move.
            if new_i == old_i:
                continue
            # If new_i == old_ip1, then i takes i+1's place.
        
        # Check adjacency with prev (i-1)
        if i > 0:
            prev = chain[i-1]
            if abs(new_i[0]-prev[0]) + abs(new_i[1]-prev[1]) + abs(new_i[2]-prev[2]) != 1:
                continue
        
        # Now try possible new positions for i+1
        for new_ip1 in _neighbours(new_i):
            # Check if new_ip1 is valid for position i+1
            # 1. Not occupied by j != i, i+1
            if new_ip1 in occupied:
                occ_idx = occupied[new_ip1]
                if occ_idx != i and occ_idx != i+1:
                    continue
                if new_ip1 == old_ip1 and new_i == old_i:
                    continue # No move
            
            # Check adjacency with next (i+2)
            if i + 1 < n - 1:
                nxt = chain[i+2]
                if abs(new_ip1[0]-nxt[0]) + abs(new_ip1[1]-nxt[1]) + abs(new_ip1[2]-nxt[2]) != 1:
                # Wait, if i+1 is the last residue, no next check.
                    if i + 2 < n:
                        if abs(new_ip1[0]-nxt[0]) + abs(new_ip1[1]-nxt[1]) + abs(new_ip1[2]-nxt[2]) != 1:
                            continue
            
            # Check self-avoidance for the pair
            # We need to ensure new_i and new_ip1 are not occupied by any other residue.
            # The checks above `if new_i in occupied` and `if new_ip1 in occupied` handle this,
            # but we must be careful if new_i == old_ip1 and new_ip1 == old_i (swap).
            # If it's a swap, the occupied set needs to be updated correctly.
            
            # Calculate delta energy
            delta = 0
            # Remove old contacts for i and i+1
            if sequence[i] == "H":
                for q in _neighbours(old_i):
                    j = occupied.get(q)
                    if j is not None and j not in [i, i+1] and sequence[j] == "H" and abs(i-j) > 1:
                        delta += 1
            if sequence[i+1] == "H":
                for q in _neighbours(old_ip1):
                    j = occupied.get(q)
                    if j is not None and j not in [i, i+1] and sequence[j] == "H" and abs(i+1-j) > 1:
                        delta += 1
            
            # Add new contacts for i and i+1
            # Note: We must consider that i and i+1 are adjacent, so they don't count as contacts.
            # Also, we must subtract the contact between i and i+1 if it existed? 
            # In HP model, contacts are only between non-adjacent residues.
            # So i and i+1 never form a contact.
            
            # When adding new contacts, we must check against the *new* occupied set.
            # This is tricky to do without updating the set.
            # We can approximate by checking against the old set, but excluding i and i+1,
            # and adding any new neighbors that are H.
            
            # Let's do a more precise calculation by simulating the move on a copy of the occupied set?
            # That's expensive. 
            # Alternative: 
            # Delta = Energy(new) - Energy(old)
            # We can compute the change in contacts for i and i+1.
            
            # Contacts lost:
            # i loses contacts with H neighbors at old_i (excluding i+1)
            # i+1 loses contacts with H neighbors at old_ip1 (excluding i)
            
            # Contacts gained:
            # i gains contacts with H neighbors at new_i (excluding i+1)
            # i+1 gains contacts with H neighbors at new_ip1 (excluding i)
            
            # However, if new_i is a neighbor of some H residue k, and k is not i+1, it's a gain.
            # If new_ip1 is a neighbor of some H residue k, and k is not i, it's a gain.
            
            # We must be careful if new_i and new_ip1 are neighbors of the same H residue.
            # That's fine, it's two separate contacts.
            
            # One edge case: If new_i is occupied by an H residue k (k != i, i+1), that's a collision, which we already checked.
            
            # So, we can sum the changes.
            
            # Lost by i:
            for q in _neighbours(old_i):
                j = occupied.get(q)
                if j is not None and j != i+1 and sequence[j] == "H" and abs(i-j) > 1:
                    delta += 1
            # Lost by i+1:
            for q in _neighbours(old_ip1):
                j = occupied.get(q)
                if j is not None and j != i and sequence[j] == "H" and abs(i+1-j) > 1:
                    delta += 1
            
            # Gained by i:
            for q in _neighbours(new_i):
                j = occupied.get(q)
                # j must be in the occupied set. 
                # But if j is i+1, we skip (adjacent).
                # If j is i, we skip (self).
                # If j is not i or i+1, and sequence[j]=='H', and abs(i-j)>1, it's a gain.
                # BUT, if new_i is the *same* as old_ip1, then occupied.get(new_i) is i+1. 
                # If new_i is *different* from old_ip1, but occupied by some k, then k is a gain.
                if j is not None and j != i+1 and j != i and sequence[j] == "H" and abs(i-j) > 1:
                    delta -= 1
            
            # Gained by i+1:
            for q in _neighbours(new_ip1):
                j = occupied.get(q)
                if j is not None and j != i and j != i+1 and sequence[j] == "H" and abs(i+1-j) > 1:
                    delta -= 1
            
            # Special case: If new_i == old_ip1 and new_ip1 == old_i (swap)
            # The "lost" and "gained" might cancel out or double count?
            # If we swap i and i+1:
            # i moves to old_ip1. i+1 moves to old_i.
            # Lost by i at old_i: contacts with H neighbors.
            # Gained by i at old_ip1: contacts with H neighbors.
            # This logic holds.
            
            return new_i, new_ip1, delta
    return None

def _local_search(sequence, chain, max_iters, rng, time_limit):
    """
    Local search with single and two-residue moves. Steepest descent.
    """
    n = len(sequence)
    if n < 3:
        return chain
    
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    occupied = {p: i for i, p in enumerate(chain)}
    
    iters = 0
    no_improve_count = 0
    max_no_improve = 1000 # Restart or stop if stuck for too long? 
    # For now, just stop if time runs out.
    
    while iters < max_iters:
        if time.time() > time_limit:
            break
        iters += 1
        
        improved = False
        # Choose move type: 70% single, 30% double
        move_type = rng.random()
        
        if move_type < 0.7:
            # Single residue move
            i = rng.randrange(1, n - 1)
            result = _try_single_move(sequence, chain, occupied, i, rng)
            if result:
                new_site, delta = result
                new_energy = current_energy + delta
                if new_energy < current_energy:
                    # Apply move
                    old_site = chain[i]
                    occupied.pop(old_site)
                    occupied[new_site] = i
                    chain[i] = new_site
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                    improved = True
                    no_improve_count = 0
        else:
            # Two residue move
            i = rng.randrange(1, n - 2)
            result = _try_two_move(sequence, chain, occupied, i, rng)
            if result:
                new_i, new_ip1, delta = result
                new_energy = current_energy + delta
                if new_energy < current_energy:
                    # Apply move
                    old_i = chain[i]
                    old_ip1 = chain[i+1]
                    occupied.pop(old_i)
                    occupied.pop(old_ip1)
                    occupied[new_i] = i
                    occupied[new_ip1] = i + 1
                    chain[i] = new_i
                    chain[i+1] = new_ip1
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                    improved = True
                    no_improve_count = 0
        
        if not improved:
            no_improve_count += 1
            # If we are stuck for a long time, we might want to do a random restart or perturb.
            # But for simplicity, we just continue.
            if no_improve_count > 5000:
                break

    return best_chain

def fold(sequence, dimension):
    """Best conformation of a fixed number of randomized greedy growths, with local search."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0
    
    n = len(sequence)
    
    # Run growths with dynamic budget based on remaining time
    if n <= 48:
        num_growths = 300
    elif n <= 80:
        num_growths = 200
    else:
        num_growths = 150
    
    for attempt in range(num_growths):
        if _remaining_time() < 5.0:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if best is None or value < best_energy:
            best, best_energy = chain, value
    
    # Try local search on the best chain if time permits
    if best is not None and _remaining_time() > 10.0:
        remaining = _remaining_time()
        ls_time_limit = min(time.time() + remaining - 5.0, _TOTAL_START + 300.0 - _SAFETY_MARGIN)
        # Scale iterations with length, but cap it
        max_ls_iters = max(500, min(20000, 100 * len(sequence)))
        improved_chain = _local_search(sequence, best, max_ls_iters, rng, ls_time_limit)
        improved_energy = _energy(sequence, improved_chain)
        if improved_energy < best_energy:
            best, best_energy = improved_chain, improved_energy
            
        # Second pass if time remains
        if _remaining_time() > 5.0:
            ls_time_limit2 = min(time.time() + _remaining_time() - 2.0, _TOTAL_START + 300.0 - _SAFETY_MARGIN)
            max_ls_iters2 = max(300, min(10000, 50 * len(sequence)))
            improved_chain2 = _local_search(sequence, best, max_ls_iters2, rng, ls_time_limit2)
            improved_energy2 = _energy(sequence, improved_chain2)
            if improved_energy2 < best_energy:
                best, best_energy = improved_chain2, improved_energy2
    
    if best is None:  # a straight line is always self-avoiding
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
