"""
Strategy: Contact-First Constructive Heuristic + Delta Local Search.
1. Constructive Phase: Grow the chain by prioritizing moves that create
   new H-H contacts (especially "pockets" where a new H touches >=2 existing Hs).
   This creates dense, high-energy initial structures, unlike the random greedy
   growth in the seed which often results in loose structures.
2. Refinement Phase: Perform local search using segment reversals and single
   residue moves, calculating energy changes in O(1) or O(k) instead of O(N)
   to allow many more steps within the time budget.
"""

import random
import time
import sys

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290  # 300s limit, keep 10s margin

def _remaining_time():
    return _DEADLINE - time.time()

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _is_adjacent(i, j):
    return abs(i - j) == 1

def _initial_grow(sequence, dimension, rng):
    """
    Constructive growth:
    - Prefer sites that create H-H contacts.
    - Strongly prefer 'pockets' (sites adjacent to 2+ Hs).
    - If no contacts possible, prefer sites with more free space to avoid traps.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    # Set of H positions for quick lookup
    h_positions = set()
    if sequence[0] == 'H':
        h_positions.add(start)

    for i in range(1, n):
        current = chain[-1]
        candidates = []
        is_h = sequence[i] == 'H'
        
        for site in _neighbours(current):
            if site in occupied:
                continue
            
            # Check if this site would create H-H contacts
            contacts = 0
            if is_h:
                for nb in _neighbours(site):
                    if nb in occupied and nb in h_positions:
                        # Ensure it's not the immediate predecessor (which is not H-H contact by definition in HP model usually, 
                        # but let's be strict: contact is non-bonded. Bonded is i-1. So if nb is chain[i-1], it's not a contact.
                        # Wait, the definition says "neighbouring lattice sites that are not adjacent along the chain".
                        # So if nb is chain[i-1], it is adjacent along the chain, so it's NOT a contact.
                        # However, chain[i-1] is the only one adjacent to current that is in the chain.
                        # Any other neighbor in occupied is non-bonded.
                        
                        # Check index
                        idx = occupied[nb]
                        if not _is_adjacent(i, idx):
                            contacts += 1
            
            # Heuristic score:
            # 1. Maximize contacts (especially pockets)
            # 2. If no contacts, maximize free space (escape routes)
            free_space = 0
            if contacts == 0:
                for nb in _neighbours(site):
                    if nb not in occupied:
                        free_space += 1
            
            # Random tie-breaker
            noise = rng.random()
            
            # Score: 
            # If H: contacts are huge. 2 contacts (pocket) is best.
            # If P: contacts don't matter directly, but free space helps.
            if is_h:
                score = contacts * 10.0 + free_space * 0.1 + noise
            else:
                score = free_space * 1.0 + noise
            
            candidates.append((score, site))
        
        if not candidates:
            return None # Trapped
        
        # Pick best
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_site = candidates[0][1]
        
        chain.append(best_site)
        occupied[best_site] = i
        if is_h:
            h_positions.add(best_site)
            
    return chain

def _compute_energy(sequence, chain):
    """Compute total energy (negative number of H-H contacts)."""
    n = len(sequence)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and not _is_adjacent(i, j) and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _local_search(sequence, chain, rng):
    """
    Local search with segment reversals and single moves.
    Uses delta energy calculation for speed.
    """
    n = len(sequence)
    if n < 4:
        return chain
    
    # Precompute H indices
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    
    # Occupied map
    occupied = {p: i for i, p in enumerate(chain)}
    
    # Current energy
    current_energy = _compute_energy(sequence, chain)
    
    # Contact count per residue (for delta updates)
    # contact_count[i] = number of H-H contacts involving residue i
    contact_count = [0] * n
    # To update contact_count efficiently, we need to know which contacts exist.
    # Let's maintain a set of contacts (i, j) with i < j
    contacts_set = set()
    for i in h_indices:
        p = chain[i]
        for q in _neighbours(p):
            j = occupied.get(q)
            if j is not None and not _is_adjacent(i, j) and sequence[j] == 'H' and i < j:
                contacts_set.add((i, j))
                contact_count[i] += 1
                contact_count[j] += 1
    
    # Verify current energy
    # current_energy should be -len(contacts_set)
    
    best_chain = list(chain)
    best_energy = current_energy
    
    # Local search loop
    # We will try moves until time runs out or no improvement for a while
    no_improve_count = 0
    max_no_improve = 5000
    
    while _remaining_time() > 0.1:
        if no_improve_count > max_no_improve:
            break
            
        # Choose a move type
        # 1. Single move (low probability, high risk)
        # 2. Segment reversal (higher probability, better for untangling)
        
        move_type = rng.random()
        if move_type < 0.3:
            # Single move
            i = rng.randint(1, n - 2) # Endpoints are fixed
            old_pos = chain[i]
            
            # Find a valid neighbor
            valid_moves = []
            for nb in _neighbours(old_pos):
                if nb not in occupied or occupied[nb] == i:
                    # nb is free or is the current position (we move away from it, so it becomes free)
                    # Wait, if we move i from old_pos to nb, old_pos becomes free.
                    # So nb is valid if nb is not occupied by any OTHER residue.
                    if nb == old_pos:
                        continue
                    if nb in occupied:
                        j = occupied[nb]
                        if j != i:
                            continue
                    # Check if moving to nb causes collision with neighbors of i?
                    # No, we just need nb to be free.
                    valid_moves.append(nb)
            
            if not valid_moves:
                continue
                
            new_pos = rng.choice(valid_moves)
            
            # Calculate delta energy
            # Remove i from old_pos, add i to new_pos
            # Contacts involving i change.
            
            # Old contacts involving i:
            old_contacts = 0
            for nb in _neighbours(old_pos):
                j = occupied.get(nb)
                if j is not None and j != i and not _is_adjacent(i, j) and sequence[j] == 'H':
                    if sequence[i] == 'H':
                        old_contacts += 1
            
            # New contacts involving i:
            new_contacts = 0
            for nb in _neighbours(new_pos):
                j = occupied.get(nb)
                if j is not None and j != i and not _is_adjacent(i, j) and sequence[j] == 'H':
                    # Note: old_pos is now free, so if nb == old_pos, j is None (or i if we haven't updated yet, but we treat it as free)
                    # Actually, occupied still has old_pos: i. We need to be careful.
                    # Let's simulate:
                    # If nb == old_pos, occupied[nb] is i. We skip j==i.
                    # If nb is some other point, occupied[nb] is j.
                    if sequence[i] == 'H':
                        new_contacts += 1
            
            delta = new_contacts - old_contacts
            if sequence[i] != 'H':
                delta = 0 # P residues don't form contacts, so moving them doesn't change energy directly?
                # Wait, moving a P residue doesn't change the number of H-H contacts directly,
                # UNLESS it was blocking a contact? No, P residues don't participate in H-H contacts.
                # So delta is always 0 for P moves?
                # Yes, energy is defined as - (# H-H contacts). Moving a P doesn't change H-H contacts.
                # So we only move H residues for energy improvement?
                # But moving P residues can untangle the chain to allow H moves later.
                # For now, let's only accept P moves if they don't break anything (delta 0) 
                # and we use them to escape local minima via random walks.
                # However, if we only accept improvements, P moves (delta 0) won't be accepted unless we allow neutral moves.
                # Let's allow neutral moves with a small probability to escape plateaus.
                pass
            
            if sequence[i] == 'H':
                if delta > 0:
                    # Accept
                    # Update structures
                    del occupied[old_pos]
                    occupied[new_pos] = i
                    chain[i] = new_pos
                    current_energy += delta
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                elif delta == 0 and rng.random() < 0.05:
                    # Neutral move
                    del occupied[old_pos]
                    occupied[new_pos] = i
                    chain[i] = new_pos
                    no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                # P move
                if rng.random() < 0.01:
                    del occupied[old_pos]
                    occupied[new_pos] = i
                    chain[i] = new_pos
                    no_improve_count += 1
                else:
                    no_improve_count += 1

        else:
            # Segment Reversal
            # Choose a segment [i, j] and reverse it
            # This is a powerful move for untangling.
            # Length of segment: small to medium
            seg_len = rng.randint(2, min(10, n // 2))
            i = rng.randint(1, n - seg_len - 1)
            j = i + seg_len - 1
            
            # Check if reversal is valid (self-avoiding)
            # The segment [i, j] is reversed.
            # New positions:
            # chain[i] -> chain[j]
            # chain[i+1] -> chain[j-1]
            # ...
            # chain[j] -> chain[i]
            
            # We need to ensure that the new positions of the internal nodes
            # do not collide with:
            # 1. The rest of the chain (indices < i or > j)
            # 2. Each other (which is guaranteed if the original was self-avoiding and the reversal is a simple flip? No.)
            #    Actually, reversing a segment in a self-avoiding walk keeps it self-avoiding internally,
            #    but might collide with the outside.
            
            # Let's check collisions with outside.
            # Outside indices: [0, i-1] and [j+1, n-1]
            # The new positions of the segment are the old positions of the segment, just in reverse order.
            # So the SET of positions occupied by the segment is unchanged.
            # Therefore, collisions with the outside are impossible!
            # The only check needed is if the move is valid topologically?
            # Wait, the positions are the same set. So no new collisions with outside.
            # But does the reversal change the bonds?
            # Bond (i-1, i) becomes (i-1, j).
            # Bond (j, j+1) becomes (i, j+1).
            # These must be unit steps.
            
            p_prev = chain[i-1]
            p_next = chain[j+1]
            p_start = chain[i]
            p_end = chain[j]
            
            # Check if (p_prev, p_end) is a unit step
            if abs(p_prev[0] - p_end[0]) + abs(p_prev[1] - p_end[1]) + abs(p_prev[2] - p_end[2]) != 1:
                continue
            # Check if (p_start, p_next) is a unit step
            if abs(p_start[0] - p_next[0]) + abs(p_start[1] - p_next[1]) + abs(p_start[2] - p_next[2]) != 1:
                continue
            
            # Calculate delta energy
            # Contacts involving the segment might change.
            # Specifically, contacts between the segment and the outside might change
            # because the "identity" of the positions is swapped? No, the positions are the same.
            # But the RESIDUES at those positions are swapped/reordered.
            # So if residue A at pos X had a contact with residue B at pos Y (outside),
            # and now residue C is at pos X, does C have a contact with B?
            # Yes, if C is H and B is H.
            
            # So we need to recalculate contacts for the boundary residues of the segment
            # and any residues inside that had contacts with the outside?
            # Actually, any H residue inside the segment that had a contact with an H residue outside
            # will lose that contact if the new residue at that position is P.
            # And gain it if the new residue is H.
            
            # This is complex. Let's just recalculate energy for the segment boundaries and affected residues.
            # Or, simpler: since the set of positions is the same, the geometry is the same.
            # The only thing that changes is which residue is at which position.
            # So for each position in the segment, check if the new residue forms a contact with an outside H.
            
            # Let's identify all H residues in the segment and outside.
            # For each H in segment (new position), check neighbors in outside.
            # For each H in outside, check neighbors in segment.
            # This is O(L_seg * 6) + O(N_out * 6).
            
            # Let's just do a full energy recalculation for simplicity if segment is small?
            # No, O(N) is too slow for many iterations.
            
            # Let's calculate delta by comparing old and new contacts involving the segment.
            
            # Old contacts:
            # For each k in [i, j]:
            #   if sequence[k] == 'H':
            #     for nb in neighbours(chain[k]):
            #       if nb in occupied and occupied[nb] not in [i, j] and sequence[occupied[nb]] == 'H' and not adjacent(k, occupied[nb]):
            #         old_count += 1
            
            # New contacts:
            # The positions are chain[i...j] but reversed.
            # So position chain[k] now holds residue sequence[j - (k - i)] = sequence[i + j - k].
            # Let new_res_at_pos_k = sequence[i + j - k].
            
            # For each k in [i, j]:
            #   if new_res_at_pos_k == 'H':
            #     for nb in neighbours(chain[k]):
            #       if nb in occupied and occupied[nb] not in [i, j] and sequence[occupied[nb]] == 'H' and not adjacent(i + j - k, occupied[nb]):
            #         new_count += 1
            
            # Also, internal contacts within the segment might change?
            # The relative positions are the same, just indices are reversed.
            # Contact between k1 and k2 (both in segment) depends on |k1 - k2| > 1.
            # In reversed segment, the distance is still |k1 - k2|.
            # So internal contacts are preserved!
            # Only boundary contacts (segment vs outside) change.
            
            old_boundary_contacts = 0
            new_boundary_contacts = 0
            
            for k in range(i, j + 1):
                pos = chain[k]
                # Old residue at pos is sequence[k]
                if sequence[k] == 'H':
                    for nb in _neighbours(pos):
                        j_out = occupied.get(nb)
                        if j_out is not None and (j_out < i or j_out > j):
                            if sequence[j_out] == 'H' and not _is_adjacent(k, j_out):
                                old_boundary_contacts += 1
                
                # New residue at pos is sequence[i + j - k]
                new_idx = i + j - k
                if sequence[new_idx] == 'H':
                    for nb in _neighbours(pos):
                        j_out = occupied.get(nb)
                        if j_out is not None and (j_out < i or j_out > j):
                            if sequence[j_out] == 'H' and not _is_adjacent(new_idx, j_out):
                                new_boundary_contacts += 1
            
            delta = new_boundary_contacts - old_boundary_contacts
            
            if delta > 0:
                # Accept reversal
                # Reverse the segment in chain
                chain[i:j+1] = reversed(chain[i:j+1])
                # Update occupied
                for k in range(i, j + 1):
                    occupied[chain[k]] = k
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            elif delta == 0 and rng.random() < 0.05:
                # Neutral reversal
                chain[i:j+1] = reversed(chain[i:j+1])
                for k in range(i, j + 1):
                    occupied[chain[k]] = k
                no_improve_count += 1
            else:
                no_improve_count += 1

    return best_chain

def fold(sequence, dimension):
    """
    Best conformation using Contact-First Growth and Local Search.
    """
    if not sequence:
        return []
    
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0 # Maximize energy (minimize negative value)
    
    # Time budget allocation:
    # Try a few different growth strategies if time permits.
    # For now, one robust growth + long local search.
    
    # 1. Constructive Growth
    chain = _initial_grow(sequence, dimension, rng)
    
    if chain is None:
        # Fallback: straight line
        chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    # 2. Local Search
    if _remaining_time() > 1.0:
        chain = _local_search(sequence, chain, rng)
    
    # Calculate final energy
    energy = _compute_energy(sequence, chain)
    
    # If we have time, try a second growth with different random seed?
    # Maybe not, local search is more important.
    
    return [list(p) for p in chain]
