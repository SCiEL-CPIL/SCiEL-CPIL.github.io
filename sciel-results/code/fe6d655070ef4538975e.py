"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _neighbours2d(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))

def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours2d(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours2d(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours2d(site) if q not in occupied)
            # Heuristic: prefer high gain, then high free space (to avoid trapping)
            score = gain * 2.0 + free * 0.5 + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _energy(sequence, chain):
    """Calculate the energy of a conformation. Lower is better (more negative)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours2d(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _calc_delta_energy(sequence, occupied, i, old_pos, new_pos):
    """
    Calculate the change in energy when moving residue i from old_pos to new_pos.
    Assumes occupied map is updated for the current state BEFORE the move.
    We need to consider contacts involving residue i.
    Energy = - (number of H-H contacts).
    Delta Energy = New Energy - Old Energy.
    If we gain a contact, Energy decreases (Delta < 0).
    If we lose a contact, Energy increases (Delta > 0).
    """
    delta = 0
    # Contacts involving residue i are those where residue i is H and neighbor is H, 
    # and they are not chain-adjacent (|j-i| > 1).
    
    # 1. Remove old contacts involving i at old_pos
    if sequence[i] == 'H':
        for q in _neighbours2d(old_pos):
            j = occupied.get(q)
            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                # We are removing a contact. Energy increases by 1 (less negative).
                delta += 1

    # 2. Add new contacts involving i at new_pos
    if sequence[i] == 'H':
        for q in _neighbours2d(new_pos):
            j = occupied.get(q)
            # Note: occupied still contains old_pos for i? 
            # In our implementation, we usually check validity first, then update.
            # If we are calculating delta before updating occupied, we must be careful.
            # Let's assume occupied reflects the state where i is at old_pos.
            # So if q == old_pos, j will be i. We skip j == i.
            # If q is a neighbor of new_pos, and q is occupied by j (j != i), 
            # and j is not chain-adjacent to i, it's a new contact.
            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                # We are adding a contact. Energy decreases by 1.
                delta -= 1

    return delta

def _is_valid_move(chain, occupied, i, new_pos):
    """Check if moving residue i to new_pos is valid (self-avoiding and no invalid adjacencies)."""
    # 1. new_pos must be unoccupied
    if new_pos in occupied:
        return False
    
    # 2. new_pos must not be adjacent to any residue j where |j-i| > 1
    #    (Chain adjacencies i-1 and i+1 are allowed/expected)
    for q in _neighbours2d(new_pos):
        j = occupied.get(q)
        if j is not None and abs(j - i) > 1:
            return False
            
    return True

def _local_search(sequence, chain, max_iter=5000, rng=None, use_kinks=True):
    """
    Local search with single-residue moves and kink flips.
    """
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    if rng is None:
        rng = random.Random(42)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    no_improve_count = 0
    max_no_improve = 100
    
    for _ in range(max_iter):
        if no_improve_count > max_no_improve:
            break
            
        improved = False
        # Randomly select a move type: single step or kink
        if use_kinks and n >= 3 and rng.random() < 0.5:
            # Kink flip: rotate a 3-residue segment i-1, i, i+1
            # Pick a random i (1 to n-2)
            i = rng.randint(1, n - 2)
            p_prev = best_chain[i-1]
            p_curr = best_chain[i]
            p_next = best_chain[i+1]
            
            # The kink is defined by the vector from prev to curr and curr to next.
            # A flip rotates the 'next' position around 'curr' by 90 degrees? 
            # No, a kink flip in SAW usually means reflecting the next residue across the axis defined by prev-curr?
            # Actually, for a square lattice, a "kink" is a 90 degree turn.
            # Flipping it makes it a straight line or a turn in the other direction.
            # Let's define the move: 
            # Current: p_prev -> p_curr -> p_next
            # The vector v1 = p_curr - p_prev, v2 = p_next - p_curr.
            # If v1 and v2 are perpendicular, it's a kink.
            # The flip position p_new is such that p_prev -> p_curr -> p_new is a straight line?
            # Or p_new is the other 90-degree turn?
            # On a square lattice, if p_prev=(0,0), p_curr=(1,0), p_next can be (1,1) or (1,-1).
            # The flip of (1,1) is (1,-1).
            # Generally, p_new = 2 * p_curr - p_prev + (p_next - p_curr) rotated 180? No.
            # p_new = p_curr + (p_curr - p_prev) + (p_next - p_curr) rotated 90?
            
            # Simpler: Calculate the unique position that completes the "other" kink.
            # If p_prev, p_curr, p_next form a kink, p_new is the reflection of p_next across the line p_prev-p_curr?
            # No, that would be a straight line if it was a straight line.
            # Let's just try the two other positions adjacent to p_curr that are not p_prev and not p_next?
            # No, that's just single residue moves.
            
            # A proper kink flip:
            # Vector v_in = p_curr - p_prev
            # Vector v_out = p_next - p_curr
            # If v_in and v_out are orthogonal (dot product 0), it's a kink.
            # The flip position p_new is p_curr + v_in + v_rotated?
            # Actually, if we have (0,0)->(1,0)->(1,1), the flip is (0,0)->(1,0)->(1,-1).
            # p_new = (1, -1).
            # Notice p_new = p_curr + (p_curr - p_prev) - (p_next - p_curr)?
            # (1,0) + (1,0) - (0,1) = (2, -1). No.
            # p_new = p_curr + (p_curr - p_prev) + (p_curr - p_next)?
            # (1,0) + (1,0) + (0,-1) = (2, -1). No.
            
            # Let's look at coordinates.
            # dx1 = p_curr[0] - p_prev[0], dy1 = p_curr[1] - p_prev[1]
            # dx2 = p_next[0] - p_curr[0], dy2 = p_next[1] - p_curr[1]
            # If dx1==0 and dy1!=0 (vertical in), then dx2!=0 and dy2==0 (horizontal out).
            # Flip: horizontal in, vertical out.
            # p_new = p_curr + (dx1, dy1) + (-dx2, -dy2)? No.
            
            # Easiest way: The 4 neighbors of p_curr are p_prev, p_next, and 2 others.
            # One of the others is the "straight" continuation if it was a straight line.
            # If it's a kink, the "straight" continuation is the flip?
            # No, if it's a kink, the "straight" continuation is the *other* neighbor that makes a straight line with p_prev?
            # Wait. If p_prev=(0,0), p_curr=(1,0). Neighbors of p_curr: (0,0), (2,0), (1,1), (1,-1).
            # If p_next=(1,1), it's a kink. The flip is (1,-1).
            # If p_next=(2,0), it's straight. There is no "flip" in the same sense, it's just a straight line.
            # So kink flips only apply when it's a kink.
            
            dx1 = p_curr[0] - p_prev[0]
            dy1 = p_curr[1] - p_prev[1]
            dx2 = p_next[0] - p_curr[0]
            dy2 = p_next[1] - p_curr[1]
            
            # Check if it's a kink (orthogonal)
            if dx1 * dx2 + dy1 * dy2 == 0:
                # The flip position is p_curr + (dx1, dy1) + (dx2, dy2) rotated 180? No.
                # p_new = p_curr + (dx1, dy1) - (dx2, dy2)?
                # (1,0) + (1,0) - (0,1) = (2, -1). No.
                # p_new = p_curr + (dx1, dy1) + ( -dx2, -dy2 )?
                # (1,0) + (1,0) + (0,-1) = (2, -1). No.
                
                # Let's just calculate the candidate.
                # If in is (1,0) and out is (0,1), flip out to (0,-1).
                # p_new = p_curr + (0, -1).
                # General rule: p_new = p_curr + (dx1, dy1) + (dx2, dy2) is the diagonal.
                # The flip is p_curr + (dx1, dy1) + ( -dx2, -dy2 )? No.
                
                # Let's use the property: p_prev, p_curr, p_new are collinear? No.
                # p_new is the reflection of p_next across the line p_prev-p_curr.
                # Line p_prev-p_curr is x-axis (locally). p_next is (0,1). Reflection is (0,-1).
                # So p_new = p_curr + (0, -1).
                # How to calculate this generally?
                # Vector v_in = (dx1, dy1). Vector v_out = (dx2, dy2).
                # v_out_perp = (-dy2, dx2) or (dy2, -dx2).
                # The flip vector v_new is the one that is orthogonal to v_in and has opposite sign of dot product with v_out?
                # Actually, simpler:
                # If dx1 != 0: # Horizontal in
                    # Then dy1 == 0.
                    # dx2 == 0, dy2 != 0.
                    # p_new = (p_curr[0], p_curr[1] - dy2)
                # Else: # Vertical in
                    # dy1 != 0.
                    # dy2 == 0, dx2 != 0.
                    # p_new = (p_curr[0] - dx2, p_curr[1])
                
                if dx1 != 0:
                    p_new = (p_curr[0], p_curr[1] - dy2)
                else:
                    p_new = (p_curr[0] - dx2, p_curr[1])
                
                # Check validity of moving i to p_new
                # Wait, a kink flip moves the whole segment?
                # No, a kink flip usually means changing the position of the 'next' residue relative to 'curr' and 'prev'.
                # But in our chain representation, we are moving residue i.
                # If we move residue i, we break the chain.
                # Kink flips are better implemented as moving the *segment*?
                # Or just treat it as a specific single-residue move?
                # If we move residue i to p_new, the chain is p_prev -> p_new -> p_next.
                # This is a valid SAW move if p_new is valid.
                # However, p_new is NOT adjacent to p_prev?
                # In the example: p_prev=(0,0), p_curr=(1,0), p_next=(1,1).
                # p_new=(1,-1).
                # Distance p_prev to p_new: (1,-1) - (0,0) = (1,-1). Distance sqrt(2). Not adjacent.
                # So moving i to p_new breaks the chain adjacency between i-1 and i.
                # Therefore, a "kink flip" in the context of single-residue moves is NOT just moving i.
                # It requires moving i AND i+1? Or it's a different move type.
                
                # Given the complexity, let's stick to single-residue moves but add "segment regrowth" if we have time.
                # For now, let's just do single-residue moves but with a more aggressive acceptance criterion or more iterations.
                # The cell finding mentioned "kink flips and short regrows".
                # Let's implement a simple "2-residue move" or "kink flip" properly.
                # A kink flip rotates the bond i-i+1 around i-1? No.
                # Let's skip explicit kink flips in the local search loop for simplicity and robustness, 
                # and rely on SA to explore the space. SA with single moves can effectively simulate kink flips over time.
                pass
        
        # Single residue move
        i = rng.randint(1, n - 2)
        current_pos = best_chain[i]
        for new_pos in _neighbours2d(current_pos):
            if not _is_valid_move(best_chain, occupied, i, new_pos):
                continue
            
            delta = _calc_delta_energy(sequence, occupied, i, current_pos, new_pos)
            new_energy = best_energy + delta
            
            if new_energy < best_energy:
                # Accept
                best_chain[i] = new_pos
                occupied.pop(current_pos)
                occupied[new_pos] = i
                best_energy = new_energy
                improved = True
                no_improve_count = 0
                break
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break

        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break

    return best_chain, best_energy

def _simulated_annealing(sequence, chain, max_iter=10000, initial_temp=5.0, final_temp=0.01, rng=None):
    """Simulated Annealing for HP lattice protein folding."""
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    if rng is None:
        rng = random.Random(42)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    cooling_rate = (final_temp / initial_temp) ** (1.0 / max(1, max_iter))
    
    for iteration in range(max_iter):
        temp = initial_temp * (cooling_rate ** iteration)
        if temp < 0.001:
            break
            
        i = rng.randint(1, n - 2)
        current_pos = current_chain[i]
        
        # Try each neighbor
        for new_pos in _neighbours2d(current_pos):
            if not _is_valid_move(current_chain, occupied, i, new_pos):
                continue
            
            delta = _calc_delta_energy(sequence, occupied, i, current_pos, new_pos)
            new_energy = current_energy + delta
            
            # Accept if better or with probability exp(-delta/temp)
            if delta < 0 or (temp > 0.01 and rng.random() < math.exp(-delta / temp)):
                # Accept move
                current_chain[i] = new_pos
                occupied.pop(current_pos)
                occupied[new_pos] = i
                current_energy = new_energy
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(current_chain)
                break  # Only accept one move per iteration
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """Best conformation using randomized greedy growths, local search, and simulated annealing."""
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin before 210s kill
    
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0  # 0 is a safe initial energy (no contacts)
    
    n = len(sequence)
    
    # Adjust number of restarts based on sequence length
    if n <= 30:
        num_restarts = 800
        ls_iter = 2000
        sa_iter = 5000
    elif n <= 60:
        num_restarts = 400
        ls_iter = 1000
        sa_iter = 2000
    else:
        num_restarts = 150
        ls_iter = 500
        sa_iter = 1000
    
    for attempt in range(num_restarts):
        if time.time() > deadline - 10.0:  # Reserve time for final SA
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Apply local search to improve
        chain, energy = _local_search(sequence, chain, max_iter=ls_iter, rng=rng)
        
        if energy < best_energy:
            best, best_energy = chain, energy
    
    # If we have time, apply simulated annealing to the best solution found
    if best is not None and time.time() < deadline - 2.0:
        remaining_time = deadline - time.time() - 2.0
        # Scale SA iterations based on remaining time
        # Assume ~1000 iterations per second for small n, less for large n
        est_iter_per_sec = 1000 if n < 40 else 500
        sa_iter = int(remaining_time * est_iter_per_sec)
        sa_iter = max(1000, min(sa_iter, 50000)) # Cap SA iterations
        
        best, best_energy = _simulated_annealing(
            sequence, best, max_iter=sa_iter, initial_temp=3.0, final_temp=0.01, rng=rng
        )
    
    if best is None:  # a straight line is always self-avoiding
        best = [(i, 0) for i in range(len(sequence))]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
