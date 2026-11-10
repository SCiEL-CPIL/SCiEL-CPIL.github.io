"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = time.time() + 170.0

def _neighbours2d(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _grow(sequence, dimension, rng):
    """
    Randomized greedy chain growth.
    Prioritizes forming H-H contacts, then maximizing free space to avoid dead ends.
    """
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    
    for i in range(1, n):
        options = []
        last = chain[-1]
        for site in _neighbours2d(last):
            if site in occupied:
                continue
            
            # 1. Immediate gain: New H-H contacts
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours2d(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            
            # 2. Future potential: If current is P, does it lead to a spot where next H can contact?
            # Simple lookahead: if next is H, check if site has neighbors that are near existing Hs
            future_gain = 0
            if i + 1 < n and sequence[i + 1] == "H":
                for q in _neighbours2d(site):
                    if q == last:
                        continue
                    if q not in occupied:
                        # Check if q's neighbors have Hs
                        for r in _neighbours2d(q):
                            j = occupied.get(r)
                            if j is not None and j < i and sequence[j] == "H":
                                future_gain += 0.5
                                break
                        if future_gain > 0:
                            break # Found one good path, don't overcount

            # 3. Free space: Avoid trapping
            free = 0
            for q in _neighbours2d(site):
                if q not in occupied and q != last:
                    free += 1
            
            # Score: Heavy weight on immediate contacts, moderate on future, light on free space
            # Add randomness to explore different paths
            score = gain * 10.0 + future_gain * 2.0 + free * 0.5 + rng.random() * 2.0
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick the best option
        site = max(options, key=lambda x: x[0])[1]
        occupied[site] = i
        chain.append(site)
    
    return chain

def _energy(sequence, chain):
    """Calculate total energy (negative number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours2d(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _compute_delta_energy(sequence, chain, occupied, i, new_site):
    """
    Compute energy change for moving residue i to new_site.
    Returns None if move is invalid (collision or breaks connectivity).
    Returns delta (new_energy - old_energy) if valid.
    """
    old_site = chain[i]
    if new_site == old_site:
        return 0.0

    # Check collision
    if new_site in occupied and occupied[new_site] != i:
        return None

    # Check connectivity (unit steps)
    if i > 0:
        px, py = chain[i-1]
        nx, ny = new_site
        if abs(nx - px) + abs(ny - py) != 1:
            return None
    if i < len(chain) - 1:
        px, py = chain[i+1]
        nx, ny = new_site
        if abs(nx - px) + abs(ny - py) != 1:
            return None

    delta = 0.0
    seq_i = sequence[i]

    # Remove old contacts involving i
    if seq_i == "H":
        ox, oy = old_site
        for q in _neighbours2d((ox, oy)):
            j = occupied.get(q)
            if j is not None and j != i and sequence[j] == "H":
                # This contact is lost
                delta += 1.0

    # Add new contacts involving i
    if seq_i == "H":
        nx, ny = new_site
        for q in _neighbours2d((nx, ny)):
            j = occupied.get(q)
            if j is not None and j != i and sequence[j] == "H":
                # This contact is gained
                delta -= 1.0

    return delta

def _local_optimize_sa(sequence, chain, occupied, time_budget):
    """
    Simulated Annealing local optimization.
    Uses single-residue moves and kink moves.
    """
    n = len(chain)
    if n < 4:
        return chain, occupied, _energy(sequence, chain)

    best_chain = list(chain)
    best_occupied = dict(occupied)
    best_energy = _energy(sequence, best_chain)
    current_energy = best_energy

    # SA Parameters
    T = max(1.0, n * 0.1)
    T_min = 0.01
    cooling = 0.998
    
    start_time = time.time()
    rng = random.Random(hash(tuple(chain)) % 100000)

    while time.time() - start_time < time_budget and time.time() < _DEADLINE:
        # Choose move type: 70% single, 30% kink
        if rng.random() < 0.7:
            # Single residue move
            i = rng.randint(1, n - 2)
            x, y = chain[i]
            candidates = [c for c in _neighbours2d((x, y)) if c != chain[i-1] and c != chain[i+1]]
            if not candidates:
                continue
            new_site = rng.choice(candidates)
            
            delta = _compute_delta_energy(sequence, chain, occupied, i, new_site)
            if delta is None:
                continue

            if delta < 0 or rng.random() < math.exp(-delta / T):
                # Accept move
                old_site = chain[i]
                chain[i] = new_site
                del occupied[old_site]
                occupied[new_site] = i
                current_energy += delta
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                    best_occupied = dict(occupied)
        else:
            # Kink move: Rotate residues i and i+1
            i = rng.randint(1, n - 3)
            p1, p2 = chain[i], chain[i+1]
            x1, y1 = p1
            x2, y2 = p2
            
            # Determine possible kink rotations
            # Vector p1->p2
            dx, dy = x2 - x1, y2 - y1
            
            # Generate 3 other orientations for the pair
            moves = []
            if dx == 1: # Right
                # Up, Down
                moves = [((x1, y1+1), (x2, y2+1)), ((x1, y1-1), (x2, y2-1))]
            elif dx == -1: # Left
                moves = [((x1, y1+1), (x2, y2+1)), ((x1, y1-1), (x2, y2-1))]
            elif dy == 1: # Up
                moves = [((x1+1, y1), (x2+1, y2)), ((x1-1, y1), (x2-1, y2))]
            else: # Down
                moves = [((x1+1, y1), (x2+1, y2)), ((x1-1, y1), (x2-1, y2))]
            
            if not moves:
                continue
                
            new_p1, new_p2 = rng.choice(moves)
            
            # Check validity for both positions
            valid = True
            # Check collision for p1
            if new_p1 in occupied and occupied[new_p1] not in [i, i+1]:
                valid = False
            # Check collision for p2
            if new_p2 in occupied and occupied[new_p2] not in [i, i+1]:
                valid = False
            # Check connectivity with i-1
            if valid and i > 0:
                px, py = chain[i-1]
                if abs(new_p1[0] - px) + abs(new_p1[1] - py) != 1:
                    valid = False
            # Check connectivity with i+2
            if valid and i+2 < n:
                px, py = chain[i+2]
                if abs(new_p2[0] - px) + abs(new_p2[1] - py) != 1:
                    valid = False
            
            if not valid:
                continue
            
            # Calculate delta energy for kink
            # This is more complex: remove old contacts for i and i+1, add new
            delta = 0.0
            
            # Helper to calc delta for a specific residue move
            def calc_res_delta(idx, old, new):
                d = 0.0
                if sequence[idx] == "H":
                    # Old
                    for q in _neighbours2d(old):
                        j = occupied.get(q)
                        if j is not None and j != idx and j != (idx-1 if idx>0 else -1) and j != (idx+1 if idx<n-1 else -2) and sequence[j] == "H":
                            # Note: Kink move moves i and i+1 together.
                            # We must be careful not to double count or miss internal contact between i and i+1.
                            # However, i and i+1 are always adjacent, so they don't form an H-H contact in the "non-adjacent" sense?
                            # Wait, the definition is "neighbouring lattice sites that are NOT adjacent along the chain".
                            # i and i+1 are adjacent along the chain, so their contact is never counted in energy.
                            # So we only care about contacts with j != i and j != i+1.
                            pass
                    
                    # Let's just do a full delta calculation for the two residues involved, excluding each other.
                    # This is safer.
                    pass
                return d

            # Simplified Delta for Kink:
            # Remove old contacts for i and i+1 (excluding i<->i+1)
            # Add new contacts for i and i+1 (excluding i<->i+1)
            
            # Since occupied map is still old, we can just compute energy contribution changes.
            
            # 1. Remove old contributions
            for idx in [i, i+1]:
                if sequence[idx] == "H":
                    old_pos = chain[idx]
                    for q in _neighbours2d(old_pos):
                        j = occupied.get(q)
                        if j is not None and j not in [i, i+1] and sequence[j] == "H":
                            delta += 1.0 # Contact lost
            
            # 2. Add new contributions
            # We need to know which positions are "occupied" by the new kink to avoid counting them as external,
            # but since we are adding contacts, we check if the new neighbor is an existing H (not part of the moving kink).
            for idx in [i, i+1]:
                if sequence[idx] == "H":
                    new_pos = new_p1 if idx == i else new_p2
                    for q in _neighbours2d(new_pos):
                        # If q is the other moving residue, skip (internal chain bond)
                        if (idx == i and q == new_p2) or (idx == i+1 and q == new_p1):
                            continue
                        j = occupied.get(q)
                        # If j is one of the moving residues at their OLD position, we already removed those.
                        # But occupied still has old positions.
                        # So if q is old_p1 or old_p2, j will be i or i+1. We skip those.
                        if j is not None and j not in [i, i+1] and sequence[j] == "H":
                            delta -= 1.0 # Contact gained

            if delta < 0 or rng.random() < math.exp(-delta / T):
                # Accept kink
                old_p1, old_p2 = chain[i], chain[i+1]
                chain[i] = new_p1
                chain[i+1] = new_p2
                del occupied[old_p1]
                del occupied[old_p2]
                occupied[new_p1] = i
                occupied[new_p2] = i+1
                current_energy += delta
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                    best_occupied = dict(occupied)

        T *= cooling
        if T < T_min:
            T = T_min

    return best_chain, best_occupied, best_energy

def fold(sequence, dimension):
    """
    Best conformation with deadline-aware search.
    Uses multiple restarts of growth + SA optimization.
    """
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 1000 # Start high (energy is negative)
    
    start_time = time.time()
    total_time_budget = 170.0 # Safety margin
    
    restarts = 0
    # Estimate time per restart based on sequence length
    # Small sequences are fast, large ones slower.
    # We will just run until deadline, allocating more time to later restarts if we are fast?
    # Better: Fixed number of restarts, or time-based.
    # Let's do time-based: run a restart, check time, if < 2s left, stop.
    
    while time.time() < _DEADLINE - 2.0:
        restarts += 1
        # Generate initial chain
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        occupied = {p: i for i, p in enumerate(chain)}
        
        # Allocate time for SA
        # If we are early, give more time. If late, less.
        elapsed = time.time() - start_time
        remaining = _DEADLINE - 2.0 - time.time()
        
        # Simple heuristic: give 10% of remaining time to this SA, min 0.1s
        sa_time = max(0.1, remaining * 0.1)
        
        # For larger sequences, SA is more important.
        if n > 50:
            sa_time = max(0.5, remaining * 0.15)
            
        chain, occupied, energy = _local_optimize_sa(sequence, chain, occupied, sa_time)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
        
        # If we've done many restarts and energy is very good (e.g., < -n/2), we might stop early?
        # No, keep going until deadline.
        
        # Optional: If we found a perfect fold (all Hs packed), stop?
        # Hard to know "perfect" without knowing best known.
        
        if restarts > 1000:
            break

    if best_chain is None:
        # Fallback: linear chain
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
