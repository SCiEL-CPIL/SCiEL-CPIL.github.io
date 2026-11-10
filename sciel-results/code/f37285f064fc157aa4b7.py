"""HP lattice protein folding: slither-move kinetic Monte Carlo with end-to-tail regrowth."""
import random
import time
import math

_DEADLINE = time.time() + 170.0  # safety margin before 180s hard limit


def _nb(p):
    x, y = p
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy(sequence, chain):
    """Compute HP contact energy (negative number of H-H non-adjacent contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _nb(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _generate_initial_saw(sequence, dimension, rng):
    """Generate a random self-avoiding walk of length n."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0)] * n
    # Start with a small random walk
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    for i in range(1, n):
        last = chain[-1]
        # Shuffle neighbors for randomness
        neighbors = list(_nb(last))
        rng.shuffle(neighbors)
        placed = False
        for site in neighbors:
            if site not in occupied:
                chain.append(site)
                occupied[site] = i
                placed = True
                break
        if not placed:
            # Stuck - restart with a different strategy
            return _generate_initial_saw(sequence, dimension, rng)
    return chain


def _try_regrow_suffix(sequence, chain, occupied, split_idx, rng, max_attempts=50):
    """
    Try to regrow the suffix from split_idx to end in a new random direction.
    Returns (new_chain, new_occupied) on success, or None on failure.
    """
    n = len(chain)
    if split_idx >= n:
        return chain, occupied
    
    # Keep prefix intact
    prefix = chain[:split_idx]
    new_occupied = {p: i for i, p in enumerate(prefix)}
    new_chain = list(prefix)
    
    # Regrow from split_idx
    last = prefix[-1]
    for i in range(split_idx, n):
        neighbors = list(_nb(last))
        rng.shuffle(neighbors)
        placed = False
        for site in neighbors:
            if site not in new_occupied:
                new_chain.append(site)
                new_occupied[site] = i
                last = site
                placed = True
                break
        if not placed:
            return None  # Failed to regrow
    return new_chain, new_occupied


def _try_slither_move(sequence, chain, occupied, seg_start, seg_len, rng):
    """
    Slither move: remove a short segment (seg_start to seg_start+seg_len-1) and regrow it.
    The segment must be at least 3 residues to allow topological change.
    Returns (new_chain, new_occupied, delta_energy) on success, or None.
    """
    n = len(chain)
    seg_end = seg_start + seg_len  # exclusive
    
    if seg_start < 1 or seg_end > n - 1:
        return None  # Can't detach from ends
    
    # Remove the segment
    old_segment = chain[seg_start:seg_end]
    # Build occupied without the segment
    new_occupied = {}
    for i, p in enumerate(chain):
        if seg_start <= i < seg_end:
            continue
        new_occupied[p] = i
    
    # The residue before the segment
    left_neighbor = chain[seg_start - 1]
    # The residue after the segment
    right_neighbor = chain[seg_end]
    
    # Try to regrow the segment from left_neighbor, connecting to right_neighbor at the end
    # This is tricky - we need the last placed residue to be adjacent to right_neighbor
    
    # Strategy: regrow randomly, then check if we can connect to right_neighbor
    # For simplicity, we'll do a random regrow and accept if it works
    
    new_chain = list(chain[:seg_start])
    last = left_neighbor
    
    for offset in range(seg_len):
        idx = seg_start + offset
        neighbors = list(_nb(last))
        rng.shuffle(neighbors)
        placed = False
        for site in neighbors:
            if site not in new_occupied:
                # Check if this is the last residue in the segment
                if offset == seg_len - 1:
                    # Must be adjacent to right_neighbor
                    if abs(site[0] - right_neighbor[0]) + abs(site[1] - right_neighbor[1]) != 1:
                        continue
                new_chain.append(site)
                new_occupied[site] = idx
                last = site
                placed = True
                break
        if not placed:
            return None
    
    if len(new_chain) != n:
        return None
    
    # Verify connectivity
    for i in range(1, n):
        if abs(new_chain[i][0] - new_chain[i-1][0]) + abs(new_chain[i][1] - new_chain[i-1][1]) != 1:
            return None
    
    # Verify no self-intersection
    if len(set(new_chain)) != n:
        return None
    
    # Compute energy delta
    old_energy = _energy(sequence, chain)
    new_energy = _energy(sequence, new_chain)
    delta = new_energy - old_energy
    
    return new_chain, new_occupied, delta


def _try_kink_move(sequence, chain, occupied, i, rng):
    """
    Kink move: rotate the pair (i, i+1) by 90 degrees around their midpoint.
    Returns (new_chain, new_occupied, delta_energy) on success, or None.
    """
    n = len(chain)
    if i < 1 or i + 1 >= n - 1:
        return None
    
    p1, p2 = chain[i], chain[i+1]
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    
    # Generate perpendicular rotation
    if abs(dx) == 1 and dy == 0:
        # Horizontal pair
        candidates = [
            ((x1, y1+1), (x2, y2+1)),
            ((x1, y1-1), (x2, y2-1)),
        ]
    elif abs(dy) == 1 and dx == 0:
        # Vertical pair
        candidates = [
            ((x1+1, y1), (x2+1, y2)),
            ((x1-1, y1), (x2-1, y2)),
        ]
    else:
        return None
    
    for new_p1, new_p2 in candidates:
        # Check validity
        if new_p1 in occupied and occupied[new_p1] not in (i, i+1):
            continue
        if new_p2 in occupied and occupied[new_p2] not in (i, i+1):
            continue
        
        # Connectivity checks
        if i > 0:
            px, py = chain[i-1]
            if abs(new_p1[0] - px) + abs(new_p1[1] - py) != 1:
                continue
        if i + 2 < n:
            px, py = chain[i+2]
            if abs(new_p2[0] - px) + abs(new_p2[1] - py) != 1:
                continue
        if abs(new_p1[0] - new_p2[0]) + abs(new_p1[1] - new_p2[1]) != 1:
            continue
        
        # Build new chain and occupied
        new_chain = list(chain)
        new_chain[i] = new_p1
        new_chain[i+1] = new_p2
        new_occupied = dict(occupied)
        del new_occupied[p1]
        del new_occupied[p2]
        new_occupied[new_p1] = i
        new_occupied[new_p2] = i + 1
        
        old_energy = _energy(sequence, chain)
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - old_energy
        
        return new_chain, new_occupied, delta
    
    return None


def _try_single_residue_move(sequence, chain, occupied, i, rng):
    """
    Single-residue move: try to move residue i to an adjacent empty site.
    Returns (new_chain, new_occupied, delta_energy) on success, or None.
    """
    n = len(chain)
    if i < 1 or i >= n - 1:
        return None
    
    old_site = chain[i]
    x, y = old_site
    
    # Try all 4 neighbors
    for new_site in _nb(old_site):
        if new_site == old_site:
            continue
        if new_site in occupied and occupied[new_site] != i:
            continue
        
        # Connectivity checks
        if i > 0:
            px, py = chain[i-1]
            if abs(new_site[0] - px) + abs(new_site[1] - py) != 1:
                continue
        if i < n - 1:
            px, py = chain[i+1]
            if abs(new_site[0] - px) + abs(new_site[1] - py) != 1:
                continue
        
        new_chain = list(chain)
        new_chain[i] = new_site
        new_occupied = dict(occupied)
        del new_occupied[old_site]
        new_occupied[new_site] = i
        
        old_energy = _energy(sequence, chain)
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - old_energy
        
        return new_chain, new_occupied, delta
    
    return None


def _local_optimize_sa(sequence, chain, occupied, max_iters=5000):
    """
    Simulated annealing with slither moves, kink moves, and single-residue moves.
    """
    n = len(chain)
    if n < 4:
        return chain, occupied, _energy(sequence, chain)
    
    best_energy = _energy(sequence, chain)
    current_energy = best_energy
    best_chain = list(chain)
    best_occupied = dict(occupied)
    
    # SA parameters
    T = max(1.0, n * 0.05)
    T_min = 0.05
    cooling = 0.999
    
    iters = 0
    while iters < max_iters and time.time() < _DEADLINE:
        iters += 1
        move_type = random.random()
        
        if move_type < 0.3:
            # Slither move: select a short segment
            seg_len = random.choice([3, 4, 5, 6, 7])
            seg_start = random.randint(1, n - seg_len - 1)
            result = _try_slither_move(sequence, chain, occupied, seg_start, seg_len, random)
            if result is None:
                continue
            new_chain, new_occupied, delta = result
        elif move_type < 0.6:
            # Kink move
            i = random.randint(1, n - 3)
            result = _try_kink_move(sequence, chain, occupied, i, random)
            if result is None:
                continue
            new_chain, new_occupied, delta = result
        else:
            # Single-residue move
            i = random.randint(1, n - 2)
            result = _try_single_residue_move(sequence, chain, occupied, i, random)
            if result is None:
                continue
            new_chain, new_occupied, delta = result
        
        # Acceptance criterion
        if delta < 0 or random.random() < math.exp(-delta / T):
            chain = new_chain
            occupied = new_occupied
            current_energy += delta
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                best_occupied = dict(occupied)
        
        # Cooling
        T = max(T_min, T * cooling)
    
    return best_chain, best_occupied, best_energy


def _end_to_tail_sa(sequence, chain, occupied, max_iters=3000):
    """
    End-to-Tail SA: bias towards regrowing long tails (split in first 20-30% of chain).
    """
    n = len(chain)
    if n < 6:
        return chain, occupied, _energy(sequence, chain)
    
    best_energy = _energy(sequence, chain)
    current_energy = best_energy
    best_chain = list(chain)
    best_occupied = dict(occupied)
    
    T = max(1.0, n * 0.06)
    T_min = 0.05
    cooling = 0.9995
    
    iters = 0
    while iters < max_iters and time.time() < _DEADLINE:
        iters += 1
        
        # Bias: 70% chance to regrow a long tail (split in first 20-30%)
        if random.random() < 0.7:
            # Split in first 20-30% of chain
            min_split = max(1, int(n * 0.1))
            max_split = min(n - 3, int(n * 0.3))
            if min_split >= max_split:
                max_split = min_split + 1
            split_idx = random.randint(min_split, max_split)
        else:
            # Random split
            split_idx = random.randint(1, n - 3)
        
        result = _try_regrow_suffix(sequence, chain, occupied, split_idx, random)
        if result is None:
            continue
        new_chain, new_occupied = result
        
        old_energy = _energy(sequence, chain)
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - old_energy
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            chain = new_chain
            occupied = new_occupied
            current_energy += delta
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                best_occupied = dict(occupied)
        
        T = max(T_min, T * cooling)
    
    return best_chain, best_occupied, best_energy


def fold(sequence, dimension):
    """Best conformation with deadline-aware search using slither moves and E2T regrowth."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    best = None
    best_energy = 0  # Start with 0 (no contacts)
    
    # Phase 1: Generate initial conformations and do local SA
    while time.time() < _DEADLINE:
        chain = _generate_initial_saw(sequence, dimension, rng)
        if chain is None or len(chain) != n:
            continue
        occupied = {p: i for i, p in enumerate(chain)}
        
        # Local SA with slither moves
        chain, occupied, value = _local_optimize_sa(sequence, chain, occupied, max_iters=2000)
        
        if value < best_energy:
            best = chain
            best_energy = value
        
        # If we've done enough restarts, break
        if time.time() > _DEADLINE - 30:
            break
    
    # Phase 2: End-to-Tail SA on the best conformation found
    if best is not None and time.time() < _DEADLINE - 10:
        best_occupied = {p: i for i, p in enumerate(best)}
        best, best_occupied, best_energy = _end_to_tail_sa(sequence, best, best_occupied, max_iters=3000)
    
    if best is None:
        best = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best]
