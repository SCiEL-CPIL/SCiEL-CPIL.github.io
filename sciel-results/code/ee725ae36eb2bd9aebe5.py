"""HP lattice protein folding - improved local search with optimized moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = time.time() + 295  # safety margin within 330s wall-clock


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy: negative count of H-H contacts (non-consecutive neighbours)."""
    n = len(chain)
    if n < 2:
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


def _energy_delta(sequence, chain, pos, old_pos, new_pos):
    """
    Compute the change in energy when moving residue at position pos from old_pos to new_pos.
    This is an approximation that only considers contacts involving the moved residue.
    For exact energy, use _energy.
    """
    n = len(chain)
    # Build position map excluding pos
    where = {}
    for i, p in enumerate(chain):
        if i != pos:
            where[p] = i
    
    # Count contacts involving pos before and after
    def count_contacts(pt, seq_idx):
        if sequence[seq_idx] != "H":
            return 0
        cnt = 0
        for q in _neighbours(pt):
            j = where.get(q)
            if j is not None and j != seq_idx and abs(j - seq_idx) > 1 and sequence[j] == "H":
                cnt += 1
        return cnt
    
    before = count_contacts(old_pos, pos)
    after = count_contacts(new_pos, pos)
    
    # Also need to account for contacts that were with pos but now aren't
    # This is complex; for simplicity, return the delta from pos's perspective
    # and let the full _energy verify
    return after - before


def _pivot_move(chain, pivot, rng, dimension):
    """Apply a pivot move: rotate the tail after pivot by 90 degrees."""
    n = len(chain)
    if pivot >= n - 1:
        return None
    
    bond = [chain[pivot + 1][a] - chain[pivot][a] for a in range(dimension)]
    bond_axis = None
    for a in range(dimension):
        if bond[a] != 0:
            bond_axis = a
            break
    if bond_axis is None:
        return None
    
    other_axes = [a for a in range(dimension) if a != bond_axis]
    rot_axis = rng.choice(other_axes)
    
    pivot_pt = chain[pivot]
    new_chain = list(chain[:pivot + 1])
    occupied = set(chain[:pivot + 1])
    
    new_tail = []
    for pt in chain[pivot + 1:]:
        rel = [pt[a] - pivot_pt[a] for a in range(dimension)]
        new_rel = list(rel)
        if rot_axis == 0:  # around x
            new_rel[1], new_rel[2] = -rel[2], rel[1]
        elif rot_axis == 1:  # around y
            new_rel[0], new_rel[2] = rel[2], -rel[0]
        else:  # around z
            new_rel[0], new_rel[1] = -rel[1], rel[0]
        
        new_pt = tuple(pivot_pt[a] + new_rel[a] for a in range(dimension))
        new_tail.append(new_pt)
    
    for pt in new_tail:
        if pt in occupied:
            return None
        occupied.add(pt)
    
    new_chain.extend(new_tail)
    return new_chain


def _local_move(chain, pos, rng, dimension):
    """
    Try to move a small segment (2-3 residues) to improve the conformation.
    This is a more targeted move than pivot.
    """
    n = len(chain)
    if pos < 1 or pos >= n - 1:
        return None
    
    # Try moving a 2-residue segment
    # The segment is chain[pos:pos+2]
    # We can try rotating this segment around chain[pos-1] or chain[pos+2]
    
    # Option 1: Rotate the bond chain[pos]->chain[pos+1] around chain[pos]
    # This is essentially a pivot at pos
    return _pivot_move(chain, pos, rng, dimension)


def _greedy_init(sequence, dimension, rng):
    """
    Greedy initialization that prefers H-H contacts.
    """
    n = len(sequence)
    if n <= 1:
        return [(0,) * dimension]
    
    chain = [(0,) * dimension]
    occupied = {chain[0]: 0}
    
    for i in range(1, n):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            # Score this site
            score = 0
            if sequence[i] == "H":
                # Check if this site is adjacent to any previous H
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        score += 10
            # Slight preference for compactness
            dist = sum(c * c for c in site)
            score -= dist * 0.1
            
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        # Pick the best candidate with some randomness
        candidates.sort(key=lambda x: -x[0])
        k = min(3, len(candidates))
        site = rng.choice([c[1] for c in candidates[:k]])
        occupied[site] = i
        chain.append(site)
    
    return chain


def _helix_init(sequence, dimension, rng):
    """Generate a compact helical initial conformation."""
    n = len(sequence)
    if n <= 1:
        return [(0,) * dimension]
    
    chain = [(0, 0, 0)]
    occupied = {chain[0]: 0}
    
    for i in range(1, n):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            dist = sum(c * c for c in site)
            candidates.append((dist, site))
        
        if not candidates:
            return None
        
        candidates.sort()
        k = min(3, len(candidates))
        site = rng.choice([c[1] for c in candidates[:k]])
        occupied[site] = i
        chain.append(site)
    
    return chain


def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n <= 1:
        return [(0,) * dimension]
    
    chain = [(0,) * dimension]
    occupied = {chain[0]: 0}
    
    for i in range(1, n):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                candidates.append(site)
        
        if not candidates:
            return None
        
        site = rng.choice(candidates)
        occupied[site] = i
        chain.append(site)
    
    return chain


def _simulated_annealing(sequence, chain, rng, time_budget, dimension):
    """
    Simulated annealing with pivot moves and local segment moves.
    """
    n = len(sequence)
    if n <= 1:
        return chain
    
    current = list(chain)
    current_energy = _energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    start_time = time.time()
    temp = 2.0
    cooling = 0.9995
    min_temp = 0.05
    
    while time.time() - start_time < time_budget and temp > min_temp:
        # Choose move type
        r = rng.random()
        
        if r < 0.6:
            # Pivot move
            pivot = rng.randint(0, n - 2)
            new_chain = _pivot_move(current, pivot, rng, dimension)
            if new_chain is None:
                continue
        elif r < 0.8:
            # Short pivot (near the end)
            pivot = rng.randint(max(0, n - 10), n - 2)
            new_chain = _pivot_move(current, pivot, rng, dimension)
            if new_chain is None:
                continue
        else:
            # Long pivot (near the beginning)
            pivot = rng.randint(0, min(n - 2, 10))
            new_chain = _pivot_move(current, pivot, rng, dimension)
            if new_chain is None:
                continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Metropolis acceptance
        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            current = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best = list(new_chain)
                best_energy = new_energy
        
        temp *= cooling
    
    return best


def fold(sequence, dimension):
    """
    Best conformation using multiple initializations and simulated annealing.
    """
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(12345 + len(sequence))
    
    # Time budget for this sequence
    # Longer sequences need more time
    base_time = 3.0
    per_residue = 0.2
    total_budget = min(15.0, base_time + per_residue * n)
    
    # Reserve some time
    remaining = _DEADLINE - time.time()
    if remaining < 1.0:
        # Very little time left, just return a valid conformation
        chain = _greedy_init(sequence, dimension, rng)
        if chain is None:
            chain = _random_walk(sequence, dimension, rng)
        if chain is None:
            chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in chain]
    
    # Cap the budget
    total_budget = min(total_budget, remaining * 0.85)
    
    best_chain = None
    best_energy = 0
    
    # Number of restarts based on budget
    n_restarts = max(3, min(8, int(total_budget / 2.0)))
    
    for restart in range(n_restarts):
        if time.time() > _DEADLINE - 5:
            break
        
        # Generate initial conformation
        if restart == 0:
            chain = _greedy_init(sequence, dimension, rng)
        elif restart == 1:
            chain = _helix_init(sequence, dimension, rng)
        else:
            chain = _random_walk(sequence, dimension, rng)
        
        if chain is None:
            continue
        
        # Verify validity
        if len(chain) != n:
            continue
        if len(set(chain)) != n:
            continue
        valid = True
        for i in range(n - 1):
            if sum(abs(a - b) for a, b in zip(chain[i], chain[i + 1])) != 1:
                valid = False
                break
        if not valid:
            continue
        
        energy = _energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = list(chain)
            best_energy = energy
        
        # Run simulated annealing with remaining budget
        time_left = _DEADLINE - time.time()
        if time_left < 1.0:
            break
        
        # Allocate budget for this restart
        restarts_left = n_restarts - restart
        search_budget = min(3.0, time_left * 0.8 / restarts_left)
        
        improved = _simulated_annealing(sequence, chain, rng, search_budget, dimension)
        imp_energy = _energy(sequence, improved)
        
        if imp_energy < best_energy:
            best_chain = improved
            best_energy = imp_energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
