"""HP lattice protein folding - local search with Monte Carlo moves."""
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
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk as initial conformation."""
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
            # dead end - try a different approach: backtrack or just fail
            return None
        site = rng.choice(candidates)
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
    
    # Make a compact helix-like structure
    # In 3D, a helix goes around a cylinder
    # For simplicity, use a zigzag in a plane then shift
    for i in range(1, n):
        # Try to make compact: alternate directions
        candidates = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                # Prefer sites that are closer to origin (compactness)
                dist = sum(c * c for c in site)
                candidates.append((dist, site))
        if not candidates:
            return None
        # Pick from closest candidates with some randomness
        candidates.sort()
        # Pick from top few
        k = min(3, len(candidates))
        site = rng.choice([c[1] for c in candidates[:k]])
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move(chain, pivot, rng, dimension):
    """Apply a pivot move: rotate the tail after pivot by 90 degrees around an axis."""
    n = len(chain)
    if pivot >= n - 1:
        return None
    
    # The tail is chain[pivot+1:]
    # We rotate the tail around the bond (chain[pivot], chain[pivot+1])
    # Actually, pivot move rotates the tail around the bond at pivot
    
    bond = [chain[pivot+1][a] - chain[pivot][a] for a in range(dimension)]
    
    # Choose a rotation axis perpendicular to the bond
    # The bond direction is along one axis
    bond_axis = None
    for a in range(dimension):
        if bond[a] != 0:
            bond_axis = a
            break
    if bond_axis is None:
        return None
    
    # Choose one of the other two axes as rotation axis
    other_axes = [a for a in range(dimension) if a != bond_axis]
    rot_axis = rng.choice(other_axes)
    
    # Rotation by 90 degrees: swap coordinates and negate one
    # For a 90-degree rotation around rot_axis:
    # The bond direction stays the same, the tail gets rotated
    
    # New positions for the tail
    new_chain = list(chain[:pivot + 1])
    occupied = set(chain[:pivot + 1])
    
    # The first tail point is at chain[pivot+1]
    # After rotation, it should still be adjacent to chain[pivot]
    # Actually, in a pivot move, we rotate the entire tail around the pivot bond
    
    # Let's think differently: the tail starts at chain[pivot+1]
    # We rotate the tail vector relative to chain[pivot]
    
    # Vector from chain[pivot] to chain[pivot+1] is the bond
    # We rotate this bond and all subsequent vectors
    
    # Actually, simpler: rotate the entire tail around the point chain[pivot]
    # by 90 degrees around an axis through chain[pivot] perpendicular to the bond
    
    # Let's use a simpler approach: try all 26 possible moves (translations + rotations)
    # For a pivot move, we rotate the tail around the bond axis
    
    # The bond is along bond_axis. We rotate around one of the other axes.
    # A 90-degree rotation around rot_axis centered at chain[pivot]:
    
    pivot_pt = chain[pivot]
    
    # For each point in the tail, compute new position after 90-degree rotation
    # around rot_axis through pivot_pt
    # Rotation by 90 degrees: (x,y,z) -> depends on axis
    
    new_tail = []
    for pt in chain[pivot + 1:]:
        rel = [pt[a] - pivot_pt[a] for a in range(dimension)]
        # Rotate rel by 90 degrees around rot_axis
        # For 90 deg around x: (x,y,z) -> (x,-z,y)
        # For 90 deg around y: (x,y,z) -> (z,y,-x)
        # For 90 deg around z: (x,y,z) -> (-y,x,z)
        new_rel = list(rel)
        if rot_axis == 0:  # around x
            new_rel[1], new_rel[2] = -rel[2], rel[1]
        elif rot_axis == 1:  # around y
            new_rel[0], new_rel[2] = rel[2], -rel[0]
        else:  # around z
            new_rel[0], new_rel[1] = -rel[1], rel[0]
        
        new_pt = tuple(pivot_pt[a] + new_rel[a] for a in range(dimension))
        new_tail.append(new_pt)
    
    # Check self-avoidance
    for pt in new_tail:
        if pt in occupied:
            return None
        occupied.add(pt)
    
    new_chain.extend(new_tail)
    return new_chain


def _translate_move(chain, rng, dimension):
    """Translate the entire chain by one unit in a random direction."""
    n = len(chain)
    if n <= 1:
        return None
    
    axis = rng.randint(0, dimension - 1)
    step = rng.choice([-1, 1])
    
    new_chain = []
    for pt in chain:
        new_pt = list(pt)
        new_pt[axis] += step
        new_chain.append(tuple(new_pt))
    
    # Check self-avoidance (translation preserves self-avoidance if original was valid)
    # But we need to check against... actually translation of a valid SAW is always valid
    return new_chain


def _local_search(sequence, chain, rng, time_budget, dimension):
    """Monte Carlo local search to improve conformation energy."""
    n = len(sequence)
    if n <= 1:
        return chain
    
    current = list(chain)
    current_energy = _energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    start_time = time.time()
    temp = 1.0  # initial temperature
    cooling = 0.999
    
    moves_made = 0
    while time.time() - start_time < time_budget:
        moves_made += 1
        
        # Choose a move type
        move_type = rng.random()
        
        if move_type < 0.7:
            # Pivot move
            pivot = rng.randint(0, n - 2)
            new_chain = _pivot_move(current, pivot, rng, dimension)
            if new_chain is None:
                continue
        else:
            # Small local move: move one internal residue
            # Pick a random internal position and try to move it
            if n < 3:
                continue
            pos = rng.randint(1, n - 2)
            old_pt = current[pos]
            # Try moving to a neighboring site
            candidates = [q for q in _neighbours(old_pt) if q not in set(current)]
            if not candidates:
                continue
            new_pt = rng.choice(candidates)
            # Check that the move doesn't break adjacency with neighbors
            # Actually, we need to move a segment, not a single point, to maintain unit steps
            # Single point move breaks the chain. Let's do a small pivot instead.
            pivot = pos - 1
            new_chain = _pivot_move(current, pivot, rng, dimension)
            if new_chain is None:
                continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Metropolis acceptance
        if delta <= 0:
            current = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best = list(new_chain)
                best_energy = new_energy
        else:
            # Accept with probability exp(-delta/temp)
            if temp > 0.01:
                prob = math.exp(-delta / temp)
                if rng.random() < prob:
                    current = new_chain
                    current_energy = new_energy
        
        temp *= cooling
    
    return best


def fold(sequence, dimension):
    """Best conformation using local search from multiple initial conformations."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(12345 + len(sequence))
    
    # Time budget for this sequence
    # Longer sequences need more time
    base_time = 2.0
    per_residue = 0.15
    total_budget = min(10.0, base_time + per_residue * n)
    
    # Reserve some time
    remaining = _DEADLINE - time.time()
    if remaining < 1.0:
        # Very little time left, just return a valid conformation
        chain = _random_walk(sequence, dimension, rng)
        if chain is None:
            chain = _helix_init(sequence, dimension, rng)
        if chain is None:
            chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in chain]
    
    # Cap the budget
    total_budget = min(total_budget, remaining * 0.8)
    
    best_chain = None
    best_energy = 0
    
    # Generate initial conformations and run local search
    n_initial = max(3, min(10, int(total_budget / 1.5)))
    
    for init_i in range(n_initial):
        if time.time() > _DEADLINE - 5:
            break
        
        # Generate initial conformation
        if init_i == 0:
            chain = _random_walk(sequence, dimension, rng)
        elif init_i == 1:
            chain = _helix_init(sequence, dimension, rng)
        else:
            chain = _random_walk(sequence, dimension, rng)
        
        if chain is None:
            continue
        
        # Verify it's valid
        if len(chain) != n:
            continue
        if len(set(chain)) != n:
            continue
        for i in range(n - 1):
            if sum(abs(a - b) for a, b in zip(chain[i], chain[i + 1])) != 1:
                break
        else:
            # Valid
            energy = _energy(sequence, chain)
            if best_chain is None or energy < best_energy:
                best_chain = list(chain)
                best_energy = energy
            
            # Run local search with remaining budget
            time_left = _DEADLINE - time.time()
            if time_left < 2.0:
                break
            search_budget = min(2.0, time_left * 0.7 / (n_initial - init_i))
            improved = _local_search(sequence, chain, rng, search_budget, dimension)
            imp_energy = _energy(sequence, improved)
            if imp_energy < best_energy:
                best_chain = improved
                best_energy = imp_energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
