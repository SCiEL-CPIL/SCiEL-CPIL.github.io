"""HP lattice protein folding: greedy growth + simulated annealing with regrow moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 170.0


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute negative count of non-adjacent H-H contacts."""
    n = len(chain)
    if n < 2:
        return 0
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _is_valid_chain(chain):
    """Check self-avoidance and unit steps."""
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            prev = chain[i - 1]
            dist = sum(abs(a - b) for a, b in zip(p, prev))
            if dist != 1:
                return False
    return True


def _greedy_grow(sequence, dimension, rng):
    """Greedy chain growth preferring new H-H contacts."""
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
            options.append((gain + 0.1 * free + rng.random() * 0.5, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move(chain, rng):
    """Random pivot: rotate a random subchain around its attachment point."""
    n = len(chain)
    if n < 3:
        return None
    i = rng.randint(1, n - 2)
    pivot = chain[i]
    subchain = chain[i:]
    
    first = subchain[0]
    direction = (first[0] - pivot[0], first[1] - pivot[1])
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    other_dirs = [d for d in directions if d != direction]
    if not other_dirs:
        return None
    new_dir = rng.choice(other_dirs)
    
    def rotate_90_ccw(v):
        return (-v[1], v[0])
    
    def rotate_180(v):
        return (-v[0], -v[1])
    
    def rotate_90_cw(v):
        return (v[1], -v[0])
    
    rot = None
    if rotate_90_ccw(direction) == new_dir:
        rot = rotate_90_ccw
    elif rotate_180(direction) == new_dir:
        rot = rotate_180
    elif rotate_90_cw(direction) == new_dir:
        rot = rotate_90_cw
    
    if rot is None:
        return None
    
    new_subchain = []
    for p in subchain:
        rel = (p[0] - pivot[0], p[1] - pivot[1])
        new_rel = rot(rel)
        new_subchain.append((pivot[0] + new_rel[0], pivot[1] + new_rel[1]))
    
    new_chain = chain[:i] + new_subchain
    
    if not _is_valid_chain(new_chain):
        return None
    
    return new_chain


def _regrow_move(chain, sequence, dimension, rng):
    """Regrow: truncate the chain at a random point and regrow greedily."""
    n = len(chain)
    if n < 5:
        return None
    cut = rng.randint(2, n - 3)
    prefix = chain[:cut]
    occupied = set(prefix)
    
    # Build a position-to-index map for the prefix
    pos_to_idx = {p: i for i, p in enumerate(prefix)}
    
    new_chain = list(prefix)
    for i in range(cut, n):
        options = []
        for site in _neighbours(new_chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in pos_to_idx:
                        if sequence[pos_to_idx[q]] == "H":
                            gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random() * 0.3, site))
        if not options:
            return None
        site = max(options)[1]
        occupied.add(site)
        pos_to_idx[site] = i
        new_chain.append(site)
    
    if not _is_valid_chain(new_chain):
        return None
    
    return new_chain


def _sa_fold(sequence, dimension, rng, time_budget):
    """Simulated annealing starting from a greedy-grown chain."""
    chain = _greedy_grow(sequence, dimension, rng)
    if chain is None:
        chain = [(i, 0) for i in range(len(sequence))]
    
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    n = len(sequence)
    T_start = 1.0
    total_time = time_budget
    start_time = time.time()
    
    while time.time() - start_time < total_time:
        if time.time() > _DEADLINE - 3:
            break
        
        move_type = rng.random()
        new_chain = None
        
        if move_type < 0.4:
            new_chain = _pivot_move(chain, rng)
        elif move_type < 0.8:
            new_chain = _regrow_move(chain, sequence, dimension, rng)
        else:
            new_chain = _pivot_move(chain, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        elapsed = time.time() - start_time
        progress = min(elapsed / total_time, 1.0)
        T = T_start * math.exp(-5.0 * progress)
        
        if delta <= 0:
            accept = True
        else:
            if T > 0.001:
                accept = rng.random() < math.exp(-delta / T)
            else:
                accept = False
        
        if accept:
            chain = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation found via greedy growth + simulated annealing."""
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    
    best_chain = None
    best_energy = 1
    
    # Greedy growth attempts
    for attempt in range(5):
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is not None:
            energy = _compute_energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
    
    # Time remaining for SA
    elapsed = time.time() - _START_TIME
    remaining = max(0.5, _DEADLINE - elapsed - 3.0)
    
    # Allocate SA time based on sequence length (longer sequences need more)
    sa_time = min(remaining, max(1.0, remaining * 0.8))
    
    sa_chain, sa_energy = _sa_fold(sequence, dimension, rng, sa_time)
    if sa_energy < best_energy:
        best_energy = sa_energy
        best_chain = sa_chain
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
