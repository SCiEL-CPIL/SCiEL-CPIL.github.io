"""HP lattice protein folding on the cubic lattice.
Combines multi-start greedy growth with a simulated-annealing local search
using pivot moves and tail re-growth, plus a deadline guard.
"""
import random
import time
import math

# Deadline: evaluator kills at 330s wall-clock. Use 320s safety margin.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0


def _neighbours(point):
    """Yield the 6 face-neighbours of point in 3D."""
    x, y, z = point
    return ((x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z), (x, y, z + 1), (x, y, z - 1))


def _energy(sequence, chain):
    """Compute the total energy of the chain. Energy = -num_HH_contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _build_occupied(chain):
    """Build dict mapping position -> index for a chain."""
    return {p: i for i, p in enumerate(chain)}


def _energy_delta_pivot(sequence, chain, occupied, i):
    """
    Compute the energy change if we pivot the tail starting at index i.
    Returns the delta (new_energy - old_energy) or None if the move is invalid.
    This is an approximation: we only check contacts involving the tail residues
    and the prefix. We assume the tail can be rotated without self-collision
    for the purpose of delta computation (actual validity is checked separately).
    
    Actually, for correctness we need to know the new positions. This function
    is a placeholder; the actual delta is computed in _pivot_move after generating
    the new chain. We keep this for interface compatibility but won't use it
    heavily in the main loop.
    """
    return None


def _pivot_move(sequence, chain, i, rng):
    """
    Try a pivot move at position i: rotate the tail starting at i around
    the bond from i-1 to i. Returns (new_chain, new_energy) or (None, None).
    """
    n = len(chain)
    if i < 2 or i >= n - 1:
        return None, None
    
    prefix = chain[:i]
    tail = chain[i:]
    
    pivot = chain[i-1]
    axis_start = chain[i]
    
    ax = axis_start[0] - pivot[0]
    ay = axis_start[1] - pivot[1]
    az = axis_start[2] - pivot[2]
    
    occupied_prefix = set(prefix)
    
    candidate_ends = []
    for site in _neighbours(pivot):
        if site == chain[i]:
            continue
        if site in occupied_prefix:
            continue
        sx = site[0] - pivot[0]
        sy = site[1] - pivot[1]
        sz = site[2] - pivot[2]
        dot = sx*ax + sy*ay + sz*az
        if dot == 0:
            candidate_ends.append((site, 90))
        elif dot == -1:
            candidate_ends.append((site, 180))
    
    if not candidate_ends:
        return None, None
    
    site, angle = candidate_ends[rng.randint(0, len(candidate_ends)-1)]
    
    new_tail = []
    for pt in tail:
        px = pt[0] - pivot[0]
        py = pt[1] - pivot[1]
        pz = pt[2] - pivot[2]
        
        if angle == 90:
            uxv_x = ay*pz - az*py
            uxv_y = az*px - ax*pz
            uxv_z = ax*py - ay*px
            udv = ax*px + ay*py + az*pz
            rx = uxv_x + ax*udv
            ry = uxv_y + ay*udv
            rz = uxv_z + az*udv
        elif angle == 180:
            udv = ax*px + ay*py + az*pz
            rx = -px + 2*ax*udv
            ry = -py + 2*ay*udv
            rz = -pz + 2*az*udv
        else:
            return None, None
        
        new_pt = (pivot[0] + rx, pivot[1] + ry, pivot[2] + rz)
        new_tail.append(new_pt)
    
    new_chain = prefix + new_tail
    
    seen = set()
    for pt in new_chain:
        if pt in seen:
            return None, None
        seen.add(pt)
    
    for k in range(len(new_chain)-1):
        diff = abs(new_chain[k][0] - new_chain[k+1][0]) + abs(new_chain[k][1] - new_chain[k+1][1]) + abs(new_chain[k][2] - new_chain[k+1][2])
        if diff != 1:
            return None, None
    
    new_energy = _energy(sequence, new_chain)
    return new_chain, new_energy


def _greedy_grow_tail(sequence, chain, i, rng, temperature=0):
    """
    Re-grow the tail from index i onwards, given the prefix up to i-1.
    If temperature > 0, use stochastic selection based on energy gain.
    """
    prefix = chain[:i]
    occupied = {p: idx for idx, p in enumerate(prefix)}
    
    new_chain = list(prefix)
    
    for k in range(i, len(sequence)):
        last = new_chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[k] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < k - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain + 0.1 * free + rng.random()
            if temperature > 0:
                # Boltzmann-like weighting
                score = gain + 0.1 * free + temperature * rng.gauss(0, 1)
            options.append((score, site))
        
        if not options:
            return None, None
        
        # Select best option (or stochastic if temperature high)
        site = max(options)[1]
        occupied[site] = k
        new_chain.append(site)
    
    energy = _energy(sequence, new_chain)
    return new_chain, energy


def _compute_energy(sequence, chain):
    """Alias for _energy."""
    return _energy(sequence, chain)


def _regrow_tail(sequence, chain, i, rng, temperature=0):
    """
    Re-grow the tail from index i onwards, given the prefix up to i-1.
    """
    return _greedy_grow_tail(sequence, chain, i, rng, temperature)


def _anneal(sequence, chain, rng, time_limit):
    """
    Run simulated annealing on the given chain, returning the best chain found.
    """
    n = len(sequence)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Temperature schedule
    t0 = max(1.0, -current_energy * 0.5)
    t_min = 0.1
    cooling_rate = 0.999
    
    temperature = t0
    max_steps = max(200, int(time_limit * 80))
    
    for step in range(max_steps):
        if time.time() > _DEADLINE - 8:
            break
        
        # Choose move type
        if rng.random() < 0.5:
            if n >= 4:
                i = rng.randint(2, n - 2)
                new_chain, new_energy = _pivot_move(sequence, current_chain, i, rng)
            else:
                new_chain, new_energy = None, None
        else:
            r = rng.random()
            if r < 0.25:
                i = rng.randint(1, max(1, n // 5))
            elif r < 0.5:
                i = rng.randint(n // 5, n // 3)
            elif r < 0.75:
                i = rng.randint(n // 3, n // 2)
            else:
                i = rng.randint(n // 2, n - 1)
            
            i = min(i, n - 1)
            new_chain, new_energy = _regrow_tail(sequence, current_chain, i, rng, temperature)
        
        if new_chain is None:
            continue
        
        delta_e = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta_e / T)
        if delta_e < 0:
            accept = True
        else:
            if temperature > t_min:
                accept = rng.random() < math.exp(-delta_e / temperature)
            else:
                accept = False
        
        if accept:
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        # Cool down
        temperature *= cooling_rate
    
    return best_chain, best_energy


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth."""
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


def fold(sequence, dimension):
    """
    Fold the sequence on the cubic lattice.
    Phase 1: Multiple randomized greedy growths.
    Phase 2: Simulated annealing local search with pivot moves and tail re-growth.
    Phase 3: Final polish pass with steeper cooling.
    """
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    # Phase 1: Multiple greedy growths
    time_remaining = _DEADLINE - time.time()
    base_growths = 400 if n < 50 else 150
    num_growths = max(30, min(800, int(base_growths * (time_remaining / 300.0))))
    
    for _ in range(num_growths):
        if time.time() > _DEADLINE - 15:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = 0
    
    # Phase 2: Simulated annealing
    time_remaining = _DEADLINE - time.time()
    sa_time = min(time_remaining * 0.6, 120.0)
    
    best_chain, best_energy = _anneal(sequence, best_chain, rng, sa_time)
    
    # Phase 3: Final polish with steeper cooling
    time_remaining = _DEADLINE - time.time()
    if time_remaining > 10:
        polish_time = min(time_remaining * 0.5, 60.0)
        best_chain, best_energy = _anneal(sequence, best_chain, rng, polish_time)
    
    return [list(p) for p in best_chain]
