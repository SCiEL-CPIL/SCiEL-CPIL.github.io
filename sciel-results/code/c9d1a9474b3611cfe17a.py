"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random
import math

_GLOBAL_DEADLINE = None


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy as minus the number of non-adjacent H-H contacts."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


class _SpatialGrid:
    """
    A lightweight spatial hash grid for O(1) neighbor lookups.
    Maps lattice coordinates to the chain index occupying that site.
    """
    def __init__(self, chain):
        self.grid = {p: i for i, p in enumerate(chain)}
    
    def get(self, point):
        return self.grid.get(point)
    
    def set(self, point, index):
        self.grid[point] = index
    
    def remove(self, point):
        self.grid.pop(point, None)
    
    def contains(self, point):
        return point in self.grid
    
    def to_dict(self):
        return dict(self.grid)


def _greedy_grow(sequence, dimension, rng, temperature=1.0):
    """Randomized greedy growth with simulated annealing-like acceptance."""
    start = (0,) * dimension
    chain = [start]
    occupied = _SpatialGrid(chain)
    
    for i in range(1, len(sequence)):
        candidates = []
        for site in _neighbours(chain[-1]):
            if occupied.contains(site):
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if not occupied.contains(q))
            score = gain + 0.1 * free + rng.random() * temperature
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        # Pick best with some randomness
        candidates.sort(reverse=True)
        if rng.random() < 0.8:
            site = candidates[0][1]
        elif len(candidates) > 1:
            site = candidates[1][1]
        else:
            site = candidates[0][1]
        
        occupied.set(site, i)
        chain.append(site)
    
    return chain


def _pivot_move(sequence, chain, i, rng):
    """
    Perform a pivot move at index i.
    Rotates the suffix chain[i:] by 90 degrees around a random axis through chain[i-1].
    Returns (new_chain, new_occupied, energy_delta) or None if invalid.
    """
    n = len(chain)
    if i <= 0 or i >= n:
        return None
    
    pivot = chain[i-1]
    suffix = chain[i:]
    
    axis = rng.randint(0, 2)
    sign = rng.choice([-1, 1])
    
    new_suffix = []
    for p in suffix:
        dx = p[0] - pivot[0]
        dy = p[1] - pivot[1]
        dz = p[2] - pivot[2]
        
        if axis == 0: # Rotate around X
            new_dx = dx
            new_dy = -sign * dz
            new_dz = sign * dy
        elif axis == 1: # Rotate around Y
            new_dx = sign * dz
            new_dy = dy
            new_dz = -sign * dx
        else: # Rotate around Z
            new_dx = -sign * dy
            new_dy = sign * dx
            new_dz = dz
            
        new_p = (pivot[0] + new_dx, pivot[1] + new_dy, pivot[2] + new_dz)
        new_suffix.append(new_p)
        
    # Check validity using a set for O(1) lookups
    prefix_set = set(chain[:i])
    seen = set()
    for p in new_suffix:
        if p in seen:
            return None
        seen.add(p)
        
    for p in new_suffix:
        if p in prefix_set:
            return None
            
    new_chain = chain[:i] + new_suffix
    
    # Compute energy delta efficiently
    old_occupied = _SpatialGrid(chain)
    new_occupied = _SpatialGrid(new_chain)
    
    # Count contacts in old state involving moved residues
    old_contacts = 0
    for idx in range(i, n):
        if sequence[idx] == 'H':
            p = chain[idx]
            for q in _neighbours(p):
                j = old_occupied.get(q)
                if j is not None and j != idx and abs(j - idx) > 1 and sequence[j] == 'H':
                    old_contacts += 1
                    
    # Count contacts in new state involving moved residues
    new_contacts = 0
    for idx in range(i, n):
        if sequence[idx] == 'H':
            p = new_chain[idx]
            for q in _neighbours(p):
                j = new_occupied.get(q)
                if j is not None and j != idx and abs(j - idx) > 1 and sequence[j] == 'H':
                    new_contacts += 1
                    
    delta = new_contacts - old_contacts
    
    return new_chain, new_occupied, delta


def _local_search(sequence, chain, rng, deadline):
    """
    Hill-climbing local search with pivot moves.
    """
    n = len(chain)
    best_chain = list(chain)
    best_occupied = _SpatialGrid(best_chain)
    best_energy = _energy(sequence, best_chain)
    
    while time.time() < deadline:
        improved = False
        # Shuffle indices to avoid bias
        indices = list(range(1, n - 1))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            result = _pivot_move(sequence, best_chain, i, rng)
            if result is None:
                continue
            
            new_chain, new_occupied, delta = result
            
            # Accept if energy improves (delta > 0 means more contacts, lower energy)
            if delta > 0:
                best_chain = new_chain
                best_occupied = new_occupied
                best_energy -= delta
                improved = True
        
        if not improved:
            break
    
    return best_chain, best_energy


def _simulated_annealing(sequence, chain, rng, deadline):
    """
    Simulated annealing with pivot moves.
    """
    n = len(chain)
    current_chain = list(chain)
    current_occupied = _SpatialGrid(current_chain)
    current_energy = _energy(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    t0 = time.time()
    t_end = deadline
    initial_temp = 5.0
    cooling_rate = 0.999
    
    temperature = initial_temp
    no_improve_count = 0
    
    while time.time() < t_end and temperature > 0.01:
        i = rng.randint(1, n - 2)
        result = _pivot_move(sequence, current_chain, i, rng)
        if result is None:
            temperature *= cooling_rate
            continue
        
        new_chain, new_occupied, delta = result
        
        # Delta > 0 means energy improves (more negative)
        if delta > 0:
            current_chain = new_chain
            current_occupied = new_occupied
            current_energy -= delta
            no_improve_count = 0
            if current_energy < best_energy:
                best_chain = list(current_chain)
                best_energy = current_energy
        else:
            # Accept worse move with probability exp(delta/temperature) where delta is negative
            # Actually, delta is the change in contacts (positive = better).
            # Energy change = -delta. So if delta < 0, energy increases.
            # Accept with probability exp(-energy_change / temperature) = exp(delta / temperature)
            if rng.random() < math.exp(delta / temperature):
                current_chain = new_chain
                current_occupied = new_occupied
                current_energy -= delta
                no_improve_count = 0
        
        temperature *= cooling_rate
        no_improve_count += 1
    
    # Final local search on best
    if time.time() < deadline - 0.5:
        best_chain, best_energy = _local_search(sequence, best_chain, rng, deadline)
    
    return best_chain, best_energy


def _heuristic_grow(sequence, dimension, rng):
    """
    More sophisticated growth that considers future potential.
    """
    n = len(sequence)
    start = (0,) * dimension
    chain = [start]
    occupied = _SpatialGrid(chain)
    
    # Precompute H positions for faster lookup
    h_positions = [i for i, r in enumerate(sequence) if r == 'H']
    
    for i in range(1, n):
        candidates = []
        for site in _neighbours(chain[-1]):
            if occupied.contains(site):
                continue
            
            # Score based on:
            # 1. Immediate H-H contacts
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            # 2. Potential for future contacts (number of H residues ahead that could contact)
            potential = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if not occupied.contains(q):
                        # Count how many future H residues could potentially be here
                        # This is a rough heuristic
                        potential += 0.5
            
            # 3. Free space for future growth
            free = sum(1 for q in _neighbours(site) if not occupied.contains(q))
            
            # 4. Avoid trapping
            # Check if the site creates a pocket that will trap the chain
            trap_score = 0
            if free < 2:
                trap_score = 1.0  # Penalize low free space
            
            score = gain * 2.0 + potential * 0.5 + 0.1 * free - trap_score + rng.random() * 0.3
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        candidates.sort(reverse=True)
        # Pick from top few with some randomness
        k = min(3, len(candidates))
        if rng.random() < 0.7:
            site = candidates[0][1]
        else:
            site = candidates[rng.randint(0, k - 1)][1]
        
        occupied.set(site, i)
        chain.append(site)
    
    return chain


def fold(sequence, dimension):
    """
    Find a good conformation for the given HP sequence.
    Returns a list of lattice points (lists of integers).
    """
    global _GLOBAL_DEADLINE
    if _GLOBAL_DEADLINE is None:
        _GLOBAL_DEADLINE = time.time() + 330.0
    
    start_time = time.time()
    remaining = _GLOBAL_DEADLINE - start_time
    if remaining < 1.0:
        # Not enough time, return a simple valid conformation
        n = len(sequence)
        return [[i, 0, 0] for i in range(n)]
    
    # Use a fraction of the remaining time
    budget = min(remaining * 0.95, 15.0)
    deadline = start_time + budget
    
    n = len(sequence)
    rng = random.Random(12345 + len(sequence) + int(time.time() * 1000) % 10000)
    
    best_chain = None
    best_energy = float('inf')
    
    # Phase 1: Generate multiple initial conformations
    n_grows = max(1, int((deadline - time.time()) * 2))  # ~2 grows per second
    n_grows = min(n_grows, 50)
    
    grow_deadline = time.time() + (deadline - time.time()) * 0.3
    
    for _ in range(n_grows):
        if time.time() > grow_deadline:
            break
        
        # Alternate between different growth strategies
        strategy = rng.random()
        if strategy < 0.4:
            chain = _greedy_grow(sequence, dimension, rng, temperature=1.0)
        elif strategy < 0.7:
            chain = _heuristic_grow(sequence, dimension, rng)
        else:
            chain = _greedy_grow(sequence, dimension, rng, temperature=2.0)
        
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [[i, 0, 0] for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local search on the best
    if time.time() < deadline - 1.0:
        best_chain, best_energy = _local_search(sequence, best_chain, rng, deadline - 1.0)
    
    # Phase 3: Simulated annealing if time remains
    if time.time() < deadline - 0.5:
        sa_chain, sa_energy = _simulated_annealing(sequence, best_chain, rng, deadline)
        if sa_energy < best_energy:
            best_chain = sa_chain
            best_energy = sa_energy
    
    return best_chain

# EVOLVE-BLOCK-END
