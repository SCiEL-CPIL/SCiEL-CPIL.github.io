"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import exp

_START_TIME = time.time()
_DEADLINE = 195.0  # Safety margin before 210s kill


def _remaining_time():
    return max(0.0, _DEADLINE - (time.time() - _START_TIME))


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute HP energy: -number of H-H contacts (non-adjacent in chain)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng, bias=1.0):
    """
    One randomized greedy chain growth.
    bias > 1 favors contacts, bias < 1 favors free space.
    """
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Add a small lookahead for potential future H neighbors
            future_h = 0
            if sequence[i] == "H" and i < n - 3:
                for q in _neighbours(site):
                    if q not in occupied:
                        future_h += 0.2
            score = gain * bias + 0.1 * free + 0.15 * future_h + rng.random() * 0.8
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _reverse_move(chain, i, j):
    """Try reversing the segment from i to j (inclusive). Return new chain or None."""
    n = len(chain)
    if j - i < 2:
        return None
    if j >= n:
        return None
    # Check boundary connections
    if i > 0:
        a, b = chain[i - 1], chain[j]
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return None
    if j < n - 1:
        a, b = chain[i], chain[j + 1]
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return None
    
    # Build the reversed segment
    rev_segment = chain[i:j + 1][::-1]
    new_chain = chain[:i] + rev_segment + chain[j + 1:]
    
    # Check self-avoidance
    seen = set()
    for p in new_chain:
        if p in seen:
            return None
        seen.add(p)
    
    return new_chain


def _rotate_move(chain, i, j, rng):
    """Try rotating the segment from i to j by 90 degrees around chain[i]. Return new chain or None."""
    n = len(chain)
    if j - i < 2 or j >= n:
        return None
    center = chain[i]
    # Try 90, 180, 270 degree rotations
    for angle in (1, 3):  # 90 CW and 90 CCW (180 is less useful)
        new_chain = chain[:i]
        for k in range(i, j + 1):
            dx = chain[k][0] - center[0]
            dy = chain[k][1] - center[1]
            if angle == 1:  # 90 CW: (dx,dy) -> (dy,-dx)
                nx, ny = dy, -dx
            else:  # 90 CCW: (dx,dy) -> (-dy,dx)
                nx, ny = -dy, dx
            new_chain.append((center[0] + nx, center[1] + ny))
        new_chain.extend(chain[j + 1:])
        
        # Check self-avoidance
        seen = set()
        valid = True
        for p in new_chain:
            if p in seen:
                valid = False
                break
            seen.add(p)
        if not valid:
            continue
        
        # Check unit steps
        for k in range(len(new_chain) - 1):
            diff = abs(new_chain[k][0] - new_chain[k + 1][0]) + abs(new_chain[k][1] - new_chain[k + 1][1])
            if diff != 1:
                valid = False
                break
        if valid:
            return new_chain
    return None


def _local_search(sequence, chain, max_iter=5000):
    """Greedy local search: only accept improving moves."""
    n = len(sequence)
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    rng = random.Random(42)
    
    for _ in range(max_iter):
        improved = False
        # Try short reversals
        for _ in range(15):
            i = rng.randint(0, n - 3)
            max_seg = min(n - 1, max(3, int(n * 0.2)))
            j = rng.randint(i + 2, min(i + max_seg, n - 1))
            new_chain = _reverse_move(chain, i, j)
            if new_chain is None:
                continue
            new_energy = _energy(sequence, new_chain)
            if new_energy < current_energy:
                chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
                improved = True
                break
        if not improved:
            # Try rotations
            for _ in range(5):
                i = rng.randint(0, n - 3)
                j = rng.randint(i + 2, min(i + 8, n - 1))
                new_chain = _rotate_move(chain, i, j, rng)
                if new_chain is None:
                    continue
                new_energy = _energy(sequence, new_chain)
                if new_energy < current_energy:
                    chain = new_chain
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = chain[:]
                    improved = True
                    break
        if not improved:
            break
    return best_chain, best_energy


def _anneal(sequence, chain, dimension, rng, time_budget):
    """Simulated annealing to improve a chain."""
    start_time = time.time()
    n = len(sequence)
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    temperature = 4.0
    cooling = 0.997
    stagnation = 0
    max_stagnation = 500
    
    while time.time() - start_time < time_budget and temperature > 0.05:
        # Choose move type
        move_type = rng.random()
        if move_type < 0.75:
            # Reverse move - bias toward shorter segments
            max_len = min(n - 1, max(3, int(n * 0.35)))
            i = rng.randint(0, n - 3)
            j = rng.randint(i + 2, min(i + max_len, n - 1))
            new_chain = _reverse_move(chain, i, j)
        else:
            # Rotate move
            i = rng.randint(0, n - 3)
            j = rng.randint(i + 2, min(i + 10, n - 1))
            new_chain = _rotate_move(chain, i, j, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or (temperature > 0.1 and rng.random() < exp(-delta / temperature)):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
                stagnation = 0
            else:
                stagnation += 1
        else:
            stagnation += 1
        
        temperature *= cooling
        
        # Adaptive cooling: if stagnating, cool faster
        if stagnation > max_stagnation:
            temperature *= 0.95
            stagnation = 0
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using greedy growth followed by simulated annealing and local search."""
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    
    # Time allocation
    base_time = 4.0
    scale = max(1, n // 12)
    total_budget = min(_remaining_time() - 5.0, base_time * scale)
    
    if total_budget < 1.0:
        # Not enough time for annealing, just do greedy
        best, best_energy = None, 1
        for _ in range(50):
            chain = _grow(sequence, dimension, rng, bias=1.5)
            if chain is None:
                continue
            value = _energy(sequence, chain)
            if value < best_energy:
                best, best_energy = chain, value
        if best is None:
            best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in best]
    
    # Generate initial chains with diverse biases
    num_growths = min(50, max(10, n // 2))
    candidates = []
    biases = [1.0, 1.5, 2.0, 0.5, 1.2, 1.8, 0.8, 2.5]
    for k in range(num_growths):
        bias = biases[k % len(biases)]
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is not None:
            candidates.append(chain)
    
    if not candidates:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in best]
    
    # Evaluate and sort candidates
    candidates_with_energy = []
    for chain in candidates:
        e = _energy(sequence, chain)
        candidates_with_energy.append((e, chain))
    candidates_with_energy.sort(key=lambda x: x[0])
    
    # Anneal top candidates with time budget
    num_to_anneal = min(len(candidates_with_energy), max(1, int(total_budget / 2.0)))
    time_per_anneal = total_budget / max(1, num_to_anneal)
    
    best_chain = candidates_with_energy[0][1]
    best_energy = candidates_with_energy[0][0]
    
    for idx in range(num_to_anneal):
        if _remaining_time() < 8.0:
            break
        e, chain = candidates_with_energy[idx]
        if e >= best_energy and idx > 0:
            break  # No point annealing worse candidates
        
        anneal_budget = min(time_per_anneal, _remaining_time() - 8.0)
        if anneal_budget < 0.5:
            break
        
        improved_chain, improved_energy = _anneal(sequence, chain, dimension, rng, anneal_budget)
        
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain
    
    # Final local search polish
    if _remaining_time() > 3.0:
        polish_budget = min(_remaining_time() - 3.0, 3.0)
        max_iter = int(polish_budget * 8000)
        best_chain, best_energy = _local_search(sequence, best_chain, max_iter)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
