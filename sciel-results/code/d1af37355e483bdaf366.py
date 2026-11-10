"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import sys
import heapq

# Global deadline guard
_DEADLINE = None
_TIME_LIMIT = 185.0  # seconds, leaving margin before 210s kill


def _init_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT


def _time_up():
    return time.time() >= _DEADLINE


def _neighbours2d(x, y):
    """Return the 4 unit-step neighbours of (x,y)."""
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _compute_energy(sequence, chain):
    """Compute energy = -number of non-adjacent H-H contacts."""
    n = len(chain)
    pos_to_idx = {}
    for i, (x, y) in enumerate(chain):
        pos_to_idx[(x, y)] = i

    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        for nx, ny in _neighbours2d(x, y):
            j = pos_to_idx.get((nx, ny))
            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                if j > i:
                    contacts += 1
    return -contacts


def _fast_energy_delta(sequence, chain, i, new_pos, occupied):
    """
    Compute the change in energy when moving residue i from chain[i] to new_pos.
    Returns delta_energy (negative is improvement).
    """
    old_pos = chain[i]
    if old_pos == new_pos:
        return 0
    
    delta = 0
    
    # Remove contacts involving old position
    if sequence[i] == 'H':
        for nx, ny in _neighbours2d(*old_pos):
            j = occupied.get((nx, ny))
            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                if j > i:
                    delta += 1  # We're losing this contact, so energy increases by 1 (less negative)
    
    # Add contacts involving new position
    if sequence[i] == 'H':
        for nx, ny in _neighbours2d(*new_pos):
            j = occupied.get((nx, ny))
            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                if j > i:
                    delta -= 1  # We're gaining this contact, so energy decreases by 1 (more negative)
    
    return delta


def _greedy_grow(sequence, rng, temperature=0.0):
    """
    Greedy chain growth with optional stochastic element.
    Returns chain (list of (x,y)) or None if stuck.
    """
    n = len(sequence)
    chain = [(0, 0)]
    occupied = {(0, 0): 0}

    for i in range(1, n):
        x, y = chain[-1]
        options = []
        for nx, ny in _neighbours2d(x, y):
            if (nx, ny) in occupied:
                continue
            # Score: number of new H-H contacts this placement would create
            gain = 0
            if sequence[i] == 'H':
                for qx, qy in _neighbours2d(nx, ny):
                    j = occupied.get((qx, qy))
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            # Free space bonus to avoid trapping
            free = 0
            for qx, qy in _neighbours2d(nx, ny):
                if (qx, qy) not in occupied:
                    free += 1
            # Stochastic component
            noise = rng.random() * temperature
            score = gain * 10.0 + free * 0.1 + noise
            options.append((score, (nx, ny)))

        if not options:
            return None

        # Pick best (or stochastically if temperature > 0)
        if temperature > 0 and len(options) > 1:
            best_score = max(o[0] for o in options)
            weights = []
            for s, _ in options:
                weights.append(1.0 / (1.0 + abs(best_score - s)))
            total = sum(weights)
            r = rng.random() * total
            cum = 0
            chosen = options[-1][1]
            for (s, site), w in zip(options, weights):
                cum += w
                if r <= cum:
                    chosen = site
                    break
        else:
            chosen = max(options)[1]

        chain.append(chosen)
        occupied[chosen] = i

    return chain


def _local_search(sequence, chain, rng, max_steps=2000):
    """
    Local search: try to improve energy by moving a single residue.
    """
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    chain = list(chain)
    occupied = {p: i for i, p in enumerate(chain)}

    for step in range(max_steps):
        if _time_up():
            break
        # Pick a random internal residue to try moving
        i = rng.randint(1, n - 2)
        x, y = chain[i]
        # Try moving to a neighbouring free site
        for nx, ny in _neighbours2d(x, y):
            if (nx, ny) in occupied:
                continue
            # Check that the move doesn't break connectivity with neighbours
            px, py = chain[i - 1]
            qx, qy = chain[i + 1]
            if abs(nx - px) + abs(ny - py) != 1:
                continue
            if abs(nx - qx) + abs(ny - qy) != 1:
                continue

            # Perform the move
            old_pos = (x, y)
            new_pos = (nx, ny)
            chain[i] = new_pos
            del occupied[old_pos]
            occupied[new_pos] = i

            new_energy = _compute_energy(sequence, chain)
            if new_energy <= current_energy:
                current_energy = new_energy
            else:
                # Revert
                chain[i] = old_pos
                del occupied[new_pos]
                occupied[old_pos] = i

    return chain, current_energy


def _simulated_annealing(sequence, chain, rng, max_steps=5000, t_start=2.0, t_end=0.1):
    """
    Simulated annealing to escape local minima.
    """
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    chain = list(chain)
    occupied = {p: i for i, p in enumerate(chain)}
    best_chain = list(chain)
    best_energy = current_energy

    for step in range(max_steps):
        if _time_up():
            break
        t = t_start * (t_end / t_start) ** (step / max_steps)
        
        # Pick a random internal residue to try moving
        i = rng.randint(1, n - 2)
        x, y = chain[i]
        # Try moving to a neighbouring free site
        moved = False
        for nx, ny in _neighbours2d(x, y):
            if (nx, ny) in occupied:
                continue
            px, py = chain[i - 1]
            qx, qy = chain[i + 1]
            if abs(nx - px) + abs(ny - py) != 1:
                continue
            if abs(nx - qx) + abs(ny - qy) != 1:
                continue

            old_pos = (x, y)
            new_pos = (nx, ny)
            chain[i] = new_pos
            del occupied[old_pos]
            occupied[new_pos] = i

            new_energy = _compute_energy(sequence, chain)
            delta = new_energy - current_energy

            if delta <= 0 or rng.random() < _exp_neg(delta / t):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                moved = True
                break
            else:
                # Revert
                chain[i] = old_pos
                del occupied[new_pos]
                occupied[old_pos] = i

    return best_chain, best_energy


def _exp_neg(x):
    """Compute exp(-x) safely."""
    if x > 50:
        return 0.0
    return 2.718281828459045 ** (-x)


def _refold_from_middle(sequence, rng):
    """
    Alternative strategy: grow from the middle outward.
    """
    n = len(sequence)
    mid = n // 2
    center = (0, 0)
    chain = [None] * n
    chain[mid] = center
    occupied = {center: mid}

    left = mid - 1
    right = mid + 1

    while left >= 0 or right < n:
        if _time_up():
            return None
        if left >= 0 and (right >= n or rng.random() < 0.5):
            x, y = chain[left + 1]
            options = []
            for nx, ny in _neighbours2d(x, y):
                if (nx, ny) in occupied:
                    continue
                gain = 0
                if sequence[left] == 'H':
                    for qx, qy in _neighbours2d(nx, ny):
                        j = occupied.get((qx, qy))
                        if j is not None and abs(j - left) > 1 and sequence[j] == 'H':
                            gain += 1
                free = sum(1 for qx, qy in _neighbours2d(nx, ny) if (qx, qy) not in occupied)
                options.append((gain * 10.0 + free * 0.1 + rng.random() * 0.5, (nx, ny)))
            if not options:
                return None
            best = max(options)
            chain[left] = best[1]
            occupied[best[1]] = left
            left -= 1
        else:
            x, y = chain[right - 1]
            options = []
            for nx, ny in _neighbours2d(x, y):
                if (nx, ny) in occupied:
                    continue
                gain = 0
                if sequence[right] == 'H':
                    for qx, qy in _neighbours2d(nx, ny):
                        j = occupied.get((qx, qy))
                        if j is not None and abs(j - right) > 1 and sequence[j] == 'H':
                            gain += 1
                free = sum(1 for qx, qy in _neighbours2d(nx, ny) if (qx, qy) not in occupied)
                options.append((gain * 10.0 + free * 0.1 + rng.random() * 0.5, (nx, ny)))
            if not options:
                return None
            best = max(options)
            chain[right] = best[1]
            occupied[best[1]] = right
            right += 1

    return chain


def fold(sequence, dimension):
    """
    Find a low-energy self-avoiding conformation for the given HP sequence.
    Returns list of len(sequence) integer lattice points (lists of length dimension).
    """
    _init_deadline()
    n = len(sequence)
    if n == 0:
        return []

    rng = random.Random(42 + len(sequence) * 137)

    best_chain = None
    best_energy = 0

    # Phase 1: Multiple greedy growths with varying temperatures
    temps = [0.0, 0.1, 0.3, 0.5, 1.0]
    num_greedy = 40

    for t_idx, temp in enumerate(temps):
        for trial in range(num_greedy // len(temps) + 1):
            if _time_up():
                break
            chain = _greedy_grow(sequence, rng, temperature=temp)
            if chain is None:
                continue
            energy = _compute_energy(sequence, chain)
            if best_chain is None or energy < best_energy:
                best_chain = chain
                best_energy = energy

    # Phase 2: Grow-from-middle strategies
    for trial in range(20):
        if _time_up():
            break
        chain = _refold_from_middle(sequence, rng)
        if chain is None:
            continue
        energy = _compute_energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy

    # Phase 3: Local search on best chains
    if best_chain is not None and not _time_up():
        improved_chain, improved_energy = _local_search(sequence, best_chain, rng, max_steps=3000)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy

        # Also try local search on a few other good chains
        for trial in range(8):
            if _time_up():
                break
            chain = _greedy_grow(sequence, rng, temperature=0.2)
            if chain is None:
                continue
            energy = _compute_energy(sequence, chain)
            if energy < best_energy:
                best_chain = chain
                best_energy = energy
            improved_chain, improved_energy = _local_search(sequence, chain, rng, max_steps=1000)
            if improved_energy < best_energy:
                best_chain = improved_chain
                best_energy = improved_energy

    # Phase 4: Simulated annealing on the best chain
    if best_chain is not None and not _time_up():
        sa_chain, sa_energy = _simulated_annealing(sequence, best_chain, rng, max_steps=8000)
        if sa_energy < best_energy:
            best_chain = sa_chain
            best_energy = sa_energy

    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]

    # Ensure output format: list of lists of integers
    return [[x, y] for x, y in best_chain]
# EVOLVE-BLOCK-END
