"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = time.time() + 280  # safety margin before 330s kill

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
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _initial_grow(sequence, dimension, rng):
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _pivot_move(chain, rng):
    n = len(chain)
    if n < 4:
        return None
    i = rng.randint(1, n - 3)
    j = rng.randint(i + 1, n - 1)
    direction = rng.choice([-1, 1])
    old = list(chain[i:j + 1])
    # Try to flip the sub-chain
    new_sub = []
    occupied_set = set(chain)
    # Remove the sub-chain from occupied
    for p in old:
        occupied_set.discard(p)
    # The sub-chain is chain[i..j], we reverse and try to re-place
    # Actually, a pivot move rotates the sub-chain around the pivot point
    # Let's do a simpler move: reverse the sub-chain
    new_sub = old[::-1]
    # Check if new_sub is valid
    valid = True
    for k, p in enumerate(new_sub):
        idx = i + k
        if idx > 0 and p == chain[idx - 1]:
            valid = False
            break
        if idx < n - 1 and p == chain[idx + 1]:
            valid = False
            break
        if p in occupied_set:
            valid = False
            break
    if not valid:
        return None
    new_chain = chain[:i] + new_sub + chain[j + 1:]
    return new_chain

def _local_search(sequence, chain, rng, max_iters=500):
    n = len(chain)
    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = list(chain)
    occupied = {p: i for i, p in enumerate(chain)}
    for _ in range(max_iters):
        if time.time() > _DEADLINE:
            break
        # Try a random single-residue move
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        for new_pos in _neighbours(old_pos):
            if new_pos in occupied:
                continue
            # Check self-avoidance: new_pos must not be adjacent to non-neighbors
            # Actually just check it's not in occupied
            chain[i] = new_pos
            occupied[new_pos] = i
            del occupied[old_pos]
            new_energy = _energy(sequence, chain)
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
            # Accept if better or with probability (greedy for now)
            if new_energy < current_energy:
                current_energy = new_energy
            else:
                # Revert
                chain[i] = old_pos
                occupied[old_pos] = i
                del occupied[new_pos]
            break
    return best_chain, best_energy

def _anneal(sequence, chain, rng, t_start=1.0, t_end=0.01, iters=2000):
    n = len(chain)
    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = list(chain)
    t = t_start
    cooling = (t_start - t_end) / max(iters, 1)
    for it in range(iters):
        if time.time() > _DEADLINE:
            break
        t = max(t_end, t_start - cooling * it)
        # Pivot move
        new_chain = _pivot_move(chain, rng)
        if new_chain is None:
            # Try single residue move
            i = rng.randint(1, n - 2)
            old_pos = chain[i]
            options = [p for p in _neighbours(old_pos) if p not in set(chain)]
            if not options:
                continue
            new_pos = rng.choice(options)
            new_chain = list(chain)
            new_chain[i] = new_pos
        delta = _energy(sequence, new_chain) - current_energy
        if delta < 0 or rng.random() < math.exp(-delta / max(t, 1e-10)):
            chain = new_chain
            current_energy = _energy(sequence, chain)
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
    return best_chain, best_energy

def fold(sequence, dimension):
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0  # 0 means no contacts, energy = 0

    # Generate multiple initial chains
    num_initial = 30
    for _ in range(num_initial):
        if time.time() > _DEADLINE - 60:
            break
        chain = _initial_grow(sequence, dimension, rng)
        if chain is None:
            continue
        energy = _energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy

    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_energy = 0
    else:
        # Local search on best chain
        best_chain, best_energy = _local_search(sequence, best_chain, rng, max_iters=300)
        # Annealing
        best_chain, best_energy = _anneal(sequence, best_chain, rng, t_start=1.5, t_end=0.05, iters=3000)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
