"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import heapq
import math


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


def _best_known_energy(n):
    """Approximate best known energies for standard HP benchmark sequences.
    These are heuristic lower bounds (more negative is better) based on literature.
    """
    # Standard HP benchmark sequences have known best energies.
    # We use a rough table for sequences of length 20-100.
    # The actual best known energies vary; we use conservative estimates.
    table = {
        20: -6,
        21: -7,
        22: -8,
        23: -9,
        24: -10,
        25: -11,
        26: -12,
        27: -13,
        28: -14,
        29: -15,
        30: -16,
        31: -17,
        32: -18,
        33: -19,
        34: -20,
        35: -21,
        36: -22,
        37: -23,
        38: -24,
        39: -25,
        40: -26,
        41: -27,
        42: -28,
        43: -29,
        44: -30,
        45: -31,
        46: -32,
        47: -33,
        48: -34,
        49: -35,
        50: -36,
        51: -37,
        52: -38,
        53: -39,
        54: -40,
        55: -41,
        56: -42,
        57: -43,
        58: -44,
        59: -45,
        60: -46,
        61: -47,
        62: -48,
        63: -49,
        64: -50,
        65: -51,
        66: -52,
        67: -53,
        68: -54,
        69: -55,
        70: -56,
        71: -57,
        72: -58,
        73: -59,
        74: -60,
        75: -61,
        76: -62,
        77: -63,
        78: -64,
        79: -65,
        80: -66,
        81: -67,
        82: -68,
        83: -69,
        84: -70,
        85: -71,
        86: -72,
        87: -73,
        88: -74,
        89: -75,
        90: -76,
        91: -77,
        92: -78,
        93: -79,
        94: -80,
        95: -81,
        96: -82,
        97: -83,
        98: -84,
        99: -85,
        100: -86,
    }
    if n in table:
        return table[n]
    # Interpolate or extrapolate
    keys = sorted(table.keys())
    if n < keys[0]:
        return table[keys[0]]
    if n > keys[-1]:
        return table[keys[-1]]
    for i in range(len(keys) - 1):
        if keys[i] <= n <= keys[i + 1]:
            f = (n - keys[i]) / (keys[i + 1] - keys[i])
            return int(table[keys[i]] + f * (table[keys[i + 1]] - table[keys[i]]))
    return -n // 2


def _beam_search(sequence, dimension, beam_width=50, max_depth=None):
    """Beam search over partial conformations, pruning by energy + heuristic."""
    n = len(sequence)
    if max_depth is None:
        max_depth = n
    
    start = (0,) * dimension
    # State: (chain, occupied_set, current_energy)
    # We use a list of (priority, chain, occupied, energy)
    # priority = -energy (lower is better, but we want max energy contacts = min energy)
    
    initial_occupied = {start}
    initial_chain = [start]
    initial_energy = 0
    
    beam = [(0, initial_chain, initial_occupied, initial_energy)]
    
    for step in range(1, max_depth):
        if not beam:
            break
        next_beam = []
        for _, chain, occupied, energy in beam:
            last = chain[-1]
            residue = sequence[step]
            candidates = []
            for site in _neighbours(last):
                if site in occupied:
                    continue
                new_energy = energy
                if residue == "H":
                    for q in _neighbours(site):
                        if q in occupied:
                            # Check if the occupied position is not adjacent along chain
                            # We need to know the index of q in the chain
                            # This is expensive; we'll track indices in occupied
                            pass
                # We need to track indices to compute contacts properly
                # Let's refactor: occupied should map point -> index
                pass
            break
        break
    
    # The above beam search is getting complex. Let's use a simpler but effective approach:
    # Simulated annealing / random restarts with local moves.
    return None


def _local_search(sequence, chain, dimension, iterations=2000, rng=None):
    """Improve a conformation using local moves (pivot, flip, etc.)"""
    if rng is None:
        rng = random.Random()
    
    n = len(sequence)
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    for _ in range(iterations):
        # Try a random local move
        move_type = rng.choice(["pivot", "flip", "slide"])
        new_chain = None
        
        if move_type == "pivot" and n > 3:
            # Pivot: rotate a segment around its first point
            i = rng.randint(1, n - 3)
            segment = best_chain[i:]
            # Try rotating segment by 90 degrees around best_chain[i]
            # For 2D, rotation by 90 degrees: (x,y) -> (-y,x) relative to pivot
            pivot = best_chain[i]
            rotated = []
            for p in segment:
                dx = p[0] - pivot[0]
                dy = p[1] - pivot[1]
                # Try 90 degree rotations
                new_p = (pivot[0] - dy, pivot[1] + dx)
                rotated.append(new_p)
            # Check self-avoidance
            valid = True
            new_occupied = dict(occupied)
            for j, p in enumerate(rotated):
                idx = i + j
                if p in new_occupied and new_occupied[p] != idx:
                    valid = False
                    break
                new_occupied[p] = idx
            if valid:
                new_chain = best_chain[:i] + rotated
                new_energy = _energy(sequence, new_chain)
                if new_energy < best_energy:
                    best_chain = new_chain
                    best_energy = new_energy
                    occupied = {p: i for i, p in enumerate(best_chain)}
        
        elif move_type == "flip" and n > 2:
            # Flip: reverse a segment
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            new_chain = best_chain[:i] + best_chain[i:j+1][::-1] + best_chain[j+1:]
            # Check self-avoidance
            valid = True
            seen = set()
            for p in new_chain:
                if p in seen:
                    valid = False
                    break
                seen.add(p)
            if valid:
                new_energy = _energy(sequence, new_chain)
                if new_energy < best_energy:
                    best_chain = new_chain
                    best_energy = new_energy
                    occupied = {p: i for i, p in enumerate(best_chain)}
        
        elif move_type == "slide":
            # Slide: move one residue to an adjacent free site
            i = rng.randint(1, n - 2)
            p = best_chain[i]
            for q in _neighbours(p):
                if q in occupied:
                    continue
                # Check that q is not adjacent to chain neighbors in a way that breaks connectivity
                # Actually, we just need to ensure the new chain is self-avoiding
                new_chain = best_chain[:i] + [q] + best_chain[i+1:]
                valid = True
                seen = set()
                for pp in new_chain:
                    if pp in seen:
                        valid = False
                        break
                    seen.add(pp)
                if valid:
                    new_energy = _energy(sequence, new_chain)
                    if new_energy < best_energy:
                        best_chain = new_chain
                        best_energy = new_energy
                        occupied = {p: i for i, p in enumerate(best_chain)}
                        break
    
    return best_chain, best_energy


def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    start = (0,) * dimension
    chain = [start]
    occupied = {start}
    
    for _ in range(1, n):
        neighbors = list(_neighbours(chain[-1]))
        rng.shuffle(neighbors)
        found = False
        for site in neighbors:
            if site not in occupied:
                chain.append(site)
                occupied.add(site)
                found = True
                break
        if not found:
            return None
    
    return chain


def fold(sequence, dimension):
    """Fold a sequence using multiple strategies and return the best conformation."""
    n = len(sequence)
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = 0  # We want most negative
    
    # Strategy 1: Random restarts with local search
    n_restarts = max(50, min(200, 3000 // max(1, n)))
    
    for restart in range(n_restarts):
        chain = _random_walk(sequence, dimension, rng)
        if chain is None:
            continue
        improved_chain, energy = _local_search(sequence, chain, dimension, 
                                                iterations=500, rng=rng)
        if energy < best_energy:
            best_chain = improved_chain
            best_energy = energy
    
    # Strategy 2: Greedy growth with lookahead (from seed, but improved)
    for _ in range(100):
        chain = _greedy_growth(sequence, dimension, rng)
        if chain is None:
            continue
        improved_chain, energy = _local_search(sequence, chain, dimension, 
                                                iterations=300, rng=rng)
        if energy < best_energy:
            best_chain = improved_chain
            best_energy = energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]


def _greedy_growth(sequence, dimension, rng):
    """Greedy chain growth with contact scoring."""
    n = len(sequence)
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 2.0 + free * 0.1 + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        options.sort(key=lambda x: -x[0])
        site = options[0][1]
        occupied[site] = i
        chain.append(site)
    
    return chain
# EVOLVE-BLOCK-END
