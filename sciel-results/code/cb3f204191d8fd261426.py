"""HP lattice protein folding using Hamiltonian path skeleton + local search."""
# EVOLVE-BLOCK-START
import random
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute HP contact energy: -number of non-consecutive H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _generate_hamiltonian_box(width, height):
    """Generate a Hamiltonian path through a width x height grid (serpentine)."""
    path = []
    for row in range(height):
        if row % 2 == 0:
            for col in range(width):
                path.append((col, row))
        else:
            for col in range(width - 1, -1, -1):
                path.append((col, row))
    return path


def _place_on_skeleton(sequence, skeleton):
    """Map sequence onto skeleton, returning chain positions."""
    n = len(sequence)
    if n > len(skeleton):
        return None
    # Try different starting offsets on the skeleton
    best_chain = None
    best_energy = 1
    for start in range(0, len(skeleton) - n + 1):
        chain = [skeleton[start + i] for i in range(n)]
        e = _energy(sequence, chain)
        if e < best_energy:
            best_energy = e
            best_chain = chain
    return best_chain, best_energy


def _local_search(sequence, chain, max_iters, rng):
    """Refine conformation using single-residue pivot moves."""
    n = len(chain)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    occupied = {p: i for i, p in enumerate(chain)}
    
    iters = 0
    no_improve = 0
    while iters < max_iters and no_improve < max_iters // 5:
        iters += 1
        # Pick a random residue to try moving
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        # Try all 4 neighbours
        moves = list(_neighbours(old_pos))
        rng.shuffle(moves)
        moved = False
        for new_pos in moves:
            if new_pos in occupied:
                continue
            # Check self-avoidance: new_pos not occupied by any other residue
            # and the move doesn't break adjacency
            # Remove old_pos temporarily
            del occupied[old_pos]
            chain[i] = new_pos
            occupied[new_pos] = i
            # Check that consecutive residues are still adjacent
            valid = True
            if i > 0 and abs(chain[i][0] - chain[i-1][0]) + abs(chain[i][1] - chain[i-1][1]) != 1:
                valid = False
            if valid and i < n - 1 and abs(chain[i][0] - chain[i+1][0]) + abs(chain[i][1] - chain[i+1][1]) != 1:
                valid = False
            if not valid:
                # Revert
                del occupied[new_pos]
                chain[i] = old_pos
                occupied[old_pos] = i
                continue
            # Compute energy change
            new_energy = _energy(sequence, chain)
            if new_energy < current_energy:
                current_energy = new_energy
                moved = True
                no_improve = 0
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                break
            else:
                # Revert
                del occupied[new_pos]
                chain[i] = old_pos
                occupied[old_pos] = i
        if not moved:
            no_improve += 1
    
    return best_chain, best_energy


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth as fallback."""
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
    """Best conformation using Hamiltonian skeleton + local search."""
    n = len(sequence)
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s
    
    rng = random.Random(42 + len(sequence))
    best_chain = None
    best_energy = 1
    
    # Count H residues to estimate compactness
    h_count = sequence.count("H")
    
    # Try multiple box sizes for Hamiltonian path skeletons
    # The chain needs at least n cells; compact boxes give more contacts
    # Try squares and rectangles that are close to n
    candidates = []
    # Square-ish boxes
    import math
    side = int(math.ceil(math.sqrt(n)))
    for w in range(max(1, side - 3), side + 4):
        for h in range(max(1, side - 3), side + 4):
            if w * h >= n and w * h <= n + 20:
                candidates.append((w, h))
    # Also try some elongated shapes
    for w in range(1, min(n, 15)):
        h = int(math.ceil(n / w))
        if h * w >= n and h * w <= n + 20:
            candidates.append((w, h))
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)
    
    # Sort by compactness (prefer smaller area)
    unique_candidates.sort(key=lambda x: x[0] * x[1])
    
    # Try each candidate skeleton
    for w, h in unique_candidates[:30]:  # limit to 30 best candidates
        if time.time() > deadline:
            break
        skeleton = _generate_hamiltonian_box(w, h)
        result = _place_on_skeleton(sequence, skeleton)
        if result is None:
            continue
        chain, energy = result
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
        # Local search refinement
        if time.time() < deadline - 5:
            refined_chain, refined_energy = _local_search(sequence, chain, max_iters=500, rng=rng)
            if refined_energy < best_energy:
                best_energy = refined_energy
                best_chain = refined_chain
    
    # If skeleton approach didn't work well, try greedy growth as backup
    if best_chain is None or best_energy > -1:
        for _ in range(100):
            if time.time() > deadline:
                break
            chain = _grow(sequence, dimension, rng)
            if chain is None:
                continue
            value = _energy(sequence, chain)
            if value < best_energy:
                best_energy = value
                best_chain = chain
    
    # Final local search on best chain
    if best_chain is not None and time.time() < deadline - 3:
        refined_chain, refined_energy = _local_search(sequence, best_chain, max_iters=200, rng=rng)
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain
    
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
