"""HP lattice protein folding: skeleton + fast local search."""
# EVOLVE-BLOCK-START
import random
import time

_DIR = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def _neighbours(point):
    x, y = point
    for dx, dy in _DIR:
        yield (x + dx, y + dy)


def _energy(sequence, chain):
    """Count H-H non-consecutive contacts; energy = -contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _generate_skeleton(n, rng):
    """Generate a snake-like space-filling path of length n."""
    # Find smallest square-ish box that fits n cells
    side = int(n ** 0.5) + 1
    path = []
    for r in range(side):
        if r % 2 == 0:
            row = [(r, c) for c in range(side)]
        else:
            row = [(r, c) for c in range(side - 1, -1, -1)]
        path.extend(row)
    path = path[:n]
    return path


def _local_search(sequence, chain, rng, deadline, max_iters):
    """Pivot-move local search with incremental energy updates."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    iters = 0

    # Precompute which indices are H
    is_h = [c == 'H' for c in sequence]

    while iters < max_iters:
        if time.time() > deadline:
            break
        iters += 1

        # Pick a random interior residue to pivot
        i = rng.randint(1, n - 2)
        old_pos = chain[i]

        # Try all 4 directions
        for dx, dy in _DIR:
            new_pos = (old_pos[0] + dx, old_pos[1] + dy)

            # Check self-avoidance: new_pos must not be occupied by any other residue
            if new_pos in where:
                occ = where[new_pos]
                if occ != i and abs(occ - i) > 1:
                    continue
                if occ == i:
                    continue
                # If occupied by adjacent in chain, it's a U-turn back - skip
                if abs(occ - i) == 1:
                    continue

            # Compute energy delta
            delta = 0
            # Remove contacts involving old_pos (if H)
            if is_h[i]:
                for q in _neighbours(old_pos):
                    j = where.get(q)
                    if j is not None and j != i and abs(j - i) > 1 and is_h[j]:
                        delta += 1  # losing a contact

            # Add contacts involving new_pos (if H)
            if is_h[i]:
                for q in _neighbours(new_pos):
                    j = where.get(q)
                    if j is not None and j != i and abs(j - i) > 1 and is_h[j]:
                        delta -= 1  # gaining a contact

            new_energy = current_energy + delta

            if new_energy < current_energy or (new_energy == current_energy and rng.random() < 0.1):
                # Accept move
                del where[old_pos]
                where[new_pos] = i
                chain[i] = new_pos
                current_energy = new_energy

                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)

                break  # accept first improving move found
        else:
            # No move accepted in this iteration
            # Occasionally do a random walk step to escape local minima
            if rng.random() < 0.05:
                i = rng.randint(1, n - 2)
                old_pos = chain[i]
                for dx, dy in _DIR:
                    new_pos = (old_pos[0] + dx, old_pos[1] + dy)
                    if new_pos not in where or where[new_pos] == i:
                        if new_pos in where and where[new_pos] != i and abs(where[new_pos] - i) <= 1:
                            continue
                        if new_pos in where and where[new_pos] != i:
                            continue
                        del where[old_pos]
                        where[new_pos] = i
                        chain[i] = new_pos
                        break

    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold sequence using skeleton generation and local search."""
    n = len(sequence)
    start_time = time.time()
    # 120s budget for all sequences; use ~10s per sequence as safety margin
    deadline = start_time + min(10.0, 115.0)

    best_chain = None
    best_energy = 0  # energy is negative, so 0 is worst (no contacts)

    num_skeletons = 50
    max_iters_per_skeleton = 20000

    rng = random.Random(42 + len(sequence) * 7)

    for s in range(num_skeletons):
        if time.time() > deadline:
            break

        # Generate skeleton with some randomization
        if s == 0:
            skeleton = _generate_skeleton(n, rng)
        else:
            # Perturb the base skeleton
            skeleton = _generate_skeleton(n, rng)
            # Shuffle some interior points
            for _ in range(min(5, n // 4)):
                idx = rng.randint(1, n - 2)
                # Swap with a nearby index
                swap_idx = rng.randint(1, n - 2)
                if abs(idx - swap_idx) > 2:
                    continue
                # Only swap if it maintains valid steps (this is complex, so just regenerate)
                skeleton = _generate_skeleton(n, rng)
                break

        # Validate skeleton: check unit steps and self-avoidance
        valid = True
        seen = set()
        for i, p in enumerate(skeleton):
            if p in seen:
                valid = False
                break
            seen.add(p)
            if i > 0:
                prev = skeleton[i - 1]
                if abs(p[0] - prev[0]) + abs(p[1] - prev[1]) != 1:
                    valid = False
                    break

        if not valid:
            continue

        # Run local search
        chain = list(skeleton)
        chain, energy = _local_search(sequence, chain, rng, deadline, max_iters_per_skeleton)

        if energy < best_energy:
            best_energy = energy
            best_chain = chain

    if best_chain is None:
        # Fallback: straight line
        best_chain = [[i, 0] for i in range(n)]

    return [[p[0], p[1]] for p in best_chain]
# EVOLVE-BLOCK-END
