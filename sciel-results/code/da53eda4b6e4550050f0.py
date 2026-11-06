"""HP lattice protein folding: fast greedy growth + local search + time guard."""
# EVOLVE-BLOCK-START
import time
import random

_T0 = time.time()
_DEADLINE = 195.0  # 210s wall-clock limit, 15s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _energy_fast(seq, chain):
    """Compute energy (negative contact count) using a position->index map."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if seq[i] == "H":
            x, y = p
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = where.get((x + dx, y + dy))
                if j is not None and j > i + 1 and seq[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng):
    """
    One randomized greedy chain growth.
    Prefer sites that create new H-H contacts, then free space, with random tie-breaking.
    Returns list of tuples or None if stuck.
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)

    for i in range(1, n):
        cx, cy = chain[-1]
        candidates = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            site = (cx + dx, cy + dy)
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                sx, sy = site
                for dx2, dy2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = (sx + dx2, sy + dy2)
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            # Count free neighbours for mild expansion bias
            free = 0
            sx, sy = site
            for dx2, dy2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (sx + dx2, sy + dy2) not in occupied:
                    free += 1
            score = gain * 10.0 + free * 0.1 + rng.random()
            candidates.append((score, site))

        if not candidates:
            return None
        candidates.sort(key=lambda t: -t[0])
        site = candidates[0][1]
        occupied[site] = i
        chain.append(site)

    return chain


def _local_search(sequence, chain, rng, max_iters=500):
    """
    Local search: try random moves (move a residue to an adjacent free site)
    that keep the chain self-avoiding and unit-step. Accept if energy improves
    or equal (with probability). Return best chain found.
    """
    n = len(sequence)
    best_chain = list(chain)
    best_energy = _energy_fast(sequence, best_chain)

    where = {p: i for i, p in enumerate(best_chain)}

    for _ in range(max_iters):
        if _remaining() < 5.0:
            break
        # Pick a random interior residue (not endpoints)
        if n <= 2:
            break
        i = rng.randint(1, n - 2)
        x, y = best_chain[i]

        # Try moving to each of 4 neighbours
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            new_pos = (x + dx, y + dy)
            if new_pos == best_chain[i - 1] or new_pos == best_chain[i + 1]:
                continue
            if new_pos in where:
                # Check if the occupant is not i-1 or i+1 (already excluded above)
                if where[new_pos] != i:
                    continue

            # Tentatively move
            old_pos = best_chain[i]
            best_chain[i] = new_pos
            del where[old_pos]
            where[new_pos] = i

            new_energy = _energy_fast(sequence, best_chain)

            if new_energy < best_energy or (new_energy == best_energy and rng.random() < 0.3):
                best_energy = new_energy
            else:
                # Revert
                best_chain[i] = old_pos
                del where[new_pos]
                where[old_pos] = i

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Best conformation using multiple greedy growths and local search.
    """
    n = len(sequence)
    rng = random.Random(12345 + n)

    best_chain = None
    best_energy = 0  # energy is negative; 0 means no contacts

    # Number of growth attempts scaled by sequence length, but bounded by time
    num_grow = min(300, max(50, 2000 // max(1, n // 10)))
    num_ls = min(20, max(5, 100 // max(1, n // 10)))

    for attempt in range(num_grow):
        if _remaining() < 10.0:
            break

        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue

        # Run local search on this chain
        chain, energy = _local_search(sequence, chain, rng, max_iters=200)

        if energy < best_energy:
            best_energy = energy
            best_chain = chain

        # If we've found a very good solution, do a few more LS rounds on it
        if best_chain is not None and attempt % 5 == 4:
            best_chain, best_energy = _local_search(sequence, best_chain, rng, max_iters=300)

    # Final polish on the best chain
    if best_chain is not None and _remaining() > 15.0:
        best_chain, best_energy = _local_search(sequence, best_chain, rng, max_iters=500)

    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
