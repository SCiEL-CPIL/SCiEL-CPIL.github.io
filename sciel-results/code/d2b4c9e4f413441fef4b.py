"""HP lattice protein folding: greedy init + local search with reversals."""
# EVOLVE-BLOCK-START
import random
import time


def _neighbours4(point):
    """The 4 axis-neighbours of a 2D lattice point."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _init_deadline():
    return time.monotonic() + 190.0  # safety margin under 210s wall


def _greedy_init(sequence, dimension, rng):
    """Randomized greedy growth: prefer sites creating new H-H contacts."""
    n = len(sequence)
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    for i in range(1, n):
        best_site = None
        best_key = None
        for site in _neighbours4(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours4(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours4(site) if q not in occupied)
            key = gain * 10 + free + rng.random()
            if best_key is None or key > best_key:
                best_key = key
                best_site = site
        if best_site is None:
            return None
        occupied[best_site] = i
        chain.append(best_site)
    return chain


def _compute_energy(sequence, chain):
    """Energy = -(number of non-consecutive H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours4(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _local_search(sequence, chain, deadline, rng):
    """Local search: single-residue moves + 2-opt reversals, accepting improvements."""
    n = len(chain)
    best = list(chain)
    best_e = _compute_energy(sequence, best)

    def valid_move(idx, new_pos):
        # Check new_pos not occupied by any other residue
        if new_pos in best_set:
            # It might be the residue itself being moved
            if best[idx] != new_pos:
                return False
        # Check neighbours in chain don't conflict
        for nb_idx in (idx - 1, idx + 1):
            if 0 <= nb_idx < n:
                nb_pos = best[nb_idx]
                if abs(nb_pos[0] - new_pos[0]) + abs(nb_pos[1] - new_pos[1]) != 1:
                    return False
        return True

    def rebuild_set():
        return set(best)

    best_set = rebuild_set()

    while time.monotonic() < deadline:
        improved = False
        # Try single-residue moves (skip ends)
        for idx in range(1, n - 1):
            if time.monotonic() >= deadline:
                return best, best_e
            old_pos = best[idx]
            for cand in _neighbours4(old_pos):
                if cand == best[idx - 1] or cand == best[idx + 1]:
                    continue
                # Check cand is free
                if cand in best_set:
                    continue
                # Apply move
                best_set.remove(old_pos)
                best_set.add(cand)
                best[idx] = cand
                new_e = _compute_energy(sequence, best)
                if new_e < best_e:
                    best_e = new_e
                    improved = True
                else:
                    # Revert
                    best[idx] = old_pos
                    best_set.remove(cand)
                    best_set.add(old_pos)

        # Try 2-opt reversals on a random subset of segments
        if not improved:
            # Random segment reversal
            for _ in range(20):
                if time.monotonic() >= deadline:
                    return best, best_e
                if n < 4:
                    break
                i = rng.randint(1, n - 3)
                j = rng.randint(i + 1, n - 2)
                # Reverse segment [i..j]
                segment = best[i:j + 1]
                segment.reverse()
                old_segment = best[i:j + 1]
                best[i:j + 1] = segment
                # Check validity: no self-avoidance violations
                valid = True
                # Build set for quick check
                new_set = set(best)
                if len(new_set) != n:
                    valid = False
                else:
                    # Check consecutive unit steps
                    for k in range(n - 1):
                        if abs(best[k][0] - best[k+1][0]) + abs(best[k][1] - best[k+1][1]) != 1:
                            valid = False
                            break
                if valid:
                    new_e = _compute_energy(sequence, best)
                    if new_e < best_e:
                        best_e = new_e
                        improved = True
                    else:
                        best[i:j + 1] = old_segment
                else:
                    best[i:j + 1] = old_segment

        if not improved:
            # Try random perturbation then re-search
            if n > 4:
                idx = rng.randint(1, n - 2)
                old_pos = best[idx]
                for cand in _neighbours4(old_pos):
                    if cand == best[idx-1] or cand == best[idx+1]:
                        continue
                    if cand not in best_set:
                        best_set.remove(old_pos)
                        best_set.add(cand)
                        best[idx] = cand
                        break
                else:
                    break
            else:
                break

    return best, best_e


def fold(sequence, dimension):
    """Best conformation via greedy init + local search, with time deadline."""
    deadline = _init_deadline()
    n = len(sequence)
    rng = random.Random(42 + len(sequence))

    best_chain = None
    best_energy = 0  # 0 = no contacts (unfolded)

    # Multiple greedy inits to find a good starting point
    for attempt in range(50):
        if time.monotonic() >= deadline - 30:
            break
        chain = _greedy_init(sequence, dimension, rng)
        if chain is None:
            continue
        e = _compute_energy(sequence, chain)
        if best_chain is None or e < best_energy:
            best_chain = chain
            best_energy = e
        if best_energy <= -n:  # very good, stop early
            break

    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)

    # Local search to improve
    if time.monotonic() < deadline - 5:
        best_chain, best_energy = _local_search(sequence, best_chain, deadline, rng)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
