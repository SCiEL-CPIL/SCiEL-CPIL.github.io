"""Stronger HP folding: randomized greedy growth + local Monte-Carlo optimisation."""
# EVOLVE-BLOCK-START
import random
import time
import heapq


_DEADLINE = time.time() + 315.0  # safety margin before 330 s hard kill


def _deadline_ok():
    return time.time() < _DEADLINE


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


def _grow(sequence, dimension, rng, beta=0.3):
    """Randomized greedy chain growth with a tunable free-space weight."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + beta * free + rng.random() * 0.5, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_search(sequence, chain, max_moves=5000, temp=0.5):
    """Metropolis-style local search on kink moves and 180-degree turns."""
    occupied = {p: i for i, p in enumerate(chain)}
    energy = _energy(sequence, chain)
    best_energy = energy
    best_chain = list(chain)

    n = len(chain)
    if n < 4:
        return best_chain, best_energy

    # Precompute H positions for quick contact delta
    h_set = set(i for i, r in enumerate(sequence) if r == "H")

    for move_num in range(max_moves):
        if not _deadline_ok():
            break

        # Pick a random residue to move (not endpoints)
        i = rng.randint(1, n - 2)
        p_old = chain[i]
        p_prev = chain[i - 1]
        p_next = chain[i + 1]

        # Candidate sites: neighbours of p_old that are not p_prev or p_next
        candidates = []
        for site in _neighbours(p_old):
            if site == p_prev or site == p_next:
                continue
            if site in occupied:
                continue
            candidates.append(site)

        if not candidates:
            continue

        p_new = candidates[rng.randint(0, len(candidates) - 1)]

        # Check self-avoidance more carefully: the new site must not be occupied
        # and the move must not create an overlap with any other residue.
        # Since we already checked `site in occupied`, and we're only moving
        # one residue, this should be fine. But we also need to ensure the
        # new position doesn't coincide with p_prev or p_next (already excluded).

        # Compute delta energy
        delta = 0
        # Remove contacts involving i
        for q in _neighbours(p_old):
            j = occupied.get(q)
            if j is not None and j != i - 1 and j != i + 1:
                if i in h_set and j in h_set:
                    delta += 1  # losing a contact (energy increases by 1, i.e. -1)
        # Add contacts at new position
        for q in _neighbours(p_new):
            j = occupied.get(q)
            if j is not None and j != i - 1 and j != i + 1:
                if i in h_set and j in h_set:
                    delta -= 1  # gaining a contact (energy decreases by 1)

        # Accept?
        if delta < 0 or rng.random() < (2.718281828 ** (-delta / temp)):
            # Apply move
            del occupied[p_old]
            occupied[p_new] = i
            chain[i] = p_new
            energy += delta
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)

    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation via randomized growth + local search, with deadline guard."""
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0  # start at 0 (no contacts) so any real chain beats it

    # Phase 1: randomized greedy growths with local search
    num_growths = 30
    for attempt in range(num_growths):
        if not _deadline_ok():
            break

        beta = 0.1 + 0.5 * (attempt / max(num_growths, 1))
        chain = _grow(sequence, dimension, rng, beta=beta)
        if chain is None:
            continue

        # Apply local search
        chain, energy = _local_search(sequence, chain, max_moves=3000, temp=0.3)

        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)

    # Phase 2: if time remains, do additional local-search-only restarts from
    # the best chain found, with perturbation
    if best_chain is not None and _deadline_ok():
        for _ in range(10):
            if not _deadline_ok():
                break
            # Perturb best_chain: pick a random residue and try a move
            perturbed = list(best_chain)
            n = len(perturbed)
            i = rng.randint(1, n - 2)
            p_old = perturbed[i]
            p_prev = perturbed[i - 1]
            p_next = perturbed[i + 1]
            occ = {p: j for j, p in enumerate(perturbed)}
            cands = [s for s in _neighbours(p_old) if s not in (p_prev, p_next) and s not in occ]
            if cands:
                p_new = rng.choice(cands)
                del occ[p_old]
                occ[p_new] = i
                perturbed[i] = p_new
            chain2, energy2 = _local_search(sequence, perturbed, max_moves=2000, temp=0.5)
            if energy2 < best_energy:
                best_energy = energy2
                best_chain = list(chain2)

    # Fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]

    return [list(p) for p in best_chain]


# Re-seed the RNG used in _local_search at module level
rng = random.Random(99999)
# EVOLVE-BLOCK-END
