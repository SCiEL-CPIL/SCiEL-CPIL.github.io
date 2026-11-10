# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = time.time() + 195.0


def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _is_valid_chain(chain, n):
    """Check self-avoidance and unit-step connectivity."""
    if len(chain) != n:
        return False
    seen = set()
    for p in chain:
        if p in seen:
            return False
        seen.add(p)
    for i in range(n - 1):
        if abs(chain[i][0] - chain[i+1][0]) + abs(chain[i][1] - chain[i+1][1]) != 1:
            return False
    return True


def _greedy_grow(sequence, dimension, rng, temperature=0.0):
    """
    Randomized greedy chain growth.
    Prefer sites that create new H-H contacts, then sites with more free space.
    Avoids dead ends for the next step.
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)

    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue

            # Avoid dead ends if there are more residues to place
            if i < n - 1:
                has_exit = False
                for q in _neighbours(site):
                    if q not in occupied and q != last:
                        has_exit = True
                        break
                if not has_exit:
                    continue

            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.5 + rng.random() * temperature
            options.append((score, site))

        if not options:
            return None

        best_score, best_site = max(options, key=lambda x: x[0])
        occupied[best_site] = i
        chain.append(best_site)

    return chain


def _greedy_grow_biased(sequence, dimension, rng, temperature=0.0, bias_center=False):
    """
    Greedy growth with optional bias toward the center of the bounding box.
    This helps create more compact conformations.
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)

    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)

            center_bias = 0.0
            if bias_center and i > n * 0.3:
                # Bias toward center of current bounding box
                xs = [p[0] for p in chain]
                ys = [p[1] for p in chain]
                cx = (min(xs) + max(xs)) / 2.0
                cy = (min(ys) + max(ys)) / 2.0
                dist_to_center = abs(site[0] - cx) + abs(site[1] - cy)
                center_bias = -dist_to_center * 0.5

            score = gain * 10.0 + free * 0.1 + center_bias + rng.random() * temperature
            options.append((score, site))

        if not options:
            return None

        best_score, best_site = max(options, key=lambda x: x[0])
        occupied[best_site] = i
        chain.append(best_site)

    return chain


def _greedy_grow_from_fixed(suffix_seq, start_pos, prefix, full_sequence, rng, temperature=0.0):
    """
    Greedy growth for suffix, with proper access to full sequence for H checks.
    """
    if not suffix_seq:
        return []

    # Build occupied set from prefix with their global indices
    occupied = {p: i for i, p in enumerate(prefix)}
    chain = [start_pos]

    for i, res in enumerate(suffix_seq):
        global_idx = len(prefix) + i
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if res == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < global_idx - 1 and full_sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))

        if not options:
            return None

        best_score, best_site = max(options, key=lambda x: x[0])
        occupied[best_site] = global_idx
        chain.append(best_site)

    return chain[1:]


def _min_conflicts_repair(sequence, chain, rng, max_iters=200):
    """
    No-park conflict resolution: fix self-avoidance and connectivity violations
    by moving only involved residues. Returns (repaired_chain, energy) or (None, None).
    """
    n = len(chain)
    if n < 3:
        return chain, _compute_energy(sequence, chain)

    chain = list(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    for _ in range(max_iters):
        if time.time() > _DEADLINE - 2:
            break

        # Check for violations
        seen = {}
        conflicts = []
        for i, p in enumerate(chain):
            if p in seen:
                conflicts.append((i, seen[p]))
            else:
                seen[p] = i

        # Check connectivity
        for i in range(n - 1):
            if abs(chain[i][0] - chain[i+1][0]) + abs(chain[i][1] - chain[i+1][1]) != 1:
                conflicts.append((i, i + 1))

        if not conflicts:
            break

        # Pick a random conflict and try to fix it
        c1, c2 = rng.choice(conflicts)
        # Try moving residue c1 to a free neighbor that maintains connectivity
        if 0 < c1 < n - 1:
            prev, nxt = chain[c1 - 1], chain[c1 + 1]
            p = chain[c1]
            for np in _neighbours(p):
                if np == prev or np == nxt:
                    continue
                # Check if np is free
                is_free = True
                for i, q in enumerate(chain):
                    if q == np and i != c1:
                        is_free = False
                        break
                if not is_free:
                    continue
                # Check connectivity
                if abs(np[0] - prev[0]) + abs(np[1] - prev[1]) != 1:
                    continue
                if abs(np[0] - nxt[0]) + abs(np[1] - nxt[1]) != 1:
                    continue
                # Perform move
                chain[c1] = np
                new_energy = _compute_energy(sequence, chain)
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                break
        elif c1 == 0:
            # Move first residue
            nxt = chain[1]
            p = chain[0]
            for np in _neighbours(p):
                if np == nxt:
                    continue
                is_free = True
                for i, q in enumerate(chain):
                    if q == np and i != 0:
                        is_free = False
                        break
                if not is_free:
                    continue
                if abs(np[0] - nxt[0]) + abs(np[1] - nxt[1]) != 1:
                    continue
                chain[0] = np
                new_energy = _compute_energy(sequence, chain)
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                break
        elif c1 == n - 1:
            # Move last residue
            prev = chain[n - 2]
            p = chain[n - 1]
            for np in _neighbours(p):
                if np == prev:
                    continue
                is_free = True
                for i, q in enumerate(chain):
                    if q == np and i != n - 1:
                        is_free = False
                        break
                if not is_free:
                    continue
                if abs(np[0] - prev[0]) + abs(np[1] - prev[1]) != 1:
                    continue
                chain[n - 1] = np
                new_energy = _compute_energy(sequence, chain)
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                break

    # Final check
    if _is_valid_chain(best_chain, n):
        return best_chain, best_energy
    return None, None


def _pivot_move(sequence, chain, occupied, rng, pivot_start=2, pivot_end=None):
    """
    Try a pivot move: rotate a subchain around a pivot point.
    Returns (new_chain, new_occupied, new_energy) if successful, else (None, None, None).
    """
    n = len(chain)
    if pivot_end is None:
        pivot_end = n - 2

    # Pick a pivot point
    pivot = rng.randint(pivot_start, pivot_end)
    if pivot >= n - 1:
        return None, None, None

    # Try different rotation lengths
    for length in range(2, min(10, n - pivot)):
        end = pivot + length - 1
        if end >= n:
            break

        subchain = chain[pivot+1:end+1]
        sub_len = len(subchain)

        for direction in [1, -1]:
            new_subchain = [subchain[0]]
            valid = True
            occupied_check = dict(occupied)
            for pos in subchain:
                if pos in occupied_check:
                    del occupied_check[pos]

            for i in range(1, sub_len):
                last = new_subchain[-1]
                found = False
                for np in _neighbours(last):
                    if np in occupied_check:
                        continue
                    new_subchain.append(np)
                    occupied_check[np] = pivot + 1 + i
                    found = True
                    break
                if not found:
                    valid = False
                    break

            if not valid:
                continue

            if end + 1 < n:
                if abs(new_subchain[-1][0] - chain[end+1][0]) + abs(new_subchain[-1][1] - chain[end+1][1]) != 1:
                    continue

            if abs(new_subchain[0][0] - chain[pivot][0]) + abs(new_subchain[0][1] - chain[pivot][1]) != 1:
                continue

            new_chain = chain[:pivot+1] + new_subchain + chain[end+1:]

            if len(set(new_chain)) != len(new_chain):
                continue

            new_occupied = {p: i for i, p in enumerate(new_chain)}
            new_energy = _compute_energy(sequence, new_chain)

            return new_chain, new_occupied, new_energy

    return None, None, None


def _local_improve(sequence, chain, rng, max_moves=5000):
    """
    Improve conformation via local moves: kink flips, short regrows, and pivot moves.
    """
    n = len(chain)
    if n < 3:
        return chain, _compute_energy(sequence, chain)

    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    no_improve_count = 0

    for _ in range(max_moves):
        if time.time() > _DEADLINE - 5:
            break

        move_type = rng.random()
        improved = False

        if move_type < 0.4:
            # Kink flip
            i = rng.randint(1, n - 2)
            p = chain[i]
            prev, nxt = chain[i - 1], chain[i + 1]

            candidates = []
            for np in _neighbours(prev):
                if np == p:
                    continue
                if abs(np[0] - nxt[0]) + abs(np[1] - nxt[1]) != 1:
                    continue
                if np in occupied and occupied[np] != i:
                    continue
                candidates.append(np)

            if candidates:
                np = rng.choice(candidates)
                del occupied[p]
                chain[i] = np
                occupied[np] = i
                new_energy = _compute_energy(sequence, chain)

                if new_energy < current_energy:
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                    no_improve_count = 0
                    improved = True
                else:
                    del occupied[np]
                    chain[i] = p
                    occupied[p] = i
                    no_improve_count += 1
        elif move_type < 0.6:
            # Short regrow
            length = rng.randint(3, min(6, n - 3))
            split = n - length
            prefix = chain[:split]
            suffix_seq = sequence[split:]

            new_suffix = _greedy_grow_from_fixed(suffix_seq, prefix[-1], prefix, sequence, rng, temperature=0.5)
            if new_suffix is not None:
                new_chain = prefix + new_suffix
                new_energy = _compute_energy(sequence, new_chain)

                if new_energy < current_energy:
                    occupied = {p: i for i, p in enumerate(new_chain)}
                    chain = new_chain
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                    no_improve_count = 0
                    improved = True
                else:
                    no_improve_count += 1
        else:
            # Pivot move
            new_chain, new_occupied, new_energy = _pivot_move(sequence, chain, occupied, rng)
            if new_chain is not None:
                if new_energy < current_energy:
                    chain = new_chain
                    occupied = new_occupied
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                    no_improve_count = 0
                    improved = True
                else:
                    no_improve_count += 1

        if not improved:
            no_improve_count += 1
            if no_improve_count > 300:
                break

    return best_chain, best_energy


def _regrow_sa(sequence, chain, rng, max_iters=5000, start_temp=5.0, cool_rate=0.997, tail_bias=0.0):
    """
    Simulated annealing using regrow moves: pick a split point, regrow the suffix.
    Uses adaptive cooling and biased split selection.
    tail_bias > 0 favors longer tails (smaller split indices).
    Includes min-conflicts repair for regrows that create transient conflicts.
    """
    n = len(sequence)
    if n < 4:
        return chain, _compute_energy(sequence, chain)

    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    temp = start_temp
    no_improve = 0
    stuck_count = 0

    for it in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
        if temp < 0.05:
            break

        min_split = 3
        max_split = max(min_split, n - 3)
        if max_split <= min_split:
            break

        if tail_bias > 0.5:
            scale = n * 0.2
            while True:
                split = min_split + int(rng.expovariate(1.0 / scale))
                if split <= max_split:
                    break
                split = max_split
        elif tail_bias > 0.2:
            u = rng.random()
            split = min_split + int(u * u * (max_split - min_split))
        else:
            mid = (min_split + max_split) // 2
            if rng.random() < 0.7:
                offset = int(rng.gauss(0, (max_split - min_split) / 4))
                split = min(max(min_split, mid + offset), max_split)
            else:
                split = rng.randint(min_split, max_split)

        prefix = chain[:split]
        suffix_seq = sequence[split:]

        new_suffix = _greedy_grow_from_fixed(suffix_seq, prefix[-1], prefix, sequence, rng, temperature=temp)
        if new_suffix is None:
            temp *= cool_rate
            no_improve += 1
            stuck_count += 1
            continue

        new_chain = prefix + new_suffix

        # Check validity; if invalid, try min-conflicts repair
        if not _is_valid_chain(new_chain, n):
            repaired, rep_energy = _min_conflicts_repair(sequence, new_chain, rng, max_iters=50)
            if repaired is None:
                temp *= cool_rate
                no_improve += 1
                stuck_count += 1
                continue
            new_chain = repaired
            new_energy = rep_energy
        else:
            new_energy = _compute_energy(sequence, new_chain)

        delta = new_energy - current_energy
        if delta < 0 or (temp > 0 and rng.random() < (10 ** (-delta / max(temp, 0.1)))):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                no_improve = 0
                stuck_count = 0
            else:
                no_improve += 1
                stuck_count += 1
        else:
            no_improve += 1
            stuck_count += 1

        if stuck_count > 100:
            temp *= cool_rate ** 1.5
            stuck_count = 0
        else:
            temp *= cool_rate

        if no_improve > 600:
            break

    return best_chain, best_energy


def _exhaustive_local_search(sequence, chain):
    """
    Exhaustively try all single-residue moves (kink flips and endpoint extensions)
    until no improving move exists. This is a deterministic final polish.
    """
    n = len(chain)
    if n < 3:
        return chain, _compute_energy(sequence, chain)

    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    improved = True

    while improved and time.time() < _DEADLINE - 1:
        improved = False
        for i in range(1, n - 1):
            if time.time() > _DEADLINE - 1:
                break
            p = chain[i]
            prev, nxt = chain[i - 1], chain[i + 1]
            for np in _neighbours(prev):
                if np == p:
                    continue
                if abs(np[0] - nxt[0]) + abs(np[1] - nxt[1]) != 1:
                    continue
                if np in occupied and occupied[np] != i:
                    continue
                del occupied[p]
                chain[i] = np
                occupied[np] = i
                new_energy = _compute_energy(sequence, chain)
                if new_energy < current_energy:
                    current_energy = new_energy
                    improved = True
                else:
                    del occupied[np]
                    chain[i] = p
                    occupied[p] = i

    return chain, current_energy


def fold(sequence, dimension):
    """
    Fold a sequence into a low-energy conformation.
    """
    if dimension != 2:
        raise ValueError("Only dimension 2 is supported")

    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]

    rng = random.Random(12345 + len(sequence))

    best_chain = None
    best_energy = float('inf')

    # Phase 1: Generate diverse initial solutions
    initial_chains = []

    for temp in [0.0, 0.3, 0.8, 1.5, 2.5]:
        for _ in range(10):
            if time.time() > _DEADLINE - 30:
                break
            chain = _greedy_grow(sequence, dimension, rng, temperature=temp)
            if chain is not None:
                initial_chains.append(chain)

        if time.time() > _DEADLINE - 30:
            break

    for temp in [0.0, 0.5, 1.0]:
        for _ in range(5):
            if time.time() > _DEADLINE - 25:
                break
            chain = _greedy_grow_biased(sequence, dimension, rng, temperature=temp, bias_center=True)
            if chain is not None:
                initial_chains.append(chain)

        if time.time() > _DEADLINE - 25:
            break

    evaluated_chains = []
    for chain in initial_chains:
        energy = _compute_energy(sequence, chain)
        evaluated_chains.append((energy, chain))

    evaluated_chains.sort(key=lambda x: x[0])

    if not evaluated_chains:
        best_chain = [[i, 0] for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
    else:
        best_energy, best_chain = evaluated_chains[0]

    # Phase 2: Local improvement on the best chain
    if time.time() < _DEADLINE - 20:
        improved_chain, improved_energy = _local_improve(sequence, best_chain, rng, max_moves=10000)
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain

    # Phase 3: SA with regrow moves (Standard)
    if time.time() < _DEADLINE - 15:
        sa_chain, sa_energy = _regrow_sa(sequence, best_chain, rng, max_iters=10000, start_temp=5.0, cool_rate=0.997)
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_chain = sa_chain

    # Phase 4: E2T SA (Long tail bias) - Structural change
    if time.time() < _DEADLINE - 10:
        e2t_chain, e2t_energy = _regrow_sa(sequence, best_chain, rng, max_iters=10000, start_temp=8.0, cool_rate=0.995, tail_bias=0.8)
        if e2t_energy < best_energy:
            best_energy = e2t_energy
            best_chain = e2t_chain

    # Phase 5: More diverse restarts with local search + SA
    if time.time() < _DEADLINE - 5:
        restarts_done = 0
        max_restarts = 50
        while restarts_done < max_restarts:
            if time.time() > _DEADLINE - 5:
                break

            if restarts_done % 3 == 0:
                chain = _greedy_grow(sequence, dimension, rng, temperature=0.5)
            elif restarts_done % 3 == 1:
                chain = _greedy_grow_biased(sequence, dimension, rng, temperature=0.5, bias_center=True)
            else:
                chain = _greedy_grow(sequence, dimension, rng, temperature=1.5)

            restarts_done += 1

            if chain is None:
                continue

            energy = _compute_energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain

            if time.time() < _DEADLINE - 5:
                improved = _local_improve(sequence, chain, rng, max_moves=3000)
                if improved[1] < best_energy:
                    best_energy = improved[1]
                    best_chain = improved[0]

            if time.time() < _DEADLINE - 4:
                sa_c, sa_e = _regrow_sa(sequence, best_chain, rng, max_iters=2000, start_temp=3.0, cool_rate=0.998, tail_bias=0.5)
                if sa_e < best_energy:
                    best_energy = sa_e
                    best_chain = sa_c

    # Final local polish
    if time.time() < _DEADLINE - 3:
        final_chain, final_energy = _local_improve(sequence, best_chain, rng, max_moves=5000)
        if final_energy < best_energy:
            best_energy = final_energy
            best_chain = final_chain

    # Min-conflicts final repair
    if time.time() < _DEADLINE - 2:
        repaired, rep_energy = _min_conflicts_repair(sequence, best_chain, rng, max_iters=100)
        if repaired is not None and rep_energy < best_energy:
            best_energy = rep_energy
            best_chain = repaired

    # Exhaustive local-search polish
    if time.time() < _DEADLINE - 1:
        best_chain, best_energy = _exhaustive_local_search(sequence, best_chain)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
