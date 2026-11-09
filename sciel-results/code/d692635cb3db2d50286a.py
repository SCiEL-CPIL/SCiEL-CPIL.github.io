"""HP lattice protein folding: constraint-satisfaction backtracking with lookahead + local refinement."""
# EVOLVE-BLOCK-START
import random
import time
import math
import heapq

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 205.0

_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    n = len(chain)
    if n < 2:
        return 0
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y = p
            for dx, dy in _DELTAS:
                q = (x + dx, y + dy)
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts // 2


def _backtrack_fold(sequence, time_limit, rng, max_depth=None):
    """
    Backtracking search with conflict-directed backjumping and heuristic placement.
    Places residues one by one, prioritizing H residues in positions with high contact potential.
    """
    n = len(sequence)
    if n < 2:
        return [(0, 0)] * n

    if max_depth is None:
        max_depth = min(n, 40)

    is_h = [c == 'H' for c in sequence]

    best_chain = None
    best_energy = 0

    def _greedy_grow(seq, rng, hh_weight=1.0, free_weight=0.1, noise=0.3):
        nn = len(seq)
        if nn < 2:
            return [(0, 0)] * nn
        start = (0, 0)
        chain = [start]
        occupied = {start: 0}

        for i in range(1, nn):
            options = []
            for site in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                x, y = chain[-1]
                cand = (x + site[0], y + site[1])
                if cand in occupied:
                    continue
                gain = 0
                if seq[i] == 'H':
                    cx, cy = cand
                    for dx, dy in _DELTAS:
                        q = (cx + dx, cy + dy)
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and seq[j] == 'H':
                            gain += 1
                free = 0
                cx, cy = cand
                for dx, dy in _DELTAS:
                    if (cx + dx, cy + dy) not in occupied:
                        free += 1
                score = hh_weight * gain + free_weight * free + noise * rng.random()
                options.append((score, cand))
            if not options:
                return None
            best_score, best_site = max(options)
            occupied[best_site] = i
            chain.append(best_site)

        return chain

    initial_chains = []
    configs = [
        (2.0, 0.0, 0.5),
        (1.0, 0.5, 0.5),
        (0.5, 1.0, 0.5),
        (1.5, 0.2, 0.3),
        (1.0, 0.3, 1.0),
        (3.0, 0.0, 0.2),
        (1.0, 0.0, 0.8),
    ]
    growth_deadline = time.time() + min(2.0, time_limit * 0.1)
    for hh_w, free_w, noise in configs:
        if time.time() > growth_deadline:
            break
        for attempt in range(3):
            if time.time() > growth_deadline:
                break
            chain = _greedy_grow(sequence, rng, hh_w, free_w, noise)
            if chain is not None:
                initial_chains.append(chain)

    if not initial_chains:
        initial_chains.append([(i, 0) for i in range(n)])

    scored = []
    for chain in initial_chains:
        e = _energy(sequence, chain)
        scored.append((e, chain))
    scored.sort(key=lambda x: x[0])
    candidates = [s[1] for s in scored[:min(6, len(scored))]]

    def _local_refine(chain, time_limit, rng):
        """Iterated local search with reversals and pivots, with move ordering near H contacts."""
        nn = len(chain)
        current_energy = _energy(sequence, chain)
        best_energy = current_energy
        best_chain = chain[:]
        T = 1.5
        cooling = 0.99997
        min_t = 0.01

        deadline = time.time() + time_limit
        no_improve = 0
        max_no_improve = 5000

        h_indices = [i for i in range(nn) if sequence[i] == 'H']

        def _try_reversal(state_chain, i, j):
            nn_local = len(state_chain)
            if i >= j:
                return None
            if i > 0:
                px, py = state_chain[i - 1]
                ex, ey = state_chain[j]
                if abs(px - ex) + abs(py - ey) != 1:
                    return None
            if j < nn_local - 1:
                sx, sy = state_chain[i]
                nx, ny = state_chain[j + 1]
                if abs(sx - nx) + abs(sy - ny) != 1:
                    return None
            segment = state_chain[i:j+1]
            for pos in segment:
                if pos in state_chain[:i] or pos in state_chain[j+1:]:
                    return None
            new_chain = state_chain[:i] + segment[::-1] + state_chain[j+1:]
            return new_chain

        def _try_pivot(state_chain, i, angle):
            nn_local = len(state_chain)
            if i <= 0 or i >= nn_local - 1:
                return None
            px, py = state_chain[i]
            tail = state_chain[i+1:]
            rotated = []
            for tx, ty in tail:
                rx, ry = tx - px, ty - py
                for _ in range(angle % 4):
                    rx, ry = -ry, rx
                rotated.append((px + rx, py + ry))
            for pos in rotated:
                if pos in state_chain[:i+1]:
                    return None
            new_chain = state_chain[:i+1] + rotated
            return new_chain

        def _generate_move():
            move_type = rng.random()
            if move_type < 0.55:
                if h_indices and rng.random() < 0.4:
                    hi = rng.choice(h_indices)
                    span = rng.randint(2, min(20, nn - 1))
                    i = max(0, hi - span // 2)
                    j = min(nn - 1, i + span)
                    if j <= i:
                        i = rng.randint(0, nn - 2)
                        j = rng.randint(i + 1, nn - 1)
                else:
                    i = rng.randint(0, nn - 2)
                    j = rng.randint(i + 1, nn - 1)
                return ('reversal', i, j)
            else:
                i = rng.randint(1, nn - 2)
                angle = rng.choice([1, 2, 3])
                return ('pivot', i, angle)

        while time.time() < deadline:
            if _remaining_time() < 2.0:
                break

            move = _generate_move()
            if move[0] == 'reversal':
                new_chain = _try_reversal(chain, move[1], move[2])
            else:
                new_chain = _try_pivot(chain, move[1], move[2])

            if new_chain is None:
                continue

            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy

            if delta <= 0 or (T > min_t and rng.random() < math.exp(-delta / T)):
                chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

            if no_improve > max_no_improve:
                break

            T *= cooling
            if T < min_t:
                break

        return best_chain, best_energy

    def _max_independent_set_contacts(chain, sequence, time_limit):
        """
        Find maximum independent set of H-H contacts using branch and bound.
        Returns the set of contact pairs that can be simultaneously realized.
        """
        nn = len(chain)
        where = {}
        for i, p in enumerate(chain):
            where[p] = i

        # Find all potential H-H contacts
        contacts = []
        for i, p in enumerate(chain):
            if sequence[i] != 'H':
                continue
            x, y = p
            for dx, dy in _DELTAS:
                q = (x + dx, y + dy)
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    if i < j:  # Avoid duplicates
                        contacts.append((i, j))

        if not contacts:
            return set()

        # Build conflict graph: two contacts conflict if they share a residue
        # or if their lattice positions are incompatible
        n_contacts = len(contacts)
        conflict = [set() for _ in range(n_contacts)]

        for a in range(n_contacts):
            ia, ja = contacts[a]
            for b in range(a + 1, n_contacts):
                ib, jb = contacts[b]
                # Conflicts if they share a residue
                if ia == ib or ia == jb or ja == ib or ja == jb:
                    conflict[a].add(b)
                    conflict[b].add(a)
                else:
                    # Check if the contact positions are geometrically compatible
                    # Contact a: (chain[ia], chain[ja]) are adjacent
                    # Contact b: (chain[ib], chain[jb]) are adjacent
                    # They conflict if any of the four positions coincide in a way
                    # that would require the chain to visit the same position twice
                    pa = chain[ia]
                    qa = chain[ja]
                    pb = chain[ib]
                    qb = chain[jb]
                    # If the positions are the same, they conflict
                    if pa == pb or pa == qb or qa == pb or qa == qb:
                        conflict[a].add(b)
                        conflict[b].add(a)

        # Branch and bound for maximum independent set
        best_set = set()
        best_size = 0

        def _branch_and_bound(remaining, current_set, depth):
            nonlocal best_set, best_size

            if time.time() > time_limit:
                return

            if not remaining:
                if len(current_set) > best_size:
                    best_set = current_set.copy()
                    best_size = len(current_set)
                return

            # Upper bound: current size + number of remaining nodes
            if len(current_set) + len(remaining) <= best_size:
                return

            # Pick the node with highest degree in remaining
            node = max(remaining, key=lambda x: len(conflict[x] & remaining))
            remaining_list = list(remaining)
            remaining_list.remove(node)

            # Branch 1: Include node
            new_remaining = set(remaining_list)
            for neighbor in conflict[node]:
                if neighbor in new_remaining:
                    new_remaining.discard(neighbor)
            current_set.add(node)
            _branch_and_bound(new_remaining, current_set, depth + 1)
            current_set.discard(node)

            # Branch 2: Exclude node
            _branch_and_bound(set(remaining_list), current_set, depth + 1)

        _branch_and_bound(set(range(n_contacts)), set(), 0)

        # Convert back to contact pairs
        result = set()
        for idx in best_set:
            result.add(contacts[idx])
        return result

    def _refine_with_contacts(chain, sequence, time_limit, rng):
        """
        Refine the chain by trying to realize the maximum independent set of contacts.
        """
        nn = len(chain)
        if nn < 4:
            return chain, _energy(sequence, chain)

        # First, find the maximum independent set of contacts
        contact_deadline = time.time() + min(2.0, time_limit * 0.2)
        target_contacts = _max_independent_set_contacts(chain, sequence, contact_deadline)

        if not target_contacts:
            return chain, _energy(sequence, chain)

        # Now try to rearrange the chain to realize these contacts
        # Use local search with a bias towards moves that increase contact count
        current_chain = chain[:]
        current_energy = _energy(sequence, chain)
        best_chain = chain[:]
        best_energy = current_energy

        deadline = time.time() + time_limit
        no_improve = 0
        max_no_improve = 3000

        h_indices = [i for i in range(nn) if sequence[i] == 'H']

        def _try_reversal(state_chain, i, j):
            nn_local = len(state_chain)
            if i >= j:
                return None
            if i > 0:
                px, py = state_chain[i - 1]
                ex, ey = state_chain[j]
                if abs(px - ex) + abs(py - ey) != 1:
                    return None
            if j < nn_local - 1:
                sx, sy = state_chain[i]
                nx, ny = state_chain[j + 1]
                if abs(sx - nx) + abs(sy - ny) != 1:
                    return None
            segment = state_chain[i:j+1]
            for pos in segment:
                if pos in state_chain[:i] or pos in state_chain[j+1:]:
                    return None
            new_chain = state_chain[:i] + segment[::-1] + state_chain[j+1:]
            return new_chain

        def _try_pivot(state_chain, i, angle):
            nn_local = len(state_chain)
            if i <= 0 or i >= nn_local - 1:
                return None
            px, py = state_chain[i]
            tail = state_chain[i+1:]
            rotated = []
            for tx, ty in tail:
                rx, ry = tx - px, ty - py
                for _ in range(angle % 4):
                    rx, ry = -ry, rx
                rotated.append((px + rx, py + ry))
            for pos in rotated:
                if pos in state_chain[:i+1]:
                    return None
            new_chain = state_chain[:i+1] + rotated
            return new_chain

        def _count_target_contacts(chain_state):
            """Count how many target contacts are realized in this chain."""
            where = {}
            for i, p in enumerate(chain_state):
                where[p] = i
            count = 0
            for i, j in target_contacts:
                pi = chain_state[i]
                pj = chain_state[j]
                if abs(pi[0] - pj[0]) + abs(pi[1] - pj[1]) == 1:
                    count += 1
            return count

        current_target_count = _count_target_contacts(current_chain)
        best_target_count = current_target_count

        while time.time() < deadline:
            if _remaining_time() < 2.0:
                break

            move_type = rng.random()
            if move_type < 0.55:
                if h_indices and rng.random() < 0.5:
                    hi = rng.choice(h_indices)
                    span = rng.randint(2, min(15, nn - 1))
                    i = max(0, hi - span // 2)
                    j = min(nn - 1, i + span)
                    if j <= i:
                        i = rng.randint(0, nn - 2)
                        j = rng.randint(i + 1, nn - 1)
                else:
                    i = rng.randint(0, nn - 2)
                    j = rng.randint(i + 1, nn - 1)
                new_chain = _try_reversal(current_chain, i, j)
            else:
                i = rng.randint(1, nn - 2)
                angle = rng.choice([1, 2, 3])
                new_chain = _try_pivot(current_chain, i, angle)

            if new_chain is None:
                continue

            new_energy = _energy(sequence, new_chain)
            new_target_count = _count_target_contacts(new_chain)

            # Accept if it improves energy or target contact count
            if new_energy < current_energy or (new_energy == current_energy and new_target_count > current_target_count):
                current_chain = new_chain
                current_energy = new_energy
                current_target_count = new_target_count

                if current_energy < best_energy or (current_energy == best_energy and current_target_count > best_target_count):
                    best_energy = current_energy
                    best_chain = current_chain[:]
                    best_target_count = current_target_count
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

            if no_improve > max_no_improve:
                break

        return best_chain, best_energy

    total_time = _remaining_time() - 3.0
    time_per = total_time / len(candidates) if candidates else 0

    for chain in candidates:
        if _remaining_time() < 2.0:
            break
        alloc = min(time_per, _remaining_time() - 1.0)
        if alloc < 0.3:
            continue
        refined_chain, refined_energy = _refine_with_contacts(chain, sequence, alloc, rng)
        if best_chain is None or refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain

    if best_chain is None:
        best_chain = candidates[0] if candidates else [(i, 0) for i in range(n)]

    return best_chain


def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    """
    rng = random.Random(42)
    n = len(sequence)

    if n < 2:
        return [[0, 0]] * n

    time_limit = max(0.5, _remaining_time() - 5.0)

    best_chain = _backtrack_fold(sequence, time_limit, rng)

    if best_chain is None or len(best_chain) != n:
        best_chain = [(i, 0) for i in range(n)]

    seen = set()
    valid = True
    for i, (x, y) in enumerate(best_chain):
        if (x, y) in seen:
            valid = False
            break
        seen.add((x, y))
        if i > 0:
            px, py = best_chain[i-1]
            if abs(px - x) + abs(py - y) != 1:
                valid = False
                break

    if not valid:
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
