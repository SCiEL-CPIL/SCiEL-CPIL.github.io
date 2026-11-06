"""HP lattice protein folding - improved with segment moves and parallel-like restarts."""
# EVOLVE-BLOCK-START
import time
import random
import math

# Global deadline: evaluator kills after 210s wall-clock. Use safety margin.
_DEADLINE = time.time() + 195.0


def _remaining():
    return _DEADLINE - time.time()


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    return [(point[0] + dx, point[1] + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]


def _energy(sequence, chain):
    """Compute HP energy: -1 for each non-consecutive H-H contact."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _contact_count(sequence, chain):
    """Count H-H contacts (non-negative)."""
    return -_energy(sequence, chain)


def _greedy_growth(sequence, dimension, rng, temperature=0.3, strategy='balanced'):
    """
    Grow a chain on the lattice using probabilistic greedy approach.
    strategy: 'contacts' (favor H-H), 'free' (favor open space), 'balanced', 'random'
    """
    n = len(sequence)
    if n == 0:
        return None
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}

    for i in range(1, n):
        res = sequence[i]
        last = chain[-1]
        options = []

        for site in _neighbours(last):
            if site in occupied:
                continue

            # Count new H-H contacts this site would form
            gain = 0
            if res == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1

            # Count free neighbours
            free = sum(1 for q in _neighbours(site) if q not in occupied)

            if strategy == 'contacts':
                score = gain * 5.0 + 0.1 * free + rng.random() * temperature
            elif strategy == 'free':
                score = gain * 1.0 + 2.0 * free + rng.random() * temperature
            elif strategy == 'random':
                score = rng.random() * 10.0
            else:  # balanced
                score = gain * 3.0 + 0.5 * free + rng.random() * temperature

            options.append((score, gain, free, site))

        if not options:
            return None

        options.sort(key=lambda x: -x[0])

        if temperature > 0.01 and len(options) > 1:
            top_score = options[0][0]
            probs = []
            total = 0.0
            for sc, _, _, _ in options[:min(4, len(options))]:
                p = math.exp((sc - top_score) / max(temperature, 0.1))
                probs.append(p)
                total += p
            if total > 0:
                probs = [p / total for p in probs]
                r = rng.random()
                cum = 0.0
                chosen_idx = len(probs) - 1
                for idx, p in enumerate(probs):
                    cum += p
                    if r <= cum:
                        chosen_idx = idx
                        break
                site = options[chosen_idx][3]
            else:
                site = options[0][3]
        else:
            site = options[0][3]

        occupied[site] = i
        chain.append(site)

    return chain


def _segment_rotation(sequence, chain, dimension, rng, max_iter=50):
    """
    Improve via segment rotation moves (move a contiguous block by 1 step perpendicular).
    Returns improved chain and energy.
    """
    if chain is None:
        return None, 1

    current_energy = _energy(sequence, chain)
    current_chain = list(chain)
    n = len(current_chain)
    pos_map = {p: i for i, p in enumerate(current_chain)}

    best_energy = current_energy
    best_chain = list(current_chain)

    for it in range(max_iter):
        if _remaining() < 0.3:
            break

        # Try several random segment moves
        found_improvement = False

        for attempt in range(5):
            seg_len = rng.choice([2, 3, 4, 5])
            if n <= seg_len + 2:
                continue

            i = rng.randint(1, n - seg_len - 2)

            # Direction from i-1 to i
            d1 = (current_chain[i][0] - current_chain[i - 1][0],
                  current_chain[i][1] - current_chain[i - 1][1])

            # Perpendicular directions
            perps = [(-d1[1], d1[0]), (d1[1], -d1[0])]

            for px, py in perps:
                # Try shifting segment [i, i+seg_len-1] by (px, py)
                new_seg = []
                valid = True

                for k in range(seg_len):
                    new_p = (current_chain[i + k][0] + px, current_chain[i + k][1] + py)
                    new_seg.append(new_p)

                # Check occupancy (excluding the segment itself)
                for new_p in new_seg:
                    j = pos_map.get(new_p)
                    if j is not None and not (i <= j < i + seg_len):
                        valid = False
                        break

                if not valid:
                    continue

                # Check unit step connections
                # i-1 to new_seg[0]
                if abs(new_seg[0][0] - current_chain[i - 1][0]) + abs(new_seg[0][1] - current_chain[i - 1][1]) != 1:
                    continue
                # new_seg[-1] to i+seg_len
                if abs(new_seg[-1][0] - current_chain[i + seg_len][0]) + abs(new_seg[-1][1] - current_chain[i + seg_len][1]) != 1:
                    continue
                # Internal steps
                for k in range(seg_len - 1):
                    if abs(new_seg[k + 1][0] - new_seg[k][0]) + abs(new_seg[k + 1][1] - new_seg[k][1]) != 1:
                        valid = False
                        break
                if not valid:
                    continue

                # Compute energy delta
                delta = 0
                # Contacts involving old segment positions (with outside residues)
                for k in range(seg_len):
                    old_p = current_chain[i + k]
                    new_p = new_seg[k]
                    if sequence[i + k] == 'H':
                        for q in _neighbours(old_p):
                            j = pos_map.get(q)
                            if j is not None and j != i + k and sequence[j] == 'H' and abs(j - (i + k)) > 1:
                                if not (i <= j < i + seg_len):
                                    delta += 1
                        for q in _neighbours(new_p):
                            j = pos_map.get(q)
                            if j is not None and j != i + k and sequence[j] == 'H' and abs(j - (i + k)) > 1:
                                if not (i <= j < i + seg_len):
                                    delta -= 1

                # Also check contacts between moved segment and fixed parts
                # Already handled above

                if delta < 0:
                    # Apply the move
                    for k in range(seg_len):
                        old_p = current_chain[i + k]
                        del pos_map[old_p]
                        current_chain[i + k] = new_seg[k]
                        pos_map[new_seg[k]] = i + k

                    current_energy += delta

                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)

                    found_improvement = True
                    break

            if found_improvement:
                break

        if not found_improvement:
            # No improving segment move; try a single residue move as perturbation
            i = rng.randint(1, n - 2)
            p = current_chain[i]
            for q in _neighbours(p):
                if q == current_chain[i - 1] or q == current_chain[i + 1]:
                    continue
                if q in pos_map and pos_map[q] != i:
                    continue
                prev = current_chain[i - 1]
                nxt = current_chain[i + 1]
                if abs(q[0] - prev[0]) + abs(q[1] - prev[1]) != 1:
                    continue
                if abs(q[0] - nxt[0]) + abs(q[1] - nxt[1]) != 1:
                    continue

                delta = 0
                if sequence[i] == 'H':
                    for old_q in _neighbours(p):
                        j = pos_map.get(old_q)
                        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                            delta += 1
                    for new_q in _neighbours(q):
                        j = pos_map.get(new_q)
                        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                            delta -= 1

                if delta <= 0:
                    old_p = current_chain[i]
                    del pos_map[old_p]
                    current_chain[i] = q
                    pos_map[q] = i
                    current_energy += delta
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    break

    return best_chain, best_energy


def _hill_climb(sequence, chain, dimension, rng, max_iter=100):
    """
    Fast hill climbing with single-residue moves only.
    """
    if chain is None:
        return None, 1

    current_energy = _energy(sequence, chain)
    current_chain = list(chain)
    n = len(current_chain)
    pos_map = {p: i for i, p in enumerate(current_chain)}

    best_energy = current_energy
    best_chain = list(current_chain)

    for it in range(max_iter):
        if _remaining() < 0.2:
            break

        improved = False
        # Try all single-residue moves once
        for i in range(1, n - 1):
            p = current_chain[i]
            for q in _neighbours(p):
                if q == current_chain[i - 1] or q == current_chain[i + 1]:
                    continue
                if q in pos_map and pos_map[q] != i:
                    continue
                prev = current_chain[i - 1]
                nxt = current_chain[i + 1]
                if abs(q[0] - prev[0]) + abs(q[1] - prev[1]) != 1:
                    continue
                if abs(q[0] - nxt[0]) + abs(q[1] - nxt[1]) != 1:
                    continue

                delta = 0
                if sequence[i] == 'H':
                    for old_q in _neighbours(p):
                        j = pos_map.get(old_q)
                        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                            delta += 1
                    for new_q in _neighbours(q):
                        j = pos_map.get(new_q)
                        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                            delta -= 1

                if delta < 0:
                    old_p = current_chain[i]
                    del pos_map[old_p]
                    current_chain[i] = q
                    pos_map[q] = i
                    current_energy += delta
                    improved = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    break

        if not improved:
            break

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Find a low-energy conformation for the given HP sequence on a square lattice.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]

    rng = random.Random(12345 + len(sequence) * 7)

    best_chain = None
    best_energy = 0  # will be set on first valid chain

    strategies = ['contacts', 'balanced', 'free', 'random']
    temps = [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.8]

    # Scale restarts by sequence length
    if n <= 25:
        max_restarts = 600
        hc_iter = 80
        seg_iter = 40
    elif n <= 40:
        max_restarts = 350
        hc_iter = 60
        seg_iter = 30
    elif n <= 60:
        max_restarts = 200
        hc_iter = 45
        seg_iter = 25
    else:
        max_restarts = 100
        hc_iter = 30
        seg_iter = 20

    for restart in range(max_restarts):
        if _remaining() < 1.0:
            break

        temp = temps[restart % len(temps)] + rng.random() * 0.3
        strategy = strategies[restart % len(strategies)]

        chain = _greedy_growth(sequence, dimension, rng, temperature=temp, strategy=strategy)
        if chain is None:
            continue

        energy = _energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy

        # Hill climb first (fast)
        if _remaining() < 1.0:
            break
        hc_chain, hc_energy = _hill_climb(sequence, best_chain if best_chain is chain else chain,
                                          dimension, rng, max_iter=hc_iter)
        if hc_chain is not None and hc_energy < best_energy:
            best_chain = hc_chain
            best_energy = hc_energy

        # Segment rotation (slower but can escape local minima)
        if _remaining() < 1.0:
            break
        seg_chain, seg_energy = _segment_rotation(sequence, best_chain, dimension, rng, max_iter=seg_iter)
        if seg_chain is not None and seg_energy < best_energy:
            best_chain = seg_chain
            best_energy = seg_energy

        # Final hill climb after segment moves
        if _remaining() < 1.0:
            break
        final_chain, final_energy = _hill_climb(sequence, best_chain, dimension, rng, max_iter=hc_iter // 2)
        if final_chain is not None and final_energy < best_energy:
            best_chain = final_chain
            best_energy = final_energy

    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
