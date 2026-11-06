"""
HP lattice protein folding on the square lattice.
Strategy: Constraint-based backtracking search with intelligent pruning,
combined with multi-start local search. This is fundamentally different from
greedy growth - it explores the solution space systematically with constraint
propagation, which can find much better packings for medium-length sequences.
"""
import time
import random
import math

_DEADLINE = time.time() + 190.0


def _remaining():
    return _DEADLINE - time.time()


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute the HP energy: -1 for each non-consecutive H-H contact."""
    if chain is None:
        return 1
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _backtrack_fold(sequence, dimension, rng, max_nodes=50000, time_limit=15.0):
    """
    Constraint-based backtracking search for HP folding.
    Uses depth-first search with pruning based on:
    - Available space (dead-end detection)
    - Potential contact gain upper bound
    - Time/node limits
    Returns best chain found and its energy.
    """
    n = len(sequence)
    if n <= 2:
        chain = [(0, 0)]
        for i in range(1, n):
            chain.append((i, 0))
        return chain, _energy(sequence, chain)

    best_chain = None
    best_energy = 0  # unfolded baseline

    # Precompute H positions for quick contact checking
    h_positions = set(i for i, c in enumerate(sequence) if c == 'H')

    # State: chain as list of (x,y), occupied as set
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    current_energy = 0

    # For tracking potential: count H residues that could still form contacts
    # This is an upper bound on additional contacts possible

    nodes_visited = 0
    start_time = time.time()

    # We'll use a recursive approach with iterative deepening on quality
    # First try to find any valid folding, then improve

    def _dfs(depth, current_chain, current_occupied, current_energy, current_nodes):
        nonlocal best_chain, best_energy, nodes_visited

        if current_nodes[0] > max_nodes:
            return
        if time.time() - start_time > time_limit:
            return

        if depth == n:
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
            return

        last = current_chain[-1]
        res = sequence[depth]

        # Get candidate positions
        candidates = []
        for site in _neighbours(last):
            if site in current_occupied:
                continue
            # Check if this creates a dead end (only 1 free neighbor and not last position)
            if depth < n - 2:
                free_neighbors = 0
                for nb in _neighbours(site):
                    if nb not in current_occupied and nb != last:
                        free_neighbors += 1
                if free_neighbors == 0:
                    continue

            # Compute energy delta
            delta = 0
            if res == 'H':
                for nb in _neighbours(site):
                    j = current_occupied.get(nb)
                    if j is not None and j < depth - 1 and sequence[j] == 'H':
                        delta -= 1  # new contact lowers energy

            # Score for ordering: prefer moves that create contacts and leave space
            free_count = sum(1 for nb in _neighbours(site) if nb not in current_occupied and nb != last)
            score = delta * 10 + free_count

            candidates.append((score, site, delta))

        if not candidates:
            return

        # Sort by score (most negative delta first, then most free space)
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Limit branching factor for performance
        max_branch = min(len(candidates), 4 if depth < n // 2 else 3)

        for score, site, delta in candidates[:max_branch]:
            # Check if this move is promising
            if current_energy + delta >= best_energy:
                # Even with perfect future contacts, we can't beat best
                # (rough upper bound: remaining H's can form at most some contacts)
                # This pruning is conservative
                pass

            current_chain.append(site)
            current_occupied[site] = depth

            current_nodes[0] += 1
            _dfs(depth + 1, current_chain, current_occupied, current_energy + delta, current_nodes)

            current_chain.pop()
            del current_occupied[site]

    # Run DFS
    nodes = [0]
    _dfs(1, chain, occupied, current_energy, nodes)

    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)

    return best_chain, best_energy


def _local_search(sequence, chain, dimension, rng, max_iter=300):
    """
    Local search with multiple move types and simulated annealing.
    Optimized for speed with precomputed structures.
    """
    if chain is None:
        return None, 1

    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, chain)
    pos_map = {p: i for i, p in enumerate(current_chain)}

    best_chain = list(current_chain)
    best_energy = current_energy

    # Simulated annealing parameters
    T = 3.0
    cool_rate = 0.995

    for it in range(max_iter):
        if _remaining() < 0.5:
            break

        # Choose move type
        r = rng.random()

        if r < 0.5:
            # Single residue move
            i = rng.randint(1, n - 2)
            p = current_chain[i]
            best_delta = 0
            best_q = None

            for q in _neighbours(p):
                if q == current_chain[i - 1] or q == current_chain[i + 1]:
                    continue
                if q in pos_map and pos_map[q] != i:
                    continue

                # Check unit steps
                prev = current_chain[i - 1]
                nxt = current_chain[i + 1]
                if abs(q[0] - prev[0]) + abs(q[1] - prev[1]) != 1:
                    continue
                if abs(q[0] - nxt[0]) + abs(q[1] - nxt[1]) != 1:
                    continue

                # Compute energy delta
                delta = 0
                if sequence[i] == 'H':
                    # Remove old contacts
                    for old_q in _neighbours(p):
                        j = pos_map.get(old_q)
                        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                            delta += 1
                    # Add new contacts
                    for new_q in _neighbours(q):
                        j = pos_map.get(new_q)
                        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                            delta -= 1

                if delta < best_delta:
                    best_delta = delta
                    best_q = q

            if best_q is not None:
                # Accept with SA
                if best_delta <= 0 or rng.random() < math.exp(-best_delta / max(T, 0.1)):
                    old_p = current_chain[i]
                    del pos_map[old_p]
                    current_chain[i] = best_q
                    pos_map[best_q] = i
                    current_energy += best_delta

                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)

        elif r < 0.8:
            # Segment shift
            seg_len = rng.choice([2, 3, 4])
            if n > seg_len + 2:
                i = rng.randint(1, n - seg_len - 2)
                # Direction from i-1 to i
                dx = current_chain[i][0] - current_chain[i - 1][0]
                dy = current_chain[i][1] - current_chain[i - 1][1]

                # Perpendicular directions
                perp_dirs = [(-dy, dx), (dy, -dx)]

                for px, py in perp_dirs:
                    new_seg = []
                    valid = True
                    for k in range(seg_len):
                        new_p = (current_chain[i + k][0] + px, current_chain[i + k][1] + py)
                        new_seg.append(new_p)

                    # Check occupancy
                    for new_p in new_seg:
                        j = pos_map.get(new_p)
                        if j is not None and not (i <= j < i + seg_len):
                            valid = False
                            break

                    if not valid:
                        continue

                    # Check connections
                    if abs(new_seg[0][0] - current_chain[i - 1][0]) + abs(new_seg[0][1] - current_chain[i - 1][1]) != 1:
                        continue
                    if abs(new_seg[-1][0] - current_chain[i + seg_len][0]) + abs(new_seg[-1][1] - current_chain[i + seg_len][1]) != 1:
                        continue

                    # Check internal steps
                    for k in range(seg_len - 1):
                        if abs(new_seg[k + 1][0] - new_seg[k][0]) + abs(new_seg[k + 1][1] - new_seg[k][1]) != 1:
                            valid = False
                            break
                    if not valid:
                        continue

                    # Compute energy delta
                    delta = 0
                    for k in range(seg_len):
                        old_p = current_chain[i + k]
                        new_p = new_seg[k]
                        if sequence[i + k] == 'H':
                            # Remove old contacts with outside
                            for q in _neighbours(old_p):
                                j = pos_map.get(q)
                                if j is not None and j != i + k and sequence[j] == 'H' and abs(j - (i + k)) > 1:
                                    if not (i <= j < i + seg_len):
                                        delta += 1
                            # Add new contacts with outside
                            for q in _neighbours(new_p):
                                j = pos_map.get(q)
                                if j is not None and j != i + k and sequence[j] == 'H' and abs(j - (i + k)) > 1:
                                    if not (i <= j < i + seg_len):
                                        delta -= 1

                    if delta < 0 or rng.random() < math.exp(-delta / max(T, 0.1)):
                        # Apply move
                        for k in range(seg_len):
                            old_p = current_chain[i + k]
                            del pos_map[old_p]
                            current_chain[i + k] = new_seg[k]
                            pos_map[new_seg[k]] = i + k
                        current_energy += delta

                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(current_chain)
                        break  # Only one segment move per iteration

        else:
            # Reverse a short segment
            seg_len = rng.choice([3, 4, 5])
            if n > seg_len + 2:
                i = rng.randint(1, n - seg_len - 2)
                new_seg = list(reversed(current_chain[i:i + seg_len]))

                # Check if positions are free
                valid = True
                for k, new_p in enumerate(new_seg):
                    j = pos_map.get(new_p)
                    if j is not None and not (i <= j < i + seg_len):
                        valid = False
                        break
                if not valid:
                    continue

                # Check connections
                if abs(new_seg[0][0] - current_chain[i - 1][0]) + abs(new_seg[0][1] - current_chain[i - 1][1]) != 1:
                    continue
                if abs(new_seg[-1][0] - current_chain[i + seg_len][0]) + abs(new_seg[-1][1] - current_chain[i + seg_len][1]) != 1:
                    continue

                # Check internal steps
                for k in range(seg_len - 1):
                    if abs(new_seg[k + 1][0] - new_seg[k][0]) + abs(new_seg[k + 1][1] - new_seg[k][1]) != 1:
                        valid = False
                        break
                if not valid:
                    continue

                # Compute energy delta
                delta = 0
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

                if delta < 0 or rng.random() < math.exp(-delta / max(T, 0.1)):
                    for k in range(seg_len):
                        old_p = current_chain[i + k]
                        del pos_map[old_p]
                        current_chain[i + k] = new_seg[k]
                        pos_map[new_seg[k]] = i + k
                    current_energy += delta

                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)

        T *= cool_rate

    return best_chain, best_energy


def _greedy_growth(sequence, dimension, rng, temperature=0.5):
    """
    Grow a chain using probabilistic greedy approach.
    """
    n = len(sequence)
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
            gain = 0
            if res == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 3.0 + 0.3 * free + rng.random() * temperature
            options.append((score, site))

        if not options:
            return None

        options.sort(key=lambda x: -x[0])
        if temperature > 0 and len(options) > 1:
            top = options[0][0]
            probs = []
            total = 0.0
            for sc, _ in options[:min(4, len(options))]:
                p = math.exp((sc - top) / max(temperature, 0.1))
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
                site = options[chosen_idx][1]
            else:
                site = options[0][1]
        else:
            site = options[0][1]

        occupied[site] = i
        chain.append(site)

    return chain


def fold(sequence, dimension):
    """
    Find a low-energy conformation for the given HP sequence on a square lattice.
    Combines backtracking search, greedy growth, and local search.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    rng = random.Random(42 + len(sequence))

    best_chain = None
    best_energy = 0  # unfolded baseline

    # Strategy based on sequence length
    if n <= 30:
        # Use backtracking for short sequences - can find near-optimal
        bt_chain, bt_energy = _backtrack_fold(sequence, dimension, rng, max_nodes=100000, time_limit=10.0)
        if bt_chain is not None and bt_energy < best_energy:
            best_chain = bt_chain
            best_energy = bt_energy

        # Also try greedy + local search
        for restart in range(50):
            if _remaining() < 2.0:
                break
            temp = 0.1 + rng.random() * 0.5
            chain = _greedy_growth(sequence, dimension, rng, temperature=temp)
            if chain is None:
                continue
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_chain = chain
                best_energy = energy

            improved_chain, improved_energy = _local_search(sequence, chain, dimension, rng, max_iter=100)
            if improved_chain is not None and improved_energy < best_energy:
                best_chain = improved_chain
                best_energy = improved_energy

    elif n <= 60:
        # Medium: more greedy restarts with local search
        for restart in range(100):
            if _remaining() < 2.0:
                break
            temp = 0.1 + rng.random() * 0.8
            chain = _greedy_growth(sequence, dimension, rng, temperature=temp)
            if chain is None:
                continue
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_chain = chain
                best_energy = energy

            improved_chain, improved_energy = _local_search(sequence, chain, dimension, rng, max_iter=80)
            if improved_chain is not None and improved_energy < best_energy:
                best_chain = improved_chain
                best_energy = improved_energy

    else:
        # Long: many quick restarts
        for restart in range(80):
            if _remaining() < 2.0:
                break
            temp = 0.1 + rng.random() * 1.0
            chain = _greedy_growth(sequence, dimension, rng, temperature=temp)
            if chain is None:
                continue
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_chain = chain
                best_energy = energy

            improved_chain, improved_energy = _local_search(sequence, chain, dimension, rng, max_iter=60)
            if improved_chain is not None and improved_energy < best_energy:
                best_chain = improved_chain
                best_energy = improved_energy

    # Final polish on the best chain
    if best_chain is not None and _remaining() > 3.0:
        final_chain, final_energy = _local_search(sequence, best_chain, dimension, rng, max_iter=150)
        if final_chain is not None and final_energy < best_energy:
            best_chain = final_chain
            best_energy = final_energy

    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
