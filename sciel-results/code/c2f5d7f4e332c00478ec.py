"""HP lattice protein folding: greedy growth + pivot-move simulated annealing."""
# EVOLVE-BLOCK-START
import time
import random
import math

# Global deadline management
_DEADLINE = None

def _remaining_time():
    if _DEADLINE is None:
        return 240.0
    return max(0.0, _DEADLINE - time.time())


def _compute_energy(sequence, chain):
    """Compute HP energy: -# of H-H non-adjacent contacts (each pair counted once)."""
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y, z = p
            for dx, dy, dz in ((1,0,0),(0,1,0),(0,0,1)):
                q = (x+dx, y+dy, z+dz)
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng):
    """Randomized greedy chain growth with bias toward compactness."""
    start = (0, 0, 0)
    chain = [start]
    occupied = {start}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        x, y, z = last
        options = []
        for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            site = (x+dx, y+dy, z+dz)
            if site in occupied:
                continue
            # Score: prefer sites near existing H residues and closer to origin
            gain = 0.0
            if sequence[i] == 'H':
                sx, sy, sz = site
                for ox, oy, oz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                    if (sx+ox, sy+oy, sz+oz) in occupied:
                        gain += 2.0
            # Mild bias toward compactness
            dist = abs(site[0]) + abs(site[1]) + abs(site[2])
            gain -= 0.05 * dist
            options.append((gain + 0.1 * rng.random(), site))
        
        if not options:
            return None
        
        # Pick best with some randomness
        options.sort(key=lambda x: -x[0])
        top_k = min(3, len(options))
        weights = [1.0 / (k+1) for k in range(top_k)]
        r = rng.random()
        cum = 0
        chosen = top_k - 1
        for k in range(top_k):
            cum += weights[k]
            if r < cum:
                chosen = k
                break
        site = options[chosen][1]
        occupied.add(site)
        chain.append(site)
    
    return chain


def _pivot_sa(sequence, chain, time_budget, rng):
    """
    Simulated annealing with pivot moves and incremental energy tracking.
    """
    n = len(chain)
    if n < 3:
        return chain

    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    current_chain = list(chain)
    current_energy = best_energy

    T0 = 1.0
    T_end = 0.005
    start_time = time.time()
    valid_pivots = list(range(1, n - 1))
    if not valid_pivots:
        return best_chain

    # Precompute H positions for faster energy delta
    is_h = [c == 'H' for c in sequence]
    
    iteration = 0
    max_iter = 200000

    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        if iteration >= max_iter:
            break
        if _remaining_time() < 5.0:
            break

        iteration += 1
        progress = min(1.0, iteration / max_iter)
        T = T0 * (T_end / T0) ** progress

        pivot_idx = rng.choice(valid_pivots)
        direction = rng.randint(0, 1)

        p_prev = current_chain[pivot_idx - 1]
        p_pivot = current_chain[pivot_idx]
        bx = p_pivot[0] - p_prev[0]
        by = p_pivot[1] - p_prev[1]
        bz = p_pivot[2] - p_prev[2]

        # Determine rotation function inline for speed
        if bx != 0:
            if direction == 0:
                # +90 in yz: (dy,dz) -> (-dz,dy)
                px, py, pz = p_prev
                new_suffix = []
                valid = True
                prefix_set = set(current_chain[:pivot_idx])
                seen = set()
                for i in range(pivot_idx, n):
                    ox, oy, oz = current_chain[i]
                    dx = ox - px
                    dy = oy - py
                    dz = oz - pz
                    new_pt = (px + dx, py - dz, pz + dy)
                    if new_pt in prefix_set or new_pt in seen:
                        valid = False
                        break
                    seen.add(new_pt)
                    new_suffix.append(new_pt)
            else:
                # -90 in yz: (dy,dz) -> (dz,-dy)
                px, py, pz = p_prev
                new_suffix = []
                valid = True
                prefix_set = set(current_chain[:pivot_idx])
                seen = set()
                for i in range(pivot_idx, n):
                    ox, oy, oz = current_chain[i]
                    dx = ox - px
                    dy = oy - py
                    dz = oz - pz
                    new_pt = (px + dx, py + dz, pz - dy)
                    if new_pt in prefix_set or new_pt in seen:
                        valid = False
                        break
                    seen.add(new_pt)
                    new_suffix.append(new_pt)
        elif by != 0:
            if direction == 0:
                # +90 in xz: (dx,dz) -> (dz,-dx)
                px, py, pz = p_prev
                new_suffix = []
                valid = True
                prefix_set = set(current_chain[:pivot_idx])
                seen = set()
                for i in range(pivot_idx, n):
                    ox, oy, oz = current_chain[i]
                    dx = ox - px
                    dy = oy - py
                    dz = oz - pz
                    new_pt = (px + dz, py + dy, pz - dx)
                    if new_pt in prefix_set or new_pt in seen:
                        valid = False
                        break
                    seen.add(new_pt)
                    new_suffix.append(new_pt)
            else:
                # -90 in xz: (dx,dz) -> (-dz,dx)
                px, py, pz = p_prev
                new_suffix = []
                valid = True
                prefix_set = set(current_chain[:pivot_idx])
                seen = set()
                for i in range(pivot_idx, n):
                    ox, oy, oz = current_chain[i]
                    dx = ox - px
                    dy = oy - py
                    dz = oz - pz
                    new_pt = (px - dz, py + dy, pz + dx)
                    if new_pt in prefix_set or new_pt in seen:
                        valid = False
                        break
                    seen.add(new_pt)
                    new_suffix.append(new_pt)
        else:
            if direction == 0:
                # +90 in xy: (dx,dy) -> (-dy,dx)
                px, py, pz = p_prev
                new_suffix = []
                valid = True
                prefix_set = set(current_chain[:pivot_idx])
                seen = set()
                for i in range(pivot_idx, n):
                    ox, oy, oz = current_chain[i]
                    dx = ox - px
                    dy = oy - py
                    dz = oz - pz
                    new_pt = (px - dy, py + dx, pz + dz)
                    if new_pt in prefix_set or new_pt in seen:
                        valid = False
                        break
                    seen.add(new_pt)
                    new_suffix.append(new_pt)
            else:
                # -90 in xy: (dx,dy) -> (dy,-dx)
                px, py, pz = p_prev
                new_suffix = []
                valid = True
                prefix_set = set(current_chain[:pivot_idx])
                seen = set()
                for i in range(pivot_idx, n):
                    ox, oy, oz = current_chain[i]
                    dx = ox - px
                    dy = oy - py
                    dz = oz - pz
                    new_pt = (px + dy, py - dx, pz + dz)
                    if new_pt in prefix_set or new_pt in seen:
                        valid = False
                        break
                    seen.add(new_pt)
                    new_suffix.append(new_pt)

        if not valid:
            continue

        # Compute energy delta
        new_chain = current_chain[:pivot_idx] + new_suffix
        new_energy = _compute_energy(sequence, new_chain)
        delta_e = new_energy - current_energy

        if delta_e < 0 or (T > 0.001 and rng.random() < math.exp(-delta_e / T)):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)

    return best_chain


def fold(sequence, dimension):
    """Fold an HP sequence on the cubic lattice using greedy growth + pivot SA."""
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 240.0

    remaining = _remaining_time()
    if remaining < 2.0:
        return [[i, 0, 0] for i in range(len(sequence))]

    n = len(sequence)
    if n <= 1:
        return [[0, 0, 0]]
    if n == 2:
        return [[0, 0, 0], [1, 0, 0]]

    # Time allocation based on chain length
    # Longer chains get more time
    if n <= 48:
        per_seq_time = min(8.0, remaining / 26.0 * 0.8)
    elif n <= 80:
        per_seq_time = min(15.0, remaining / 26.0 * 1.2)
    else:
        per_seq_time = min(25.0, remaining / 26.0 * 1.5)
    per_seq_time = max(per_seq_time, 1.0)

    rng = random.Random(42 + n)

    # Phase 1: Greedy growth with multiple restarts
    n_greedy = max(20, int(100 * per_seq_time))
    n_greedy = min(n_greedy, 300)

    best_chain = None
    best_energy = 0

    for attempt in range(n_greedy):
        if _remaining_time() < 5.0:
            break
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            continue
        energy = _compute_energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy

    if best_chain is None:
        return [[i, 0, 0] for i in range(n)]

    # Phase 2: Local search with pivot moves
    remaining_for_ls = _remaining_time() - 5.0
    if remaining_for_ls > 1.0:
        # Run SA from the best chain
        sa_time = max(0.5, remaining_for_ls * 0.9)
        improved = _pivot_sa(sequence, best_chain, sa_time, rng)
        imp_energy = _compute_energy(sequence, improved)
        if imp_energy < best_energy:
            best_chain = improved
            best_energy = imp_energy

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
