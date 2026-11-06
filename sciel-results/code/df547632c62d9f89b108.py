"""HP lattice protein folding: fast delta-energy search with progressive segment relocation."""
# EVOLVE-BLOCK-START
import random
import time
import math

_T0 = time.time()
_DEADLINE = 195.0  # wall-clock limit with safety margin (210s kill, leave 15s buffer)


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _compute_energy(sequence, chain):
    """Compute energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    if n < 3:
        return 0
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = (x + dx, y + dy)
            j = pos_to_idx.get(nb)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _build_pos_to_idx(chain):
    """Build position-to-index mapping."""
    return {p: i for i, p in enumerate(chain)}


def _energy_delta_for_relocation(sequence, chain, pos_to_idx, start, end, insert_k):
    """
    Compute energy delta when cutting segment [start..end] and inserting at position k in the gap.
    Returns (delta, new_chain) or (None, None) if invalid.
    """
    n = len(chain)
    seg_len = end - start + 1
    gap_len = n - seg_len
    
    # Build gap list
    gap = chain[:start] + chain[end + 1:]
    
    # Check connectivity at insertion point
    # Left connection: gap[k-1] to segment[0]
    if k > 0:
        left_pos = gap[k - 1]
        seg_first = chain[start]
        if abs(left_pos[0] - seg_first[0]) + abs(left_pos[1] - seg_first[1]) != 1:
            return None, None
    # Right connection: segment[-1] to gap[k]
    if k < gap_len:
        right_pos = gap[k]
        seg_last = chain[end]
        if abs(right_pos[0] - seg_last[0]) + abs(right_pos[1] - seg_last[1]) != 1:
            return None, None
    
    # Build new chain
    new_chain = gap[:k] + chain[start:end + 1] + gap[k:]
    
    # Check self-avoidance
    if len(set(new_chain)) != n:
        return None, None
    
    # Compute energy delta efficiently
    # Contacts lost: any H-H contact involving positions in the old segment that are broken
    # Contacts gained: any new H-H contact formed
    # For simplicity with relocation, recompute affected contacts
    
    # Build new pos_to_idx
    new_pos_to_idx = _build_pos_to_idx(new_chain)
    
    # Compute energy change by checking neighbors of moved segment endpoints and all H in segment
    # Actually for correctness, let's compute full energy difference
    # But we can optimize: only contacts involving indices that changed position matter
    
    # Old positions of segment
    old_positions = set(chain[start:end + 1])
    
    # Find all H residues in the segment
    h_in_seg = [i for i in range(start, end + 1) if sequence[i] == 'H']
    
    # Also need to check H residues outside that were adjacent to segment
    affected_indices = set(h_in_seg)
    for i in h_in_seg:
        x, y = chain[i]
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = (x + dx, y + dy)
            j = pos_to_idx.get(nb)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                affected_indices.add(j)
    
    # Also check H residues adjacent to the insertion point in the new chain
    for i in range(n):
        if sequence[i] != 'H':
            continue
        if new_pos_to_idx.get(chain[i]) != i:  # position changed
            affected_indices.add(i)
    
    # Compute old contacts involving affected indices
    old_contacts = 0
    for i in affected_indices:
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = (x + dx, y + dy)
            j = pos_to_idx.get(nb)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H' and i < j:
                old_contacts += 1
    
    # Compute new contacts involving affected indices
    new_contacts = 0
    for i in affected_indices:
        if sequence[i] != 'H':
            continue
        x, y = new_chain[i]
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = (x + dx, y + dy)
            j = new_pos_to_idx.get(nb)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H' and i < j:
                new_contacts += 1
    
    delta = -(new_contacts - old_contacts)
    return delta, new_chain


def _kink_move(chain, rng):
    """Kink move: pick 3 consecutive residues and move the middle one to an adjacent empty position."""
    n = len(chain)
    if n < 4:
        return None
    
    occupied = set(chain)
    
    # Pick a random interior index
    i = rng.randint(1, n - 2)
    mid = chain[i]
    left = chain[i - 1]
    right = chain[i + 1]
    
    # Try all 4 directions for the middle residue
    candidates = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        new_mid = (mid[0] + dx, mid[1] + dy)
        if new_mid in occupied:
            continue
        # Must be adjacent to left and right
        if abs(new_mid[0] - left[0]) + abs(new_mid[1] - left[1]) != 1:
            continue
        if abs(new_mid[0] - right[0]) + abs(new_mid[1] - right[1]) != 1:
            continue
        candidates.append(new_mid)
    
    if not candidates:
        return None
    
    new_mid = rng.choice(candidates)
    new_chain = list(chain)
    new_chain[i] = new_mid
    
    # Verify self-avoidance (should be fine since we checked)
    if len(set(new_chain)) != n:
        return None
    
    return new_chain


def _pivot_move_fast(chain, pos_to_idx, rng):
    """Fast pivot move using pos_to_idx for O(1) lookups."""
    n = len(chain)
    if n < 5:
        return None
    
    pivot = rng.randint(1, n - 2)
    pivot_pos = chain[pivot]
    suffix = chain[pivot + 1:]
    prefix_set = set(chain[:pivot + 1])
    
    # Try 90-degree CCW rotation
    rotated = []
    valid = True
    for (x, y) in suffix:
        rx, ry = x - pivot_pos[0], y - pivot_pos[1]
        new_x, new_y = -ry, rx
        pos = (pivot_pos[0] + new_x, pivot_pos[1] + new_y)
        if pos in prefix_set:
            valid = False
            break
        rotated.append(pos)
    
    if valid:
        new_chain = chain[:pivot + 1] + rotated
        if len(set(new_chain)) == n:
            return new_chain
    
    # Try 180-degree rotation
    rotated = []
    valid = True
    for (x, y) in suffix:
        rx, ry = x - pivot_pos[0], y - pivot_pos[1]
        new_x, new_y = -rx, -ry
        pos = (pivot_pos[0] + new_x, pivot_pos[1] + new_y)
        if pos in prefix_set:
            valid = False
            break
        rotated.append(pos)
    
    if valid:
        new_chain = chain[:pivot + 1] + rotated
        if len(set(new_chain)) == n:
            return new_chain
    
    return None


def _grow_initial(sequence, rng):
    """Generate initial self-avoiding walk with greedy H-contact preference."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0)}
    pos_to_idx = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = (last[0] + dx, last[1] + dy)
            if nb not in occupied:
                options.append(nb)
        
        if not options:
            return None  # Signal failure
        
        if sequence[i] == 'H':
            # Score by potential H contacts
            scored = []
            for nb in options:
                gain = 0
                x, y = nb
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    pp = (x + dx, y + dy)
                    if pp in pos_to_idx:
                        idx = pos_to_idx[pp]
                        if sequence[idx] == 'H':
                            gain += 1
                scored.append((gain * 10 + rng.random(), nb))
            scored.sort(reverse=True)
            next_pos = scored[0][1]
        else:
            next_pos = rng.choice(options)
        
        chain.append(next_pos)
        occupied.add(next_pos)
        pos_to_idx[next_pos] = i
    
    return chain


def _search_with_delta(sequence, chain, rng, max_iters, time_limit):
    """
    Main search loop using fast energy delta computation.
    Uses progressive segment sizes: start large, end small.
    """
    n = len(chain)
    pos_to_idx = _build_pos_to_idx(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    T = 3.0
    cooling = 0.9995
    min_T = 0.05
    
    # Progressive segment sizes: phase 1 (large), phase 2 (medium), phase 3 (small)
    # Phase boundaries based on iteration progress
    it = 0
    start_time = time.time()
    
    while it < max_iters:
        if _remaining() < 3.0 or (time.time() - start_time) > time_limit:
            break
        
        progress = it / max_iters
        
        # Choose move type based on progress
        r = rng.random()
        new_chain = None
        
        if progress < 0.3:
            # Phase 1: Large segment relocations and pivots
            if r < 0.5:
                # Large segment relocation
                seg_len = rng.randint(max(3, n // 8), min(n // 3, n - 4))
                if seg_len >= 2:
                    start = rng.randint(1, n - seg_len - 1)
                    end = start + seg_len - 1
                    gap_len = n - seg_len
                    gap = chain[:start] + chain[end + 1:]
                    
                    # Try several insertion points
                    for _attempt in range(8):
                        k = rng.randint(0, gap_len)
                        if k == start:
                            continue
                        # Check connectivity
                        if k > 0:
                            lp = gap[k - 1]
                            sf = chain[start]
                            if abs(lp[0] - sf[0]) + abs(lp[1] - sf[1]) != 1:
                                continue
                        if k < gap_len:
                            rp = gap[k]
                            sl = chain[end]
                            if abs(rp[0] - sl[0]) + abs(rp[1] - sl[1]) != 1:
                                continue
                        
                        new_chain = gap[:k] + chain[start:end + 1] + gap[k:]
                        if len(set(new_chain)) == n:
                            break
                        new_chain = None
            elif r < 0.8:
                new_chain = _pivot_move_fast(chain, pos_to_idx, rng)
            else:
                # Reversal of large segment
                seg_len = rng.randint(max(3, n // 6), min(n // 3, n - 2))
                if seg_len >= 2:
                    start = rng.randint(0, n - seg_len - 1)
                    end = start + seg_len - 1
                    new_seg = list(reversed(chain[start:end + 1]))
                    if start > 0:
                        lp = chain[start - 1]
                        fs = new_seg[0]
                        if abs(lp[0] - fs[0]) + abs(lp[1] - fs[1]) != 1:
                            new_chain = None
                        else:
                            nc = chain[:start] + new_seg + chain[end + 1:]
                            if len(set(nc)) == n:
                                new_chain = nc
                    else:
                        nc = chain[:start] + new_seg + chain[end + 1:]
                        if len(set(nc)) == n:
                            new_chain = nc
        elif progress < 0.7:
            # Phase 2: Medium moves
            if r < 0.4:
                # Medium segment relocation
                seg_len = rng.randint(2, min(5, n - 4))
                if seg_len >= 2:
                    start = rng.randint(1, n - seg_len - 1)
                    end = start + seg_len - 1
                    gap_len = n - seg_len
                    gap = chain[:start] + chain[end + 1:]
                    
                    for _attempt in range(6):
                        k = rng.randint(0, gap_len)
                        if k == start:
                            continue
                        if k > 0:
                            lp = gap[k - 1]
                            sf = chain[start]
                            if abs(lp[0] - sf[0]) + abs(lp[1] - sf[1]) != 1:
                                continue
                        if k < gap_len:
                            rp = gap[k]
                            sl = chain[end]
                            if abs(rp[0] - sl[0]) + abs(rp[1] - sl[1]) != 1:
                                continue
                        
                        new_chain = gap[:k] + chain[start:end + 1] + gap[k:]
                        if len(set(new_chain)) == n:
                            break
                        new_chain = None
            elif r < 0.7:
                new_chain = _kink_move(chain, rng)
            else:
                new_chain = _pivot_move_fast(chain, pos_to_idx, rng)
        else:
            # Phase 3: Fine-grained polish
            if r < 0.5:
                new_chain = _kink_move(chain, rng)
            elif r < 0.8:
                # Small segment relocation (length 2-3)
                seg_len = rng.randint(2, 3)
                if n > seg_len + 2:
                    start = rng.randint(1, n - seg_len - 1)
                    end = start + seg_len - 1
                    gap_len = n - seg_len
                    gap = chain[:start] + chain[end + 1:]
                    
                    for _attempt in range(5):
                        k = rng.randint(0, gap_len)
                        if k == start:
                            continue
                        if k > 0:
                            lp = gap[k - 1]
                            sf = chain[start]
                            if abs(lp[0] - sf[0]) + abs(lp[1] - sf[1]) != 1:
                                continue
                        if k < gap_len:
                            rp = gap[k]
                            sl = chain[end]
                            if abs(rp[0] - sl[0]) + abs(rp[1] - sl[1]) != 1:
                                continue
                        
                        new_chain = gap[:k] + chain[start:end + 1] + gap[k:]
                        if len(set(new_chain)) == n:
                            break
                        new_chain = None
            else:
                new_chain = _pivot_move_fast(chain, pos_to_idx, rng)
        
        if new_chain is None:
            it += 1
            continue
        
        # Verify validity
        if len(set(new_chain)) != n:
            it += 1
            continue
        
        valid = True
        for i in range(n - 1):
            if abs(new_chain[i][0] - new_chain[i + 1][0]) + abs(new_chain[i][1] - new_chain[i + 1][1]) != 1:
                valid = False
                break
        if not valid:
            it += 1
            continue
        
        # Compute energy (full recompute for correctness, but infrequent enough)
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Acceptance criterion
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            pos_to_idx = _build_pos_to_idx(chain)
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(new_chain)
        
        T *= cooling
        if T < min_T:
            T = min_T
        it += 1
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold sequence into low-energy conformation."""
    if dimension != 2:
        return [[i] + [0] * (dimension - 1) for i in range(len(sequence))]
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    rng = random.Random(42 + len(sequence) * 7)
    
    best_chain = None
    best_energy = 0
    
    start_time = time.time()
    time_budget = min(_remaining() - 5.0, 180.0)
    
    # Adaptive restarts: more for shorter sequences
    if n <= 25:
        num_restarts = 40
        iters_per = 8000
    elif n <= 40:
        num_restarts = 30
        iters_per = 6000
    elif n <= 60:
        num_restarts = 20
        iters_per = 5000
    elif n <= 80:
        num_restarts = 15
        iters_per = 4000
    else:
        num_restarts = 10
        iters_per = 3000
    
    for restart in range(num_restarts):
        if _remaining() < 8.0:
            break
        if (time.time() - start_time) > time_budget:
            break
        
        # Generate initial conformation
        chain = None
        for _attempt in range(20):
            chain = _grow_initial(sequence, rng)
            if chain is not None and len(chain) == n:
                break
            chain = None
        
        if chain is None or len(chain) != n:
            continue
        
        # Verify
        if len(set(chain)) != n:
            continue
        valid = True
        for i in range(n - 1):
            if abs(chain[i][0] - chain[i + 1][0]) + abs(chain[i][1] - chain[i + 1][1]) != 1:
                valid = False
                break
        if not valid:
            continue
        
        # Run search
        time_left = min(time_budget - (time.time() - start_time), _remaining() - 5.0)
        if time_left < 1.0:
            break
        
        chain, energy = _search_with_delta(sequence, chain, rng, max_iters=iters_per, time_limit=time_left)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [[x, y] for x, y in best_chain]
# EVOLVE-BLOCK-END
