"""HP lattice protein folding on the square lattice.
Entry point: fold(sequence, dimension) -> list of lattice points.
"""
# EVOLVE-BLOCK-START
import time
import random
import math

_GLOBAL_DEADLINE = time.time() + 175.0  # Safety margin before 210s kill


def _time_left():
    return _GLOBAL_DEADLINE - time.time()


def _is_deadline_exceeded():
    return time.time() >= _GLOBAL_DEADLINE


def _neighbours_2d(x, y):
    """Yield the 4 neighbours of (x, y)."""
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _calculate_energy(sequence, chain):
    """Compute H-H contact energy. Returns negative number of contacts."""
    n = len(chain)
    where = {}
    for i, (x, y) in enumerate(chain):
        where[(x, y)] = i
    contacts = 0
    h_indices = set(i for i, r in enumerate(sequence) if r == 'H')
    for i in h_indices:
        x, y = chain[i]
        for (nx, ny) in _neighbours_2d(x, y):
            j = where.get((nx, ny))
            if j is not None and j != i - 1 and j != i + 1 and j in h_indices:
                contacts += 1
    return -contacts // 2  # Each contact counted twice


def _fast_grow(sequence, dimension, rng, bias_contacts=2.0, bias_open=0.5):
    """Grow a chain using randomized greedy with contact bias.
    Returns a list of (x,y) tuples or None if stuck."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    # Use a bounding box for occupancy
    size = n * 2 + 10
    offset = n + 5
    occupied = [[False] * size for _ in range(size)]
    chain = [(offset, offset)]
    occupied[offset][offset] = True
    
    h_set = set(i for i, r in enumerate(sequence) if r == 'H')
    
    for i in range(1, n):
        cx, cy = chain[-1]
        options = []
        for (nx, ny) in _neighbours_2d(cx, cy):
            if 0 <= nx < size and 0 <= ny < size and not occupied[ny][nx]:
                # Score this position
                score = rng.random() * 0.1
                if i in h_set:
                    # Count H neighbours (excluding previous residue)
                    for (hx, hy) in _neighbours_2d(nx, ny):
                        if 0 <= hx < size and 0 <= hy < size and occupied[hy][hx]:
                            j = None  # We need to know which residue
                            # Check if it's an H residue not adjacent in chain
                            # We'll track positions differently
                            pass
                # Count open neighbours
                open_count = 0
                for (ox, oy) in _neighbours_2d(nx, ny):
                    if 0 <= ox < size and 0 <= oy < size and not occupied[oy][ox]:
                        open_count += 1
                score += bias_open * open_count
                options.append((score, nx, ny))
        
        if not options:
            return None
        
        options.sort(key=lambda t: -t[0])
        # Pick from top options with some randomness
        top_k = min(3, len(options))
        chosen = options[rng.randint(0, top_k - 1)]
        chain.append((chosen[1], chosen[2]))
        occupied[chosen[2]][chosen[1]] = True
    
    # Re-center the chain
    min_x = min(p[0] for p in chain)
    min_y = min(p[1] for p in chain)
    return [(p[0] - min_x, p[1] - min_y) for p in chain]


def _grow_with_contacts(sequence, dimension, rng, contact_weight=1.5, open_weight=0.3):
    """Improved growth that actually counts H-H contacts."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    size = n * 2 + 10
    offset = n + 5
    occupied_pos = [[None] * size for _ in range(size)]  # Store residue index
    chain = [(offset, offset)]
    occupied_pos[offset][offset] = 0
    
    h_set = set(i for i, r in enumerate(sequence) if r == 'H')
    
    for i in range(1, n):
        cx, cy = chain[-1]
        options = []
        for (nx, ny) in _neighbours_2d(cx, cy):
            if 0 <= nx < size and 0 <= ny < size and occupied_pos[ny][nx] is None:
                score = rng.random() * 0.05
                # Count potential H-H contacts
                if i in h_set:
                    for (hx, hy) in _neighbours_2d(nx, ny):
                        if 0 <= hx < size and 0 <= hy < size:
                            j = occupied_pos[hy][hx]
                            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                                score += contact_weight
                # Count open neighbours
                open_count = 0
                for (ox, oy) in _neighbours_2d(nx, ny):
                    if 0 <= ox < size and 0 <= oy < size and occupied_pos[oy][ox] is None:
                        open_count += 1
                score += open_weight * open_count
                options.append((score, nx, ny))
        
        if not options:
            return None
        
        # Pick best with some randomness
        options.sort(key=lambda t: -t[0])
        top_k = min(4, len(options))
        # Weighted random pick from top
        if top_k == 1:
            chosen = options[0]
        else:
            # Prefer higher scores
            r = rng.random()
            if r < 0.5:
                chosen = options[0]
            elif r < 0.8:
                chosen = options[1]
            else:
                chosen = options[rng.randint(2, top_k - 1)]
        
        chain.append((chosen[1], chosen[2]))
        occupied_pos[chosen[2]][chosen[1]] = i
    
    min_x = min(p[0] for p in chain)
    min_y = min(p[1] for p in chain)
    return [(p[0] - min_x, p[1] - min_y) for p in chain]


def _spiral_grow(sequence, dimension, rng):
    """Grow a compact spiral-like conformation."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0)] * n
    
    # Generate a spiral path
    path = [(0, 0)]
    dx, dy = 1, 0
    for step in range(1, n * 2 + 5):
        x = path[-1][0] + dx
        y = path[-1][1] + dy
        if len(path) < n:
            path.append((x, y))
        # Turn every few steps
        if step % 2 == 0:
            dx, dy = -dy, dx
        if len(path) >= n:
            break
    
    if len(path) < n:
        # Extend straight
        while len(path) < n:
            path.append((path[-1][0] + dx, path[-1][1] + dy))
    
    # Shuffle the assignment of residues to positions using random walk
    # Actually, just return the spiral as is - it's compact
    return path[:n]


def _single_residue_move(chain, occupied, idx, rng):
    """Try to move residue idx to a neighbouring free site.
    Returns (new_chain, new_occupied, energy_delta) or None."""
    n = len(chain)
    x, y = chain[idx]
    for (nx, ny) in _neighbours_2d(x, y):
        if (nx, ny) in occupied:
            continue
        # Check self-avoidance: new position must not be occupied
        # (already checked above)
        # Apply move
        del occupied[(x, y)]
        occupied[(nx, ny)] = idx
        chain[idx] = (nx, ny)
        return True
    return False


def _pivot_move(chain, occupied, rng, max_depth=15):
    """Pivot move: rotate a suffix of the chain around a pivot.
    Returns True if move was made, False otherwise."""
    n = len(chain)
    if n < 5:
        return False
    
    # Choose pivot index
    pivot_idx = rng.randint(2, min(n - 3, 15))
    pivot = chain[pivot_idx]
    prev_pos = chain[pivot_idx - 1]
    
    # Remove suffix from occupied
    suffix = chain[pivot_idx:]
    for pos in suffix:
        del occupied[pos]
    
    # Try to regrow the suffix from a neighbour of prev_pos
    # Use DFS with depth limit
    import sys
    sys.setrecursionlimit(10000)
    
    def dfs(pos, path, visited):
        if len(path) == len(suffix):
            # Check if last position connects to next (none for suffix)
            # Actually for a pivot, the suffix is the end, so no right anchor
            return True
        x, y = pos
        for (nx, ny) in _neighbours_2d(x, y):
            if (nx, ny) in visited or (nx, ny) in occupied:
                continue
            if len(path) > max_depth:
                continue
            visited.add((nx, ny))
            path.append((nx, ny))
            if dfs((nx, ny), path, visited):
                return True
            path.pop()
            visited.remove((nx, ny))
        return False
    
    start_options = [(nx, ny) for (nx, ny) in _neighbours_2d(prev_pos[0], prev_pos[1])
                     if (nx, ny) not in occupied]
    rng.shuffle(start_options)
    
    found = False
    for start_pos in start_options[:3]:
        visited = {start_pos}
        path = [start_pos]
        if dfs(start_pos, path, visited):
            # Apply new suffix
            for i, pos in enumerate(path):
                occupied[pos] = pivot_idx + i
                chain[pivot_idx + i] = pos
            found = True
            break
    
    if not found:
        # Revert
        for i, pos in enumerate(suffix):
            occupied[pos] = pivot_idx + i
            chain[pivot_idx + i] = pos
        return False
    
    return True


def _simulate_annealing(sequence, chain, time_budget):
    """Run simulated annealing on a conformation.
    Returns (best_chain, best_energy)."""
    n = len(chain)
    if n <= 2:
        return list(chain), _calculate_energy(sequence, chain)
    
    rng = random.Random(hash(tuple(chain)) % 100000)
    current_chain = list(chain)
    occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = _calculate_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    h_set = set(i for i, r in enumerate(sequence) if r == 'H')
    
    # Temperature schedule
    t0 = max(1.0, -current_energy * 0.5 + 1.0)
    t_min = 0.05
    start_time = time.time()
    step = 0
    no_improve_steps = 0
    
    while time.time() - start_time < time_budget and _time_left() > 5.0:
        step += 1
        T = t0 * (t_min / t0) ** (step / max(1, 10000))
        
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.55:
            # Single residue move
            idx = rng.randint(1, n - 2)
            x, y = current_chain[idx]
            moved = False
            for (nx, ny) in _neighbours_2d(x, y):
                if (nx, ny) in occupied:
                    continue
                # Compute energy delta
                delta = 0
                if idx in h_set:
                    # Remove contacts at old position
                    for (hx, hy) in _neighbours_2d(x, y):
                        j = occupied.get((hx, hy))
                        if j is not None and abs(j - idx) > 1 and j in h_set:
                            delta -= 1
                    # Add contacts at new position
                    for (hx, hy) in _neighbours_2d(nx, ny):
                        j = occupied.get((hx, hy))
                        if j is not None and abs(j - idx) > 1 and j in h_set:
                            delta += 1
                    # Also check neighbours of old position that might gain/lose contacts
                    # (this is complex; for simplicity, approximate)
                
                new_energy = current_energy + delta
                if new_energy < current_energy or rng.random() < math.exp(-(new_energy - current_energy) / T):
                    # Apply
                    del occupied[(x, y)]
                    occupied[(nx, ny)] = idx
                    current_chain[idx] = (nx, ny)
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(current_chain)
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                    moved = True
                    break
            if not moved:
                no_improve_steps += 1
        elif move_type < 0.85:
            # Pivot move
            if _pivot_move(current_chain, occupied, rng):
                new_energy = _calculate_energy(sequence, current_chain)
                delta = new_energy - current_energy
                if new_energy < best_energy or rng.random() < math.exp(-delta / T):
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(current_chain)
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                else:
                    # Revert pivot (re-run to restore)
                    # This is tricky; for simplicity, accept the move but note it's worse
                    # Actually, let's just always accept for now and rely on SA
                    current_energy = new_energy
                    no_improve_steps += 1
            else:
                no_improve_steps += 1
        else:
            # Crankshaft: move a short segment
            if n >= 5:
                seg_len = rng.randint(2, min(3, n - 2))
                seg_start = rng.randint(1, n - seg_len - 1)
                seg_end = seg_start + seg_len - 1
                
                left_anchor = current_chain[seg_start - 1]
                right_anchor = current_chain[seg_end + 1]
                
                seg_positions = [current_chain[i] for i in range(seg_start, seg_end + 1)]
                for pos in seg_positions:
                    del occupied[pos]
                
                # Try to find new path from left_anchor to right_anchor
                import sys
                sys.setrecursionlimit(5000)
                
                def find_path(pos, path, visited, remaining):
                    if remaining == 0:
                        # Check if adjacent to right_anchor
                        for (qx, qy) in _neighbours_2d(pos[0], pos[1]):
                            if (qx, qy) == right_anchor:
                                return True
                        return False
                    x, y = pos
                    for (nx, ny) in _neighbours_2d(x, y):
                        if (nx, ny) in visited or (nx, ny) in occupied:
                            continue
                        if (nx, ny) == right_anchor:
                            continue  # Can't step on right anchor
                        visited.add((nx, ny))
                        path.append((nx, ny))
                        if find_path((nx, ny), path, visited, remaining - 1):
                            return True
                        path.pop()
                        visited.remove((nx, ny))
                    return False
                
                start_options = [(nx, ny) for (nx, ny) in _neighbours_2d(left_anchor[0], left_anchor[1])
                                 if (nx, ny) not in occupied and (nx, ny) != right_anchor]
                rng.shuffle(start_options)
                
                found = False
                for start_pos in start_options[:3]:
                    visited = {start_pos}
                    path = [start_pos]
                    if find_path(start_pos, path, visited, seg_len - 1):
                        for i, pos in enumerate(path):
                            occupied[pos] = seg_start + i
                            current_chain[seg_start + i] = pos
                        found = True
                        break
                
                if found:
                    new_energy = _calculate_energy(sequence, current_chain)
                    delta = new_energy - current_energy
                    if new_energy < best_energy or rng.random() < math.exp(-delta / T):
                        current_energy = new_energy
                        if new_energy < best_energy:
                            best_energy = new_energy
                            best_chain = list(current_chain)
                            no_improve_steps = 0
                        else:
                            no_improve_steps += 1
                    else:
                        current_energy = new_energy
                        no_improve_steps += 1
                else:
                    # Revert
                    for i, pos in enumerate(seg_positions):
                        occupied[pos] = seg_start + i
                        current_chain[seg_start + i] = pos
                    no_improve_steps += 1
            else:
                no_improve_steps += 1
        
        # Early exit if stuck for too long
        if no_improve_steps > 5000:
            break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold an HP sequence into a low-energy conformation on the square lattice."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    # Phase 1: Generate diverse starting conformations
    starting_chains = []
    
    # Strategy 1: Contact-biased growth with various parameters
    for cw, ow in [(1.5, 0.3), (2.0, 0.2), (1.0, 0.5), (3.0, 0.1)]:
        for _ in range(15):
            if _is_deadline_exceeded():
                break
            chain = _grow_with_contacts(sequence, dimension, random.Random(hash((cw, ow, _)) % 100000), cw, ow)
            if chain is not None and len(chain) == n:
                # Verify self-avoidance
                if len(set(chain)) == n:
                    energy = _calculate_energy(sequence, chain)
                    starting_chains.append((energy, chain))
    
    # Strategy 2: Spiral growth
    for _ in range(5):
        if _is_deadline_exceeded():
            break
        chain = _spiral_grow(sequence, dimension, random.Random())
        if len(chain) == n and len(set(chain)) == n:
            energy = _calculate_energy(sequence, chain)
            starting_chains.append((energy, chain))
    
    # Strategy 3: Simple random growth
    for _ in range(10):
        if _is_deadline_exceeded():
            break
        chain = _fast_grow(sequence, dimension, random.Random())
        if chain is not None and len(chain) == n and len(set(chain)) == n:
            energy = _calculate_energy(sequence, chain)
            starting_chains.append((energy, chain))
    
    if not starting_chains:
        # Fallback: straight line
        chain = [(i, 0) for i in range(n)]
        return [[p[0], p[1]] for p in chain]
    
    # Sort by energy and keep top
    starting_chains.sort(key=lambda x: x[0])
    top_chains = starting_chains[:min(8, len(starting_chains))]
    
    # Phase 2: Simulated annealing on each starting conformation
    best_chain = top_chains[0][1]
    best_energy = top_chains[0][0]
    
    remaining_time = _time_left() - 10.0  # Leave 10s buffer
    if remaining_time <= 0:
        return [[p[0], p[1]] for p in best_chain]
    
    num_sa = len(top_chains)
    time_per_sa = max(0.5, remaining_time * 0.85 / num_sa)
    
    for energy, chain in top_chains:
        if _is_deadline_exceeded() or _time_left() < 8.0:
            break
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
        
        sa_chain, sa_energy = _simulate_annealing(sequence, chain, time_per_sa)
        
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_chain = list(sa_chain)
        
        # Update time budget based on actual remaining time
        time_per_sa = max(0.5, (_time_left() - 10.0) * 0.85 / max(1, num_sa - 1))
    
    return [[p[0], p[1]] for p in best_chain]
# EVOLVE-BLOCK-END
