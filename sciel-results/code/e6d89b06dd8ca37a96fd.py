# EVOLVE-BLOCK-START
import random
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute HP contact energy: -number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _is_valid_chain(chain):
    """Check if the chain is self-avoiding and has unit steps."""
    if not chain:
        return True
    occupied = set()
    for i, p in enumerate(chain):
        if p in occupied:
            return False
        occupied.add(p)
        if i > 0:
            prev = chain[i - 1]
            if abs(p[0] - prev[0]) + abs(p[1] - prev[1]) != 1:
                return False
    return True


def _local_search(sequence, chain, time_budget, rng):
    """Simulated annealing local search with pivot and reversal moves."""
    n = len(chain)
    if n <= 2:
        return chain
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, chain)
    current_chain = list(chain)
    current_energy = best_energy
    
    start_time = time.time()
    max_time = time_budget
    initial_temp = 2.0
    cooling_rate = 0.999
    
    temp = initial_temp
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() - start_time < max_time:
        # Choose move type: 50% pivot, 50% reversal
        if rng.random() < 0.5:
            # Pivot Move
            if n > 2:
                pivot_idx = rng.randint(1, n - 1)
                pivot = current_chain[pivot_idx - 1]
                tail = current_chain[pivot_idx:]
                
                # Choose a random rotation: 1, 2, or 3 (90, 180, 270 deg)
                rot = rng.choice([1, 2, 3])
                rotated = []
                valid = True
                occupied_set = set(current_chain[:pivot_idx])
                
                # Apply rotation
                # Rotation matrix for 90 deg clockwise: (x, y) -> (y, -x)
                # We need to apply this 'rot' times relative to pivot
                # Or just calculate directly
                # Let's do it step by step for clarity or use a helper
                # For 2D:
                # 1 rot: (dx, dy) -> (dy, -dx)
                # 2 rot: (dx, dy) -> (-dx, -dy)
                # 3 rot: (dx, dy) -> (-dy, dx)
                
                for pos in tail:
                    dx = pos[0] - pivot[0]
                    dy = pos[1] - pivot[1]
                    
                    if rot == 1:
                        ndx, ndy = dy, -dx
                    elif rot == 2:
                        ndx, ndy = -dx, -dy
                    else: # rot == 3
                        ndx, ndy = -dy, dx
                    
                    new_pos = (pivot[0] + ndx, pivot[1] + ndy)
                    
                    if new_pos in occupied_set:
                        valid = False
                        break
                    occupied_set.add(new_pos)
                    
                    # Check unit step from previous in rotated tail
                    if rotated:
                        prev_rot = rotated[-1]
                        if abs(new_pos[0] - prev_rot[0]) + abs(new_pos[1] - prev_rot[1]) != 1:
                            valid = False
                            break
                    else:
                        # Check unit step from pivot
                        if abs(new_pos[0] - pivot[0]) + abs(new_pos[1] - pivot[1]) != 1:
                            valid = False
                            break
                            
                    rotated.append(new_pos)
                
                if valid:
                    new_chain = current_chain[:pivot_idx] + rotated
                    new_energy = _compute_energy(sequence, new_chain)
                else:
                    continue
            else:
                continue
        else:
            # Reversal Move
            if n > 3:
                start_idx = rng.randint(1, n - 2)
                end_idx = rng.randint(start_idx + 1, n - 1)
                
                # Reverse the segment
                new_chain = current_chain[:start_idx] + list(reversed(current_chain[start_idx:end_idx + 1])) + current_chain[end_idx + 1:]
                
                # Check validity
                if _is_valid_chain(new_chain):
                    new_energy = _compute_energy(sequence, new_chain)
                else:
                    continue
            else:
                continue

        # Accept or reject based on simulated annealing
        delta = new_energy - current_energy
        if delta < 0 or (temp > 0.01 and rng.random() < pow(2.718281828, -delta / temp)):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_chain = list(current_chain)
                best_energy = current_energy
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        temp *= cooling_rate
        
        if no_improve_count > max_no_improve:
            # Restart with best chain
            current_chain = list(best_chain)
            current_energy = best_energy
            temp = initial_temp * 0.5
            no_improve_count = 0
    
    return best_chain


def _generate_initial_chain(sequence, rng):
    """Generate a random self-avoiding walk as initial conformation."""
    n = len(sequence)
    if n <= 2:
        return [(0, 0), (1, 0)][:n]
    
    # Start at origin, grow randomly
    chain = [(0, 0)]
    occupied = {(0, 0)}
    direction = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
    
    for i in range(1, n):
        candidates = []
        last = chain[-1]
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_pos = (last[0] + dx, last[1] + dy)
            if new_pos not in occupied:
                candidates.append(new_pos)
        
        if not candidates:
            return None
        
        # Prefer continuing in similar direction
        if direction in [(c[0] - last[0], c[1] - last[1]) for c in candidates]:
            new_pos = [c for c in candidates if (c[0] - last[0], c[1] - last[1]) == direction][0]
        else:
            new_pos = rng.choice(candidates)
        
        direction = (new_pos[0] - last[0], new_pos[1] - last[1])
        chain.append(new_pos)
        occupied.add(new_pos)
    
    return chain


def fold(sequence, dimension):
    """Fold HP sequence using multi-start simulated annealing local search."""
    n = len(sequence)
    if n <= 2:
        return [[0, 0], [1, 0]][:n]
    
    # Time budget: distribute based on sequence length
    # Total budget is 120s. 10 sequences.
    # Let's give each sequence ~10s max, min 3s.
    time_per_seq = min(3.0 + n * 0.07, 10.0)
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = float('inf')
    
    # Multi-start: try different initial conformations
    num_starts = max(3, min(15, int(time_per_seq / 0.8)))
    time_per_start = time_per_seq / num_starts
    
    for start in range(num_starts):
        # Generate initial chain
        initial = _generate_initial_chain(sequence, rng)
        if initial is None:
            # Fallback: straight line
            initial = [(i, 0) for i in range(n)]
        
        # Local search
        improved = _local_search(sequence, initial, time_per_start, rng)
        
        energy = _compute_energy(sequence, improved)
        if energy < best_energy:
            best_energy = energy
            best_chain = improved
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
