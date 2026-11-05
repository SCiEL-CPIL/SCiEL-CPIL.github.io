"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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


def _is_valid_move(chain, occupied, i, new_pos):
    """Check if placing residue i at new_pos is valid (self-avoiding, unit step)."""
    if new_pos in occupied:
        return False
    # Check unit step from previous
    if i > 0:
        prev = chain[i - 1]
        if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
            return False
    # Check unit step to next
    if i < len(chain) - 1:
        nxt = chain[i + 1]
        if abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) != 1:
            return False
    return True


def _local_search(sequence, chain, time_budget, rng):
    """Simulated annealing local search with pivot, reversal, and shift moves."""
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
    cooling_rate = 0.995
    
    temp = initial_temp
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < max_time:
        move_type = rng.choice(['pivot', 'reversal', 'shift'])
        new_chain = list(current_chain)
        new_energy = current_energy
        valid = True
        
        if move_type == 'pivot':
            # Pick a pivot point and rotate the tail
            if n > 3:
                pivot_idx = rng.randint(1, n - 2)
                # Try rotating the tail around the pivot
                # Get the tail segment
                tail = new_chain[pivot_idx:]
                # Try 4 rotations (0, 90, 180, 270 degrees) around pivot
                pivot = new_chain[pivot_idx - 1]
                for rot in range(4):
                    # Rotate tail relative to pivot
                    rotated = []
                    for pos in tail:
                        # Translate to origin
                        dx = pos[0] - pivot[0]
                        dy = pos[1] - pivot[1]
                        # Rotate 90 degrees clockwise
                        dx, dy = dy, -dx
                        rotated.append((pivot[0] + dx, pivot[1] + dy))
                    
                    # Check if rotated tail is valid
                    valid_rot = True
                    occupied_set = set(new_chain[:pivot_idx])
                    for i, pos in enumerate(rotated):
                        if pos in occupied_set:
                            valid_rot = False
                            break
                        occupied_set.add(pos)
                        # Check adjacency
                        if i == 0:
                            if abs(pos[0] - pivot[0]) + abs(pos[1] - pivot[1]) != 1:
                                valid_rot = False
                                break
                        else:
                            prev = rotated[i - 1]
                            if abs(pos[0] - prev[0]) + abs(pos[1] - prev[1]) != 1:
                                valid_rot = False
                                break
                    
                    if valid_rot:
                        new_chain = new_chain[:pivot_idx] + rotated
                        # Recompute energy
                        new_energy = _compute_energy(sequence, new_chain)
                        break
                if not valid_rot:
                    continue
        
        elif move_type == 'reversal':
            # Reverse a segment of the chain
            if n > 4:
                start_idx = rng.randint(1, n - 3)
                end_idx = rng.randint(start_idx + 1, n - 1)
                # Reverse the segment
                segment = new_chain[start_idx:end_idx + 1]
                reversed_segment = list(reversed(segment))
                new_chain = new_chain[:start_idx] + reversed_segment + new_chain[end_idx + 1:]
                
                # Check validity
                valid = True
                occupied_set = set(new_chain)
                if len(occupied_set) != len(new_chain):
                    valid = False
                else:
                    for i in range(1, len(new_chain)):
                        if abs(new_chain[i][0] - new_chain[i-1][0]) + abs(new_chain[i][1] - new_chain[i-1][1]) != 1:
                            valid = False
                            break
                if valid:
                    new_energy = _compute_energy(sequence, new_chain)
                else:
                    continue
        
        else:  # shift
            # Move a single residue to a neighboring empty site
            if n > 2:
                idx = rng.randint(1, n - 2)
                current_pos = new_chain[idx]
                for neighbor in _neighbours(current_pos):
                    # Check if neighbor is empty
                    occupied_set = set(new_chain)
                    if neighbor in occupied_set:
                        continue
                    # Check adjacency constraints
                    prev_pos = new_chain[idx - 1]
                    next_pos = new_chain[idx + 1]
                    if abs(neighbor[0] - prev_pos[0]) + abs(neighbor[1] - prev_pos[1]) != 1:
                        continue
                    if abs(neighbor[0] - next_pos[0]) + abs(neighbor[1] - next_pos[1]) != 1:
                        continue
                    
                    new_chain[idx] = neighbor
                    new_energy = _compute_energy(sequence, new_chain)
                    break
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
        # Try to continue in same direction first, then random
        candidates = []
        last = chain[-1]
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_pos = (last[0] + dx, last[1] + dy)
            if new_pos not in occupied:
                candidates.append(new_pos)
        
        if not candidates:
            # Dead end, try to backtrack or fail
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
    # Shorter sequences get less time, longer get more
    base_time = 2.0
    time_per_seq = min(base_time + n * 0.05, 10.0)
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = float('inf')
    
    # Multi-start: try different initial conformations
    num_starts = max(3, min(10, int(time_per_seq / 1.5)))
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
