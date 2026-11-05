"""HP lattice protein folding with simulated annealing and spiral initialization."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = 200.0  # safety margin before 210s kill
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy: -number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _spiral_conformation(n, dimension):
    """Generate a compact spiral-like conformation for n residues."""
    if n <= 1:
        return [(0,)] * dimension
    if dimension == 2:
        # Generate a spiral that fills a roughly square region
        chain = [(0, 0)]
        occupied = {(0, 0)}
        direction = (1, 0)  # start going right
        step_count = 0
        steps_in_direction = 1
        direction_changes = 0
        
        for i in range(1, n):
            # Try to continue in current direction
            placed = False
            for _ in range(steps_in_direction):
                nx = chain[-1][0] + direction[0]
                ny = chain[-1][1] + direction[1]
                if (nx, ny) not in occupied:
                    chain.append((nx, ny))
                    occupied.add((nx, ny))
                    placed = True
                    break
            if not placed:
                # Turn right and try again
                # Rotate direction 90 degrees clockwise: (x,y) -> (y, -x)
                direction = (direction[1], -direction[0])
                direction_changes += 1
                if direction_changes % 2 == 0:
                    steps_in_direction += 1
                # Retry with new direction
                for _ in range(steps_in_direction):
                    nx = chain[-1][0] + direction[0]
                    ny = chain[-1][1] + direction[1]
                    if (nx, ny) not in occupied:
                        chain.append((nx, ny))
                        occupied.add((nx, ny))
                        placed = True
                        break
            if not placed:
                # Fallback: try all 4 directions
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx = chain[-1][0] + dx
                    ny = chain[-1][1] + dy
                    if (nx, ny) not in occupied:
                        chain.append((nx, ny))
                        occupied.add((nx, ny))
                        direction = (dx, dy)
                        steps_in_direction = 1
                        break
                else:
                    # Should not happen for reasonable n
                    break
        return chain[:n]
    else:
        # For higher dimensions, use a simple line with some perturbation
        return [(i, 0)] * dimension


def _segment_move(chain, start, length, direction, occupied):
    """Try to move a segment of the chain by one step in a direction.
    Returns True if move was successful and chain/occupied are updated."""
    n = len(chain)
    if start < 0 or start + length > n:
        return False
    
    # Check if the move is valid
    # The segment is from index start to start+length-1
    # We want to shift each residue in the segment by 'direction'
    # Need to check:
    # 1. The new positions are not occupied (except by the segment itself)
    # 2. The new positions maintain unit steps within the segment
    # 3. The new positions maintain unit steps at the boundaries
    
    seg_indices = list(range(start, start + length))
    new_positions = []
    for idx in seg_indices:
        new_pos = (chain[idx][0] + direction[0], chain[idx][1] + direction[1])
        new_positions.append(new_pos)
    
    # Check self-avoidance: new positions must not conflict with occupied
    # except for positions that are part of the segment being moved
    seg_set = set(chain[idx] for idx in seg_indices)
    for new_pos in new_positions:
        if new_pos in occupied and new_pos not in seg_set:
            return False
    
    # Check that consecutive residues in the segment remain unit steps
    for k in range(len(new_positions) - 1):
        dx = abs(new_positions[k][0] - new_positions[k+1][0])
        dy = abs(new_positions[k][1] - new_positions[k+1][1])
        if dx + dy != 1:
            return False
    
    # Check boundary connections
    if start > 0:
        prev_pos = chain[start - 1]
        dx = abs(new_positions[0][0] - prev_pos[0])
        dy = abs(new_positions[0][1] - prev_pos[1])
        if dx + dy != 1:
            return False
    if start + length < n:
        next_pos = chain[start + length]
        dx = abs(new_positions[-1][0] - next_pos[0])
        dy = abs(new_positions[-1][1] - next_pos[1])
        if dx + dy != 1:
            return False
    
    # Move is valid: apply it
    for k, idx in enumerate(seg_indices):
        occupied.discard(chain[idx])
        chain[idx] = new_positions[k]
        occupied.add(new_positions[k])
    return True


def _undo_segment_move(chain, start, length, old_positions, occupied):
    """Undo a segment move."""
    for k, idx in enumerate(range(start, start + length)):
        occupied.discard(chain[idx])
        chain[idx] = old_positions[k]
        occupied.add(old_positions[k])


def _simulated_annealing(sequence, chain, occupied, max_time):
    """Apply simulated annealing to improve the conformation."""
    n = len(chain)
    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = list(chain)
    
    # Temperature schedule
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.995
    
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    while _remaining() > max_time:
        if T < T_min:
            T = 1.0  # restart annealing
        
        # Choose a random move
        move_type = random.random()
        if move_type < 0.3:
            # Single residue move (if possible)
            idx = random.randint(1, n - 2)  # avoid ends for simplicity
            old_pos = chain[idx]
            for d in directions:
                new_pos = (old_pos[0] + d[0], old_pos[1] + d[1])
                if new_pos in occupied and new_pos != old_pos:
                    continue
                # Check if new_pos is adjacent to neighbors
                if idx > 0:
                    prev = chain[idx - 1]
                    if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
                        continue
                if idx < n - 1:
                    nxt = chain[idx + 1]
                    if abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) != 1:
                        continue
                # Check if new_pos is not occupied by non-adjacent residues
                if new_pos in occupied:
                    # It's occupied by some other residue - check if it's adjacent
                    # We can only move if new_pos is free
                    continue
                # Try the move
                occupied.discard(old_pos)
                chain[idx] = new_pos
                occupied.add(new_pos)
                new_energy = _energy(sequence, chain)
                delta = new_energy - current_energy
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                else:
                    # Revert
                    occupied.discard(new_pos)
                    chain[idx] = old_pos
                    occupied.add(old_pos)
                break
        else:
            # Segment move
            length = random.randint(2, min(10, n // 3))
            start = random.randint(0, n - length)
            d = random.choice(directions)
            old_positions = [chain[start + k] for k in range(length)]
            if _segment_move(chain, start, length, d, occupied):
                new_energy = _energy(sequence, chain)
                delta = new_energy - current_energy
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                else:
                    _undo_segment_move(chain, start, length, old_positions, occupied)
        
        T *= cooling_rate
    
    return best_chain, best_energy


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def fold(sequence, dimension):
    """Best conformation found using multiple strategies and simulated annealing."""
    global _start_time
    _start_time = time.time()
    
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension for _ in range(n)]
    
    best_chain = None
    best_energy = 0  # 0 means no contacts (energy = 0)
    
    # Strategy 1: Spiral initialization + SA
    spiral = _spiral_conformation(n, dimension)
    if len(spiral) == n:
        occupied = set(spiral)
        # Verify self-avoidance
        if len(occupied) == n:
            e = _energy(sequence, spiral)
            if e < best_energy:
                best_energy = e
                best_chain = list(spiral)
            # Apply SA
            sa_chain, sa_energy = _simulated_annealing(sequence, spiral, occupied, 15.0)
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_chain = sa_chain
    
    # Strategy 2: Multiple greedy growths + SA
    rng = random.Random(12345 + len(sequence))
    num_growths = 50 if n < 50 else 20
    for _ in range(num_growths):
        if _remaining() < 5.0:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        e = _energy(sequence, chain)
        if e < best_energy:
            best_energy = e
            best_chain = list(chain)
        # Apply brief SA
        occupied = set(chain)
        sa_chain, sa_energy = _simulated_annealing(sequence, chain, occupied, 3.0)
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_chain = sa_chain
    
    # Strategy 3: More greedy growths if time permits
    if _remaining() > 10.0:
        for _ in range(100):
            if _remaining() < 5.0:
                break
            chain = _grow(sequence, dimension, rng)
            if chain is None:
                continue
            e = _energy(sequence, chain)
            if e < best_energy:
                best_energy = e
                best_chain = list(chain)
            occupied = set(chain)
            sa_chain, sa_energy = _simulated_annealing(sequence, chain, occupied, 2.0)
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_chain = sa_chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
