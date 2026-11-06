"""HP lattice protein folding: simulated annealing with local search moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

_T0 = time.time()
_DEADLINE = 195.0  # wall-clock limit with safety margin (210s kill, leave 15s buffer)


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _neighbours(point):
    """Yield 4-neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _compute_energy(sequence, chain):
    """Compute energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    if n < 3:
        return 0
    # Map position -> index
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
    
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        # Check 4 neighbors
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = (x + dx, y + dy)
            j = pos_to_idx.get(nb)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _initial_conformation(sequence, rng):
    """Generate a random self-avoiding walk as initial conformation."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0)}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for nb in _neighbours(last):
            if nb not in occupied:
                options.append(nb)
        
        if not options:
            # Backtrack and try different path - for simplicity, restart
            return _initial_conformation(sequence, rng)
        
        # Prefer moves that create H-H contacts
        if sequence[i] == 'H':
            scored = []
            for nb in options:
                gain = 0
                x, y = nb
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    pp = (x + dx, y + dy)
                    if pp in occupied:
                        # Check if the residue at pp is H
                        idx = chain.index(pp) if pp in chain else -1
                        if idx >= 0 and sequence[idx] == 'H':
                            gain += 1
                scored.append((gain + rng.random(), nb))
            scored.sort(reverse=True)
            next_pos = scored[0][1]
        else:
            next_pos = rng.choice(options)
        
        chain.append(next_pos)
        occupied.add(next_pos)
    
    return chain


def _pivot_move(chain, rng):
    """Perform a pivot move: rotate a suffix of the chain around a pivot point."""
    n = len(chain)
    if n < 4:
        return None
    
    # Choose pivot index (not at ends)
    pivot = rng.randint(1, n - 2)
    pivot_pos = chain[pivot]
    suffix = chain[pivot + 1:]
    
    # Generate all 4 rotations of the suffix relative to pivot
    rotations = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        rotated = []
        valid = True
        for j, (x, y) in enumerate(suffix):
            # Rotate relative to pivot
            rx, ry = x - pivot_pos[0], y - pivot_pos[1]
            # Apply 90-degree rotation
            if dx == 1 and dy == 0:  # This is just translation, not rotation
                pass
            # Let's do proper rotation: 90 deg CCW: (x,y) -> (-y,x)
            # Actually, let's rotate the vector
            if j == 0:
                # First element must be adjacent to pivot
                new_x, new_y = -ry, rx  # 90 deg CCW
                if (pivot_pos[0] + new_x, pivot_pos[1] + new_y) in set(chain[:pivot+1]):
                    valid = False
                    break
                rotated.append((pivot_pos[0] + new_x, pivot_pos[1] + new_y))
            else:
                prev = rotated[-1]
                # The new position should be adjacent to prev
                # We rotate the entire suffix
                new_x, new_y = -ry, rx
                pos = (pivot_pos[0] + new_x, pivot_pos[1] + new_y)
                if pos in set(chain[:pivot+1]):
                    valid = False
                    break
                rotated.append(pos)
        
        if valid and len(rotated) == len(suffix):
            new_chain = chain[:pivot + 1] + rotated
            # Check self-avoidance for the new suffix
            if len(set(new_chain)) == n:
                rotations.append(new_chain)
    
    if not rotations:
        return None
    return rng.choice(rotations)


def _crankshaft_move(chain, rng):
    """Perform a crankshaft move: rotate a short segment (length 2-4) in the middle."""
    n = len(chain)
    if n < 5:
        return None
    
    # Choose a segment length
    seg_len = rng.choice([2, 3, 4])
    
    # Choose start index
    if n <= seg_len + 2:
        return None
    start = rng.randint(1, n - seg_len - 1)
    
    segment = chain[start:start + seg_len]
    before = chain[:start]
    after = chain[start + seg_len:]
    
    if not before or not after:
        return None
    
    # The segment must be rotated around its first point
    pivot = segment[0]
    rest = segment[1:]
    
    # Generate rotations
    candidates = []
    for angle in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        # This is a simplified rotation - in 2D, we can try reflecting or rotating
        # Let's try all 4 possible directions for the first step
        new_segment = [pivot]
        valid = True
        for j, (x, y) in enumerate(rest):
            # The relative direction from previous point
            if j == 0:
                # First step: try all 4 directions
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    new_pos = (pivot[0] + dx, pivot[1] + dy)
                    if new_pos in set(before) or new_pos in set(after):
                        continue
                    # Check if this leads to a valid continuation
                    test_seg = [new_pos]
                    valid_test = True
                    for k, (px, py) in enumerate(rest[1:]):
                        # Maintain relative geometry - this is complex
                        # Simplify: just check if we can place it
                        # For now, skip complex crankshaft
                        valid_test = False
                        break
                    if valid_test:
                        new_segment.append(new_pos)
            # This approach is getting complex, let's use a simpler move
        # Skip complex crankshaft for now, use pivot moves instead
        pass
    
    return None  # Fall back to pivot moves


def _try_local_search(sequence, chain, rng, max_iters=500):
    """Perform local search with simulated annealing."""
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    T = 5.0
    cooling = 0.999
    
    for it in range(max_iters):
        if _remaining() < 5.0:
            break
        
        # Choose a move
        if rng.random() < 0.7:
            new_chain = _pivot_move(chain, rng)
        else:
            # Simple random move: try to move one residue
            idx = rng.randint(1, n - 1)
            new_chain = list(chain)
            x, y = chain[idx]
            moved = False
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_pos = (x + dx, y + dy)
                # Check if new_pos is not occupied by other residues
                # And if the old position becomes valid
                occupied_without_idx = set(chain[:idx] + chain[idx+1:])
                if new_pos not in occupied_without_idx:
                    # Also check that the new position is adjacent to both neighbors
                    if idx > 0 and idx < n - 1:
                        prev = chain[idx - 1]
                        next_pos = chain[idx + 1]
                        # Check adjacency
                        if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
                            continue
                        if abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1]) != 1:
                            continue
                    new_chain[idx] = new_pos
                    moved = True
                    break
            
            if not moved:
                continue
        
        if new_chain is None:
            continue
        
        # Verify self-avoidance
        if len(set(new_chain)) != n:
            continue
        
        # Verify connectivity
        valid = True
        for i in range(n - 1):
            if abs(new_chain[i][0] - new_chain[i+1][0]) + abs(new_chain[i][1] - new_chain[i+1][1]) != 1:
                valid = False
                break
        if not valid:
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Simulated annealing acceptance
        if delta < 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(new_chain)
        
        T *= cooling
        if T < 0.1:
            T = 0.1
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold sequence into low-energy conformation using simulated annealing."""
    if dimension != 2:
        # Fallback for non-2D (shouldn't happen per task)
        return [[i] + [0] * (dimension - 1) for i in range(len(sequence))]
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0  # Start with unfolded (energy 0)
    
    # Time budget: use most of the available time
    start_time = time.time()
    time_budget = min(_remaining() - 5.0, 180.0)  # Leave 5s buffer
    
    # Number of restarts based on sequence length
    if n <= 30:
        num_restarts = 50
    elif n <= 50:
        num_restarts = 30
    elif n <= 70:
        num_restarts = 20
    else:
        num_restarts = 15
    
    iters_per_restart = max(100, int(500000 / n))
    
    for restart in range(num_restarts):
        if _remaining() < 10.0:
            break
        
        # Generate initial conformation
        chain = _initial_conformation(sequence, rng)
        if chain is None or len(chain) != n:
            continue
        
        # Verify it's valid
        if len(set(chain)) != n:
            continue
        valid = True
        for i in range(n - 1):
            if abs(chain[i][0] - chain[i+1][0]) + abs(chain[i][1] - chain[i+1][1]) != 1:
                valid = False
                break
        if not valid:
            continue
        
        # Run local search
        chain, energy = _try_local_search(sequence, chain, rng, max_iters=iters_per_restart)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
        
        # Check time
        if time.time() - start_time > time_budget:
            break
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [[x, y] for x, y in best_chain]
# EVOLVE-BLOCK-END
