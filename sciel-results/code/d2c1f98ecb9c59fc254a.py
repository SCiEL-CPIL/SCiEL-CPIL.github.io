# EVOLVE-BLOCK-START
import random
import time
import math

def _compute_energy(sequence, chain):
    """Compute HP contact energy: -number of non-adjacent H-H contacts."""
    n = len(chain)
    if n <= 2:
        return 0
    
    # Map position to index
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    contacts = 0
    # Only check H residues to save time
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y = p
            # Check 4 neighbors
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                q = (x + dx, y + dy)
                j = pos_to_idx.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _is_valid_move_pivot(chain, pivot_idx, rot):
    """Check if a pivot move at pivot_idx with rotation rot is valid."""
    n = len(chain)
    if pivot_idx <= 0 or pivot_idx >= n:
        return False
    
    head = chain[:pivot_idx]
    tail = chain[pivot_idx:]
    
    # Rotation logic
    # rot 0: identity
    # rot 1: (x,y) -> (y, -x)
    # rot 2: (x,y) -> (-x, -y)
    # rot 3: (x,y) -> (-y, x)
    
    pivot = head[-1]
    px, py = pivot
    
    new_tail = []
    occupied = set(head)
    
    # First element of new tail must be adjacent to pivot
    # And not in occupied
    
    for i, p in enumerate(tail):
        x, y = p
        dx, dy = x - px, y - py
        
        if rot == 1:
            ndx, ndy = dy, -dx
        elif rot == 2:
            ndx, ndy = -dx, -dy
        elif rot == 3:
            ndx, ndy = -dy, dx
        else:
            continue
            
        new_p = (px + ndx, py + ndy)
        
        if new_p in occupied:
            return False
        occupied.add(new_p)
        
        if i > 0:
            prev = new_tail[-1]
            if abs(new_p[0] - prev[0]) + abs(new_p[1] - prev[1]) != 1:
                return False
        
        new_tail.append(new_p)
        
    return True

def _apply_pivot(chain, pivot_idx, rot):
    """Apply pivot move and return new chain."""
    head = chain[:pivot_idx]
    tail = chain[pivot_idx:]
    pivot = head[-1]
    px, py = pivot
    
    new_tail = []
    for p in tail:
        x, y = p
        dx, dy = x - px, y - py
        
        if rot == 1:
            ndx, ndy = dy, -dx
        elif rot == 2:
            ndx, ndy = -dx, -dy
        elif rot == 3:
            ndx, ndy = -dy, dx
        else:
            continue
            
        new_p = (px + ndx, py + ndy)
        new_tail.append(new_p)
        
    return head + new_tail

def _is_valid_move_reversal(chain, start, end):
    """Check if a reversal move from start to end is valid."""
    n = len(chain)
    if start >= end or start < 0 or end >= n:
        return False
        
    # The segment [start:end+1] is reversed.
    # We need to check if any new position in the reversed segment
    # collides with any position in the rest of the chain.
    
    # New positions are chain[start:end+1] reversed.
    # Old positions in that segment are removed.
    # So we check if new_pos is in (chain[:start] U chain[end+1:])
    
    rest = set(chain[:start]) | set(chain[end+1:])
    
    # Also check connectivity at the boundaries
    # Boundary 1: chain[start-1] to new_chain[start] (which is old chain[end])
    # Boundary 2: new_chain[end] (which is old chain[start]) to chain[end+1]
    
    # 1. Check collision
    for i in range(start, end + 1):
        new_p = chain[end - (i - start)]
        if new_p in rest:
            return False
            
    # 2. Check connectivity
    if start > 0:
        p_prev = chain[start-1]
        p_new_start = chain[end]
        if abs(p_prev[0] - p_new_start[0]) + abs(p_prev[1] - p_new_start[1]) != 1:
            return False
            
    if end < n - 1:
        p_new_end = chain[start]
        p_next = chain[end+1]
        if abs(p_new_end[0] - p_next[0]) + abs(p_new_end[1] - p_next[1]) != 1:
            return False
            
    return True

def _apply_reversal(chain, start, end):
    """Apply reversal move and return new chain."""
    return chain[:start] + list(reversed(chain[start:end+1])) + chain[end+1:]

def _generate_initial_chain(sequence, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n <= 2:
        return [(0, 0), (1, 0)][:n]
    
    chain = [(0, 0)]
    occupied = {(0, 0)}
    
    for _ in range(1, n):
        last = chain[-1]
        candidates = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            p = (last[0] + dx, last[1] + dy)
            if p not in occupied:
                candidates.append(p)
        
        if not candidates:
            return None
        
        next_p = rng.choice(candidates)
        chain.append(next_p)
        occupied.add(next_p)
        
    return chain

def fold(sequence, dimension):
    """Fold HP sequence using Parallel Tempering (Replica Exchange Monte Carlo)."""
    n = len(sequence)
    if n <= 2:
        return [[0, 0], [1, 0]][:n]
    
    rng = random.Random(123)
    
    # Time budget management
    start_time = time.time()
    total_budget = 110.0 # Leave 10s buffer
    
    # Number of replicas
    # More replicas for longer sequences?
    num_replicas = 8
    
    # Temperatures
    # T_min should be low enough to converge, T_min high enough to explore
    # E is integer (negative).
    T_min = 0.5
    T_max = 5.0
    temps = []
    for i in range(num_replicas):
        # Geometric spacing
        t = T_min * (T_max / T_min) ** (i / (num_replicas - 1))
        temps.append(t)
    
    # Initialize replicas
    replicas = []
    energies = []
    for _ in range(num_replicas):
        init_chain = _generate_initial_chain(sequence, rng)
        if init_chain is None:
            init_chain = [(i, 0) for i in range(n)]
        e = _compute_energy(sequence, init_chain)
        replicas.append(init_chain)
        energies.append(e)
        
    best_chain = min(replicas, key=lambda c: _compute_energy(sequence, c))
    best_energy = _compute_energy(sequence, best_chain)
    
    # MC steps per replica per cycle
    # Adjust based on n
    steps_per_cycle = max(5, 500 // n)
    
    while time.time() - start_time < total_budget:
        # 1. Update each replica
        for i in range(num_replicas):
            T = temps[i]
            current_chain = replicas[i]
            current_energy = energies[i]
            
            for _ in range(steps_per_cycle):
                move_type = rng.random()
                
                if move_type < 0.5:
                    # Pivot
                    if n > 2:
                        pivot_idx = rng.randint(1, n - 1)
                        rot = rng.choice([1, 2, 3])
                        if _is_valid_move_pivot(current_chain, pivot_idx, rot):
                            new_chain = _apply_pivot(current_chain, pivot_idx, rot)
                            new_energy = _compute_energy(sequence, new_chain)
                            delta = new_energy - current_energy
                            if delta < 0 or (T > 0.1 and rng.random() < math.exp(-delta / T)):
                                current_chain = new_chain
                                current_energy = new_energy
                else:
                    # Reversal
                    if n > 3:
                        start_idx = rng.randint(1, n - 2)
                        end_idx = rng.randint(start_idx + 1, n - 1)
                        if _is_valid_move_reversal(current_chain, start_idx, end_idx):
                            new_chain = _apply_reversal(current_chain, start_idx, end_idx)
                            new_energy = _compute_energy(sequence, new_chain)
                            delta = new_energy - current_energy
                            if delta < 0 or (T > 0.1 and rng.random() < math.exp(-delta / T)):
                                current_chain = new_chain
                                current_energy = new_energy
                                
            replicas[i] = current_chain
            energies[i] = current_energy
            
        # 2. Attempt swaps between adjacent replicas
        for i in range(num_replicas - 1):
            i1, i2 = i, i + 1
            T1, T2 = temps[i1], temps[i2]
            E1, E2 = energies[i1], energies[i2]
            
            # Delta = (1/T1 - 1/T2) * (E2 - E1)
            # If T1 < T2 (1/T1 > 1/T2), then (1/T1 - 1/T2) is positive.
            # If E2 > E1 (E2-E1 positive), Delta > 0 -> Reject (likely)
            # If E2 < E1 (E2-E1 negative), Delta < 0 -> Accept
            # This encourages lower energy chains to be at lower temperatures.
            
            delta = (1.0 / T1 - 1.0 / T2) * (E2 - E1)
            
            if delta < 0 or rng.random() < math.exp(-delta):
                # Swap
                replicas[i1], replicas[i2] = replicas[i2], replicas[i1]
                energies[i1], energies[i2] = energies[i2], energies[i1]
                
                # Update best
                # The chain at index 0 is the lowest temperature, likely best
                # But we track global best
                if energies[i1] < best_energy:
                    best_energy = energies[i1]
                    best_chain = replicas[i1]
                if energies[i2] < best_energy:
                    best_energy = energies[i2]
                    best_chain = replicas[i2]
            else:
                # Even if no swap, check if current states are better
                if energies[i1] < best_energy:
                    best_energy = energies[i1]
                    best_chain = replicas[i1]
                if energies[i2] < best_energy:
                    best_energy = energies[i2]
                    best_chain = replicas[i2]
                    
        # Periodic check for best
        if rng.random() < 0.1:
            for i in range(num_replicas):
                if energies[i] < best_energy:
                    best_energy = energies[i]
                    best_chain = replicas[i]
                    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
