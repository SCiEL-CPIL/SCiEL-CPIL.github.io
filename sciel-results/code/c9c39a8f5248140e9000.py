import random
import time
import math

def _energy(sequence, chain):
    """Compute HP contact energy."""
    n = len(chain)
    if n == 0:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y = p
            # Check 4 neighbors
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                q = (x+dx, y+dy)
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _generate_snake(n):
    """Generate a compact snake-like initial conformation."""
    if n == 0:
        return []
    # Approximate square side
    side = int(math.ceil(math.sqrt(n)))
    
    chain = []
    # We will fill a grid of size side x side
    # Snake pattern: Row 0 left-to-right, Row 1 right-to-left, etc.
    # But we only need n points.
    
    # Let's just do a standard boustrophedon in a bounding box
    # Start at (0,0)
    # Move Right, Down, Left, Down, Right...
    
    # A simpler robust generator:
    # Use a "spiral" or "snake" that fits in ceil(sqrt(n)) x ceil(sqrt(n))
    
    # Let's use a simple greedy growth that avoids dead ends by checking next moves
    # Actually, snake is deterministic and fast.
    
    x, y = 0, 0
    chain.append((x, y))
    
    # Directions: Right, Down, Left, Down, Right...
    # We need to handle the last row if it's not full.
    
    # Let's just iterate rows
    for row in range(side):
        if row % 2 == 0:
            # Left to Right
            for col in range(side):
                if len(chain) == n:
                    return chain
                # If we are in the first row, we already added (0,0)
                if row == 0 and col == 0:
                    continue
                x, y = col, row
                chain.append((x, y))
        else:
            # Right to Left
            for col in range(side-1, -1, -1):
                if len(chain) == n:
                    return chain
                x, y = col, row
                chain.append((x, y))
                
    # Fallback if n > side*side (should not happen with ceil sqrt)
    # But just in case, extend
    # This simple snake might have issues with connectivity if n is not a multiple of side?
    # Yes, the transition between rows is valid (down step).
    # The transition between last point of row i and first point of row i+1 is a unit step?
    # Row 0 ends at (side-1, 0). Row 1 starts at (side-1, 1). Step is (0, 1). Valid.
    # Row 1 ends at (0, 1). Row 2 starts at (0, 2). Step is (0, 1). Valid.
    # So this is a valid SAW.
    
    return chain

def _try_pivot_move(chain, seq, k, rot):
    """
    Try a pivot move at index k with rotation rot (1, 2, or 3 for 90, 180, 270 deg).
    Returns (new_chain, new_energy) if valid, else (None, None).
    """
    n = len(chain)
    if k < 1 or k >= n - 1:
        return None, None
        
    pivot_x, pivot_y = chain[k]
    
    # Head is 0..k, Tail is k+1..n-1
    # We rotate the tail around the pivot.
    
    # Pre-compute head positions for collision check
    head_positions = set(chain[:k+1])
    
    new_chain = chain[:k+1] # Copy head
    
    # Calculate new tail positions
    # Rotation matrices:
    # 90 deg: (x, y) -> (-y, x) relative to pivot
    # 180 deg: (x, y) -> (-x, -y)
    # 270 deg: (x, y) -> (y, -x)
    
    tail_points = []
    for i in range(k+1, n):
        px, py = chain[i]
        dx = px - pivot_x
        dy = py - pivot_y
        
        if rot == 1: # 90
            nx = -dy
            ny = dx
        elif rot == 2: # 180
            nx = -dx
            ny = -dy
        else: # 270
            nx = dy
            ny = -dx
            
        new_pos = (pivot_x + nx, pivot_y + ny)
        tail_points.append(new_pos)
        
    # Check collisions
    # The new tail points must not collide with the head.
    # Also, the new tail points must not collide with each other?
    # No, rigid rotation preserves internal distances, so no internal collisions.
    # But we must ensure the new tail points do not overlap with the head.
    
    # Optimization: Check if any new_pos is in head_positions.
    # Also, we must ensure the new tail does not "cross" the head in a way that breaks unit steps?
    # The step from k to k+1 must be a unit step.
    # The step from k+1 to k+2 is preserved.
    # So we only need to check:
    # 1. new_tail[0] is not in head_positions.
    # 2. All new_tail points are not in head_positions.
    # 3. (Implicitly) The unit step k->k+1 is preserved?
    #    Distance between pivot and new_tail[0] is same as old distance, which is 1.
    #    So yes.
    
    for pos in tail_points:
        if pos in head_positions:
            return None, None
            
    new_chain.extend(tail_points)
    
    # Calculate energy
    # We can calculate full energy
    e = _energy(seq, new_chain)
    
    return new_chain, e

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    best_chain = None
    best_energy = 0 # 0 is no contacts (energy 0). We want negative energy.
    # Actually, best_energy should be initialized to a very high value (bad)
    # or we can just track the chain with lowest energy.
    # Let's initialize with the snake chain.
    
    init_chain = _generate_snake(n)
    init_energy = _energy(sequence, init_chain)
    
    best_chain = init_chain
    best_energy = init_energy
    
    # Perform local search with restarts
    # We will do several SA runs
    
    num_restarts = 10
    # Dynamic restarts: if we have time, do more
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Generate a new initial chain (perturbed snake or random growth)
        # For simplicity, let's use the snake for the first restart,
        # and for subsequent restarts, we can perturb the best chain found so far
        # or just use the snake again but with a different random seed for the search.
        
        # Actually, restarting from the same snake might lead to the same local minima.
        # Let's use a "Random Greedy Growth" for initialization in later restarts.
        
        if restart == 0:
            current_chain = _generate_snake(n)
        else:
            # Try to generate a different compact chain
            # Use a random walk with bias towards compactness
            current_chain = _random_grow(sequence, n)
            if current_chain is None or len(current_chain) < n:
                current_chain = _generate_snake(n)
                
        current_energy = _energy(sequence, current_chain)
        
        # Simulated Annealing
        # T_start, T_end
        T_start = 1.0
        T_end = 0.01
        steps = 5000 # Adjust based on time
        
        # Check time per restart
        # We want to finish all restarts before deadline.
        # Let's allocate time per restart.
        
        t0 = time.time()
        max_time_per_restart = 10.0 # seconds
        
        # SA Loop
        T = T_start
        cooling = (T_start - T_end) / steps if steps > 0 else 0
        
        for step in range(steps):
            if time.time() - t0 > max_time_per_restart:
                break
                
            # Pick pivot
            k = random.randint(1, n - 2)
            rot = random.randint(1, 3)
            
            new_chain, new_energy = _try_pivot_move(current_chain, sequence, k, rot)
            
            if new_chain is None:
                # Invalid move, skip
                # T still cools? Or stay same?
                # Usually T cools regardless.
                T -= cooling
                continue
                
            delta_e = new_energy - current_energy
            
            if delta_e < 0:
                # Improvement
                current_chain = new_chain
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = current_chain
            else:
                # Accept with probability exp(-delta_e / T)
                # delta_e is positive (energy increase, worse)
                # We want to minimize energy, so energy is negative.
                # delta_e > 0 means new energy is less negative (worse).
                # Prob = exp(-delta_e / T)
                if T > 0:
                    prob = math.exp(-delta_e / T)
                    if random.random() < prob:
                        current_chain = new_chain
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = current_chain
                
            T -= cooling
            if T < T_end:
                T = T_end
                
    return [list(p) for p in best_chain]

def _random_grow(sequence, n):
    """Grow a chain using random moves with dead-end avoidance."""
    if n == 0:
        return []
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for i in range(1, n):
        last_pos = chain[-1]
        prev_pos = chain[-2] if i > 1 else None
        
        valid_moves = []
        for dx, dy in directions:
            # Avoid backtracking
            if i > 1:
                if (dx, dy) == (-prev_pos[0] + last_pos[0], -prev_pos[1] + last_pos[1]):
                    continue
            
            new_pos = (last_pos[0] + dx, last_pos[1] + dy)
            if new_pos in occupied:
                continue
            
            # Dead end check: does new_pos have at least one free neighbor (excluding last_pos)?
            next_moves = 0
            for ndx, ndy in directions:
                if (ndx, ndy) == (-dx, -dy):
                    continue
                next_pos = (new_pos[0] + ndx, new_pos[1] + ndy)
                if next_pos not in occupied:
                    next_moves += 1
            
            if next_moves > 0:
                valid_moves.append(new_pos)
        
        if not valid_moves:
            return None # Failed to grow
        
        # Pick randomly
        new_pos = random.choice(valid_moves)
        chain.append(new_pos)
        occupied[new_pos] = i
        
    return chain
