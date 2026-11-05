"""
HP Lattice Protein Folding using Simulated Annealing.
Strategy: Instead of one-shot greedy growth, we use a robust local search method
(Simulated Annealing) that can escape local minima by accepting worse solutions
with a decaying probability. This is a fundamentally different design family
from the current randomized greedy construction.
"""
# EVOLVE-BLOCK-START
import random
import time
import math

def _neighbours(point):
    """Generate the 4 neighbors of a point on a 2D square lattice."""
    x, y = point
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

def _energy(sequence, chain):
    """
    Compute the energy of a conformation.
    Energy = - (number of H-H contacts between non-adjacent residues).
    """
    n = len(chain)
    if n == 0:
        return 0
    
    # Map position to index for O(1) lookup
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            # Check 4 neighbors
            for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                p = (nx, ny)
                j = pos_to_idx.get(p)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _generate_initial_chain(n):
    """
    Generate a random self-avoiding walk of length n.
    Uses a simple random walk with backtracking if stuck.
    """
    if n == 0:
        return []
    
    # Start at origin
    chain = [(0, 0)]
    visited = {(0, 0)}
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for _ in range(n - 1):
        # Shuffle directions to randomize
        random.shuffle(directions)
        moved = False
        for dx, dy in directions:
            last_x, last_y = chain[-1]
            nx, ny = last_x + dx, last_y + dy
            if (nx, ny) not in visited:
                chain.append((nx, ny))
                visited.add((nx, ny))
                moved = True
                break
        if not moved:
            # Backtrack: this simple method might get stuck.
            # For robustness in SA initialization, we can just return what we have
            # or try a different start. But for SA, we usually need a full chain.
            # A better init: Use a snake-like pattern if random walk fails.
            # However, random walks on 2D lattice are recurrent and might trap.
            # Let's use a more robust init: a random "spiral" or just a straight line if stuck.
            # Actually, for short sequences (<=100), random walk usually works if we restart.
            # But we need to return a valid chain of length n.
            # If stuck, we can't easily extend. 
            # Alternative: Use a deterministic "snake" fill for init to guarantee length.
            # Let's fall back to a simple snake if we get stuck.
            # But we are already in the middle. 
            # Let's just restart the whole generation if we get stuck.
            return _generate_initial_chain(n)
    return chain

def _apply_move(chain, move_type, rng):
    """
    Apply a random local move to the chain.
    Returns (new_chain, success)
    Move types:
    0: Pivot (rotate a subchain around a bond)
    1: Internal Move (move a subchain to a new location)
    2: Reptation (move the end of the chain)
    """
    n = len(chain)
    if n <= 1:
        return chain, False
    
    # Build visited set for collision check (excluding the moving segment)
    # This is O(N) per move, which is acceptable for N<=100.
    
    if move_type == 0: # Pivot
        if n < 3:
            return chain, False
        # Pick a pivot point i (1 to n-2)
        i = rng.randint(1, n - 2)
        # Pick a rotation angle (90, 180, 270 degrees)
        # Directions: 0:Right, 1:Up, 2:Left, 3:Down
        # We need to rotate the vector from chain[i] to chain[i+1]
        p0 = chain[i]
        p1 = chain[i+1]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        
        # Rotate (dx, dy)
        if rng.randint(0, 2) == 0:
            # 90 deg CCW: (x, y) -> (-y, x)
            ndx, ndy = -dy, dx
        elif rng.randint(0, 1) == 0:
            # 180 deg: (x, y) -> (-x, -y)
            ndx, ndy = -dx, -dy
        else:
            # 90 deg CW: (x, y) -> (y, -x)
            ndx, ndy = dy, -dx
        
        new_dx = ndx
        new_dy = ndy
        
        # Generate new segment
        new_segment = []
        cx, cy = p0[0], p0[1]
        # The original segment is chain[i+1:]
        # We need to map the relative coordinates
        # Original relative coords:
        # rel[0] = (0,0) is p1? No, p1 is the first point after pivot.
        # Let's store relative coords of the tail
        tail = chain[i+1:]
        rel_tail = [(tx - p1[0], ty - p1[1]) for tx, ty in tail]
        
        # Apply rotation to relative coords? 
        # Actually, it's easier to just rotate the entire tail around p0.
        # But rotation is not just scaling.
        # Let's just rotate the first step and then translate the rest?
        # No, pivot rotation rotates the entire subchain.
        
        # Let's construct the new tail by rotating each point around p0.
        # Rotation matrix for 90 CCW: [0 -1; 1 0]
        # 180: [-1 0; 0 -1]
        # 90 CW: [0 1; -1 0]
        
        def rot_90_ccw(x, y): return (-y, x)
        def rot_180(x, y): return (-x, -y)
        def rot_90_cw(x, y): return (y, -x)
        
        if ndx == -dy and ndy == dx: # 90 CCW
            rot_func = rot_90_ccw
        elif ndx == -dx and ndy == -dy: # 180
            rot_func = rot_180
        else: # 90 CW
            rot_func = rot_90_cw
            
        new_tail = []
        for tx, ty in tail:
            rx, ry = tx - p0[0], ty - p0[1]
            nrx, nry = rot_func(rx, ry)
            new_tail.append((p0[0] + nrx, p0[1] + nry))
            
        new_chain = chain[:i+1] + new_tail
        new_visited = set(new_chain)
        if len(new_visited) != len(new_chain):
            return chain, False
        return new_chain, True
        
    elif move_type == 1: # Internal Move
        if n < 3:
            return chain, False
        # Pick start and end of subchain (i, j) with i > 0 and j < n-1
        i = rng.randint(1, n - 2)
        j = rng.randint(i + 1, n - 2)
        
        subchain = chain[i:j+1]
        # Pick a random shift (dx, dy) from a small set
        # Usually shifts are small. Let's try a random neighbor of the current start or end?
        # Or just a random integer in [-2, 2]
        dx = rng.randint(-2, 2)
        dy = rng.randint(-2, 2)
        if dx == 0 and dy == 0:
            return chain, False
            
        new_subchain = [(x + dx, y + dy) for x, y in subchain]
        
        # Check collision with the rest of the chain (excluding the subchain itself)
        # The rest is chain[:i] + chain[j+1:]
        rest = chain[:i] + chain[j+1:]
        rest_set = set(rest)
        
        # The new subchain must not intersect the rest.
        # Also, the connection points must be valid?
        # The move is valid if the new subchain doesn't collide with the rest.
        # The connections (chain[i-1] to new_subchain[0]) and (new_subchain[-1] to chain[j+1])
        # are NOT required to be contiguous in the lattice in the standard definition of "move"
        # wait, NO. The chain must remain a valid self-avoiding walk.
        # So new_subchain[0] MUST be a neighbor of chain[i-1].
        # And new_subchain[-1] MUST be a neighbor of chain[j+1].
        
        p_before = chain[i-1]
        p_after = chain[j+1]
        
        if p_before not in [(nx+dx, ny+dy) for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)] for x, y in [p_before]]:
             # This check is wrong. 
             # p_before is fixed. new_start = subchain[0] + (dx, dy).
             # new_start must be neighbor of p_before.
             new_start = (subchain[0][0] + dx, subchain[0][1] + dy)
             new_end = (subchain[-1][0] + dx, subchain[-1][1] + dy)
             
             if new_start not in _neighbours(p_before):
                 return chain, False
             if new_end not in _neighbours(p_after):
                 return chain, False
                 
             # Check collision
             for p in new_subchain:
                 if p in rest_set:
                     return chain, False
             
             new_chain = chain[:i] + new_subchain + chain[j+1:]
             # Final self-avoidance check (just in case)
             if len(set(new_chain)) != len(new_chain):
                 return chain, False
             return new_chain, True
             
    elif move_type == 2: # Reptation
        # Move the last residue to a neighbor of the second-to-last residue
        if n < 2:
            return chain, False
        p_second_last = chain[-2]
        neighbors = _neighbours(p_second_last)
        # Filter out the current last position
        current_last = chain[-1]
        valid_neighbors = [p for p in neighbors if p != current_last]
        if not valid_neighbors:
            return chain, False
        new_last = valid_neighbors[rng.randint(0, len(valid_neighbors)-1)]
        
        # Check if new_last is occupied by any other part of the chain (excluding last)
        chain_prefix = chain[:-1]
        if new_last in chain_prefix:
            return chain, False
            
        new_chain = chain_prefix + [new_last]
        return new_chain, True
        
    return chain, False

def fold(sequence, dimension):
    """
    Fold the sequence using Simulated Annealing.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Time budget: 210s total for all sequences.
    # We don't know how many sequences are coming, but the contract says 
    # "fold is called once per benchmark sequence... 120s CPU budget for all sequences together"
    # Wait, the prompt says "120 s CPU budget for all sequences together" in the task description,
    # but the directive says "evaluator KILLS the program after 210s wall-clock".
    # And "Keep an EXPLICIT deadline guard".
    # Since fold is called per sequence, we should estimate time per sequence.
    # If there are 10 sequences, we have ~12s per sequence.
    # Let's use a global deadline if possible, but since we can't easily share state 
    # between calls in a simple function without globals, let's use a heuristic time limit per call.
    # However, the prompt says "Spare time does not establish a bottleneck...".
    # Let's assume we have ~10-15 seconds per sequence to be safe.
    
    start_time = time.time()
    time_limit = 10.0 # seconds per sequence
    
    # Initialize
    current_chain = _generate_initial_chain(n)
    if len(current_chain) < n:
        # Fallback to straight line if init fails
        current_chain = [(i, 0) for i in range(n)]
        
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain
    best_energy = current_energy
    
    # SA Parameters
    T = 1.0 # Initial temperature
    T_min = 0.01
    alpha = 0.99 # Cooling rate
    # Estimate steps: we want to run until time limit
    # Rough estimate: 10000-50000 steps might be feasible in 10s for N=100?
    # Let's just loop until time limit.
    
    rng = random.Random(42) # Fixed seed for reproducibility? 
    # Actually, randomizing might help escape specific bad local minima. 
    # But fixed seed ensures deterministic behavior for the evaluator if it runs twice.
    # Let's use a seed based on sequence hash for variety but determinism.
    seed_val = sum(ord(c) for c in sequence)
    rng = random.Random(seed_val)
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
            
        # Pick a move
        move_type = rng.randint(0, 2)
        new_chain, success = _apply_move(current_chain, move_type, rng)
        
        if not success:
            # Try a few more times? Or just skip?
            # Skipping is fine, it just wastes a bit of time.
            # To be efficient, if we fail many times, we might want to change strategy.
            # But for now, just continue.
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Acceptance criterion
        if delta_e < 0:
            # Better solution
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain
        else:
            # Worse solution, accept with probability exp(-delta_e / T)
            if T > T_min:
                prob = math.exp(-delta_e / T)
                if rng.random() < prob:
                    current_chain = new_chain
                    current_energy = new_energy
        
        # Cool down
        T *= alpha
        
    # Return best chain found
    # Ensure format: list of lists of integers
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
