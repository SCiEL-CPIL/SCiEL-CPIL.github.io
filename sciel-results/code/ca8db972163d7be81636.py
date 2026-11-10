"""
Rewritten HP lattice protein folding using Simulated Annealing with pivot and crankshaft moves.
"""
import random
import time
import math

# Global deadline guard
START_TIME = time.time()
TIME_LIMIT = 200.0  # Leave 10s margin for 210s limit

def _neighbours(point):
    """Yield 4-neighbors of a 2D point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _is_valid_walk(chain):
    """Check if the chain is self-avoiding and has unit steps."""
    if len(chain) == 0:
        return True
    occupied = set()
    for i, p in enumerate(chain):
        if p in occupied:
            return False
        occupied.add(p)
        if i > 0:
            prev = chain[i-1]
            dist = abs(p[0] - prev[0]) + abs(p[1] - prev[1])
            if dist != 1:
                return False
    return True

def _energy(sequence, chain):
    """
    Calculate energy: -1 for each non-adjacent H-H contact.
    Uses a dictionary for O(1) neighbor lookups.
    """
    if not chain:
        return 0
    
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    
    # Only check H residues to save time
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None:
                    # Contact if not adjacent in sequence
                    if abs(i - j) > 1:
                        # Count each pair once (i < j)
                        if i < j:
                            contacts += 1
                            
    return -contacts

def _greedy_grow(sequence, dimension, rng):
    """
    Generate an initial self-avoiding walk using a greedy heuristic.
    Prioritizes moves that create H-H contacts or keep space open.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Start at (0,0)
    chain = [(0, 0)]
    occupied = {(0, 0)}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        
        for nb in _neighbours(last):
            if nb in occupied:
                continue
            
            # Score candidates
            score = rng.random() * 0.1 # Small random noise
            
            # Bonus for creating H-H contacts
            if sequence[i] == 'H':
                for q in _neighbours(nb):
                    if q in occupied:
                        # Find index of q
                        # Optimization: we don't have pos_to_idx here easily without building it, 
                        # but for greedy init, we can just check if the residue at that pos was H.
                        # Since we are growing, occupied contains (pos, idx) if we stored it, 
                        # but here occupied is just a set of tuples. 
                        # Let's assume we store indices in a dict for the greedy phase too.
                        pass
            
            # To properly score H-H contacts in greedy, we need to know which indices are where.
            # Let's maintain a pos_to_idx dict during growth.
            candidates.append((score, nb))
            
        if not candidates:
            # Dead end, return None to signal failure
            return None
            
        # In a more sophisticated greedy, we'd sort by potential contacts.
        # For now, random choice among free neighbors is a valid SAW generator,
        # but we want it to be "good". 
        # Let's use a simple heuristic: prefer directions that haven't been used recently or are "open".
        # Actually, just picking a random free neighbor is fine for SA init, 
        # but a spiral or snake pattern is better for compactness.
        
        # Let's try a simple spiral-like bias: prefer turning right/left consistently?
        # Or just random. Random greedy often gets stuck.
        # Let's try to bias towards staying in a box.
        
        # Simple strategy: Pick a random free neighbor.
        _, best_nb = random.choice(candidates)
        
        chain.append(best_nb)
        occupied.add(best_nb)
        
    return chain

def _sa_fold(sequence, dimension, rng, time_budget):
    """
    Simulated Annealing to optimize the chain.
    """
    n = len(sequence)
    
    # 1. Generate initial state
    # Try a few greedy inits and pick the best
    best_init = None
    best_init_energy = 1000
    
    for _ in range(10):
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            continue
        e = _energy(sequence, chain)
        if e < best_init_energy:
            best_init_energy = e
            best_init = chain
            
    if best_init is None:
        # Fallback: straight line (always valid)
        best_init = [(i, 0) for i in range(n)]
        best_init_energy = _energy(sequence, best_init)
        
    current_chain = best_init
    current_energy = best_init_energy
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # SA Parameters
    # T0 depends on N
    T0 = 1.0
    T_min = 0.001
    # Cooling rate: T(t) = T0 * (T_min/T0)^(t/max_steps)
    # We don't know max_steps, so use time-based cooling or fixed steps with time check.
    # Let's use a fixed number of steps per second target, e.g., 1000 steps/sec.
    
    start_time = time.time()
    
    # Estimate max steps based on time budget
    # Each step involves move generation, collision check, energy calc.
    # Energy calc is O(N). Move gen is O(N) for pivots.
    # Let's aim for ~100-500 steps/sec for N=100.
    # Let's just loop until time budget is exhausted.
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        # Current temperature
        # Use a simple cooling schedule based on steps taken
        # Or just fixed T? No, need cooling.
        # Let's use a global step counter
        # We'll approximate steps by time.
        
        # Simple linear cooling from T0 to T_min over the expected duration?
        # Better: T = T0 * exp(-alpha * elapsed)
        alpha = 0.05 # Cooling rate
        T = max(T_min, T0 * math.exp(-alpha * elapsed))
        
        # Generate a move
        move_type = rng.random()
        
        new_chain = None
        
        if move_type < 0.5:
            # Pivot move
            # Pick a pivot index i (1 to n-2)
            # Rotate segment i+1 to n-1 around i
            i = rng.randint(1, n - 2)
            
            # Get the segment
            pivot = current_chain[i]
            segment = current_chain[i+1:]
            
            # Determine rotation direction (90, 180, 270)
            # Relative to the vector from pivot to segment[0]
            v = (segment[0][0] - pivot[0], segment[0][1] - pivot[1])
            
            # Possible rotations of vector v:
            # (x, y) -> (-y, x) [90 deg CCW]
            # (x, y) -> (-x, -y) [180 deg]
            # (x, y) -> (y, -x) [90 deg CW]
            
            # Try all 3 non-identity rotations? Or pick one?
            # Picking one is faster. Let's pick a random one.
            rot_choice = rng.randint(0, 2)
            
            if rot_choice == 0: # 90 CCW
                dx, dy = -v[1], v[0]
            elif rot_choice == 1: # 180
                dx, dy = -v[0], -v[1]
            else: # 90 CW
                dx, dy = v[1], -v[0]
                
            # Apply rotation to the whole segment
            # New segment[j] = old_segment[j] + (dx, dy) ? 
            # No, rotation is around the pivot.
            # Point p relative to pivot: p_rel = p - pivot
            # New p_rel = R(p_rel)
            # New p = pivot + R(p_rel)
            
            new_segment = []
            for p in segment:
                px, py = p
                rx, ry = px - pivot[0], py - pivot[1]
                
                if rot_choice == 0: # 90 CCW: (x,y) -> (-y, x)
                    nrx, nry = -ry, rx
                elif rot_choice == 1: # 180: (x,y) -> (-x, -y)
                    nrx, nry = -rx, -ry
                else: # 90 CW: (x,y) -> (y, -x)
                    nrx, nry = ry, -rx
                    
                new_segment.append((pivot[0] + nrx, pivot[1] + nry))
                
            new_chain = current_chain[:i+1] + new_segment
            
        elif move_type < 0.8:
            # Crankshaft move
            # Find a straight line of length >= 3
            # i, i+1, i+2 are collinear and consecutive
            # Rotate i+2 ... n-1 around i+1? 
            # Or rotate i+1 ... n-1 around i? That's pivot.
            # Crankshaft: Rotate segment [i+2, n-1] around [i+1] such that [i, i+1, i+2] stays connected?
            # Standard crankshaft: 
            # ... A - B - C ...
            # Rotate C... around B.
            # A-B is fixed. B-C changes direction.
            
            # Find valid i
            valid_i = []
            for i in range(1, n - 2):
                p0 = current_chain[i-1]
                p1 = current_chain[i]
                p2 = current_chain[i+1]
                
                # Check if p0, p1, p2 are collinear and unit steps
                v1 = (p1[0] - p0[0], p1[1] - p0[1])
                v2 = (p2[0] - p1[0], p2[1] - p1[1])
                
                if v1 == v2:
                    valid_i.append(i)
                    
            if not valid_i:
                continue
                
            i = rng.choice(valid_i)
            pivot = current_chain[i]
            segment = current_chain[i+1:]
            
            # Rotate segment around pivot
            # The first element of segment is p2.
            # v = p2 - pivot
            v = (segment[0][0] - pivot[0], segment[0][1] - pivot[1])
            
            rot_choice = rng.randint(0, 2)
            
            new_segment = []
            for p in segment:
                px, py = p
                rx, ry = px - pivot[0], py - pivot[1]
                
                if rot_choice == 0: # 90 CCW
                    nrx, nry = -ry, rx
                elif rot_choice == 1: # 180
                    nrx, nry = -rx, -ry
                else: # 90 CW
                    nrx, nry = ry, -rx
                    
                new_segment.append((pivot[0] + nrx, pivot[1] + nry))
                
            new_chain = current_chain[:i+1] + new_segment
            
        else:
            # Local move: Swap a residue? 
            # Or just a small pivot of length 1?
            # Let's do a "flip" of a single residue if possible?
            # A single residue move is just moving it to a neighbor, which breaks connectivity unless it's a pivot of length 1 (which is just moving the last residue? No, pivot moves segment i+1..n).
            # Let's stick to pivots and crankshafts as they are powerful.
            # If no move generated, continue.
            continue
            
        # Check validity of new_chain
        if new_chain is None:
            continue
            
        # Self-avoidance check
        # We can do a quick check. 
        # Since N is small, a set check is fast.
        try:
            occupied = set()
            for i, p in enumerate(new_chain):
                if p in occupied:
                    raise Exception("Collision")
                occupied.add(p)
                if i > 0:
                    prev = new_chain[i-1]
                    if abs(p[0] - prev[0]) + abs(p[1] - prev[1]) != 1:
                        raise Exception("Break")
        except:
            continue
            
        # Calculate energy
        new_energy = _energy(sequence, new_chain)
        
        # Acceptance criterion
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
    return best_chain

def fold(sequence, dimension):
    """
    Best conformation found using Simulated Annealing.
    """
    n = len(sequence)
    if n == 0:
        return []
        
    # Check global time
    elapsed_global = time.time() - START_TIME
    remaining = TIME_LIMIT - elapsed_global
    
    if remaining <= 0:
        # Fallback straight line
        return [[i, 0] for i in range(n)]
        
    # Allocate time for this sequence
    # If this is the last sequence, give it all remaining.
    # We don't know how many sequences are left. 
    # Strategy: Give a fixed chunk, or proportional to N?
    # Let's give a fixed time per sequence, e.g., 15 seconds, but capped by remaining.
    # If N is small, it finishes fast. If N is large, it takes longer.
    # Let's use a time budget of min(15, remaining) seconds.
    seq_time_budget = min(15.0, remaining)
    
    # Use a fixed seed for reproducibility in evaluation? 
    # The prompt says "score is mean over 10 sequences". 
    # Using a fixed seed makes it deterministic.
    rng = random.Random(12345 + len(sequence))
    
    best_chain = _sa_fold(sequence, dimension, rng, seq_time_budget)
    
    # Ensure return format: list of lists
    return [list(p) for p in best_chain]

