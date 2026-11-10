import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = 580.0  # safety margin before 630s kill

    p = 7
    k = 3  # alpha^2 = 3
    total_points = p ** 4  # 2401
    num_dirs = p * p + 1  # 50 directions (1 for (0,1), 49 for (1,m))

    def point_flat(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    # Precompute line_other: for each direction d and point x,
    # the 48 other points on the line through x in direction d
    line_other = np.zeros((num_dirs, total_points, p * p - 1), dtype=np.int32)

    # Direction 0: v = (0, 1)
    d = 0
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    x_idx = point_flat(xa, xb, ya, yb)
                    t_idx = 0
                    for ta in range(p):
                        for tb in range(p):
                            if ta == 0 and tb == 0:
                                continue
                            nya = (ya + ta) % p
                            nyb = (yb + tb) % p
                            line_other[d, x_idx, t_idx] = point_flat(xa, xb, nya, nyb)
                            t_idx += 1

    # Directions 1..49: v = (1, m) for m in F_49
    d = 1
    for ma in range(p):
        for mb in range(p):
            for xa in range(p):
                for xb in range(p):
                    for ya in range(p):
                        for yb in range(p):
                            x_idx = point_flat(xa, xb, ya, yb)
                            t_idx = 0
                            for ta in range(p):
                                for tb in range(p):
                                    if ta == 0 and tb == 0:
                                        continue
                                    nxa = (xa + ta) % p
                                    nxb = (xb + tb) % p
                                    tma = (ta * ma + tb * mb * k) % p
                                    tmb = (ta * mb + tb * ma) % p
                                    nya = (ya + tma) % p
                                    nyb = (yb + tmb) % p
                                    line_other[d, x_idx, t_idx] = point_flat(nxa, nxb, nya, nyb)
                                    t_idx += 1
            d += 1

    # Precompute for each point, the list of (direction, position) pairs
    # where that point appears as a "other" point on some line
    # This helps with efficient constraint tracking
    # point_line_refs[point_idx] = list of (d, x_idx, t_idx) where point is at line_other[d][x_idx][t_idx]
    point_line_refs = [[] for _ in range(total_points)]
    for d in range(num_dirs):
        for x_idx in range(total_points):
            for t_idx in range(p * p - 1):
                pt = line_other[d, x_idx, t_idx]
                point_line_refs[pt].append((d, x_idx, t_idx))

    # Precompute: for each point x, the set of (d, x_idx) pairs it belongs to
    # as the center point of a line
    # Actually, every point is the center of exactly one line per direction
    # So for point x_idx, it's the center for all 50 directions

    # Fast verification using vectorized operations
    def count_violations(in_S):
        """Count the number of (x, d) pairs where x is NOT covered by direction d.
        A point x is covered by direction d if all 48 other points on L(d,x) are in S.
        """
        # in_S[line_other[d]] has shape (2401, 48)
        # all_in[d][x] = True if all 48 are in S
        all_in = np.all(in_S[line_other], axis=2)  # shape (50, 2401)
        # A point x is valid if ANY direction covers it
        # Violation: point x where NO direction covers it
        valid_points = np.any(all_in, axis=0)  # shape (2401,)
        return np.sum(~valid_points)

    def is_valid(in_S):
        return count_violations(in_S) == 0

    # Start with the Hermitian unital construction
    # U = {(xa, xb, ya, yb) : 2*xa + ya^2 - 3*yb^2 = 0 mod 7}
    # S = {points in U where ya is a square mod 7}
    # Squares mod 7: {0, 1, 2, 4}
    in_S = np.zeros(total_points, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - k * yb * yb) % p == 0:
                        if ya in {0, 1, 2, 4}:
                            in_S[point_flat(xa, xb, ya, yb)] = True

    initial_S_size = np.sum(in_S)

    # Verify initial construction
    if not is_valid(in_S):
        # Fallback: return all points as N (S is empty)
        points = []
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)

    best_in_S = in_S.copy()
    best_S_size = initial_S_size
    best_violations = 0

    # WalkSAT-style local search
    # State: in_S (boolean array)
    # Objective: maximize |S| subject to zero violations
    # Strategy: 
    # 1. Try to add random points to S, keep if valid
    # 2. If stuck, try removing points from S to enable adding more
    # 3. Use simulated annealing to escape local optima

    np.random.seed(42)
    
    # Phase 1: Greedy addition with random restarts
    for restart in range(30):
        if time.time() - start_time > deadline:
            break
        
        current_in_S = in_S.copy()
        current_S_size = initial_S_size
        
        # Create random order of all points
        all_point_indices = np.arange(total_points)
        np.random.shuffle(all_point_indices)
        
        # Try to add each point to S
        for p_idx in all_point_indices:
            if time.time() - start_time > deadline:
                break
            
            if current_in_S[p_idx]:
                continue
            
            # Try adding p_idx to S
            current_in_S[p_idx] = True
            if is_valid(current_in_S):
                current_S_size += 1
            else:
                current_in_S[p_idx] = False
        
        if current_S_size > best_S_size:
            best_S_size = current_S_size
            best_in_S = current_in_S.copy()

    # Phase 2: Simulated annealing with swaps
    # Try to find better configurations by swapping points in/out of S
    if time.time() - start_time < deadline - 60:
        current_in_S = best_in_S.copy()
        current_S_size = best_S_size
        temperature = 1.0
        cooling_rate = 0.995
        
        for step in range(20000):
            if time.time() - start_time > deadline:
                break
            
            if step % 100 == 0:
                temperature *= cooling_rate
                if temperature < 0.01:
                    break
            
            # Choose a random point to flip
            flip_idx = np.random.randint(total_points)
            
            # Try flipping this point
            old_state = current_in_S[flip_idx]
            current_in_S[flip_idx] = not old_state
            
            new_S_size = np.sum(current_in_S)
            new_violations = count_violations(current_in_S)
            
            # Calculate energy change
            # We want to maximize S_size and minimize violations
            # Energy = -S_size * 100 + violations * 1000
            old_energy = -current_S_size * 100 + best_violations * 1000
            new_energy = -new_S_size * 100 + new_violations * 1000
            delta_energy = new_energy - old_energy
            
            # Accept if better or with probability exp(-delta/T)
            if delta_energy < 0 or np.random.random() < np.exp(-delta_energy / temperature):
                current_S_size = new_S_size
                best_violations = new_violations
                
                if new_violations == 0 and new_S_size > best_S_size:
                    best_S_size = new_S_size
                    best_in_S = current_in_S.copy()
            else:
                # Revert
                current_in_S[flip_idx] = old_state

    # Phase 3: Try to add more points using constraint propagation
    # For each point not in S, check if adding it would cause violations
    # If yes, identify which constraints are violated and try to fix them
    if time.time() - start_time < deadline - 30:
        current_in_S = best_in_S.copy()
        current_S_size = best_S_size
        
        # Get all points not in S
        not_in_S = np.where(~current_in_S)[0]
        np.random.shuffle(not_in_S)
        
        for p_idx in not_in_S:
            if time.time() - start_time > deadline:
                break
            
            # Try adding this point
            current_in_S[p_idx] = True
            violations = count_violations(current_in_S)
            
            if violations == 0:
                current_S_size += 1
                if current_S_size > best_S_size:
                    best_S_size = current_S_size
                    best_in_S = current_in_S.copy()
            else:
                # Revert
                current_in_S[p_idx] = False

    # Build output from best_in_S
    in_N_final = ~best_in_S
    
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if not best_in_S[point_flat(xa, xb, ya, yb)]:
                        points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)
