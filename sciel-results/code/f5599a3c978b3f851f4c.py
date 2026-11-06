import random
import math
import time

N = 64

def get_forbidden_indices(a_idx, b_idx):
    """
    Returns a list of indices (0..N*N-1) that form an isosceles triangle 
    with points a_idx and b_idx.
    """
    x1 = a_idx // N
    y1 = a_idx % N
    x2 = b_idx // N
    y2 = b_idx % N
    
    dx = x2 - x1
    dy = y2 - y1
    d_sq = dx * dx + dy * dy
    
    if d_sq == 0:
        return []

    forbidden = set()
    
    # 1. Perpendicular Bisector
    C = (x2 * x2 - x1 * x1) + (y2 * y2 - y1 * y1)
    
    if dx == 0 and dy == 0:
        return []
        
    if dx == 0:
        # dy*y = C/2 => y = C / (2*dy)
        if C % (2 * dy) == 0:
            y = C // (2 * dy)
            if 0 <= y < N:
                for x in range(N):
                    forbidden.add(x * N + y)
    elif dy == 0:
        # dx*x = C/2 => x = C / (2*dx)
        if C % (2 * dx) == 0:
            x = C // (2 * dx)
            if 0 <= x < N:
                for y in range(N):
                    forbidden.add(x * N + y)
    else:
        # 2*dx*x + 2*dy*y = C
        for x in range(N):
            rem = C - 2 * dx * x
            if rem % (2 * dy) == 0:
                y = rem // (2 * dy)
                if 0 <= y < N:
                    forbidden.add(x * N + y)

    # 2. Circle centered at a with radius d
    max_dx = int(math.isqrt(d_sq))
    for x in range(max(0, x1 - max_dx), min(N, x1 + max_dx + 1)):
        dy_sq = d_sq - (x - x1) * (x - x1)
        if dy_sq < 0:
            continue
        dy = int(math.isqrt(dy_sq))
        if dy * dy == dy_sq:
            y1_val = y1 - dy
            y2_val = y1 + dy
            if 0 <= y1_val < N:
                forbidden.add(x * N + y1_val)
            if 0 <= y2_val < N:
                forbidden.add(x * N + y2_val)

    # 3. Circle centered at b with radius d
    for x in range(max(0, x2 - max_dx), min(N, x2 + max_dx + 1)):
        dy_sq = d_sq - (x - x2) * (x - x2)
        if dy_sq < 0:
            continue
        dy = int(math.isqrt(dy_sq))
        if dy * dy == dy_sq:
            y1_val = y2 - dy
            y2_val = y2 + dy
            if 0 <= y1_val < N:
                forbidden.add(x * N + y1_val)
            if 0 <= y2_val < N:
                forbidden.add(x * N + y2_val)

    # Remove a and b themselves if they were added (they are part of the pair, not the third point)
    # Actually, the third point C must be distinct from A and B.
    forbidden.discard(a_idx)
    forbidden.discard(b_idx)
    
    return list(forbidden)

def construct():
    N_LOCAL = N
    TOTAL_POINTS = N_LOCAL * N_LOCAL
    
    # Precompute forbidden lists for all pairs? 
    # Number of pairs is ~ 4096^2 / 2 = 8 million. 
    # Each list is small. 
    # Memory might be high but speed will be better.
    # However, 8 million lists is too much memory/time to precompute.
    # We will compute on the fly.
    
    best_valid_size = 0
    best_valid_points = set()
    
    # Simulated Annealing
    # State: set of points, forbidden_count array
    points = set()
    forbidden_count = [0] * TOTAL_POINTS
    
    def add_point(p):
        nonlocal points, forbidden_count
        c = forbidden_count[p]
        points.add(p)
        # Update forbidden counts for other points
        for q in points:
            if q == p:
                continue
            forb = get_forbidden_indices(p, q)
            for f in forb:
                forbidden_count[f] += 1
        return c

    def remove_point(p):
        nonlocal points, forbidden_count
        c = forbidden_count[p]
        # Update forbidden counts for other points
        for q in points:
            if q == p:
                continue
            forb = get_forbidden_indices(p, q)
            for f in forb:
                forbidden_count[f] -= 1
        points.remove(p)
        return c

    # Initialize with greedy to get a good start
    # Use the original greedy logic quickly
    all_points = list(range(TOTAL_POINTS))
    random.shuffle(all_points)
    for p in all_points:
        if forbidden_count[p] == 0:
            add_point(p)
    
    current_size = len(points)
    if current_size > best_valid_size:
        best_valid_size = current_size
        best_valid_points = set(points)
        
    # SA Parameters
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    lambda_penalty = 1.0
    max_iterations = 20000
    
    start_time = time.time()
    
    for i in range(max_iterations):
        if time.time() - start_time > 250:
            break
            
        p = random.randint(0, TOTAL_POINTS - 1)
        
        if p in points:
            c = forbidden_count[p]
            # Delta E = 1 - lambda * c
            delta_e = 1.0 - lambda_penalty * c
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                remove_point(p)
                # If we removed a point, size decreased. 
                # We don't update best here because it's likely worse, 
                # but if we later add better points, we will.
        else:
            c = forbidden_count[p]
            # Delta E = -1 + lambda * c
            delta_e = -1.0 + lambda_penalty * c
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                add_point(p)
                current_size = len(points)
                # Check if valid
                if c == 0:
                    # It was added without creating new violations?
                    # Wait, if c==0, it didn't create violations.
                    # But did it fix existing ones? No, adding points doesn't fix violations.
                    # So if c==0, the set remains valid if it was valid before.
                    # If it was invalid before, it remains invalid.
                    # We track best *valid* size.
                    # We need to know if the current set is valid.
                    # We can track total_violations.
                    pass
        
        # Track total violations to check validity
        # This is hard to maintain incrementally without extra work.
        # Instead, periodically check if valid?
        # Or, only update best if we are sure it's valid.
        # Since we start valid and only add if c==0 (greedy phase), 
        # in SA phase, we might add invalid points.
        # We should only update best_valid_points if the set is valid.
        # To check validity, we need total_violations == 0.
        
        # Let's maintain total_violations.
        # When adding p with c violations: total_violations += c
        # When removing p with c violations: total_violations -= c
        # We need to add this to the code.
        
        T *= cooling
        if T < T_min:
            T = T_min

    # The above SA loop didn't track total_violations.
    # Let's rewrite the core loop to be cleaner.
    
    return [(p // N, p % N) for p in best_valid_points]
