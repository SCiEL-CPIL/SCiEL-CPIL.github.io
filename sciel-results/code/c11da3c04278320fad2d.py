"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def _precompute_dist2():
    """Precompute squared distances between all pairs of grid points."""
    points = list(itertools.product(range(N), repeat=2))
    V = len(points)
    dist2 = np.zeros((V, V), dtype=np.int64)
    for i in range(V):
        xi, yi = points[i]
        for j in range(i + 1, V):
            xj, yj = points[j]
            d = (xi - xj) ** 2 + (yi - yj) ** 2
            dist2[i, j] = d
            dist2[j, i] = d
    return points, dist2

# Precompute at module load time
POINTS, DIST2 = _precompute_dist2()
V = len(POINTS)

def _is_valid_no_isosceles(S_indices):
    """Check if the set contains no isosceles triangles."""
    S_list = list(S_indices)
    n = len(S_list)
    if n < 3:
        return True
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            d_ab = DIST2[a, b]
            for k in range(j + 1, n):
                c = S_list[k]
                d_ac = DIST2[a, c]
                d_bc = DIST2[b, c]
                if d_ab == d_ac or d_ab == d_bc or d_ac == d_bc:
                    return False
    return True

def _get_unblocked_points(S_indices):
    """
    Returns a list of points u not in S_indices that are NOT blocked by S_indices.
    u is blocked if there exist a, b in S_indices such that (u, a, b) is isosceles.
    """
    S_set = set(S_indices)
    unblocked = []
    S_list = list(S_indices)
    n = len(S_list)
    
    for u in range(V):
        if u in S_set:
            continue
        
        is_blocked = False
        if n >= 2:
            # Check if u forms isosceles with any pair in S
            # Optimization: Group points in S by distance to u?
            # d(u, a) == d(u, b) -> isosceles
            # d(u, a) == d(a, b) -> isosceles
            # d(u, b) == d(a, b) -> isosceles
            
            # We can just check all pairs
            for i in range(n):
                a = S_list[i]
                d_ua = DIST2[u, a]
                for j in range(i + 1, n):
                    b = S_list[j]
                    d_ub = DIST2[u, b]
                    d_ab = DIST2[a, b]
                    if d_ua == d_ub or d_ua == d_ab or d_ub == d_ab:
                        is_blocked = True
                        break
                if is_blocked:
                    break
        
        if not is_blocked:
            unblocked.append(u)
            
    return unblocked

def construct():
    start_time = time.time()
    deadline = start_time + 290.0 # 10s buffer
    
    best_size = V
    best_set = []
    
    # Initialize with a greedy maximal set
    # Greedy: pick points that block the most others? Or just random maximal?
    # Let's use a fast greedy to get a valid maximal set start point.
    def get_initial_maximal_set(seed):
        random.seed(seed)
        S = []
        S_set = set()
        # Shuffle points
        candidates = list(range(V))
        random.shuffle(candidates)
        
        for p in candidates:
            # Check if p can be added to S without creating isosceles
            can_add = True
            if len(S) >= 2:
                # Check against all pairs in S
                for i in range(len(S)):
                    a = S[i]
                    d_pa = DIST2[p, a]
                    for j in range(i + 1, len(S)):
                        b = S[j]
                        d_pb = DIST2[p, b]
                        d_ab = DIST2[a, b]
                        if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                            can_add = False
                            break
                    if not can_add:
                        break
            
            if can_add:
                S.append(p)
                S_set.add(p)
        
        # Now S is a maximal no-isosceles set?
        # Greedy construction by adding all possible points in random order results in a maximal set.
        # Why? If a point p was skipped, it was because it formed an isosceles with existing points.
        # Those existing points are in the final set. So p is still blocked.
        # Thus, no point can be added.
        return S

    # Simulated Annealing
    # State: S (list of indices)
    # Energy: len(S) + LAMBDA * len(unblocked(S))
    # We want to minimize Energy.
    # If unblocked(S) is empty, S is maximal.
    
    LAMBDA = 100.0 # High penalty for non-maximality
    
    def calculate_energy(S_list):
        unblocked = _get_unblocked_points(S_list)
        return len(S_list) + LAMBDA * len(unblocked)
    
    # Run SA for multiple restarts
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        current_S = get_initial_maximal_set(restart * 12345)
        current_energy = calculate_energy(current_S)
        
        if len(current_S) < best_size and len(_get_unblocked_points(current_S)) == 0:
            best_size = len(current_S)
            best_set = current_S[:]
            if best_size <= 8:
                break
        
        # SA Parameters
        T_start = 5.0
        T_end = 0.01
        cooling_rate = 0.99
        T = T_start
        
        # Number of steps per temperature
        steps_per_temp = 200
        
        while T > T_end:
            for _ in range(steps_per_temp):
                if time.time() > deadline:
                    break
                
                # Generate a neighbor
                # Move types: Add, Remove
                if random.random() < 0.5 and len(current_S) < V:
                    # Add a random point not in S
                    candidates = [i for i in range(V) if i not in set(current_S)]
                    if not candidates:
                        continue
                    p = random.choice(candidates)
                    new_S = current_S + [p]
                else:
                    # Remove a random point from S
                    if not current_S:
                        continue
                    p = random.choice(current_S)
                    new_S = [x for x in current_S if x != p]
                
                # Check validity (no isosceles)
                if not _is_valid_no_isosceles(new_S):
                    # Reject invalid states immediately (or high penalty)
                    # In SA, we can accept invalid states with penalty, but it's often better to stay valid
                    # Let's reject invalid states to keep search space clean
                    continue
                
                new_energy = calculate_energy(new_S)
                delta_e = new_energy - current_energy
                
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                    current_S = new_S
                    current_energy = new_energy
                    
                    # Update best if maximal
                    if len(_get_unblocked_points(current_S)) == 0:
                        if len(current_S) < best_size:
                            best_size = len(current_S)
                            best_set = current_S[:]
                            if best_size <= 8:
                                break
            
            T *= cooling_rate
            
            if best_size <= 8:
                break
            
            if time.time() > deadline:
                break

    return [POINTS[i] for i in best_set]

# EVOLVE-BLOCK-END
