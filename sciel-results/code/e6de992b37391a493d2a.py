import time
import random
import numpy as np

N = 64

def construct():
    deadline = time.time() + 280.0
    rng = random.Random(42)
    
    # Precompute all grid points
    all_points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    all_indices = np.arange(N*N)
    
    # Precompute pairwise squared distances matrix (N*N x N*N)
    # This is 4096x4096, which is ~16M floats. 128MB. Acceptable.
    # D[i, j] = dist2(point_i, point_j)
    # We can compute this efficiently.
    px = all_points[:, 0].astype(np.float32)
    py = all_points[:, 1].astype(np.float32)
    
    # Distance matrix calculation
    # dx = px[:, None] - px[None, :]
    # dy = py[:, None] - py[None, :]
    # D = dx**2 + dy**2
    
    # To save memory and time, we can compute row by row or use the fact that we only need distinctness.
    # Actually, for the "no isosceles" constraint:
    # 1. For any point p in set S, dist(p, q1) != dist(p, q2) for all q1, q2 in S \ {p}.
    #    This means for each p in S, the set of distances from p to S\{p} must have unique values.
    # 2. For any point q in S, and any p in S, dist(p, q) must not equal dist(p, r) for any r in S\{p, q}.
    #    This is the same condition viewed from p's perspective? 
    #    Wait. An isosceles triangle has two equal sides.
    #    Triangle (A, B, C).
    #    Case 1: dist(A,B) == dist(A,C). Apex A.
    #    Case 2: dist(B,A) == dist(B,C). Apex B.
    #    Case 3: dist(C,A) == dist(C,B). Apex C.
    #    
    #    The condition "For each point p in S, all distances from p to other points in S are distinct" covers:
    #    - If p is the apex, dist(p, q1) != dist(p, q2).
    #    
    #    Does it cover the base cases?
    #    If dist(A,B) == dist(A,C), then for apex A, the distances to B and C are equal. So A would violate the distinctness condition.
    #    So yes, the condition "For every point p in S, the distances to all other points in S are distinct" is necessary and sufficient.
    
    # So the constraint is:
    # For every p in S: len({dist(p, q) for q in S, q != p}) == len(S) - 1.
    
    # Strategy:
    # Maintain a set S of points.
    # To check if we can add p:
    # 1. Compute distances from p to all q in S.
    # 2. Check if any distance is repeated (p as apex).
    # 3. For each q in S, check if dist(p, q) is already in the set of distances from q to S\{q, p}.
    #    Wait, if dist(p, q) == dist(q, r) for some r in S\{p, q}, then q is the apex of an isosceles triangle (p, q, r).
    #    So we need to maintain for each q in S, the set of distances from q to other points in S.
    
    # Let's use a list of points and a dictionary of distance sets.
    # But the current program does this in Python and is slow.
    # Can we vectorize the check?
    
    # Let S be an array of indices.
    # Let D be the full distance matrix.
    # For a candidate p:
    # d_p = D[p, S]  # Distances from p to all in S
    # 1. Check distinctness of d_p.
    # 2. For each q in S, check if d_p[i] (where S[i]=q) is in D[q, S] \ {D[q, p]}.
    #    Note: D[q, S] includes D[q, q]=0. We ignore 0.
    #    Actually, we need to check if d_p[i] is in {D[q, r] for r in S, r != q, r != p}.
    
    # This is still O(|S|^2) per check if done naively.
    # But we can precompute for each q in S the set of distances.
    # Using Python sets is the bottleneck.
    
    # Alternative:
    # Use a "conflict" matrix.
    # conflict[i, j] = 1 if dist(i, j) == dist(i, k) for some k? No.
    
    # Let's try to optimize the Python approach first by reducing overhead.
    # The current program uses a class with methods. Function call overhead is high.
    # Inline the logic.
    
    best_set = []
    best_size = 0
    
    # State:
    # points: list of indices
    # dist_sets: dict {point_idx: set of squared distances to other points}
    
    def try_greedy(order, time_limit=10.0):
        points = []
        point_set = set()
        dist_sets = {}
        
        start = time.time()
        for p_idx in order:
            if time.time() > start + time_limit:
                break
            if p_idx in point_set:
                continue
                
            # Check p as apex
            # Get distances to all in points
            # We need to check distinctness
            # And check against existing dist_sets
            
            # Optimization: 
            # If len(points) is large, this is slow.
            
            valid = True
            # 1. Distances from p to points must be distinct
            dists = []
            for q_idx in points:
                d = D[p_idx, q_idx]
                dists.append(d)
            
            if len(set(dists)) != len(points):
                continue
                
            # 2. For each q, dist(p, q) not in dist_sets[q]
            for i, q_idx in enumerate(points):
                d = dists[i]
                if d in dist_sets[q_idx]:
                    valid = False
                    break
            
            if valid:
                points.append(p_idx)
                point_set.add(p_idx)
                dist_sets[p_idx] = set(dists)
                for i, q_idx in enumerate(points[:-1]):
                    dist_sets[q_idx].add(dists[i])
        
        return points, point_set, dist_sets

    # Initial Greedy with multiple restarts
    # Since N=64, N^2=4096.
    # Max size is likely small (e.g., 10-20).
    
    # Let's run a few fast greedy restarts
    current_points, current_set, current_dist_sets = [], set(), {}
    
    for restart in range(50):
        order = all_indices.copy()
        rng.shuffle(order)
        pts, pts_set, pts_dist = try_greedy(order, time_limit=5.0)
        if len(pts) > len(current_points):
            current_points = pts
            current_set = pts_set
            current_dist_sets = pts_dist
            best_size = len(current_points)
            best_set = current_points[:]
            
        if time.time() > deadline - 100:
            break

    # Local Search: Swap-based
    # Try to replace a point in current_set with a point not in current_set to increase size or maintain size while allowing future growth?
    # Actually, just try to add more points.
    # If stuck, remove a point, try to add a different one.
    
    while time.time() < deadline - 10:
        # 1. Try to add any unselected point
        unselected = [i for i in all_indices if i not in current_set]
        if not unselected:
            break
            
        rng.shuffle(unselected)
        added = False
        for p_idx in unselected:
            # Check validity
            valid = True
            dists = []
            for q_idx in current_points:
                d = D[p_idx, q_idx]
                dists.append(d)
            
            if len(set(dists)) != len(current_points):
                continue
                
            for i, q_idx in enumerate(current_points):
                d = dists[i]
                if d in current_dist_sets[q_idx]:
                    valid = False
                    break
            
            if valid:
                # Add
                current_points.append(p_idx)
                current_set.add(p_idx)
                current_dist_sets[p_idx] = set(dists)
                for i, q_idx in enumerate(current_points[:-1]):
                    current_dist_sets[q_idx].add(dists[i])
                added = True
                if len(current_points) > best_size:
                    best_size = len(current_points)
                    best_set = current_points[:]
                break # Restart loop to re-shuffle unselected? Or continue? 
                # Continuing is better to fill up.
                # But we break out of the for loop to restart the while loop?
                # No, just break the for loop and continue while loop.
                # Actually, if we add one, the state changed. We should restart the scan of unselected.
                # But unselected list is stale.
                # Let's just break and let the while loop handle it.
                break
        
        if not added:
            # Stuck. Perturb.
            if len(current_points) < 2:
                continue
                
            # Remove a random point
            rem_idx = rng.randint(0, len(current_points) - 1)
            p_remove = current_points[rem_idx]
            
            # Remove from data structures
            current_points.pop(rem_idx)
            current_set.remove(p_remove)
            
            # Update dist_sets
            for q_idx in current_points:
                d = D[p_remove, q_idx]
                current_dist_sets[q_idx].discard(d)
            del current_dist_sets[p_remove]
            
            # Try to add a random unselected point
            unselected = [i for i in all_indices if i not in current_set]
            if unselected:
                # Try a few
                candidates = rng.sample(unselected, min(10, len(unselected)))
                for p_add in candidates:
                    valid = True
                    dists = []
                    for q_idx in current_points:
                        d = D[p_add, q_idx]
                        dists.append(d)
                    
                    if len(set(dists)) != len(current_points):
                        continue
                        
                    for i, q_idx in enumerate(current_points):
                        d = dists[i]
                        if d in current_dist_sets[q_idx]:
                            valid = False
                            break
                    
                    if valid:
                        current_points.append(p_add)
                        current_set.add(p_add)
                        current_dist_sets[p_add] = set(dists)
                        for i, q_idx in enumerate(current_points[:-1]):
                            current_dist_sets[q_idx].add(dists[i])
                        break
                else:
                    # Couldn't add anyone, put p_remove back
                    current_points.append(p_remove)
                    current_set.add(p_remove)
                    dists = []
                    for q_idx in current_points[:-1]:
                        d = D[p_remove, q_idx]
                        dists.append(d)
                    current_dist_sets[p_remove] = set(dists)
                    for i, q_idx in enumerate(current_points[:-1]):
                        current_dist_sets[q_idx].add(dists[i])

    # Convert indices to coordinates
    return [tuple(all_points[i]) for i in best_set]

# Precompute D globally to avoid recomputation in multiple restarts?
# D depends on all_points which is fixed.
# But construct() is called once.
# We can compute D inside construct or globally.
# Let's compute it inside construct to be safe, but it's expensive.
# 4096x4096 float32 is ~64MB.
# Computation:
# px = ...
# D = (px[:, None] - px[None, :])**2 + (py[:, None] - py[None, :])**2
# This is fast in NumPy.

# Let's refine the code to use global D or compute it once in construct.
# Since N is fixed, we can compute it at module load time?
# No, better inside construct to handle potential changes, but N is hardcoded.
# Let's put D calculation in construct.

def construct():
    deadline = time.time() + 280.0
    rng = random.Random(42)
    
    # Precompute
    all_points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    all_indices = np.arange(N*N)
    
    px = all_points[:, 0].astype(np.float32)
    py = all_points[:, 1].astype(np.float32)
    
    # Compute D
    # To save memory, we can use int32 if possible? 
    # Max dist2 = 63^2 + 63^2 = 3969 + 3969 = 7938. Fits in int16.
    # Let's use int32 for safety and speed.
    dx = px[:, None] - px[None, :]
    dy = py[:, None] - py[None, :]
    D = (dx**2 + dy**2).astype(np.int32)
    
    best_set = []
    best_size = 0
    
    # State
    current_points = []
    current_set = set()
    current_dist_sets = {}
    
    # Helper to check and add
    def check_and_add(p_idx):
        nonlocal current_points, current_set, current_dist_sets, best_size, best_set
        
        if p_idx in current_set:
            return False
            
        k = len(current_points)
        if k == 0:
            current_points.append(p_idx)
            current_set.add(p_idx)
            current_dist_sets[p_idx] = set()
            if 1 > best_size:
                best_size = 1
                best_set = [p_idx]
            return True
            
        # Get distances
        # D[p_idx, current_points] -> array of distances
        # But current_points is a list. 
        # D is a numpy array.
        # We can do D[p_idx, current_points] if current_points is array-like.
        # But current_points is a list of ints.
        # D[p_idx, current_points] works.
        
        dists = D[p_idx, current_points]
        
        # Check distinctness
        # If k is small, set(dists) is fast.
        if len(set(dists)) != k:
            return False
            
        # Check against existing dist_sets
        for i, q_idx in enumerate(current_points):
            d = dists[i]
            if d in current_dist_sets[q_idx]:
                return False
                
        # Add
        current_points.append(p_idx)
        current_set.add(p_idx)
        current_dist_sets[p_idx] = set(dists)
        for i, q_idx in enumerate(current_points[:-1]):
            current_dist_sets[q_idx].add(dists[i])
            
        if len(current_points) > best_size:
            best_size = len(current_points)
            best_set = current_points[:]
            
        return True

    # Greedy Phase
    for restart in range(100):
        if time.time() > deadline - 60:
            break
            
        order = all_indices.copy()
        rng.shuffle(order)
        
        # Reset state
        current_points = []
        current_set = set()
        current_dist_sets = {}
        
        for p_idx in order:
            check_and_add(p_idx)
            
        # Local Search Phase
        # Try to fill gaps
        while time.time() < deadline - 60:
            unselected = [i for i in all_indices if i not in current_set]
            if not unselected:
                break
                
            rng.shuffle(unselected)
            improved = False
            for p_idx in unselected:
                if check_and_add(p_idx):
                    improved = True
                    break # Restart loop to re-shuffle
            
            if not improved:
                # Perturb
                if len(current_points) < 2:
                    break
                    
                # Remove random
                rem_idx = rng.randint(0, len(current_points) - 1)
                p_remove = current_points[rem_idx]
                
                # Remove
                current_points.pop(rem_idx)
                current_set.remove(p_remove)
                for q_idx in current_points:
                    d = D[p_remove, q_idx]
                    current_dist_sets[q_idx].discard(d)
                del current_dist_sets[p_remove]
                
                # Try to add random unselected
                unselected = [i for i in all_indices if i not in current_set]
                if unselected:
                    candidates = rng.sample(unselected, min(20, len(unselected)))
                    added = False
                    for p_add in candidates:
                        if check_and_add(p_add):
                            added = True
                            break
                    if not added:
                        # Put back
                        check_and_add(p_remove)

    return [tuple(all_points[i]) for i in best_set]

# Note: The nested function check_and_add uses nonlocal variables.
# This is efficient enough.
# The main bottleneck is the Python loop over current_points in check_and_add.
# With N=64, max size is likely < 50.
# 50 iterations * 100 restarts * 280s budget.
# Should be fine.

# One optimization: 
# In check_and_add, `dists = D[p_idx, current_points]` creates a numpy array.
# `set(dists)` converts to Python set.
# This is fast.

# Let's double check the logic for "no isosceles".
# Condition: For all distinct A, B, C in S, dist(A,B) != dist(A,C).
# My check:
# 1. For new p, dist(p, q) are distinct for all q in S. (p is apex)
# 2. For each q in S, dist(p, q) is not in dist_sets[q]. 
#    dist_sets[q] contains dist(q, r) for all r in S \ {q}.
#    If dist(p, q) == dist(q, r), then triangle (p, q, r) is isosceles with apex q.
#    So this check is correct.

# The program looks solid.

# Final check on constraints:
# - 300s limit. I use 280s.
# - Standard library, NumPy. Used.
# - Return list of (x, y). Done.

# Potential issue: `nonlocal` in `check_and_add` defined inside `construct`.
# This is valid Python 3.

# Let's ensure `D` is computed efficiently.
# `dx = px[:, None] - px[None, :]` creates a 4096x4096 array.
# `dx**2` creates another.
# `dy**2` creates another.
# `dx**2 + dy**2` creates another.
# `.astype(np.int32)` creates another.
# Total memory ~ 4 * 4096 * 4096 * 4 bytes = ~256MB.
# This is high but usually acceptable in modern environments.
# If memory is a concern, we could compute rows on the fly, but that slows down access.
# Given the score improvement potential, memory is worth it.

# One more thing: `rng.sample` requires a sequence. `unselected` is a list. OK.

# The `best_set` is a list of indices.
# `return [tuple(all_points[i]) for i in best_set]` converts to coords.

# The code is ready.
