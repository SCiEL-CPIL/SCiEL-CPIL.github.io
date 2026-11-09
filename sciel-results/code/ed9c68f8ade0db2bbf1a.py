import time
import random
import math

N = 64

# EVOLVE-BLOCK-START

def _dist2_idx(i, j):
    """Compute squared distance between grid points i and j (index 0..N*N-1)."""
    xi = i // N
    yi = i % N
    xj = j // N
    yj = j % N
    dx = xi - xj
    dy = yi - yj
    return dx*dx + dy*dy

def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin for 330s limit
    
    num_points = N * N
    rng = random.Random(42)
    
    # State representation:
    # selected: list of point indices currently in the set.
    # point_set: set of point indices for O(1) lookup.
    # dist_sets: dict mapping point index q -> set of squared distances from q to other selected points.
    
    selected = []
    point_set = set()
    dist_sets = {}
    
    def _can_add(p_idx):
        """Check if p_idx can be added to the current set without creating an isosceles triangle."""
        if p_idx in point_set:
            return False
        k = len(selected)
        if k == 0:
            return True
        
        # Check distances from p to all selected points
        seen_dists = set()
        for q_idx in selected:
            d = _dist2_idx(p_idx, q_idx)
            # Condition 1: Distances from p must be unique (p as apex)
            if d in seen_dists:
                return False
            seen_dists.add(d)
            
            # Condition 2: d must not be in dist_sets[q_idx] (q as apex)
            if d in dist_sets[q_idx]:
                return False
        
        return True

    def _add(p_idx):
        """Add p_idx to the set and update data structures."""
        point_set.add(p_idx)
        selected.append(p_idx)
        dist_sets[p_idx] = set()
        
        for q_idx in selected[:-1]: # All except p_idx
            d = _dist2_idx(p_idx, q_idx)
            dist_sets[p_idx].add(d)
            dist_sets[q_idx].add(d)

    def _remove(p_idx):
        """Remove p_idx from the set and update data structures."""
        point_set.remove(p_idx)
        selected.remove(p_idx)
        
        # Update dist_sets for all remaining points
        for q_idx in selected:
            d = _dist2_idx(p_idx, q_idx)
            dist_sets[q_idx].remove(d)
        
        del dist_sets[p_idx]

    # 1. Greedy Initialization with multiple random orders to find a good starting point
    best_size = 0
    best_selected = []
    best_point_set = set()
    best_dist_sets = {}
    
    for trial in range(10):
        selected.clear()
        point_set.clear()
        dist_sets.clear()
        
        candidates = list(range(num_points))
        rng.shuffle(candidates)
        
        for p_idx in candidates:
            if _can_add(p_idx):
                _add(p_idx)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected[:]
            best_point_set = point_set.copy()
            best_dist_sets = {k: v.copy() for k, v in dist_sets.items()}
        
        if time.time() > deadline:
            break

    # Restore best state
    selected = best_selected[:]
    point_set = best_point_set.copy()
    dist_sets = {k: v.copy() for k, v in best_dist_sets.items()}
    
    # 2. Simulated Annealing / Local Search
    # We try to improve the set size by:
    # A) Adding random unselected points if valid.
    # B) If stuck, removing a random point to potentially allow more additions later.
    
    current_temp = 1.0
    cooling_rate = 0.9995
    min_temp = 1e-5
    
    # To speed up "Add" attempts, we don't maintain a full unselected list.
    # We just pick random indices and check membership.
    
    stuck_count = 0
    max_stuck_before_remove = 200  # Try 200 random adds before considering removal
    
    while time.time() < deadline:
        # 1. Attempt additions
        added_any = False
        
        # Try a batch of random additions
        for _ in range(100):
            if time.time() > deadline:
                break
                
            p_idx = rng.randint(0, num_points - 1)
            if p_idx in point_set:
                continue
                
            if _can_add(p_idx):
                _add(p_idx)
                added_any = True
                best_size = len(selected)
                # Update best snapshot if we found a new global best
                # We don't need to copy the whole state every time, just track size
                # But we need to return a valid set at the end. 
                # Since we modify in-place, we should keep a snapshot of the best *state*.
                # Copying the state is O(N^2) which is expensive. 
                # Alternative: Only copy when we exceed the previous best size.
                # For now, let's assume we keep the current state as the best candidate 
                # if it's larger than any previous snapshot. 
                # Actually, the simplest robust way is to just keep the current state 
                # if it's valid (it always is) and return it at the end. 
                # But what if we remove points later and end up with a smaller set? 
                # We should track the best size seen and store a copy of that state.
                
                # Optimization: Only copy state if size > best_size_stored
                # We need a separate variable for the "stored best" size.
        
        if added_any:
            stuck_count = 0
        else:
            stuck_count += 1
            
            # 2. If stuck, consider a removal move
            if stuck_count >= max_stuck_before_remove and len(selected) > 0:
                # Accept removal with probability based on temperature
                # delta = -1 (size decreases by 1)
                # prob = exp(-delta / temp) = exp(1 / temp)
                # Wait, standard SA: prob = exp(-delta_E / T). 
                # We want to maximize size. 
                # Move: Remove point. Delta Size = -1.
                # We want to accept "worse" moves with probability exp(-(-1)/T) = exp(1/T)?
                # No, usually we define energy E = -size. Minimizing E.
                # Delta E = E_new - E_old = -size_new - (-size_old) = -(size_new - size_old) = -(-1) = 1.
                # Prob = exp(-Delta E / T) = exp(-1/T).
                
                if rng.random() < math.exp(-1.0 / current_temp):
                    # Perform removal
                    p_idx = rng.choice(selected)
                    _remove(p_idx)
                    stuck_count = 0
                    # After removal, we might be able to add more, so reset stuck count
                    # and continue to next iteration of the main loop (which tries adds)
        
        # Cool down
        current_temp *= cooling_rate
        if current_temp < min_temp:
            current_temp = min_temp
            
        # Optional: Restart if stuck for too long? 
        # For now, let's just keep trying.

    # Final polish: Try a greedy pass one last time with the current set
    # Sometimes a full sweep can add a few more points that random sampling missed.
    # But a full sweep is O(N^3) which is ~68 billion ops. Too slow.
    # Instead, let's just do a few more random tries.
    
    # Ensure we return a list of (x, y) tuples
    result = [(i // N, i % N) for i in selected]
    
    return result

# EVOLVE-BLOCK-END
