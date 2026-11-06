import numpy as np
import time
import random

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s kill

    # Grid points
    xs = np.arange(N)
    ys = np.arange(N)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    coords = np.column_stack((X.ravel(), Y.ravel()))
    n_pts = len(coords)

    # Precompute distance matrix
    # dist_mat[i, j] = squared distance between point i and point j
    # Using int32 to avoid overflow and keep memory reasonable (4096*4096*4 = 64MB)
    # Calculation: (x1-x2)^2 + (y1-y2)^2
    # Vectorized calculation might be slow/memory heavy, let's do it in chunks or optimized loop
    # Actually, 4096^2 is 16M. 
    # Let's try the vectorized way first, it's usually fast enough.
    try:
        # coords shape (n_pts, 2)
        # diff shape (n_pts, n_pts, 2) -> 16M * 2 * 8 bytes = 256MB. Might be too much.
        # Let's compute row by row to save memory.
        dist_mat = np.zeros((n_pts, n_pts), dtype=np.int32)
        for i in range(n_pts):
            dx = coords[i, 0] - coords[:, 0]
            dy = coords[i, 1] - coords[:, 1]
            dist_mat[i, :] = dx**2 + dy**2
    except Exception:
        # Fallback to pure python if numpy fails or is too slow
        dist_mat = np.zeros((n_pts, n_pts), dtype=np.int32)
        for i in range(n_pts):
            for j in range(n_pts):
                dx = coords[i, 0] - coords[j, 0]
                dy = coords[i, 1] - coords[j, 1]
                dist_mat[i, j] = dx*dx + dy*dy

    # Helper to count isosceles triangles in a set of indices
    # This is O(k^2)
    def count_isosceles(selected_indices):
        if len(selected_indices) < 3:
            return 0
        count = 0
        sel = list(selected_indices)
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                for k in range(j + 1, len(sel)):
                    d1 = dist_mat[sel[i], sel[j]]
                    d2 = dist_mat[sel[i], sel[k]]
                    d3 = dist_mat[sel[j], sel[k]]
                    # Check if any two are equal
                    if d1 == d2 or d1 == d3 or d2 == d3:
                        count += 1
        return count

    # Greedy Initialization with Minimum Conflict
    # We want to build a set with 0 isosceles triangles.
    # Start with empty set.
    # At each step, pick the point that adds the fewest isosceles triangles.
    # If min is 0, pick one. If all > 0, stop (or pick min if we allow temp invalid, but for init we want valid).
    
    def greedy_init():
        selected = set()
        # Pick first point randomly
        p1 = random.randint(0, n_pts - 1)
        selected.add(p1)
        
        # Pick second point randomly
        while True:
            p2 = random.randint(0, n_pts - 1)
            if p2 != p1:
                selected.add(p2)
                break
        
        while True:
            if time.time() > deadline - 30:
                break
            best_p = -1
            best_conflicts = float('inf')
            best_min_dist = -1 # Tie-breaker: maximize min dist to existing
            
            # To speed up, we can sample candidates? No, let's check all.
            # n_pts is 4096. k is small initially.
            # Checking all 4096 points for each step is 4096 * k^2 ops.
            # If k=10, 4096 * 100 = 400k ops. Fast.
            
            for p in range(n_pts):
                if p in selected:
                    continue
                
                # Calculate conflicts if we add p
                # We need to check triples (p, a, b) for a, b in selected
                # And (a, b, c) in selected are already 0.
                # So we only check triples involving p.
                conflicts = 0
                sel_list = list(selected)
                d_p_to_sel = {}
                for a in sel_list:
                    d_pa = dist_mat[p, a]
                    if d_pa in d_p_to_sel:
                        # d_pa == d_pb for some b. This forms isosceles with apex p.
                        conflicts += 1
                        # We don't break immediately because we want exact count? 
                        # For greedy "min conflict", we just need to know if it's 0 or not, or the count.
                        # If we just want valid, we can break. But min conflict is better.
                    d_p_to_sel[d_pa] = a
                    
                # Check d_pa == d_ab
                for i in range(len(sel_list)):
                    a = sel_list[i]
                    d_pa = dist_mat[p, a]
                    for j in range(i + 1, len(sel_list)):
                        b = sel_list[j]
                        d_ab = dist_mat[a, b]
                        d_pb = dist_mat[p, b]
                        if d_pa == d_ab:
                            conflicts += 1
                        if d_pb == d_ab:
                            conflicts += 1
                        if d_pa == d_pb:
                            # Already counted in d_p_to_sel check? 
                            # d_p_to_sel check counts pairs (a,b) where d_pa==d_pb.
                            # The loop above counts specific pairs.
                            # Let's be careful not to double count or miss.
                            # The condition "no isosceles" means for ANY triple, all 3 distances distinct.
                            # My conflict count above:
                            # 1. d_pa == d_pb (apex p)
                            # 2. d_pa == d_ab (apex a)
                            # 3. d_pb == d_ab (apex b)
                            # These are the 3 conditions for triple (p, a, b).
                            # If any is true, it's isosceles.
                            # My logic:
                            # d_p_to_sel check: if d_pa == d_pb, conflicts++. This handles condition 1.
                            # Loop: if d_pa == d_ab, conflicts++. This handles condition 2.
                            # Loop: if d_pb == d_ab, conflicts++. This handles condition 3.
                            # This seems correct.
                            pass

                if conflicts < best_conflicts:
                    # Tie-breaker: maximize minimum distance to existing points
                    min_d = min(dist_mat[p, a] for a in selected) if selected else 0
                    if min_d > best_min_dist:
                        best_min_dist = min_d
                        best_conflicts = conflicts
                        best_p = p
                elif conflicts == best_conflicts:
                    min_d = min(dist_mat[p, a] for a in selected) if selected else 0
                    if min_d > best_min_dist:
                        best_min_dist = min_d
                        best_conflicts = conflicts
                        best_p = p
            
            if best_p == -1 or best_conflicts > 0:
                # No point can be added without creating an isosceles triangle
                break
            
            selected.add(best_p)
            
            # Check time
            if time.time() > deadline - 30:
                break
                
        return selected

    best_selected = greedy_init()
    best_size = len(best_selected)
    
    # Simulated Annealing
    # State: set of indices
    # Energy: -size * 100 + conflicts * 1000
    # We want to minimize Energy.
    
    current_selected = set(best_selected)
    current_conflicts = count_isosceles(current_selected)
    current_energy = -len(current_selected) * 100 + current_conflicts * 1000
    
    best_energy = current_energy
    best_set_sa = set(current_selected)
    
    T = 100.0
    cooling = 0.999
    
    # Precompute list of all points for random sampling
    all_points = list(range(n_pts))
    
    steps = 0
    while time.time() < deadline - 10:
        steps += 1
        if T < 0.1:
            break
            
        # Move: Swap
        p_out = random.choice(list(current_selected))
        p_in = random.choice(all_points)
        
        if p_in in current_selected:
            continue
            
        # Try swap
        new_selected = set(current_selected)
        new_selected.remove(p_out)
        new_selected.add(p_in)
        
        # Calculate new conflicts
        # Optimization: Only recompute conflicts involving p_in or removed p_out?
        # count_isosceles is O(k^2). k is small (~10-20).
        # So just recompute.
        new_conflicts = count_isosceles(new_selected)
        new_size = len(new_selected)
        new_energy = -new_size * 100 + new_conflicts * 1000
        
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or random.random() < np.exp(-delta_e / T):
            current_selected = new_selected
            current_conflicts = new_conflicts
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_set_sa = set(current_selected)
        
        T *= cooling
        
    # Final check: ensure best_set_sa is valid (0 conflicts)
    # If not, try to repair or fall back to greedy result
    if count_isosceles(best_set_sa) > 0:
        # Fallback to greedy if SA didn't find a valid set
        result_indices = best_selected
    else:
        result_indices = best_set_sa
    
    # Convert to list of tuples
    result = [tuple(coords[i]) for i in result_indices]
    return result
