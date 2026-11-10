import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    sq_norms = (grid ** 2).sum(axis=1)
    
    # Precompute all 4-combinations of indices 0..511 once for reuse in can_add_to_set
    # We'll use a chunked approach: for a given set of m points, we need C(m,4) determinants.
    # Strategy: build the 4x4 matrix rows from the selected points, then check determinants in chunks.
    
    def can_add_to_set(points_indices, new_point_idx, deadline=None):
        """
        Check if new_point_idx can be added to the set of points given by indices in points_indices.
        Returns True if no 5 points (including new_point_idx) are cospherical.
        Uses chunked vectorized determinant with early exit.
        """
        m = len(points_indices)
        if m < 4:
            return True
        
        p_new = grid[new_point_idx]
        sq_new = sq_norms[new_point_idx]
        
        # Build the m x 4 matrix w
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = grid[points_indices] - p_new
        w[:, 3] = sq_norms[points_indices] - sq_new
        
        # Generate all 4-combinations of range(m)
        # For efficiency, we process in chunks
        chunk_size = 50000  # tune: ~200KB per chunk for (C,4,4) float64
        
        combos = itertools.combinations(range(m), 4)
        
        # We'll process in chunks to limit memory and allow early exit
        chunk = []
        for c in combos:
            chunk.append(c)
            if len(chunk) >= chunk_size:
                arr = np.array(chunk, dtype=np.intp)
                sub = w[arr]  # (C, 4, 4)
                dets = np.linalg.det(sub)
                if np.any(np.abs(dets) < 0.5):
                    return False
                chunk = []
                if deadline is not None and time.time() > deadline:
                    # Time pressure: conservatively reject to avoid timeout
                    return False
        
        if chunk:
            arr = np.array(chunk, dtype=np.intp)
            sub = w[arr]
            dets = np.linalg.det(sub)
            if np.any(np.abs(dets) < 0.5):
                return False
        
        return True

    def is_valid_set(indices):
        """
        Check if the set of points given by indices is valid.
        """
        # We can just check all 5-combinations? No, that's O(k^5).
        # But we build it incrementally. 
        # For the initial check of a greedy set, we can just assume it's valid if built correctly.
        # For swaps, we use can_add_to_set logic on the reduced set.
        pass

    # Geometric Initialization using structured lines
    def geometric_init(deadline=None):
        """
        Construct initial set using arithmetic progressions along specific directions.
        Points on lines with rational directions have algebraic properties that
        reduce cospherical/coplanar 5-tuples.
        """
        selected_indices = []
        selected_set = set()
        
        # Define several line directions with rational slopes
        # These directions are chosen to minimize algebraic dependencies
        directions = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, 0, 1), (0, 1, 1),
            (1, 1, 1), (1, 2, 3), (2, 3, 5),
            (1, 3, 2), (3, 1, 2), (2, 1, 3),
        ]
        
        # For each direction, generate points along lines in that direction
        for dx, dy, dz in directions:
            if deadline is not None and time.time() > deadline:
                break
            
            # Generate all lines in this direction within the grid
            # A line is determined by its starting point (x0, y0, z0) such that
            # (x0 + t*dx, y0 + t*dy, z0 + t*dz) stays in [0,7] for t=0..k
            for x0 in range(n):
                for y0 in range(n):
                    for z0 in range(n):
                        if deadline is not None and time.time() > deadline:
                            break
                        
                        # Check if this starting point is valid and generates a line
                        line_points = []
                        t = 0
                        while True:
                            x = x0 + t * dx
                            y = y0 + t * dy
                            z = z0 + t * dz
                            if x < 0 or x >= n or y < 0 or y >= n or z < 0 or z >= n:
                                break
                            line_points.append((x, y, z))
                            t += 1
                        
                        if len(line_points) < 3:
                            continue
                        
                        # Try to add points from this line
                        for (x, y, z) in line_points:
                            if deadline is not None and time.time() > deadline:
                                break
                            
                            # Find the index of this point
                            idx = x * 64 + y * 8 + z
                            if idx in selected_set:
                                continue
                            
                            if can_add_to_set(selected_indices, idx, deadline):
                                selected_indices.append(idx)
                                selected_set.add(idx)
        
        return selected_indices

    # Greedy Initialization
    def greedy_init(seed, deadline=None):
        np.random.seed(seed)
        order = np.random.permutation(N)
        selected_indices = []
        for idx in order:
            if deadline is not None and time.time() > deadline:
                break
            if can_add_to_set(selected_indices, idx, deadline):
                selected_indices.append(idx)
        return selected_indices

    # Local Search
    def local_search(initial_indices, max_time, deadline=None):
        current_indices = list(initial_indices)
        current_set = set(current_indices)
        best_indices = list(current_indices)
        best_size = len(current_indices)
        
        start_time = time.time()
        no_improve_count = 0
        
        while True:
            elapsed = time.time() - start_time
            if elapsed >= max_time:
                break
            if deadline is not None and time.time() > deadline:
                break
            
            # 1. Try to add any point
            added = False
            candidates = [i for i in range(N) if i not in current_set]
            if candidates:
                np.random.shuffle(candidates)
                
                for idx in candidates:
                    if deadline is not None and time.time() > deadline:
                        break
                    if can_add_to_set(current_indices, idx, deadline):
                        current_indices.append(idx)
                        current_set.add(idx)
                        added = True
                        if len(current_indices) > best_size:
                            best_indices = list(current_indices)
                            best_size = len(current_indices)
                            no_improve_count = 0
                        break
            
            if added:
                continue
            
            # 2. No point could be added. Try 1-swap.
            swapped = False
            # Pick a random point to remove
            out_idx = int(np.random.choice(current_indices))
            out_set = current_set - {out_idx}
            remaining_indices = [i for i in current_indices if i != out_idx]
            
            # Try to add a random point from candidates
            candidates = [i for i in range(N) if i not in out_set]
            np.random.shuffle(candidates)
            
            for in_idx in candidates:
                if deadline is not None and time.time() > deadline:
                    break
                if can_add_to_set(remaining_indices, in_idx, deadline):
                    current_indices = remaining_indices + [in_idx]
                    current_set = out_set | {in_idx}
                    swapped = True
                    no_improve_count += 1
                    break
            
            if not swapped:
                # Try a few more random out_idx if first one failed
                found_swap = False
                for _ in range(10):
                    if deadline is not None and time.time() > deadline:
                        break
                    out_idx = int(np.random.choice(current_indices))
                    out_set = current_set - {out_idx}
                    remaining_indices = [i for i in current_indices if i != out_idx]
                    candidates = [i for i in range(N) if i not in out_set]
                    np.random.shuffle(candidates)
                    for in_idx in candidates[:30]:
                        if deadline is not None and time.time() > deadline:
                            break
                        if can_add_to_set(remaining_indices, in_idx, deadline):
                            current_indices = remaining_indices + [in_idx]
                            current_set = out_set | {in_idx}
                            found_swap = True
                            no_improve_count += 1
                            break
                    if found_swap:
                        break
                
                if not found_swap:
                    break

        return best_indices

    # Run multiple restarts
    best_overall = []
    start_total = time.time()
    time_limit = 630.0
    safety_margin = 30.0  # seconds
    
    deadline = start_total + time_limit - safety_margin
    
    # First, try geometric initialization
    geo_init = geometric_init(deadline)
    if len(geo_init) > len(best_overall):
        best_overall = geo_init
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        seed = num_restarts
        
        # Alternate between geometric and random initialization
        if num_restarts % 3 == 1:
            init_indices = geometric_init(deadline)
        else:
            init_indices = greedy_init(seed, deadline)
        
        # Allocate remaining time to local search
        elapsed = time.time() - start_total
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Local search with a portion of the time
        # Give more time to local search for larger sets
        ls_time = min(remaining * 0.6, 90.0)
        if ls_time < 1:
            break
        
        search_result = local_search(init_indices, ls_time, deadline)
        
        if len(search_result) > len(best_overall):
            best_overall = search_result
            
            # Deterministic Polish Pass
            # Try to add points in sorted order
            current = list(best_overall)
            current_set = set(current)
            for idx in range(N):
                if time.time() > deadline:
                    break
                if idx not in current_set:
                    if can_add_to_set(current, idx, deadline):
                        current.append(idx)
                        current_set.add(idx)
                        
            if len(current) > len(best_overall):
                best_overall = current

    # Final polish on the absolute best
    if best_overall and time.time() < deadline:
        current = list(best_overall)
        current_set = set(current)
        # Try adding in sorted order
        for idx in range(N):
            if time.time() > deadline:
                break
            if idx not in current_set:
                if can_add_to_set(current, idx, deadline):
                    current.append(idx)
                    current_set.add(idx)
        best_overall = current

    if not best_overall:
        return np.empty((0, 3), dtype=np.int64)
        
    return grid[best_overall].astype(np.int64)
