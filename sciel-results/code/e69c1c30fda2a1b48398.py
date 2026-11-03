import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def exact_score(U):
        """
        Compute exact score for a set U.
        Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
        """
        if not U:
            return 0.0
        U_set = set(U)
        max_U = max(U_set)
        if max_U == 0:
            return 0.0
        
        n = len(U_set)
        sums = set()
        diffs = set()
        for a in U_set:
            for b in U_set:
                sums.add(a + b)
                diffs.add(a - b)
        
        if not sums:
            return 0.0
            
        ratio = len(diffs) / len(sums)
        if ratio <= 0:
            return 0.0
        
        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_U + 1)
        except (ValueError, OverflowError, ZeroDivisionError):
            return 0.0
        return exponent

    def fast_score_numpy(U):
        """
        Fast score using numpy for sets up to size ~500.
        """
        if len(U) < 2:
            return 0.0
        U_arr = np.array(sorted(U), dtype=np.int64)
        max_U = int(U_arr.max())
        if max_U == 0:
            return 0.0
        
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        
        sumset_size = len(np.unique(sums))
        diffset_size = len(np.unique(diffs))
        
        ratio = diffset_size / sumset_size
        if ratio <= 0:
            return 0.0
        
        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_U + 1)
        except (ValueError, OverflowError, ZeroDivisionError):
            return 0.0
        return exponent

    def simulated_annealing_lattice(time_limit):
        """
        Simulated Annealing on a 2D lattice representation.
        We represent the set U as a list of (x, y) coordinates.
        The actual integer value is v = x * W + y.
        We optimize the score by moving points in the 2D space.
        
        Strategy:
        1. Initialize a set of points in a 2D grid (lattice packing).
        2. Moves: 
           - Move a point to a random nearby lattice position.
           - Add a new point.
           - Remove a point.
        3. Objective: Maximize score.
        4. Cooling schedule: Exponential decay.
        """
        # Initial setup
        # We want to balance |U-U| and |U+U|.
        # In 2D, if we pick points (i, j) such that i and j are "independent", 
        # the sumset is roughly the Minkowski sum of the projections.
        
        # Let's start with a small grid-like structure.
        # Size of U should be moderate. Let's aim for N=50 to 100.
        N = 60
        W = 100 # Width of the lattice (max y)
        
        # Initial set: random points in a grid
        current_points = []
        for _ in range(N):
            x = random.randint(0, 20)
            y = random.randint(0, W)
            current_points.append((x, y))
        
        # Convert to set of integers
        def points_to_set(points):
            vals = set()
            for x, y in points:
                vals.add(x * (W + 1) + y)
            # Ensure 0 is in the set
            if 0 not in vals:
                vals.add(0)
            return sorted(list(vals))
        
        current_set = points_to_set(current_points)
        current_score = fast_score_numpy(current_set)
        
        best_set = current_set[:]
        best_score = current_score
        
        T = 1.0
        T_min = 1e-6
        alpha = 0.999 # Cooling rate
        
        steps = 0
        while time.time() < time_limit and T > T_min:
            steps += 1
            
            # Choose a move
            move_type = random.random()
            new_points = current_points[:]
            
            if move_type < 0.7:
                # Move a point
                if not new_points:
                    continue
                idx = random.randint(0, len(new_points) - 1)
                x, y = new_points[idx]
                # Small random step
                dx = random.randint(-2, 2)
                dy = random.randint(-2, 2)
                nx = max(0, x + dx)
                ny = max(0, min(W, y + dy))
                new_points[idx] = (nx, ny)
                
            elif move_type < 0.9:
                # Add a point
                if len(new_points) < 150:
                    x = random.randint(0, 30)
                    y = random.randint(0, W)
                    new_points.append((x, y))
                    
            else:
                # Remove a point (keep at least 2)
                if len(new_points) > 2:
                    idx = random.randint(0, len(new_points) - 1)
                    new_points.pop(idx)
            
            # Convert and evaluate
            new_set = points_to_set(new_points)
            
            # Check if set changed significantly (to avoid redundant evals)
            # Simple check: if length is same and max is same, it might be same set? 
            # Better: just evaluate.
            
            new_score = fast_score_numpy(new_set)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                current_points = new_points
                current_set = new_set
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set[:]
                    
            T *= alpha
            
            # Periodic restart if stuck?
            if steps % 1000 == 0 and T > T_min:
                # Slight perturbation
                pass

        return best_set, best_score

    def lattice_heuristic_init():
        """
        Create a structured initial set based on lattice packing principles.
        We want to minimize sum collisions. 
        A "sparse" set in terms of additive structure is good.
        Let's try a set based on powers of a base, or a 2D grid projection.
        """
        # Grid projection: (i, j) -> i * B + j
        # If we pick i and j from sets with small sumsets, the total sumset is small.
        # But we want LARGE diffset.
        
        # Let's try a "Sidon-like" set in 2D.
        # Actually, just a random set of points in a large box is a decent start for SA.
        # But let's try a specific structure:
        # U = { i * 100 + j | i in I, j in J }
        # If I and J are small, U is small.
        # Let's just return a random set for SA to refine.
        
        N = 50
        W = 100
        points = []
        for _ in range(N):
            x = random.randint(0, 20)
            y = random.randint(0, W)
            points.append((x, y))
        
        vals = set()
        for x, y in points:
            vals.add(x * (W + 1) + y)
        if 0 not in vals:
            vals.add(0)
        return sorted(list(vals))

    # Strategy:
    # 1. Run Simulated Annealing on Lattice representation.
    # 2. Run Local Search on the result to polish.
    
    best_overall_score = 0.0
    best_overall_set = [0, 1]
    
    # Phase 1: Simulated Annealing
    sa_time = 100.0
    sa_set, sa_score = simulated_annealing_lattice(start_time + sa_time)
    
    if sa_score > best_overall_score:
        best_overall_score = sa_score
        best_overall_set = sa_set[:]

    # Phase 2: Local Search Refinement (1D, as in current program)
    # Reuse the logic from the current program for refinement
    def local_search_refinement(initial_set, time_limit):
        current_set = sorted(set(initial_set))
        current_score = fast_score_numpy(current_set)
        best_set = current_set[:]
        best_score = current_score
        
        max_iter = 0
        while time.time() < time_limit and max_iter < 50:
            max_iter += 1
            improved = False
            
            # 1. Try removals
            n = len(current_set)
            if n > 1:
                best_removal = None
                best_removal_score = current_score
                # Sample removals if large
                indices_to_check = list(range(1, n))
                if n > 50:
                    indices_to_check = random.sample(indices_to_check, 20)
                
                for i in indices_to_check:
                    new_set = current_set[:i] + current_set[i+1:]
                    s = fast_score_numpy(new_set)
                    if s > best_removal_score + 1e-12:
                        best_removal_score = s
                        best_removal = i
                if best_removal is not None:
                    current_set = current_set[:best_removal] + current_set[best_removal+1:]
                    current_score = best_removal_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = current_set[:]

            # 2. Try replacements
            if n > 1:
                best_replace_idx = None
                best_replace_val = None
                best_replace_score = current_score
                
                indices_to_check = list(range(1, n))
                if n > 20:
                    indices_to_check = random.sample(indices_to_check, min(10, len(indices_to_check)))
                
                for i in indices_to_check:
                    old_val = current_set[i]
                    for delta in [-10, -5, -1, 1, 5, 10, -20, 20]:
                        new_val = old_val + delta
                        if new_val <= 0:
                            continue
                        if new_val in current_set:
                            continue
                        
                        new_set = current_set[:]
                        new_set[i] = new_val
                        new_set = sorted(set(new_set))
                        
                        s = fast_score_numpy(new_set)
                        if s > best_replace_score + 1e-12:
                            best_replace_score = s
                            best_replace_idx = i
                            best_replace_val = new_val
                
                if best_replace_idx is not None:
                    current_set[best_replace_idx] = best_replace_val
                    current_set = sorted(set(current_set))
                    current_score = best_replace_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = current_set[:]

            # 3. Try additions
            if n < 100:
                candidates = set()
                for u in current_set:
                    for d in [-5, -1, 1, 5]:
                        nx = u + d
                        if nx > 0:
                            candidates.add(nx)
                candidates.update(random.sample(range(1, max(current_set)+10), min(10, max(current_set)+10)))
                
                best_add_val = None
                best_add_score = current_score
                for x in candidates:
                    if x in current_set:
                        continue
                    new_set = current_set + [x]
                    s = fast_score_numpy(new_set)
                    if s > best_add_score + 1e-12:
                        best_add_score = s
                        best_add_val = x
                
                if best_add_val is not None:
                    current_set.append(best_add_val)
                    current_set = sorted(current_set)
                    current_score = best_add_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = current_set[:]

            if not improved:
                break

        return best_set, best_score

    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        ls_time = remaining_time * 0.8
        ls_set, ls_score = local_search_refinement(sa_set, time.time() + ls_time)
        
        if ls_score > best_overall_score:
            best_overall_score = ls_score
            best_overall_set = ls_set[:]

    # Final verification
    final_score = exact_score(best_overall_set)
    
    return best_overall_set

if __name__ == "__main__":
    U = construct()
    print(f"Result: {U}")
    if U:
        max_U = max(U)
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a+b)
                diffs.add(a-b)
        if sums:
            ratio = len(diffs)/len(sums)
            score = 1 + math.log(ratio)/math.log(2*max_U+1)
            print(f"Score: {score}")
