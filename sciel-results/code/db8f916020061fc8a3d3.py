"""
Strategy: Model set construction as a lattice packing problem in Z^2.
We represent elements of U as u_i = x_i + y_i * B, where B is a large base.
This maps U to a set of points P = {(x_i, y_i)} in Z^2.
The sumset U+U corresponds to the Minkowski sum P+P.
The difference set U-U corresponds to P-P.
We want to maximize |P-P| / |P+P| while keeping the "cost" (max U) low.
Max U is roughly proportional to max(x_i) + max(y_i)*B.
We use Simulated Annealing to optimize the coordinates (x_i, y_i) for a fixed size n.
The objective function is the score.
"""
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    def score_set(U):
        if not U:
            return 0.0
        U_set = set(U)
        if 0 not in U_set:
            # Task requires containing 0. If not present, we can shift, but construct returns list.
            # The evaluator likely checks if 0 is in the list or handles it? 
            # "construct() -> list of nonnegative integers containing 0"
            # So we must ensure 0 is in the list.
            pass
        
        max_U = max(U_set)
        if max_U == 0:
            return 0.0
        
        # Compute sumset and difference set
        # For small sets, O(n^2) is fine.
        sumset = set()
        diffset = set()
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
                diffset.add(a - b)
        
        n = len(U_set)
        if n == 0:
            return 0.0
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0
        
        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_U + 1)
        except:
            return 0.0
        return exponent

    best_score = 0.0
    best_set = [0, 1]
    
    # Base B for the lattice mapping u = x + y*B
    # We want B large enough to avoid collisions in the "digit" sense for sums/diffs
    # but not so large that max_U explodes.
    # If we fix n, and want to search x,y in a small box, B should be > 2*max_coord.
    
    # We will try different sizes n and different B values.
    # For each, we run a short SA.
    
    # Initial heuristic: Geometric progression is a specific lattice point configuration
    # U = {0, 1, B, B^2...} -> Points (0,0), (1,0), (0,1), (0,2)...
    # This is a "corner" packing.
    # SA can explore better packings.
    
    n_max = 12
    B_values = [10, 20, 50, 100]
    
    for n in range(2, n_max + 1):
        if time.time() > deadline:
            return best_set
            
        for B in B_values:
            if time.time() > deadline:
                return best_set
                
            # Initialize coordinates
            # One point is (0,0) -> u=0
            # Others are random or structured.
            # Let's start with a "line" or "grid" structure to give SA a good start.
            # Or just random within [-5, 5] for x and y.
            
            # We need to ensure U contains 0.
            # Let's fix u_0 = 0.
            # The rest are u_i.
            
            # Represent state as list of (x, y) for i=1..n-1.
            # u_i = x_i + y_i * B.
            # We must ensure u_i >= 0 and distinct.
            # To simplify, we can search for u_i directly but use lattice structure for initialization?
            # No, the directive says "Model ... by mapping elements to points".
            # So we optimize x_i, y_i.
            
            # Constraints:
            # 1. u_i >= 0
            # 2. u_i distinct
            # 3. 0 is in U (fixed at index 0, x=0, y=0)
            
            # Objective: Score(U)
            
            # SA Parameters
            T = 1.0
            cooling = 0.99
            iterations = 2000
            max_coord = 10 # Search space for x, y
            
            # Initialize with a structured set (e.g. arithmetic progression in y, x=0)
            # u_i = i * B. Points (0, i).
            # This is a geometric progression with base B.
            coords = [(0, i) for i in range(n)]
            # coords[0] is (0,0).
            
            def get_U_from_coords(coords):
                U = set()
                for x, y in coords:
                    u = x + y * B
                    if u >= 0:
                        U.add(u)
                    else:
                        # If u < 0, it's invalid. 
                        # We can penalize heavily or reject.
                        # Let's just return a dummy set to get low score.
                        return None
                if 0 not in U:
                    return None
                return list(U)
            
            current_U = get_U_from_coords(coords)
            if current_U is None:
                current_U = [0] + [i*B for i in range(1, n)]
            current_score = score_set(current_U)
            
            if current_score > best_score:
                best_score = current_score
                best_set = current_U
            
            for it in range(iterations):
                if time.time() > deadline:
                    return best_set
                    
                if it % 100 == 0:
                    T *= cooling
                    if T < 1e-6:
                        break
                
                # Propose change: Pick a random coordinate (i > 0) and perturb x or y
                idx = random.randint(1, n-1)
                x, y = coords[idx]
                
                # Perturbation
                # Small step
                delta_x = random.choice([-1, 0, 1])
                delta_y = random.choice([-1, 0, 1])
                if delta_x == 0 and delta_y == 0:
                    continue
                    
                new_x = x + delta_x
                new_y = y + delta_y
                
                # Optional: Larger jump
                if random.random() < 0.1:
                    new_x += random.randint(-3, 3)
                    new_y += random.randint(-3, 3)
                    
                # Check bounds
                if new_x < -max_coord or new_x > max_coord or new_y < -max_coord or new_y > max_coord:
                    continue
                
                new_coords = list(coords)
                new_coords[idx] = (new_x, new_y)
                
                new_U = get_U_from_coords(new_coords)
                if new_U is None:
                    # Invalid (negative or missing 0)
                    # Accept with very low probability? Or reject.
                    # Reject to keep search valid.
                    continue
                
                # Check distinctness? 
                # get_U_from_coords uses a set, so duplicates are merged.
                # If size changes, it means a collision.
                # We want to maximize |U-U|/|U+U|.
                # Collisions in U reduce n, which usually hurts the ratio potential.
                # But maybe a smaller n with better packing is better?
                # The score function handles it.
                
                new_score = score_set(new_U)
                
                delta = new_score - current_score
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    # Accept
                    current_score = new_score
                    current_U = new_U
                    coords = new_coords
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_set = current_U
                        
    return best_set

# EVOLVE-BLOCK-END
