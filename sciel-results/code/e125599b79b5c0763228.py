import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin for 150s limit

    def compute_score(U):
        """Compute the score for a given set U using vectorized NumPy operations."""
        if not U:
            return 0.0
        arr = np.array(U, dtype=np.int64)
        max_u = int(arr.max())
        if max_u == 0:
            return 0.0
        
        # Compute all pairwise sums and differences
        # For a set of size n, we have n^2 pairs
        # Use broadcasting for efficiency
        a = arr[:, np.newaxis]
        b = arr[np.newaxis, :]
        
        sums = (a + b).ravel()
        diffs = (a - b).ravel()
        
        # Get unique counts
        size_sum = len(np.unique(sums))
        size_diff = len(np.unique(diffs))
        
        if size_sum == 0:
            return 0.0
            
        ratio = size_diff / size_sum
        if ratio <= 1.0:
            return 1.0
            
        denom = math.log(2 * max_u + 1)
        if denom <= 0:
            return 1.0
            
        return 1.0 + math.log(ratio) / denom

    best_score = 0.0
    best_set = [0]

    # Phase 1: Evaluate initial candidates and small exhaustive searches
    # Start with some known good structures
    initial_sets = [
        [0, 1, 3],
        [0, 1, 2, 4],
        [0, 1, 3, 4],
        [0, 1, 2, 5],
        [0, 1, 4, 6],
        [0, 1, 2, 3, 7],
        [0, 1, 3, 7],
        [0, 1, 2, 5, 10],
        [0, 1, 3, 9],
        [0, 1, 4, 13],
        [0, 1, 4, 10, 12],
        [0, 1, 2, 6, 10],
        [0, 1, 5, 7, 11],
        [0, 1, 2, 7, 12],
        [0, 1, 3, 8, 14],
        [0, 1, 4, 11, 16],
    ]
    
    for s in initial_sets:
        score = compute_score(s)
        if score > best_score:
            best_score = score
            best_set = list(s)
    
    # Also try all subsets of size 3 and 4 from small ranges
    # This is feasible for small max values
    for max_val in range(2, 20):
        if time.time() > deadline - 30:
            break
        # Try all sets of size 3 with 0 < a < b <= max_val
        for a in range(1, max_val + 1):
            if time.time() > deadline - 30:
                break
            for b in range(a + 1, max_val + 1):
                U = [0, a, b]
                score = compute_score(U)
                if score > best_score:
                    best_score = score
                    best_set = list(U)
                    
        # Try all sets of size 4 with 0 < a < b < c <= max_val
        for a in range(1, max_val):
            if time.time() > deadline - 30:
                break
            for b in range(a + 1, max_val):
                if time.time() > deadline - 30:
                    break
                for c in range(b + 1, max_val + 1):
                    U = [0, a, b, c]
                    score = compute_score(U)
                    if score > best_score:
                        best_score = score
                        best_set = list(U)

    # Phase 2: Hill Climbing with vectorized scoring
    # Since compute_score is now fast (O(n^2) with NumPy), we can do many more iterations
    max_steps = 200000
    current_set = list(best_set)
    current_score = best_score
    
    # Use simulated annealing to escape local minima
    T = 0.01  # Initial temperature
    T_min = 0.001
    cooling_rate = 0.9995
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        # Check time budget for scoring (should be fast now)
        neighbor = list(current_set)
        op = random.randint(0, 3)
        
        if op == 0 and len(neighbor) > 1:
            # Remove a random element (not 0)
            if len(neighbor) > 1:
                idx = random.randint(1, len(neighbor) - 1)
                neighbor.pop(idx)
                if len(neighbor) == 0:
                    continue
                if 0 not in neighbor:
                    neighbor.append(0)
                    neighbor.sort()
        elif op == 1:
            # Add a random element
            max_val = max(neighbor) if neighbor else 0
            # Scale the range based on current max
            upper_bound = max(2 * max_val + 20, 50)
            new_val = random.randint(0, upper_bound)
            if new_val not in neighbor:
                neighbor.append(new_val)
                neighbor.sort()
        elif op == 2:
            # Modify a random element (not 0) to a nearby value
            if len(neighbor) > 1:
                idx = random.randint(1, len(neighbor) - 1)
                old_val = neighbor[idx]
                # Try larger jumps for better exploration
                jump = random.randint(-5, 5)
                new_val = max(0, old_val + jump)
                if new_val != old_val and new_val not in neighbor:
                    neighbor[idx] = new_val
                    neighbor.sort()
                else:
                    # If collision, try a different value
                    for _ in range(5):
                        jump = random.randint(-10, 10)
                        new_val = max(0, old_val + jump)
                        if new_val != old_val and new_val not in neighbor:
                            neighbor[idx] = new_val
                            neighbor.sort()
                            break
        else:
            # Insert a new element at a random position
            if neighbor:
                max_val = max(neighbor)
                upper_bound = max(2 * max_val + 20, 50)
                new_val = random.randint(0, upper_bound)
                if new_val not in neighbor:
                    neighbor.append(new_val)
                    neighbor.sort()
                    # Optionally remove the largest element to keep size bounded
                    if len(neighbor) > 12 and random.random() < 0.3:
                        # Remove the element farthest from the median
                        med = np.median(neighbor)
                        idx = int(np.argmin([abs(x - med) for x in neighbor]))
                        # Actually, let's just remove a random non-zero element
                        if len(neighbor) > 1:
                            remove_idx = random.randint(1, len(neighbor) - 1)
                            neighbor.pop(remove_idx)
                            neighbor.sort()

        if not neighbor or 0 not in neighbor:
            continue
            
        new_score = compute_score(neighbor)
        
        # Simulated Annealing acceptance criterion
        if new_score > current_score:
            current_set = neighbor
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
        else:
            # Accept worse moves with probability exp(-delta/T)
            delta = current_score - new_score
            if delta < 0.0001:
                continue
            prob = math.exp(-delta / max(T, T_min))
            if random.random() < prob:
                current_set = neighbor
                current_score = new_score
        
        # Cool down
        T = max(T * cooling_rate, T_min)

    # Phase 3: Final polish - try to improve the best set with more aggressive moves
    if time.time() < deadline - 10:
        polish_steps = 50000
        for step in range(polish_steps):
            if time.time() > deadline - 10:
                break
            
            neighbor = list(best_set)
            op = random.randint(0, 2)
            
            if op == 0 and len(neighbor) > 1:
                idx = random.randint(1, len(neighbor) - 1)
                neighbor.pop(idx)
                if 0 not in neighbor and neighbor:
                    neighbor.append(0)
                    neighbor.sort()
            elif op == 1:
                max_val = max(neighbor) if neighbor else 0
                upper_bound = max(2 * max_val + 30, 80)
                new_val = random.randint(0, upper_bound)
                if new_val not in neighbor:
                    neighbor.append(new_val)
                    neighbor.sort()
            else:
                if len(neighbor) > 1:
                    idx = random.randint(1, len(neighbor) - 1)
                    old_val = neighbor[idx]
                    jump = random.randint(-8, 8)
                    new_val = max(0, old_val + jump)
                    if new_val != old_val and new_val not in neighbor:
                        neighbor[idx] = new_val
                        neighbor.sort()
            
            if not neighbor or 0 not in neighbor:
                continue
                
            new_score = compute_score(neighbor)
            if new_score > best_score:
                best_score = new_score
                best_set = list(neighbor)

    return best_set
# EVOLVE-BLOCK-END
