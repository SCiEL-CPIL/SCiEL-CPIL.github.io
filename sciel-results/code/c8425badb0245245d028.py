import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill

    def compute_score(U):
        """
        Compute the score for a set U.
        Score = 1 + log(|U-U|/|U+U|) / log(2 * max(U) + 1)
        """
        if not U or len(U) < 2:
            return 0.0
        
        U_arr = np.array(sorted(U), dtype=np.int64)
        max_U = int(U_arr.max())
        if max_U == 0:
            return 0.0
        
        # Compute sumset
        # For N up to ~100, N^2 is 10,000, which is fast for numpy unique
        sums = U_arr[:, None] + U_arr[None, :]
        sumset_size = len(np.unique(sums))
        
        # Compute diffset
        diffs = U_arr[:, None] - U_arr[None, :]
        diffset_size = len(np.unique(diffs))
        
        if sumset_size == 0:
            return 0.0
            
        ratio = diffset_size / sumset_size
        if ratio <= 0:
            return 0.0
        
        denom = math.log(2 * max_U + 1)
        if denom == 0:
            return 0.0
            
        exponent = 1 + math.log(ratio) / denom
        return exponent

    def simulate_annealing(M, N_target, time_limit):
        """
        Run simulated annealing on a binary vector of length M+1.
        We try to select exactly N_target elements including 0.
        """
        # Initialize candidate pool
        # 0 is always in the set
        # We choose M as the upper bound of the search space
        # We want to pick N_target-1 other elements from [1, M]
        
        # Initial state: random selection
        candidates = list(range(1, M + 1))
        random.shuffle(candidates)
        current_elements = [0] + candidates[:N_target - 1]
        current_elements.sort()
        current_score = compute_score(current_elements)
        
        best_elements = current_elements[:]
        best_score = current_score
        
        T = 0.1  # Initial temperature
        cooling = 0.999  # Cooling factor
        
        # Precompute indices for fast swapping
        # We represent the set as a sorted list. 
        # To speed up, we can maintain a boolean mask for membership in [0, M]
        # But since N is small relative to M potentially, list operations are okay.
        
        # Optimization: Use a set for O(1) lookup, but we need sorted for numpy.
        # We will maintain a sorted list. Insertion/deletion is O(N).
        # N is ~20-50, so O(N) is fine.
        
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < time_limit:
            if no_improve_count > max_no_improve:
                # Restart with slightly different parameters or break
                T = 0.1
                no_improve_count = 0
                # Re-randomize slightly
                random.shuffle(candidates)
                current_elements = [0] + candidates[:N_target - 1]
                current_elements.sort()
                current_score = compute_score(current_elements)
                if current_score > best_score:
                    best_score = current_score
                    best_elements = current_elements[:]
                continue

            # Pick a random element to remove (not 0)
            # And a random candidate to add (not in set)
            # To avoid O(M) scan for candidates, we can just pick a random index in [1, M]
            # and check if it's in the set. If it is, pick another.
            
            # Remove an element
            remove_idx = random.randint(1, len(current_elements) - 1)
            elem_to_remove = current_elements[remove_idx]
            
            # Add an element
            # Pick a random value from [1, M]
            # If it's already in the set, we might just swap or reject.
            # Let's do a swap: pick a random element in set (not 0) and a random value in [1, M].
            # If the new value is already in the set, it's a no-op or duplicate, reject.
            
            new_val = random.randint(1, M)
            
            if new_val == elem_to_remove:
                continue
            if new_val in current_elements:
                continue
                
            # Perform swap
            old_elements = current_elements[:]
            current_elements[remove_idx] = new_val
            current_elements.sort()
            
            new_score = compute_score(current_elements)
            
            delta = new_score - current_score
            
            # Acceptance criterion
            if delta > 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_elements = current_elements[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Revert
                current_elements = old_elements
                no_improve_count += 1
            
            T *= cooling
            if T < 0.001:
                T = 0.001

        return best_elements, best_score

    # Strategy:
    # Try different sizes N (set cardinality) and M (max value).
    # The score depends on N (via |U+U|, |U-U|) and M (via denominator).
    # Generally, larger N increases |U-U|/|U+U| towards 1 (or slightly above for structured sets),
    # but increases the denominator if M grows.
    # We want a set that is "Golomb-like" to maximize differences.
    
    # Heuristic: N around 20-50, M around 2*N^2 to 5*N^2 for Golomb rulers.
    # Let's try a few configurations.
    
    configs = [
        (20, 1000),
        (30, 2000),
        (40, 4000),
        (50, 6000),
        (30, 1500),
        (40, 3000),
    ]
    
    # Shuffle to ensure robustness
    random.shuffle(configs)
    
    best_overall_score = 0.0
    best_overall_set = [0, 1]
    
    # Allocate time to each config
    time_per_config = (deadline - time.time()) / len(configs)
    
    for N, M in configs:
        if time.time() >= deadline - 2.0:
            break
            
        sub_deadline = time.time() + time_per_config
        try:
            improved_set, improved_score = simulate_annealing(M, N, sub_deadline)
            if improved_score > best_overall_score:
                best_overall_score = improved_score
                best_overall_set = improved_set
        except Exception:
            pass

    # Final check: ensure 0 is in the set and it's a list of ints
    if 0 not in best_overall_set:
        best_overall_set.append(0)
    
    return sorted(best_overall_set)

if __name__ == "__main__":
    # Quick self-test
    pass
