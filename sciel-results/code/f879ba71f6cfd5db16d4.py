import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    N = 129
    TARGET_LOW = 128
    TARGET_HIGH = 256
    TARGET_SLICE = slice(TARGET_LOW, TARGET_HIGH + 1)
    
    best_A = None
    best_score = float('inf')
    
    # Helper to compute counts from mask
    def get_counts(mask):
        vals = np.where(mask)[0]
        if len(vals) == 0:
            return np.zeros(257, dtype=int)
        # Outer sum
        sums = vals[:, None] + vals[None, :]
        # Upper triangle (including diagonal)
        i, j = np.triu_indices(len(vals))
        upper_sums = sums[i, j]
        # Filter range
        valid_sums = upper_sums[(upper_sums >= TARGET_LOW) & (upper_sums <= TARGET_HIGH)]
        counts = np.bincount(valid_sums, minlength=257)
        return counts

    # Helper to evaluate
    def evaluate(counts):
        if np.min(counts[TARGET_SLICE]) == 0:
            return float('inf')
        return np.max(counts[TARGET_SLICE])

    # SA Loop with Restarts
    # We will try multiple restarts if time permits
    for restart in range(200): # Max restarts
        if time.time() > deadline:
            break
        
        # Initialize
        # Try different sizes to cover the space
        size = random.randint(35, 55)
        A_mask = np.zeros(N, dtype=bool)
        indices = random.sample(range(N), size)
        for i in indices:
            A_mask[i] = True
        
        A_vals = np.array(indices)
        counts = get_counts(A_mask)
        current_score = evaluate(counts)
        
        if current_score < best_score:
            best_score = current_score
            best_A = A_mask.copy()
        
        # If we found a very good solution, we can stop early?
        # Or just keep refining.
        
        T = 15.0
        T_min = 0.01
        cooling = 0.99995 # Slow cooling to allow many steps
        
        stuck_time = time.time()
        last_improvement_time = time.time()
        
        while time.time() < deadline:
            # Choose move
            if random.random() < 0.5:
                # Add
                candidates = np.where(~A_mask)[0]
                if len(candidates) == 0: continue
                k = candidates[random.randint(0, len(candidates)-1)]
                
                # Calculate delta
                # Pairs added: (k, j) for j in A_vals, and (k, k)
                # Note: A_vals currently does NOT contain k
                new_indices = k + A_vals
                new_indices = np.append(new_indices, 2*k)
                
                # Update counts
                counts[new_indices] += 1
                A_vals = np.append(A_vals, k)
                A_mask[k] = True
                
                new_score = evaluate(counts)
                
                # Acceptance
                accept = False
                if current_score == float('inf'):
                    if new_score < float('inf'):
                        accept = True
                    else:
                        accept = random.random() < 0.5
                else:
                    delta = new_score - current_score
                    if delta < 0:
                        accept = True
                    else:
                        if T > 0:
                            accept = random.random() < np.exp(-delta / T)
                        else:
                            accept = False
                
                if accept:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_A = A_mask.copy()
                        stuck_time = time.time()
                        last_improvement_time = time.time()
                else:
                    # Revert
                    counts[new_indices] -= 1
                    # Remove k from A_vals
                    # Since k was just appended, it's the last element?
                    # Not necessarily if we use np.append.
                    # But we know k is in A_vals.
                    # Using list logic is safer for removal, but np.array is fast for addition.
                    # Let's convert to list for easier removal?
                    # Or use np.where
                    A_vals = A_vals[A_vals != k]
                    A_mask[k] = False
            else:
                # Remove
                if len(A_vals) == 0: continue
                k_idx = random.randint(0, len(A_vals)-1)
                k = A_vals[k_idx]
                
                # Pairs removed: (k, j) for j in A_vals (which includes k)
                # So indices are k + A_vals
                old_indices = k + A_vals
                
                counts[old_indices] -= 1
                A_vals = np.delete(A_vals, k_idx)
                A_mask[k] = False
                
                new_score = evaluate(counts)
                
                # Acceptance
                accept = False
                if current_score == float('inf'):
                    # Removing from an invalid state usually keeps it invalid or makes it worse
                    # But if it was invalid, we want to get out.
                    # However, removing elements makes coverage harder.
                    # So we should rarely accept removals from invalid states.
                    accept = random.random() < 0.1
                else:
                    delta = new_score - current_score
                    if delta < 0:
                        accept = True
                    else:
                        if T > 0:
                            accept = random.random() < np.exp(-delta / T)
                        else:
                            accept = False
                
                if accept:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_A = A_mask.copy()
                        stuck_time = time.time()
                        last_improvement_time = time.time()
                else:
                    # Revert
                    counts[old_indices] += 1
                    A_vals = np.append(A_vals, k)
                    A_mask[k] = True
            
            T *= cooling
            if T < T_min:
                break # End of this SA run, restart
            
            # If stuck for a while, break early to restart
            if time.time() - stuck_time > 3.0:
                break

    if best_A is None:
        # Fallback to seed
        return list(range(64, 129))
    
    return np.where(best_A)[0].tolist()
