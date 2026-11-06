import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin: 120s limit, use 118s
    
    N = 1000
    max_val = 2 * N + 1  # Max possible sum is 2000, need index up to 2000
    
    def greedy_build(strategy_func, deadline_check):
        """
        Build a B2[2] set using a greedy approach with a specific strategy for choosing candidates.
        strategy_func: function that returns the next candidate to try.
        """
        chosen = np.array([], dtype=np.int32)
        counts = np.zeros(max_val, dtype=np.int8)
        chosen_set = set()
        
        candidates = strategy_func()
        
        for x in candidates:
            if time.time() > deadline_check:
                break
            if x in chosen_set:
                continue
            
            # Check if adding x violates constraints
            # New sums: x + y for y in chosen, and 2*x
            if len(chosen) == 0:
                new_sums = np.array([2 * x], dtype=np.int32)
            else:
                # Vectorized sum calculation
                new_sums = x + chosen
                new_sums = np.concatenate([new_sums, [2 * x]])
            
            # Check counts
            # If any count + 1 > 2, invalid
            new_counts = counts[new_sums] + 1
            if np.any(new_counts > 2):
                continue
            
            # Valid, add it
            chosen = np.append(chosen, x)
            chosen_set.add(x)
            counts[new_sums] = new_counts
            
        return chosen, counts, chosen_set

    def local_search(chosen, counts, chosen_set, deadline):
        """
        Vectorized local search to improve the set size.
        Moves: Add, Swap, Remove.
        """
        best_set = chosen.copy()
        best_size = len(chosen)
        
        rng = random.Random(42)
        
        # Precompute some stats if needed? No, dynamic.
        
        while time.time() < deadline:
            # 1. Try to add a random element
            # To speed up rejection sampling, we can just pick a random int and check.
            x = rng.randint(1, 1000)
            if x not in chosen_set:
                if len(chosen) > 0:
                    new_sums = x + chosen
                    new_sums = np.concatenate([new_sums, [2 * x]])
                else:
                    new_sums = np.array([2 * x], dtype=np.int32)
                
                new_counts = counts[new_sums] + 1
                if not np.any(new_counts > 2):
                    # Valid add
                    chosen = np.append(chosen, x)
                    chosen_set.add(x)
                    counts[new_sums] = new_counts
                    if len(chosen) > best_size:
                        best_set = chosen.copy()
                        best_size = len(chosen)
            
            # 2. Try a swap
            # Remove a random element, try to add a random element
            if len(chosen) > 0:
                idx_rm = rng.randint(0, len(chosen) - 1)
                y_rm = chosen[idx_rm]
                
                # Temporarily remove y_rm
                # Update counts
                if len(chosen) > 1:
                    rem_sums = y_rm + np.delete(chosen, idx_rm)
                    rem_sums = np.concatenate([rem_sums, [2 * y_rm]])
                else:
                    rem_sums = np.array([2 * y_rm], dtype=np.int32)
                
                # Decrement counts
                counts[rem_sums] -= 1
                chosen = np.delete(chosen, idx_rm)
                chosen_set.remove(y_rm)
                
                # Try to add x
                x = rng.randint(1, 1000)
                if x not in chosen_set:
                    if len(chosen) > 0:
                        add_sums = x + chosen
                        add_sums = np.concatenate([add_sums, [2 * x]])
                    else:
                        add_sums = np.array([2 * x], dtype=np.int32)
                    
                    add_counts = counts[add_sums] + 1
                    if not np.any(add_counts > 2):
                        # Valid swap
                        chosen = np.append(chosen, x)
                        chosen_set.add(x)
                        counts[add_sums] = add_counts
                        if len(chosen) > best_size:
                            best_set = chosen.copy()
                            best_size = len(chosen)
                    else:
                        # Invalid, revert removal of y_rm
                        counts[rem_sums] += 1
                        chosen = np.append(chosen, y_rm)
                        chosen_set.add(y_rm)
                else:
                    # x was in set, revert removal
                    counts[rem_sums] += 1
                    chosen = np.append(chosen, y_rm)
                    chosen_set.add(y_rm)
            
            # 3. Occasionally remove an element to escape local optima (20% chance)
            if rng.random() < 0.2 and len(chosen) > 0:
                idx_rm = rng.randint(0, len(chosen) - 1)
                y_rm = chosen[idx_rm]
                
                if len(chosen) > 1:
                    rem_sums = y_rm + np.delete(chosen, idx_rm)
                    rem_sums = np.concatenate([rem_sums, [2 * y_rm]])
                else:
                    rem_sums = np.array([2 * y_rm], dtype=np.int32)
                
                counts[rem_sums] -= 1
                chosen = np.delete(chosen, idx_rm)
                chosen_set.remove(y_rm)
                
                # Do not revert, just let it be smaller to allow future adds
                # This is a risky move but helps escape local optima

        return best_set

    # Strategy 1: Natural order 1..1000
    # We can try to optimize the greedy phase slightly.
    # Standard greedy is usually good enough as a baseline.
    
    # Generate candidates: 1 to 1000
    def gen_natural():
        return range(1, 1001)
        
    # Alternative: Random shuffled order for diversity? 
    # Let's stick to natural for the first pass to ensure a strong baseline, 
    # then use local search to improve.
    
    chosen, counts, chosen_set = greedy_build(gen_natural, start_time + 5)
    
    if time.time() > deadline:
        return list(chosen)
        
    best_set = local_search(chosen, counts, chosen_set, deadline)
    
    return list(best_set)

# EVOLVE-BLOCK-END
