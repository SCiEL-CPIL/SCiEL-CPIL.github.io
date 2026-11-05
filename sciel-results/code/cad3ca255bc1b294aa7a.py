import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5-second safety margin before 120s limit
    
    cap = 1000
    best_set = set()
    best_len = 0
    rng = random.Random(42)
    
    # Pre-allocate numpy arrays for speed
    # We will use a greedy approach with numpy vectorization for the feasibility check
    
    while time.time() < deadline:
        # Random order of candidates
        order = list(range(1, cap + 1))
        rng.shuffle(order)
        
        chosen = []
        chosen_set = set()
        # counts[s] = number of representations of s as a+b with a<=b from chosen
        # Max sum is 2000
        counts = np.zeros(2 * cap + 1, dtype=np.int32)
        
        # We maintain a numpy array of chosen elements for fast vectorized sum checks
        # But appending to numpy is slow. We can use a list and convert, or just use list for checks if size is small?
        # Size will be ~700. 
        # Vectorized check: new_sums = chosen_np + x. 
        # Let's keep chosen as a list and a numpy array.
        
        chosen_np = np.empty(0, dtype=np.int64)
        
        for x in order:
            if time.time() > deadline:
                break
            
            # Check if x can be added
            # New sums: x + y for y in chosen, and 2*x
            if chosen_np.size == 0:
                new_sums = np.array([2 * x], dtype=np.int64)
            else:
                # x + every element in chosen
                # We need to check counts[new_sums] < 2
                new_sums = chosen_np + x
                # Also add 2*x
                # To avoid allocation overhead of concatenate in tight loop, 
                # we can check 2*x separately and the rest together.
                
                # Check 2*x
                if counts[2 * x] >= 2:
                    continue
                    
                # Check x + y for y in chosen
                # If any counts[x+y] >= 2, invalid
                # Vectorized: np.any(counts[new_sums] >= 2)
                if np.any(counts[new_sums] >= 2):
                    continue
                
                # Valid
                counts[new_sums] += 1
                counts[2 * x] += 1
                chosen.append(x)
                chosen_set.add(x)
                chosen_np = np.append(chosen_np, x)
                
        # Local Improvement Phase
        # Try to add any remaining elements that fit
        # This is a simple greedy pass over all 1..1000
        # If we can add more, great.
        
        # Sort chosen for deterministic processing in improvement? Not strictly necessary.
        # But let's try to add any x not in chosen_set
        
        # We need to be careful with counts if we modify the set.
        # Let's do a second pass.
        
        # To make this faster, we can iterate through 1..1000 and check.
        # This is O(N * |S|) which is 1000 * 700 = 700k operations. Very fast.
        
        # However, we must update counts correctly.
        
        # Let's try a "fill" phase
        # We'll iterate 1 to 1000
        # If x not in chosen_set, check if it fits.
        
        # Note: The order of filling might matter. 
        # We can try multiple orders for the fill phase too, or just ascending.
        # Ascending is often good for packing.
        
        filled = False
        # We'll do a few passes of filling to catch interactions
        for _ in range(3):
            if time.time() > deadline:
                break
            for x in range(1, cap + 1):
                if x in chosen_set:
                    continue
                if chosen_np.size == 0:
                    new_sums = np.array([2 * x], dtype=np.int64)
                else:
                    new_sums = chosen_np + x
                    if counts[2 * x] >= 2:
                        continue
                    if np.any(counts[new_sums] >= 2):
                        continue
                    counts[new_sums] += 1
                    counts[2 * x] += 1
                    chosen_set.add(x)
                    chosen.append(x)
                    chosen_np = np.append(chosen_np, x)
                    filled = True
            if not filled:
                break
        
        # Try swaps?
        # If we have time, try to remove one element and add two others?
        # This is complex. Let's stick to the greedy + fill for now, 
        # as the current program (score 0.979) uses pure greedy.
        # The inspirations used numpy greedy. 
        # The current program uses Python set/dict greedy.
        # Numpy should be faster, allowing more restarts.
        
        if len(chosen_set) > best_len:
            best_len = len(chosen_set)
            best_set = set(chosen_set)
            
            # Early exit if we reach a very high count?
            # Theoretical max is around 1000 * sqrt(2) / something?
            # For B2[2], density is higher than Sidon (B2[1]).
            # Sidon in [1, N] is ~ sqrt(N). For N=1000, ~31.
            # B2[2] allows 2 reps. Density is higher.
            # Asymptotic constant is at least 3/2 for the density? No, that's for something else.
            # Let's just maximize.
            
    return list(best_set)
