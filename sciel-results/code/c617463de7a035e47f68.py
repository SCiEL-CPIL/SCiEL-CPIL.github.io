import time
import random
import math
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 20
    max_val = 10000
    
    # Precompute all pairs (i, j) with i <= j for n=20
    pairs_i = np.array([i for i in range(n) for j in range(i, n)], dtype=np.int32)
    pairs_j = np.array([j for i in range(n) for j in range(i, n)], dtype=np.int32)
    
    # Generate all 7-smooth numbers up to 10000
    smooths = set()
    for a in range(14):  # 2^13 > 8000
        p2 = 2 ** a
        if p2 > max_val: break
        for b in range(9): # 3^8 > 6000
            p3 = 3 ** b
            if p2 * p3 > max_val: break
            for c in range(6): # 5^5 > 3000
                p5 = 5 ** c
                if p2 * p3 * p5 > max_val: break
                for d in range(5): # 7^4 > 2400
                    p7 = 7 ** d
                    val = p2 * p3 * p5 * p7
                    if val > max_val: break
                    smooths.add(val)
    
    smooth_list = sorted(list(smooths))
    smooth_set = set(smooth_list)
    
    # Map value to index in smooth_list for quick lookup? 
    # Actually, we just need to check membership in smooth_set and uniqueness in A.
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A using NumPy vectorization."""
        A_arr = np.array(A, dtype=np.int64)
        sums = A_arr[pairs_i] + A_arr[pairs_j]
        prods = A_arr[pairs_i] * A_arr[pairs_j]
        sumset_size = len(np.unique(sums))
        prodset_size = len(np.unique(prods))
        return max(sumset_size, prodset_size)
    
    # Initial set: pick n random smooth numbers, biased towards smaller ones
    # Smaller numbers tend to have more collisions in products and sums
    # Let's sort smooth_list and pick from the first half
    half_index = len(smooth_list) // 2
    initial_pool = smooth_list[:half_index]
    
    best_set = None
    best_score = float('inf')
    
    # Generate a few initial candidates
    for _ in range(5):
        # Pick n distinct random numbers from the pool
        A = random.sample(initial_pool, n)
        A.sort()
        score = evaluate(A)
        if score < best_score:
            best_score = score
            best_set = A[:]
            
    # Hill Climbing / Simulated Annealing on the smooth lattice
    # We restrict moves to swapping an element with another smooth number
    # This keeps us in the "good" region of the space where product sets are small
    
    current_set = best_set[:]
    current_score = best_score
    
    # Precompute neighbors? No, too many. 
    # Instead, for each position, pick a random smooth number from the pool.
    # To be efficient, we can pre-shuffle the pool or just pick random indices.
    
    # Let's use a simple SA with temperature decay
    temperature = 10.0
    cooling_rate = 0.99
    min_temp = 0.1
    
    # To avoid expensive random.sample every time, we can just pick a random index 
    # from the smooth_list and check if it's valid.
    # But smooth_list can be large (~1000+). 
    # Let's count how many 7-smooth numbers <= 10000 there are.
    # It's not huge. 
    num_smooths = len(smooth_list)
    
    while time.time() < deadline and temperature > min_temp:
        # Pick a random index to modify
        idx = random.randint(0, n - 1)
        old_val = current_set[idx]
        
        # Pick a new value from smooth_list
        # To bias towards smaller numbers (which are denser), we can pick from a random prefix
        # or just pick uniformly from the whole list.
        # Uniform is fine, but let's try picking from the first 80% to stay in the dense region
        max_idx = int(num_smooths * 0.8)
        new_idx = random.randint(0, max_idx - 1)
        new_val = smooth_list[new_idx]
        
        if new_val == old_val:
            continue
            
        # Check if new_val is already in current_set (excluding idx)
        if new_val in current_set:
            continue
            
        # Perform move
        current_set[idx] = new_val
        new_score = evaluate(current_set)
        
        delta_score = new_score - current_score
        
        if delta_score < 0 or random.random() < math.exp(-delta_score / temperature):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_set = current_set[:]
        else:
            current_set[idx] = old_val
            
        temperature *= cooling_rate
        
        # Safety check
        if time.time() > deadline - 2:
            break
            
    return sorted(best_set)
