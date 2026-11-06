import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    N = 32
    M = 512
    max_d = (M - 1) // 2  # 255
    
    # Pre-allocate numpy array for speed
    present_arr = np.zeros(M + 1, dtype=bool)
    
    def calculate_score(s):
        """
        Calculate the number of distinct d > 0 such that 
        a, a+d, a+2d are in s for some a.
        Uses NumPy for vectorized checks.
        """
        # Reset present_arr
        present_arr[:] = False
        for x in s:
            present_arr[x] = True
        
        count = 0
        for d in range(1, max_d + 1):
            # a ranges from 1 to M - 2d
            # If M - 2d < 1, no valid a
            if M - 2 * d < 1:
                break
                
            # Slices for a, a+d, a+2d
            # a: 1 to M-2d
            # a+d: 1+d to M-d
            # a+2d: 1+2d to M
            
            # Vectorized check
            if np.any(present_arr[1 : M - 2*d + 1] & 
                      present_arr[1+d : M - d + 1] & 
                      present_arr[1+2*d : M + 1]):
                count += 1
        return count

    def generate_random_set():
        return set(random.sample(range(1, M + 1), N))

    best_set = generate_random_set()
    best_score = calculate_score(best_set)
    
    if best_score == max_d:
        return list(best_set)

    # Simulated Annealing with multiple restarts
    while time.time() < deadline:
        current_set = generate_random_set()
        current_score = calculate_score(current_set)
        
        if current_score > best_score:
            best_set = current_set
            best_score = current_score
            if best_score == max_d:
                break
        
        # Local Search
        temperature = 5.0
        min_temperature = 0.01
        cooling_rate = 0.995
        steps = 2000 # Adjust based on speed
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Propose move: swap one element
            # Convert set to list for easy indexing
            curr_list = list(current_set)
            curr_set_obj = set(curr_list)
            
            idx = random.randint(0, N - 1)
            old_val = curr_list[idx]
            
            # Find a new value not in the set
            while True:
                new_val = random.randint(1, M)
                if new_val not in curr_set_obj:
                    break
            
            # Apply swap
            curr_list[idx] = new_val
            curr_set_obj.remove(old_val)
            curr_set_obj.add(new_val)
            
            new_score = calculate_score(curr_set_obj)
            
            # Acceptance
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / temperature):
                current_score = new_score
                current_set = curr_set_obj
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set
                    if best_score == max_d:
                        break
                        
            temperature *= cooling_rate
            if temperature < min_temperature:
                break

    return list(best_set)
