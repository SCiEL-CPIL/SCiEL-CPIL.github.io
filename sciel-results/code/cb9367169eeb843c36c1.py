import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit

    def compute_score(U):
        if not U or len(U) <= 1:
            return 1.0
        max_u = max(U)
        if max_u == 0:
            return 1.0
        
        n = len(U)
        # For small n, explicit set computation is fast
        # Use sets for exact counting
        sumset = set()
        diffset = set()
        
        # Optimization: If n is small, loop is fine.
        # If n is larger, we might need numpy, but n is likely small for best score.
        for i in range(n):
            ui = U[i]
            for j in range(i, n):
                uj = U[j]
                sumset.add(ui + uj)
                if i != j:
                    d = ui - uj
                    diffset.add(d)
                    diffset.add(-d)
        
        diffset.add(0)
        
        size_sum = len(sumset)
        size_diff = len(diffset)
        
        if size_sum == 0:
            return 1.0
            
        ratio = size_diff / size_sum
        if ratio <= 1.0:
            return 1.0
        
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
            
        return 1.0 + math.log(ratio) / denom

    def optimize_for_n(n, time_limit=10.0):
        """Local search for a fixed set size n."""
        # Initialize with a Golomb ruler if available, else random
        # Golomb rulers for n up to 12 are known
        golomb = {
            1: [0],
            2: [0, 1],
            3: [0, 1, 3],
            4: [0, 1, 4, 6],
            5: [0, 1, 4, 10, 12],
            6: [0, 1, 4, 10, 12, 17],
            7: [0, 1, 4, 10, 12, 17, 25],
            8: [0, 1, 4, 10, 12, 17, 25, 29],
            9: [0, 1, 4, 10, 12, 17, 25, 29, 34],
            10: [0, 1, 4, 10, 12, 17, 25, 29, 34, 41],
            11: [0, 1, 4, 10, 12, 17, 25, 29, 34, 41, 48],
            12: [0, 1, 4, 10, 12, 17, 25, 29, 34, 41, 48, 55],
        }
        
        if n in golomb:
            current_set = list(golomb[n])
        else:
            # Random start
            current_set = [0] + sorted(random.sample(range(1, 100), n-1))
        
        current_score = compute_score(current_set)
        best_score = current_score
        best_set = list(current_set)
        
        # Simulated Annealing / Hill Climbing
        temp = 1.0
        cooling = 0.99
        no_improve = 0
        
        while time.time() < start_time + time_limit:
            # Generate neighbor: change one random element (index > 0)
            idx = random.randint(1, n-1)
            old_val = current_set[idx]
            
            # New value: random in a range around old_val or global range
            # Range should cover potential improvements
            # Since we want small max_u, keep values small
            max_val = max(current_set)
            upper_bound = max(2 * max_val + 10, 50)
            
            # Bias towards smaller values
            if random.random() < 0.5:
                new_val = random.randint(1, max(old_val + 5, 10))
            else:
                new_val = random.randint(1, upper_bound)
            
            if new_val == old_val or new_val in current_set:
                continue
                
            new_set = current_set.copy()
            new_set[idx] = new_val
            new_set.sort()
            
            new_score = compute_score(new_set)
            
            if new_score > current_score:
                current_set = new_set
                current_score = new_score
                no_improve = 0
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current_set)
            else:
                no_improve += 1
                # Accept worse with probability
                delta = current_score - new_score
                if random.random() < math.exp(-delta / temp):
                    current_set = new_set
                    current_score = new_score
            
            temp *= cooling
            if no_improve > 100:
                break
                
        return best_score, best_set

    best_global_score = 1.0
    best_global_set = [0]
    
    # Search for n from 3 to 15
    # Allocate time based on remaining budget
    # Prioritize smaller n as they have higher potential scores
    
    for n in range(3, 16):
        if time.time() > deadline - 5:
            break
            
        # Estimate time per n
        # Smaller n converges faster?
        time_budget = min(10.0, (deadline - time.time()) / (16 - n))
        if time_budget < 1.0:
            continue
            
        score, s = optimize_for_n(n, time_budget)
        
        if score > best_global_score:
            best_global_score = score
            best_global_set = s

    return best_global_set
