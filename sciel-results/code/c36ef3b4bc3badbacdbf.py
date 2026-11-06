import math
import random
import time

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of denominators
    # LCM(2..48)
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def lcm(a, b):
        return a * b // gcd(a, b)

    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Weights are L / d
    weights = [L // d for d in denominators]
    
    # Target: Maximize sum(weights[i] for i in S) such that sum < L
    # Deficit = L - sum
    
    best_sum = 0
    best_set = set()
    
    start_time = time.time()
    max_time = 55.0
    
    def get_sum(s):
        return sum(weights[i] for i in s)
        
    # 1. Greedy Initialization
    # Sort indices by weight descending
    indices_by_weight = sorted(range(n), key=lambda i: weights[i], reverse=True)
    
    current_set = set()
    current_sum = 0
    for i in indices_by_weight:
        if current_sum + weights[i] < L:
            current_set.add(i)
            current_sum += weights[i]
            
    best_set = set(current_set)
    best_sum = current_sum
    
    # 2. Local Search (Hill Climbing) with Restarts
    # We use integer arithmetic for speed.
    
    def local_search(start_set, max_iters=10000):
        curr_set = set(start_set)
        curr_sum = get_sum(curr_set)
        if curr_sum >= L:
            return None, 0
            
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() - start_time < max_time:
            improved = False
            iters += 1
            
            # 1. Try adding
            best_add_i = -1
            best_add_sum = curr_sum
            for i in range(n):
                if i not in curr_set:
                    new_sum = curr_sum + weights[i]
                    if new_sum < L:
                        if new_sum > best_add_sum:
                            best_add_sum = new_sum
                            best_add_i = i
                            
            if best_add_i != -1:
                curr_set.add(best_add_i)
                curr_sum = best_add_sum
                improved = True
                continue
                
            # 2. Try swapping
            best_swap_i = -1
            best_swap_j = -1
            best_swap_sum = curr_sum
            
            # Optimization: Pre-compute list of in/out
            in_list = list(curr_set)
            out_list = [i for i in range(n) if i not in curr_set]
            
            for i in in_list:
                for j in out_list:
                    new_sum = curr_sum - weights[i] + weights[j]
                    if new_sum < L:
                        if new_sum > best_swap_sum:
                            best_swap_sum = new_sum
                            best_swap_i = i
                            best_swap_j = j
                            
            if best_swap_i != -1:
                curr_set.remove(best_swap_i)
                curr_set.add(best_swap_j)
                curr_sum = best_swap_sum
                improved = True
                continue
                
            # 3. Try removing (usually not helpful for maximizing, but might be needed if we get stuck? 
            #    Actually, if we can't add or swap to improve, removing decreases sum, so it's worse.
            #    So we stop if no add/swap improves.
            
        return curr_set, curr_sum

    # Run local search from greedy solution
    res_set, res_sum = local_search(current_set)
    if res_sum > best_sum:
        best_set = res_set
        best_sum = res_sum
        
    # Random Restarts
    # We can generate random subsets and run local search
    # To ensure diversity, we can start with random sets of varying sizes
    
    random.seed(42)
    
    while time.time() - start_time < max_time:
        # Generate a random starting set
        # Strategy: Take a random subset of size k
        # Or perturb the best solution found so far
        
        # Let's try perturbing the best solution
        if best_set:
            start_set = set(best_set)
            # Randomly remove some elements
            for _ in range(random.randint(1, 5)):
                if start_set:
                    item = random.choice(list(start_set))
                    start_set.remove(item)
            # Randomly add some elements
            for _ in range(random.randint(1, 5)):
                candidates = [i for i in range(n) if i not in start_set]
                if candidates:
                    item = random.choice(candidates)
                    if get_sum(start_set) + weights[item] < L:
                        start_set.add(item)
        else:
            start_set = set()
            
        res_set, res_sum = local_search(start_set)
        if res_sum > best_sum:
            best_set = res_set
            best_sum = res_sum
            
    return sorted(best_set)
