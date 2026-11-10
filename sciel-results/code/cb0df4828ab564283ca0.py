import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    N = 128
    TARGET_START = 128
    TARGET_END = 256
    TARGETS = np.array(range(TARGET_START, TARGET_END + 1))
    
    # Fallback: Seed solution
    seed_A = list(range(64, 129))
    
    def calculate_metrics(sum_counts):
        target_counts = sum_counts[TARGET_START:TARGET_END+1]
        num_missing = np.sum(target_counts == 0)
        max_mult = np.max(target_counts)
        return max_mult, num_missing

    def cost_from_metrics(max_mult, num_missing):
        return max_mult + 10000 * num_missing

    best_A = seed_A
    best_cost = float('inf')
    
    # Calculate cost of seed
    sum_counts_seed = np.zeros(257, dtype=int)
    seed_set = set(seed_A)
    seed_list = list(seed_set)
    for i in range(len(seed_list)):
        a = seed_list[i]
        sum_counts_seed[2*a] += 1
        for j in range(i+1, len(seed_list)):
            b = seed_list[j]
            sum_counts_seed[a+b] += 1
    seed_max, seed_missing = calculate_metrics(sum_counts_seed)
    best_cost = cost_from_metrics(seed_max, seed_missing)
    
    candidate_sizes = [20, 25, 30, 35, 40, 45]
    
    for k in candidate_sizes:
        if time.time() > deadline - 2:
            break
            
        # Initialize
        A = set(random.sample(range(N+1), k))
        A_list = list(A)
        
        # Initialize sum_counts
        sum_counts = np.zeros(257, dtype=int)
        for i in range(len(A_list)):
            a = A_list[i]
            sum_counts[2*a] += 1
            for j in range(i+1, len(A_list)):
                b = A_list[j]
                sum_counts[a+b] += 1
                
        max_mult, num_missing = calculate_metrics(sum_counts)
        current_cost = cost_from_metrics(max_mult, num_missing)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_A = list(A)
            
        # Annealing
        T = 5.0
        decay = 0.9995
        min_T = 0.1
        
        while time.time() < deadline - 2:
            # Pick random element to remove
            idx_remove = random.randint(0, len(A_list)-1)
            elem_remove = A_list[idx_remove]
            
            # Pick random candidate to add
            candidate = random.randint(0, N)
            if candidate == elem_remove:
                continue
            if candidate in A:
                continue
                
            # Remove elem_remove from sum_counts
            for w in A:
                if w == elem_remove:
                    continue
                sum_counts[elem_remove + w] -= 1
            sum_counts[2*elem_remove] -= 1
            
            # Add candidate to sum_counts
            # Interactions with remaining elements (A - {elem_remove})
            for w in A:
                if w == elem_remove:
                    continue
                sum_counts[candidate + w] += 1
            sum_counts[2*candidate] += 1
            
            # Update A structures
            A.remove(elem_remove)
            A.add(candidate)
            A_list[idx_remove] = candidate
            
            # Calculate new cost
            new_max, new_missing = calculate_metrics(sum_counts)
            new_cost = cost_from_metrics(new_max, new_missing)
            
            delta = new_cost - current_cost
            
            if delta < 0:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_A = list(A)
            else:
                if random.random() < np.exp(-delta / T):
                    current_cost = new_cost
                else:
                    # Revert
                    # Remove candidate
                    for w in A:
                        if w == candidate:
                            continue
                        sum_counts[candidate + w] -= 1
                    sum_counts[2*candidate] -= 1
                    
                    # Add back elem_remove
                    for w in A:
                        if w == candidate:
                            continue
                        sum_counts[elem_remove + w] += 1
                    sum_counts[2*elem_remove] += 1
                    
                    A.remove(candidate)
                    A.add(elem_remove)
                    A_list[idx_remove] = elem_remove
                    
            T = max(T * decay, min_T)
            
    return best_A
