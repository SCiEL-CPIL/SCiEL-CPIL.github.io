import time
import random
import math

def set_sizes(A):
    A = list(A)
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return len(sums), len(prods)

def objective(A):
    s, p = set_sizes(A)
    return max(s, p)

def construct():
    start = time.time()
    deadline = start + 55.0  # Safety margin under 60s
    
    # Initial best: Standard AP
    best_set = list(range(1, 21))
    best_obj = objective(best_set)
    
    def try_candidate(A):
        nonlocal best_set, best_obj
        if len(set(A)) != 20:
            return
        if any(x < 1 or x > 10000 for x in A):
            return
        A = list(A)
        obj = objective(A)
        if obj < best_obj:
            best_obj = obj
            best_set = A[:]
            
    # 1. Quick evaluation of structured baselines
    # APs
    for k in range(1, 20):
        if time.time() > deadline:
            break
        A = list(range(k, k + 20))
        try_candidate(A)
        
    # APs with step
    for d in [2, 3, 4, 5]:
        for a in range(1, 10):
            if time.time() > deadline:
                break
            A = [a + i * d for i in range(20)]
            if A[-1] > 10000: continue
            try_candidate(A)
            
    # "Log-uniform" sets (approximating GP)
    # This creates sets with smaller product sets potentially
    random.seed(42)
    for _ in range(10):
        if time.time() > deadline:
            break
        # Generate 20 points in log space [0, log(10000)]
        log_max = math.log(10000)
        points = sorted(random.uniform(0, log_max) for _ in range(20))
        A = [max(1, int(math.exp(p))) for p in points]
        # Ensure distinct
        if len(set(A)) == 20:
            try_candidate(A)
        else:
            # Fix duplicates by jittering
            A_unique = list(set(A))
            while len(A_unique) < 20:
                A_unique.append(random.randint(1, 10000))
            try_candidate(A_unique[:20])

    # 2. Simulated Annealing
    # We use SA to escape local minima found by simple hill climbing.
    
    def simulated_annealing(start_set, max_iters=1000):
        current_set = list(start_set)
        current_obj = objective(current_set)
        T = 50.0  # Initial temperature
        T_min = 0.1
        cooling_rate = 0.995
        
        best_local_set = current_set[:]
        best_local_obj = current_obj
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            # Generate a neighbor
            move_type = random.random()
            neighbor = current_set[:]
            
            if move_type < 0.5:
                # Replace one element
                idx = random.randint(0, 19)
                new_val = random.randint(1, 10000)
                if new_val not in current_set or new_val == current_set[idx]:
                    neighbor[idx] = new_val
            elif move_type < 0.8:
                # Swap two elements
                i, j = random.sample(range(20), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            else:
                # Perturb one element
                idx = random.randint(0, 19)
                delta = random.choice([-1, 1, -2, 2, -5, 5, -10, 10])
                new_val = current_set[idx] + delta
                if 1 <= new_val <= 10000 and new_val not in current_set:
                    neighbor[idx] = new_val
                else:
                    # If invalid, just replace with random
                    new_val = random.randint(1, 10000)
                    if new_val not in current_set:
                        neighbor[idx] = new_val
                    else:
                        continue # Skip this iteration if we can't make a valid move easily

            # Validate distinctness
            if len(set(neighbor)) != 20:
                continue
                
            neighbor_obj = objective(neighbor)
            delta_obj = neighbor_obj - current_obj
            
            # Acceptance criterion
            if delta_obj < 0 or random.random() < math.exp(-delta_obj / T):
                current_set = neighbor
                current_obj = neighbor_obj
                
                if current_obj < best_local_obj:
                    best_local_obj = current_obj
                    best_local_set = current_set[:]
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        return best_local_set, best_local_obj

    # Run SA from multiple diverse starts
    random.seed(123)
    
    # Start 1: AP
    seed1 = list(range(1, 21))
    # Start 2: Log-uniform
    log_max = math.log(10000)
    points = sorted(random.uniform(0, log_max) for _ in range(20))
    seed2 = [max(1, int(math.exp(p))) for p in points]
    if len(set(seed2)) < 20:
        seed2 = list(set(seed2))
        while len(seed2) < 20:
            seed2.append(random.randint(1, 10000))
        seed2 = seed2[:20]
    # Start 3: Random
    seed3 = [random.randint(1, 10000) for _ in range(20)]
    if len(set(seed3)) < 20:
        seed3 = list(set(seed3))
        while len(seed3) < 20:
            seed3.append(random.randint(1, 10000))
        seed3 = seed3[:20]
    # Start 4: Small numbers (dense)
    seed4 = list(range(1, 21))
    random.shuffle(seed4)
    # Start 5: Multiples of 2
    seed5 = [i * 2 for i in range(1, 21)]

    seeds = [seed1, seed2, seed3, seed4, seed5]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        try_candidate(seed)
        sa_set, sa_obj = simulated_annealing(seed, max_iters=500)
        if sa_obj < best_obj:
            best_obj = sa_obj
            best_set = sa_set[:]
            
    # Final polish: Local search (greedy) on the best found set
    if time.time() < deadline - 5.0:
        current_set = best_set[:]
        current_obj = best_obj
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Try replacing each element with a few random values
            for i in range(20):
                for _ in range(10):
                    new_val = random.randint(1, 10000)
                    if new_val in current_set and new_val != current_set[i]:
                        continue
                    new_set = current_set[:]
                    new_set[i] = new_val
                    if len(set(new_set)) == 20:
                        obj = objective(new_set)
                        if obj < current_obj:
                            current_set = new_set
                            current_obj = obj
                            improved = True
                            break # Restart outer loop to re-evaluate with new context? Or just continue. Let's break to restart.
                if improved:
                    break
            # Try small perturbations
            if not improved:
                for i in range(20):
                    for delta in [-1, 1, -2, 2, -5, 5]:
                        new_val = current_set[i] + delta
                        if 1 <= new_val <= 10000 and new_val not in current_set:
                            new_set = current_set[:]
                            new_set[i] = new_val
                            obj = objective(new_set)
                            if obj < current_obj:
                                current_set = new_set
                                current_obj = obj
                                improved = True
                                break
                    if improved:
                        break
            
        if current_obj < best_obj:
            best_obj = current_obj
            best_set = current_set[:]

    return best_set
