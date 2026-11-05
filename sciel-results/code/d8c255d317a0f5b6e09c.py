import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    N = 128
    min_sum = 128
    max_sum = 256
    target_sums = range(min_sum, max_sum + 1)
    
    def get_counts(A):
        """Returns array of counts for sums in [0, 256]"""
        counts = np.zeros(max_sum + 1, dtype=int)
        A_list = list(A)
        k = len(A_list)
        for i in range(k):
            ai = A_list[i]
            s = ai + ai
            if s <= max_sum:
                counts[s] += 1
            for j in range(i + 1, k):
                aj = A_list[j]
                s = ai + aj
                if s <= max_sum:
                    counts[s] += 1
        return counts

    def get_cost(A):
        counts = get_counts(A)
        target_counts = counts[min_sum:max_sum+1]
        if np.any(target_counts == 0):
            return 1000 + np.sum(target_counts == 0)
        return np.max(target_counts)

    # Multi-start Simulated Annealing
    best_A = None
    best_cost = 10000
    
    random.seed(42)
    
    # Estimate good set size
    # k^2/2 >= 129 => k >= 16. 
    # Let's try sizes 18 to 25.
    
    while time.time() < deadline:
        # Choose a random initial size
        k_init = random.randint(18, 25)
        A = set(random.sample(range(0, N + 1), k_init))
        
        # Initial greedy improvement to get close to feasible
        # 1. Ensure coverage by adding elements
        while time.time() < deadline - 5.0: # Reserve time for SA
            counts = get_counts(A)
            target_counts = counts[min_sum:max_sum+1]
            if np.any(target_counts > 0) == False: # Not fully covered
                 # Find uncovered
                 uncovered = [s for s in target_sums if target_counts[s - min_sum] == 0]
                 if not uncovered: break
                 
                 # Add best element
                 best_x = None
                 best_gain = -1
                 for x in range(0, N + 1):
                     if x in A: continue
                     gain = 0
                     for y in A:
                         s = x + y
                         if min_sum <= s <= max_sum and target_counts[s - min_sum] == 0:
                             gain += 1
                     if gain > best_gain:
                         best_gain = gain
                         best_x = x
                 if best_x is not None and best_gain > 0:
                     A.add(best_x)
                 else:
                     break # Stuck in greedy
            else:
                break # Covered
        
        if get_cost(A) >= 1000: # Still not covered
            continue

        # Simulated Annealing
        current_cost = get_cost(A)
        current_A = set(A)
        
        T = 10.0
        cooling = 0.99
        min_T = 0.1
        
        # Run SA for a fixed number of steps or time
        sa_start = time.time()
        while time.time() - sa_start < 2.0 and T > min_T: # 2 seconds per restart
            # Generate move
            move_type = random.random()
            new_A = set(current_A)
            
            if move_type < 0.5:
                # Swap
                if len(new_A) > 0:
                    x_in = random.choice(list(new_A))
                    x_out = random.choice([x for x in range(0, N + 1) if x not in new_A])
                    new_A.remove(x_in)
                    new_A.add(x_out)
            elif move_type < 0.8:
                # Add
                x_out = random.choice([x for x in range(0, N + 1) if x not in new_A])
                new_A.add(x_out)
            else:
                # Remove
                if len(new_A) > 15:
                    x_in = random.choice(list(new_A))
                    new_A.remove(x_in)
            
            new_cost = get_cost(new_A)
            
            # Acceptance
            if new_cost < current_cost:
                current_cost = new_cost
                current_A = new_A
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = set(current_A)
            else:
                # Metropolis criterion
                # If cost is 1000+, it's a big penalty.
                delta = new_cost - current_cost
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_cost = new_cost
                    current_A = new_A
            
            T *= cooling
        
        # If we found a very good solution, we can break early?
        if best_cost <= 2:
             break

    if best_A is None:
        # Fallback: Return a valid set if possible, or best effort
        # If best_A is None, it means we never found a covered set in the loop?
        # Unlikely with greedy phase.
        # Let's do one last greedy fill
        A = set(random.sample(range(0, N + 1), 20))
        while time.time() < deadline - 1.0:
            counts = get_counts(A)
            target_counts = counts[min_sum:max_sum+1]
            if np.any(target_counts == 0):
                 uncovered = [s for s in target_sums if target_counts[s - min_sum] == 0]
                 if not uncovered: break
                 best_x = None
                 best_gain = -1
                 for x in range(0, N + 1):
                     if x in A: continue
                     gain = 0
                     for y in A:
                         s = x + y
                         if min_sum <= s <= max_sum and target_counts[s - min_sum] == 0:
                             gain += 1
                     if gain > best_gain:
                         best_gain = gain
                         best_x = x
                 if best_x is not None and best_gain > 0:
                     A.add(best_x)
                 else:
                     break
            else:
                break
        best_A = A

    return list(best_A)
