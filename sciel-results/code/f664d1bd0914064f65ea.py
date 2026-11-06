import numpy as np
import time
import random
import math

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    # Precompute all 4-APs as index tuples
    all_aps = []
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                all_aps.append((a - 1, b - 1, c - 1, e - 1))
    
    n_aps = len(all_aps)
    
    # Precompute aps_containing[i] = list of (j, k, l) for each index i
    aps_containing = [[] for _ in range(N)]
    for ap in all_aps:
        i0, i1, i2, i3 = ap
        aps_containing[i0].append((i1, i2, i3))
        aps_containing[i1].append((i0, i2, i3))
        aps_containing[i2].append((i0, i1, i3))
        aps_containing[i3].append((i0, i1, i2))

    def count_mono(x):
        """Count monochromatic 4-APs. x is a numpy array of 0/1."""
        count = 0
        for i0, i1, i2, i3 in all_aps:
            if x[i0] == x[i1] == x[i2] == x[i3]:
                count += 1
        return count

    def get_delta(x, i):
        """Compute change in mono count if we flip x[i]."""
        old_val = x[i]
        new_val = 1 - old_val
        delta = 0
        for j, k, l in aps_containing[i]:
            if x[j] == new_val and x[k] == new_val and x[l] == new_val:
                delta += 1
            if x[j] == old_val and x[k] == old_val and x[l] == old_val:
                delta -= 1
        return delta

    def greedy_local_search(x, max_time=None):
        """Greedy single-flip hill climbing. Returns improved x and score."""
        current_score = count_mono(x)
        if max_time is None:
            max_time = deadline - 5.0
        
        improved = True
        while improved:
            if time.time() > max_time:
                break
            improved = False
            indices = list(range(N))
            random.shuffle(indices)
            for i in indices:
                if time.time() > max_time:
                    return x, current_score
                delta = get_delta(x, i)
                if delta < 0:
                    x[i] = 1 - x[i]
                    current_score += delta
                    improved = True
        return x, current_score

    def iterated_local_search(x, iterations, max_time=None):
        """ILS: perturb then local search."""
        if max_time is None:
            max_time = deadline - 5.0
        
        best_x = x.copy()
        best_score = count_mono(x)
        
        for _ in range(iterations):
            if time.time() > max_time:
                break
            
            # Perturbation: flip 3-8 random bits
            n_flips = random.randint(3, 8)
            x_pert = best_x.copy()
            for _ in range(n_flips):
                i = random.randint(0, N - 1)
                x_pert[i] = 1 - x_pert[i]
            
            # Local search from perturbed state
            x_ls, score_ls = greedy_local_search(x_pert, max_time)
            
            if score_ls < best_score:
                best_x = x_ls.copy()
                best_score = score_ls
            elif score_ls == best_score and random.random() < 0.1:
                # Occasionally accept equal to escape plateaus
                best_x = x_ls.copy()
        
        return best_x, best_score

    def simulated_annealing(x, steps, T0, cooling, max_time=None):
        """Simulated annealing with single-bit flips."""
        if max_time is None:
            max_time = deadline - 5.0
        
        current_score = count_mono(x)
        T = T0
        
        for _ in range(steps):
            if time.time() > max_time:
                break
            
            i = random.randint(0, N - 1)
            delta = get_delta(x, i)
            
            if delta < 0:
                x[i] = 1 - x[i]
                current_score += delta
            else:
                if delta == 0:
                    if random.random() < 0.3:
                        x[i] = 1 - x[i]
                else:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        x[i] = 1 - x[i]
                        current_score += delta
            
            T *= cooling
        
        return x, current_score

    best_score = float('inf')
    best_x = None

    # Multiple restarts with different strategies
    for restart in range(50):
        if time.time() > deadline - 15.0:
            break
        
        # Random initialization
        x = np.random.randint(0, 2, N)
        
        # Phase 1: Greedy local search
        x, score = greedy_local_search(x)
        
        # Phase 2: ILS for deeper exploration
        x, score = iterated_local_search(x, iterations=5)
        
        # Phase 3: SA polish
        x, score = simulated_annealing(x, steps=2000, T0=1.5, cooling=0.995)
        
        # Final greedy polish
        x, score = greedy_local_search(x)
        
        if score < best_score:
            best_score = score
            best_x = x.copy()
        
        # If we've found a very good solution, do more SA from it
        if best_score <= 10 and time.time() < deadline - 30.0:
            x2 = best_x.copy()
            x2, score2 = simulated_annealing(x2, steps=3000, T0=0.8, cooling=0.998)
            x2, score2 = greedy_local_search(x2)
            if score2 < best_score:
                best_score = score2
                best_x = x2.copy()

    if best_x is None:
        return [(i // 2) % 2 for i in range(N)]
    
    return best_x.tolist()
