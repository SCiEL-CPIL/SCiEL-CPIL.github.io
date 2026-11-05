"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin out of 60s
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    def is_compatible(a, b):
        return lcm(a, b) > 256
    
    def calc_sum(S):
        return sum(1.0/x for x in S)
    
    candidates = list(range(2, 257))
    n = len(candidates)
    
    # Precompute compatibility matrix for speed
    # compat[i][j] = True if candidates[i] and candidates[j] are compatible
    compat = [[True] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if not is_compatible(candidates[i], candidates[j]):
                compat[i][j] = False
                compat[j][i] = False
    
    # Greedy initialization
    def greedy_init():
        selected = set()
        # Sort by ascending value (highest reciprocal first)
        for x in sorted(candidates):
            idx = x - 2
            can_add = True
            for y in selected:
                j = y - 2
                if not compat[idx][j]:
                    can_add = False
                    break
            if can_add:
                selected.add(x)
        return selected
    
    # Local search: try swaps
    def local_search(selected, max_iter=10000):
        selected = set(selected)
        current_sum = calc_sum(selected)
        selected_list = list(selected)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            # Try to add a new element
            for x in candidates:
                if x in selected:
                    continue
                idx = x - 2
                can_add = True
                for y in selected:
                    j = y - 2
                    if not compat[idx][j]:
                        can_add = False
                        break
                if can_add:
                    selected.add(x)
                    current_sum += 1.0/x
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-for-1 swap
            for y in list(selected):
                for x in candidates:
                    if x in selected or x >= y:
                        continue
                    idx_x = x - 2
                    idx_y = y - 2
                    # Check if x is compatible with all in selected except y
                    valid = True
                    for z in selected:
                        if z == y:
                            continue
                        j = z - 2
                        if not compat[idx_x][j]:
                            valid = False
                            break
                    if valid:
                        selected.remove(y)
                        selected.add(x)
                        current_sum += (1.0/x - 1.0/y)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return selected, current_sum
    
    # Simulated annealing
    def simulated_annealing(initial_selected, start_temp=1.0, end_temp=0.01, max_iter=50000):
        selected = set(initial_selected)
        current_sum = calc_sum(selected)
        best_selected = set(selected)
        best_sum = current_sum
        
        temp = start_temp
        cooling = (end_temp / start_temp) ** (1.0 / max_iter)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Random move: remove a random element, or swap
            move_type = random.random()
            
            if move_type < 0.3:
                # Remove a random element
                if len(selected) > 1:
                    y = random.choice(list(selected))
                    new_selected = set(selected)
                    new_selected.remove(y)
                    new_sum = current_sum - 1.0/y
                    delta = new_sum - current_sum
                    if delta > 0 or random.random() < math.exp(delta / temp):
                        selected = new_selected
                        current_sum = new_sum
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_selected = set(selected)
            
            elif move_type < 0.7:
                # Add a random element
                x = random.choice(candidates)
                if x not in selected:
                    idx = x - 2
                    can_add = True
                    for y in selected:
                        j = y - 2
                        if not compat[idx][j]:
                            can_add = False
                            break
                    if can_add:
                        new_selected = set(selected)
                        new_selected.add(x)
                        new_sum = current_sum + 1.0/x
                        delta = new_sum - current_sum
                        if delta > 0 or random.random() < math.exp(delta / temp):
                            selected = new_selected
                            current_sum = new_sum
                            if current_sum > best_sum:
                                best_sum = current_sum
                                best_selected = set(selected)
            
            else:
                # Swap: remove random y, add random x
                if len(selected) > 0:
                    y = random.choice(list(selected))
                    x = random.choice(candidates)
                    if x != y and x not in selected:
                        idx_x = x - 2
                        valid = True
                        for z in selected:
                            if z == y:
                                continue
                            j = z - 2
                            if not compat[idx_x][j]:
                                valid = False
                                break
                        if valid:
                            new_selected = set(selected)
                            new_selected.remove(y)
                            new_selected.add(x)
                            new_sum = current_sum + (1.0/x - 1.0/y)
                            delta = new_sum - current_sum
                            if delta > 0 or random.random() < math.exp(delta / temp):
                                selected = new_selected
                                current_sum = new_sum
                                if current_sum > best_sum:
                                    best_sum = current_sum
                                    best_selected = set(selected)
            
            temp *= cooling
        
        return best_selected, best_sum
    
    # Get initial solution
    initial = greedy_init()
    initial, initial_sum = local_search(initial)
    
    best_selected = set(initial)
    best_sum = initial_sum
    
    # Run simulated annealing with multiple restarts
    restarts = 0
    while time.time() < deadline:
        # Perturb the current best solution
        perturbed = set(best_selected)
        # Remove 1-3 random elements
        for _ in range(random.randint(1, 3)):
            if len(perturbed) > 1:
                y = random.choice(list(perturbed))
                perturbed.remove(y)
        
        # Run SA from perturbed state
        sa_result, sa_sum = simulated_annealing(perturbed, start_temp=0.5, max_iter=20000)
        
        if sa_sum > best_sum:
            best_sum = sa_sum
            best_selected = sa_result
        
        # Also try local search on the SA result
        ls_result, ls_sum = local_search(sa_result, max_iter=5000)
        if ls_sum > best_sum:
            best_sum = ls_sum
            best_selected = ls_result
        
        restarts += 1
        if time.time() > deadline - 10:
            break
    
    return list(best_selected)

# EVOLVE-BLOCK-END
