import time
from math import gcd
from functools import reduce
import random

def construct():
    start_time = time.time()
    timeout = 55.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(2, 49))
    weights = [L // d for d in denominators]
    
    def greedy_init():
        current_sum = 0
        selected = [False] * n
        for i in range(n):
            if current_sum + weights[i] < L:
                current_sum += weights[i]
                selected[i] = True
        return selected, current_sum
    
    def local_search(selected, current_sum, max_time):
        best_sum = current_sum
        best_selected = selected[:]
        
        # Precompute counts or just use list operations
        # For speed, we will use a list of booleans
        
        while time.time() < max_time:
            improved = False
            
            # Try all possible single flips
            # This is O(n^2) per iteration if we recompute sum, but we can track deltas.
            # Let's do a pass over all indices.
            for i in range(n):
                # If selected[i] is False, try adding it
                # If selected[i] is True, try removing it
                
                # Adding
                if not selected[i]:
                    new_sum = current_sum + weights[i]
                    if new_sum < L:
                        # Valid. Update if better.
                        if new_sum > best_sum:
                            best_sum = new_sum
                            best_selected = selected[:]
                            best_selected[i] = True
                            # Update current state to track this? 
                            # No, local search usually does one step at a time or hill climbing.
                            # Let's do proper hill climbing: find the best single move, apply it.
                            # But n=47 is small. We can just check all and pick the best.
                            pass
                else:
                    # Removing
                    new_sum = current_sum - weights[i]
                    # Always valid since weights are positive and sum < L implies removal < L.
                    # But we want to MAXIMIZE sum < L. Removing lowers sum.
                    # So removing is only useful if we can then add something else?
                    # Standard local search: move to neighbor.
                    # Neighbor: flip one bit.
                    # If we flip a 1 to 0, sum decreases. Is it better? No, we want max sum < L.
                    # So removing is only good if it allows adding a larger item?
                    # That's a 2-opt move (swap).
                    
            # Let's implement 1-opt (flip) and 2-opt (swap) properly.
            # 1-opt:
            # - If sum + w_i < L, new sum is larger. Good.
            # - If sum - w_i is valid, it's smaller. Bad, unless it enables a better swap.
            
            # 2-opt:
            # - Swap i (selected) with j (unselected).
            # - new_sum = current_sum - w_i + w_j
            # - Must be < L.
            
            best_gain = 0
            best_move = None # (type, i, j)
            
            # 1-opt: Add unselected
            for i in range(n):
                if not selected[i]:
                    new_sum = current_sum + weights[i]
                    if new_sum < L:
                        gain = new_sum - current_sum
                        if gain > best_gain:
                            best_gain = gain
                            best_move = ('add', i, -1)
            
            # 2-opt: Swap
            # Only worth checking if there are selected and unselected items.
            # To speed up, we can just iterate.
            for i in range(n):
                if selected[i]:
                    for j in range(n):
                        if not selected[j]:
                            new_sum = current_sum - weights[i] + weights[j]
                            if new_sum < L:
                                gain = new_sum - current_sum
                                if gain > best_gain:
                                    best_gain = gain
                                    best_move = ('swap', i, j)
            
            if best_move is None:
                break
            
            # Apply move
            if best_move[0] == 'add':
                _, i, _ = best_move
                selected[i] = True
                current_sum += weights[i]
            else:
                _, i, j = best_move
                selected[i] = False
                selected[j] = True
                current_sum -= weights[i]
                current_sum += weights[j]
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_selected = selected[:]
            else:
                # If no improvement in sum (shouldn't happen with gain > 0), break
                break
        
        return best_selected, best_sum
    
    def simulated_annealing(selected, current_sum, max_time):
        best_sum = current_sum
        best_selected = selected[:]
        T = 1000.0
        min_T = 0.1
        cooling_rate = 0.9995
        
        while T > min_T and time.time() < max_time:
            # Generate neighbor
            # Random swap or flip
            if random.random() < 0.5:
                # Swap
                selected_indices = [i for i in range(n) if selected[i]]
                unselected_indices = [i for i in range(n) if not selected[i]]
                
                if not selected_indices or not unselected_indices:
                    T *= cooling_rate
                    continue
                
                i = random.choice(selected_indices)
                j = random.choice(unselected_indices)
                
                new_sum = current_sum - weights[i] + weights[j]
                
                if new_sum < L:
                    delta = new_sum - current_sum
                    # We want to maximize sum, so delta > 0 is good.
                    # Accept if delta > 0 or with prob exp(delta/T)
                    if delta > 0 or random.random() < pow(2.718, delta / T):
                        selected[i] = False
                        selected[j] = True
                        current_sum = new_sum
                        
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_selected = selected[:]
                # else: new_sum >= L, reject (or accept with high penalty, but let's reject for simplicity in constraint)
            else:
                # Flip
                i = random.randint(0, n - 1)
                if selected[i]:
                    new_sum = current_sum - weights[i]
                    # Removing is always valid
                    delta = new_sum - current_sum
                    if delta > 0 or random.random() < pow(2.718, delta / T):
                        selected[i] = False
                        current_sum = new_sum
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_selected = selected[:]
                else:
                    new_sum = current_sum + weights[i]
                    if new_sum < L:
                        delta = new_sum - current_sum
                        if delta > 0 or random.random() < pow(2.718, delta / T):
                            selected[i] = True
                            current_sum = new_sum
                            if current_sum > best_sum:
                                best_sum = current_sum
                                best_selected = selected[:]
            
            T *= cooling_rate
        
        return best_selected, best_sum
    
    # Strategy:
    # 1. Greedy Init
    # 2. Local Search (Hill Climbing) to find local optimum
    # 3. Simulated Annealing from that point to escape local optima
    # 4. Repeat with different random seeds/restarts if time permits
    
    best_overall_sum = 0
    best_overall_indices = []
    
    initial_selected, initial_sum = greedy_init()
    
    # Phase 1: Deterministic Hill Climbing
    ls_selected, ls_sum = local_search(initial_selected, initial_sum, start_time + 5.0)
    
    if ls_sum > best_overall_sum:
        best_overall_sum = ls_sum
        best_overall_indices = [i for i in range(n) if ls_selected[i]]
    
    # Phase 2: Simulated Annealing with restarts
    # We have about 50s left.
    # Let's run SA for a while.
    # If time is tight, skip SA and just return LS result.
    
    if time.time() < start_time + timeout - 5:
        # Use SA
        # Start from LS result
        sa_selected = ls_selected[:]
        sa_sum = ls_sum
        
        # Run SA
        sa_selected, sa_sum = simulated_annealing(sa_selected, sa_sum, start_time + timeout - 5)
        
        if sa_sum > best_overall_sum:
            best_overall_sum = sa_sum
            best_overall_indices = [i for i in range(n) if sa_selected[i]]
        
        # Optional: Another restart from greedy with different random order?
        # Maybe just run SA again from a random valid start?
        # Generating a random valid start is hard (sum < L).
        # Let's just try another SA run from the best found so far?
        # No, SA cools down.
        
        # Let's try a "Random Hill Climbing" start
        # Pick random subset, repair if sum >= L by removing largest elements.
        # Then hill climb.
        
        if time.time() < start_time + timeout - 2:
            # Random Restart
            random_selected = [random.random() < 0.5 for _ in range(n)]
            random_sum = sum(weights[i] for i in range(n) if random_selected[i])
            
            # Repair: while sum >= L, remove largest weight selected
            while random_sum >= L:
                # Find largest weight among selected
                max_w = -1
                max_i = -1
                for i in range(n):
                    if random_selected[i] and weights[i] > max_w:
                        max_w = weights[i]
                        max_i = i
                if max_i == -1:
                    break
                random_selected[max_i] = False
                random_sum -= max_w
            
            # Now hill climb
            rr_selected, rr_sum = local_search(random_selected, random_sum, start_time + timeout - 2)
            
            if rr_sum > best_overall_sum:
                best_overall_sum = rr_sum
                best_overall_indices = [i for i in range(n) if rr_selected[i]]
    
    if not best_overall_indices:
        # Fallback: just [0] (1/2)
        best_overall_indices = [0]
    
    return best_overall_indices

# Quick test to ensure it runs fast and returns valid output
if __name__ == "__main__":
    res = construct()
    denominators = list(range(2, 49))
    s = sum(1.0 / denominators[i] for i in res)
    print(f"Result indices: {res}")
    print(f"Sum: {s}")
    print(f"Deficit: {1.0 - s}")
    print(f"Valid: {s < 1.0}")
