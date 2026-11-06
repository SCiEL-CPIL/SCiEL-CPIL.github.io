import time
from math import lcm
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)

    # Calculate LCM of all denominators for exact integer arithmetic
    L = 1
    for d in denominators:
        L = lcm(L, d)

    weights = [L // d for d in denominators]
    
    # Target sum is L - 1 (strictly below L)
    # We want to maximize sum < L
    
    # Precompute weights sorted by value descending for greedy
    # But we need to keep track of original indices
    # Greedy: Pick largest weight that fits
    
    def greedy_solve():
        selected = set()
        current_sum = 0
        # Try all items in descending order of weight (ascending denominator)
        # Indices 0..46 correspond to denominators 2..48
        # Smaller denominator -> Larger weight
        for i in range(n):
            if current_sum + weights[i] < L:
                selected.add(i)
                current_sum += weights[i]
        return selected, current_sum

    best_sum = -1
    best_mask = 0
    best_indices = []
    
    # Initialize with greedy
    init_set, init_sum = greedy_solve()
    if init_sum > best_sum:
        best_sum = init_sum
        best_indices = list(init_set)
        for i in init_set:
            best_mask |= (1 << i)

    # Local Search: Simulated Annealing or Random Walk
    # State: mask (integer bitset), current_sum
    # Move: 
    # 1. Remove one item (sum decreases)
    # 2. Add one item (sum increases, check < L)
    # 3. Swap one item for another
    # 4. Swap two items for two others (more complex, maybe later)
    
    # To improve sum, we usually want to ADD items.
    # But if we are close to L, we might need to SWAP to fill the gap exactly.
    # The gap is L - current_sum.
    # We want to find a subset of remaining items that sums to exactly Gap, or as close as possible < Gap.
    # This is still subset sum, but on the "complement" or "gap".
    # However, simple local search with swaps can work.
    
    # Let's use a simple hill climbing with random restarts.
    # We will maintain a set of "active" indices.
    
    def local_search(start_set, time_limit):
        current_set = set(start_set)
        current_sum = sum(weights[i] for i in current_set)
        best_local_sum = current_sum
        best_local_set = set(current_set)
        
        # Precompute list of items not in current_set for fast addition?
        # Or just iterate all i. n=47 is small.
        
        while time.time() < time_limit:
            # Randomly pick a move type
            # Type 1: Add a random unused item
            # Type 2: Remove a random used item
            # Type 3: Swap a used item with an unused item
            
            move_type = random.randint(1, 3)
            
            if move_type == 1:
                # Add
                # Pick random i not in current_set
                # To make it efficient, we can just pick random i and check
                i = random.randint(0, n - 1)
                if i not in current_set:
                    if current_sum + weights[i] < L:
                        current_set.add(i)
                        current_sum += weights[i]
                        if current_sum > best_local_sum:
                            best_local_sum = current_sum
                            best_local_set = set(current_set)
                    else:
                        # If it doesn't fit, maybe try to swap?
                        pass
            elif move_type == 2:
                # Remove
                if current_set:
                    i = random.choice(list(current_set))
                    current_set.remove(i)
                    current_sum -= weights[i]
            else:
                # Swap
                if current_set:
                    i_out = random.choice(list(current_set))
                    i_in = random.randint(0, n - 1)
                    if i_in not in current_set:
                        # Check if swap improves sum
                        # new_sum = current_sum - w[i_out] + w[i_in]
                        # We want new_sum < L
                        if current_sum - weights[i_out] + weights[i_in] < L:
                            current_set.remove(i_out)
                            current_set.add(i_in)
                            current_sum = current_sum - weights[i_out] + weights[i_in]
                            if current_sum > best_local_sum:
                                best_local_sum = current_sum
                                best_local_set = set(current_set)
        
        return best_local_sum, best_local_set

    # Run multiple restarts
    # We can restart with the best found so far, or with slightly perturbed versions
    # Or just run one long local search from the greedy start.
    # Given n=47, the search space is large, but greedy is a very good start.
    # The gap is small. 
    # Greedy sum for 2..48:
    # 1/2+1/3+1/7+1/43 = 1805/1806. Gap 1/1806.
    # Can we fill 1/1806?
    # 1/1806 = 1/(2*3*7*43).
    # We have denominators up to 48.
    # 1806 is large. We need fractions that sum to 1/1806.
    # e.g. 1/3612 + 1/3612? No, distinct indices.
    # 1/1806 is very small.
    # It might be impossible to fill it exactly with distinct unit fractions from 2..48.
    # But we can get closer.
    
    # Let's run local search for a fixed duration, then restart if stuck or just continue.
    # Since we have 85s, we can run local search for 80s.
    
    # Optimization: 
    # The "Add" move is the most useful. 
    # If we can't add, we are likely at a local optimum where all remaining items are too big.
    # In that case, we MUST swap or remove to make space for smaller items.
    
    # Let's refine the local search to be more directed.
    # If we can't add any single item, we look for swaps that reduce the "overshoot" or allow a smaller item to fit.
    
    # Actually, a simpler and very effective method for this specific "fill the gap" problem:
    # 1. Start with Greedy.
    # 2. Gap = L - current_sum.
    # 3. Try to find a subset of UNUSED items that sums to <= Gap.
    #    If we find one, add it.
    #    This is a smaller subset sum problem on the unused items, targeting Gap.
    #    Gap is small (initially ~ L/1806).
    #    L is huge, but Gap is relatively small compared to L.
    #    However, weights are integers.
    #    We can use DP or Meet-in-the-Middle on the UNUSED items to find the best subset sum <= Gap.
    #    Number of unused items is ~40. MITM on 40 is $2^{20} \approx 10^6$. This is doable in Python if optimized, or maybe too slow.
    #    But we can do a simpler search: 
    #    Since Gap is small, we can just iterate through unused items and try to add them if they fit.
    #    If multiple fit, pick the one that minimizes the new gap?
    #    Greedy on the gap: Pick largest unused item that fits in Gap.
    #    Repeat until no item fits.
    #    This is essentially running Greedy again on the remaining items, targeting the remaining Gap.
    
    # Let's try "Iterative Greedy on Gap":
    # 1. current_set, current_sum from initial Greedy.
    # 2. Gap = L - current_sum.
    # 3. While True:
    #      Find unused item i with max weight[i] such that weight[i] <= Gap.
    #      If found: add i, update Gap.
    #      Else: break.
    # 4. This might get stuck if the largest fitting item leaves a gap that can't be filled by smaller ones, but a combination of smaller ones could.
    #    Example: Gap = 10. Items: 6, 5, 4. 
    #    Greedy picks 6. Gap=4. Picks 4. Gap=0. Sum=10.
    #    Example: Gap = 10. Items: 7, 6, 4.
    #    Greedy picks 7. Gap=3. No fit. Sum=7.
    #    But 6+4=10. 
    #    So simple Greedy on Gap is not optimal.
    
    # We need local search to fix these.
    # The previous local search code is okay but `random.choice(list(current_set))` is slow.
    # Let's use a list of indices and a boolean array for membership.
    
    def fast_local_search(start_set, time_limit):
        current_set = set(start_set)
        in_set = [False] * n
        for i in current_set:
            in_set[i] = True
        
        current_sum = sum(weights[i] for i in current_set)
        best_local_sum = current_sum
        best_local_set = set(current_set)
        
        # Precompute unused items list for faster sampling?
        # Or just sample random i.
        
        while time.time() < time_limit:
            # Try to Add
            i = random.randint(0, n - 1)
            if not in_set[i]:
                if current_sum + weights[i] < L:
                    current_set.add(i)
                    in_set[i] = True
                    current_sum += weights[i]
                    if current_sum > best_local_sum:
                        best_local_sum = current_sum
                        best_local_set = set(current_set)
            else:
                # Try to Swap
                # Pick another random j not in set
                j = random.randint(0, n - 1)
                if not in_set[j]:
                    # Check swap i -> j
                    new_sum = current_sum - weights[i] + weights[j]
                    if new_sum < L:
                        # Accept swap? 
                        # We want to maximize sum. 
                        # If new_sum > current_sum, it's an improvement.
                        # If new_sum < current_sum, it's a regression, but might enable better additions later.
                        # Simple Hill Climbing: only accept if new_sum > current_sum?
                        # Or accept with probability?
                        # Given the small gap, we probably want to accept swaps that allow filling the gap.
                        # Let's accept if new_sum > current_sum (improvement) 
                        # OR if new_sum < current_sum but we are stuck (no adds possible)?
                        
                        # Let's just do Hill Climbing with random restarts.
                        if new_sum > current_sum:
                            current_set.remove(i)
                            in_set[i] = False
                            current_set.add(j)
                            in_set[j] = True
                            current_sum = new_sum
                            if current_sum > best_local_sum:
                                best_local_sum = current_sum
                                best_local_set = set(current_set)
                        else:
                            # If it's a regression, we might still take it to escape local optima?
                            # For now, let's stick to strict improvement to be safe and fast.
                            # But this might get stuck.
                            # Let's add a "Random Regret" move: 
                            # If we haven't improved in X steps, allow a regression.
                            pass
            
            # Also try Remove?
            # If we are stuck, removing a large item might allow adding two smaller ones that sum to more.
            # e.g. Remove 1/5 (0.2). Add 1/6 (0.166) + 1/12 (0.083) = 0.25. Improvement!
            # So Remove is important.
            
            # Let's add a Remove move with lower probability
            if random.random() < 0.2:
                if current_set:
                    # Pick random item to remove
                    # To avoid O(N) list conversion, we can just pick random i and check if in_set
                    # But we need to pick one that IS in set.
                    # Resample until found? Or maintain a list of active indices?
                    # Maintaining a list is better.
                    
                    # Let's restart the loop structure with a list of active indices.
                    pass
        
        return best_local_sum, best_local_set

    # The above is getting messy. Let's use a cleaner Simulated Annealing implementation.
    
    def simulated_annealing(start_set, time_limit):
        current_set = set(start_set)
        in_set = [False] * n
        for i in current_set:
            in_set[i] = True
        
        current_sum = sum(weights[i] for i in current_set)
        best_sum = current_sum
        best_set = set(current_set)
        
        T = 1000.0 # Initial temperature (in weight units? No, deficit units)
        # Let's use deficit as energy. Lower is better.
        # E = L - sum.
        # We want to minimize E.
        
        # Actually, just maximizing sum is fine.
        
        # To pick a random element from current_set efficiently, we can use a list and swap-remove.
        active_indices = list(current_set)
        
        def remove_from_active(idx):
            # Find idx in active_indices
            # This is O(N). N is small (max 47).
            for k, val in enumerate(active_indices):
                if val == idx:
                    active_indices.pop(k)
                    break
        
        def add_to_active(idx):
            active_indices.append(idx)
            
        # Initial
        for i in current_set:
            in_set[i] = True
            
        steps = 0
        last_improve = time.time()
        
        while time.time() < time_limit:
            steps += 1
            if steps % 1000 == 0:
                T *= 0.99 # Cooling
                if T < 0.1:
                    T = 0.1 # Floor
            
            # Generate neighbor
            # 1. Add
            # 2. Remove
            # 3. Swap
            
            move = random.random()
            if move < 0.4: # Add
                if not active_indices: # Empty set? Unlikely
                    continue
                # Pick random candidate from ALL indices
                i = random.randint(0, n - 1)
                if not in_set[i]:
                    if current_sum + weights[i] < L:
                        # Valid add
                        new_sum = current_sum + weights[i]
                        # Always accept if valid? 
                        # It increases sum, so it's an improvement.
                        current_set.add(i)
                        in_set[i] = True
                        add_to_active(i)
                        current_sum = new_sum
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_set = set(current_set)
                            last_improve = time.time()
                    else:
                        # Invalid add. Try swap?
                        pass
            elif move < 0.7: # Remove
                if active_indices:
                    i = random.choice(active_indices)
                    # Removing decreases sum. 
                    # Only accept if it allows a better state later?
                    # In SA, we accept with prob exp(-delta/T).
                    # Here delta = -weights[i] (change in sum).
                    # We want to maximize sum. So delta is negative.
                    # We accept with prob exp(delta/T) = exp(-w[i]/T).
                    # This is very small for large w[i].
                    # This prevents us from removing large items unless T is high.
                    # But removing large items is key to repacking.
                    # Let's force some removals.
                    
                    # Alternative: 
                    # If we are stuck (no adds possible), we MUST remove.
                    # Check if any add is possible.
                    # This check is O(N).
                    can_add = False
                    for j in range(n):
                        if not in_set[j] and current_sum + weights[j] < L:
                            can_add = True
                            break
                    
                    if not can_add:
                        # Stuck. Remove a random item to break deadlock.
                        current_set.remove(i)
                        in_set[i] = False
                        remove_from_active(i)
                        current_sum -= weights[i]
                    else:
                        # Not stuck. Only remove if it leads to improvement (impossible) or with low prob.
                        # Let's just skip removal if not stuck, to preserve sum.
                        pass
            else: # Swap
                if active_indices:
                    i = random.choice(active_indices)
                    j = random.randint(0, n - 1)
                    if not in_set[j]:
                        new_sum = current_sum - weights[i] + weights[j]
                        if new_sum < L:
                            # Valid swap
                            # Accept if new_sum > current_sum (improvement)
                            # Or with prob exp((new_sum - current_sum)/T)
                            delta = new_sum - current_sum
                            if delta > 0:
                                # Improvement
                                current_set.remove(i)
                                in_set[i] = False
                                remove_from_active(i)
                                current_set.add(j)
                                in_set[j] = True
                                add_to_active(j)
                                current_sum = new_sum
                                if current_sum > best_sum:
                                    best_sum = current_sum
                                    best_set = set(current_set)
                                    last_improve = time.time()
                            else:
                                # Regression
                                # Accept with prob exp(delta/T)
                                # delta is negative.
                                # If T is high, we accept.
                                if random.random() < math.exp(delta / T):
                                    current_set.remove(i)
                                    in_set[i] = False
                                    remove_from_active(i)
                                    current_set.add(j)
                                    in_set[j] = True
                                    add_to_active(j)
                                    current_sum = new_sum
                                    if current_sum > best_sum:
                                        best_sum = current_sum
                                        best_set = set(current_set)
                                        last_improve = time.time()
            
            # Check for restart?
            # If no improvement for 5s, restart from best_set with new T?
            if time.time() - last_improve > 5.0:
                # Restart
                current_set = set(best_set)
                in_set = [False] * n
                for i in current_set:
                    in_set[i] = True
                active_indices = list(current_set)
                current_sum = best_sum
                T = 1000.0
                last_improve = time.time()
        
        return best_sum, best_set

    import math
    
    # Run SA
    final_sum, final_set = simulated_annealing(init_set, deadline)
    
    if final_sum > best_sum:
        best_sum = final_sum
        best_indices = list(final_set)
        
    return best_indices

