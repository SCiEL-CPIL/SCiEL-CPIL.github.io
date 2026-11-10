"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    n = 512
    
    # Precompute all 4-APs (a, b, c, d) with a < b < c < d in [1, n]
    # This allows O(1) conflict checking via a set of tuples or a boolean matrix
    # For n=512, number of 4-APs is approx sum_{d=1}^{n/3} (n - 3d) ~ n^2/6 ~ 43k
    # This is very fast to precompute.
    
    ap_set = set()
    # Map from element to list of AP indices it belongs to? 
    # Or better: for each element x, store the set of APs (as indices or tuples) containing x.
    # Actually, for SA, we need to quickly check if adding x creates a 4-AP.
    # A 4-AP is formed if x completes an existing 3-term AP in the set.
    # Or more generally, x is part of a 4-AP if any of the 4 positions (1st, 2nd, 3rd, 4th)
    # have the other 3 elements in the set.
    
    # Let's precompute for each x, the list of (other_elements_tuple) that would form a 4-AP with x.
    # If any of these tuples are fully contained in the current set, x cannot be added.
    
    # conflicts[x] = list of sets of 3 other elements. If any set is subset of current_set, conflict.
    # To speed up, we can represent the current set as a bitset or boolean array.
    # Checking if a specific 3-tuple is in the set is O(1) with boolean array.
    
    # Precompute conflicts for each x
    # x is 1st: (x+d, x+2d, x+3d)
    # x is 2nd: (x-d, x+d, x+2d)
    # x is 3rd: (x-2d, x-d, x+d)
    # x is 4th: (x-3d, x-2d, x-d)
    
    conflicts = [[] for _ in range(n + 1)]
    
    for x in range(1, n + 1):
        # x as 4th term: x-3d, x-2d, x-d
        for d in range(1, (x // 3) + 1):
            conflicts[x].append((x-d, x-2*d, x-3*d))
        # x as 1st term: x+d, x+2d, x+3d
        for d in range(1, (n - x) // 3 + 1):
            conflicts[x].append((x+d, x+2*d, x+3*d))
        # x as 2nd term: x-d, x+d, x+2d
        max_d = min(x, (n - x) // 2)
        for d in range(1, max_d + 1):
            conflicts[x].append((x-d, x+d, x+2*d))
        # x as 3rd term: x-2d, x-d, x+d
        max_d = min(x // 2, n - x)
        for d in range(1, max_d + 1):
            conflicts[x].append((x-2*d, x-d, x+d))
            
    # Helper to check if x can be added
    def can_add(x, in_set):
        for t in conflicts[x]:
            if in_set[t[0]] and in_set[t[1]] and in_set[t[2]]:
                return False
        return True

    # Phase 1: Greedy Initialization
    # Use base-3 digit sum ordering as a good heuristic for 4-AP free sets
    def base3_sum(x):
        s = 0
        while x > 0:
            s += x % 3
            x //= 3
        return s

    order = sorted(range(1, n + 1), key=lambda x: (base3_sum(x), x))
    
    in_set = [False] * (n + 1)
    current_set = set()
    
    for x in order:
        if time.time() > deadline:
            break
        if can_add(x, in_set):
            in_set[x] = True
            current_set.add(x)
            
    best_set = set(current_set)
    best_size = len(current_set)
    
    # Phase 2: Simulated Annealing
    # State: current_set, in_set
    # Move: Add a random element not in set (if valid) OR Remove a random element from set
    # Cost: -size (we want to maximize size)
    
    # Parameters
    T_start = 1.0
    T_end = 0.01
    # Number of steps depends on time budget
    # We will run until deadline
    
    # To make removals useful, we should probably only remove elements that are "bad" (involved in many potential conflicts)
    # Or just random removal to allow exploration.
    
    # Let's use a cooling schedule
    # Total steps ~ 50000? 
    # Each step is O(1) check for add, O(1) for remove.
    # can_add is O(deg(x)) where deg(x) is number of 4-APs x is in. Max deg ~ 200.
    # So 50000 * 200 = 10^7 operations. Should be fast enough in Python? Maybe tight.
    # Let's optimize can_add slightly or reduce steps if needed.
    
    # Actually, let's just run as many steps as possible.
    
    random.seed(42) # For reproducibility if needed, though SA is stochastic
    
    T = T_start
    # Estimate steps: if we assume 1ms per step, 80s = 80000 steps.
    # Let's try to run until time runs out.
    
    while time.time() < deadline - 2.0:
        # Determine move
        # 80% chance to try adding, 20% chance to remove
        if len(current_set) > 0 and random.random() < 0.2:
            # Remove a random element
            to_remove = random.choice(list(current_set))
            current_set.remove(to_remove)
            in_set[to_remove] = False
            new_size = len(current_set)
            
            # Acceptance for removal (worse move)
            # delta_cost = new_size - old_size = -1
            # Accept with prob exp(-delta_cost / T) = exp(1/T)
            # Since delta_cost is negative (size decreased), -delta_cost is positive.
            # Wait, cost function is -size.
            # old_cost = -old_size
            # new_cost = -new_size = -(old_size - 1) = -old_size + 1
            # delta_cost = new_cost - old_cost = 1
            # Accept with prob exp(-1/T)
            if random.random() < math.exp(-1.0 / T):
                # Keep removal
                pass
            else:
                # Reject removal, add back
                current_set.add(to_remove)
                in_set[to_remove] = True
        else:
            # Try to add a random element
            # Pick random candidate
            candidates = [i for i in range(1, n + 1) if not in_set[i]]
            if not candidates:
                break
            to_add = random.choice(candidates)
            
            if can_add(to_add, in_set):
                current_set.add(to_add)
                in_set[to_add] = True
                new_size = len(current_set)
                
                # This is an improvement (size increased)
                # Always accept
                if new_size > best_size:
                    best_size = new_size
                    best_set = set(current_set)
            else:
                # Conflict, do nothing
                pass
        
        # Cooling
        # Exponential cooling: T = T_start * (T_end/T_start)^step
        # Or linear: T = T_start - (T_start - T_end) * step / total_steps
        # Since total steps unknown, use a simpler scheme or fixed number of iterations per "epoch"
        # Let's use a fixed number of iterations for cooling, say 10000, and reset if stuck?
        # No, just cool down.
        T = T_end + (T_start - T_end) * (0.99 ** (time.time() - start_time)) 
        # This is a bit hacky. Let's use a standard geometric cooling with a step counter.
        
    # To ensure we don't waste time, let's structure it as fixed steps if possible, 
    # but the deadline guard is safer.
    
    # Let's refine the cooling. 
    # A common approach: T = T0 * alpha^k
    # Let's just decrement T slightly each step.
    # Actually, the loop above doesn't update T correctly based on steps, but on time.
    # Let's rewrite the SA loop to be cleaner.
    
    # Reset SA for a cleaner implementation within the time limit
    # The previous loop was a bit messy with T update.
    
    # Let's restart SA properly if we have time, or just use the best found so far.
    # The greedy init is likely decent. SA should improve it.
    
    # Let's implement a proper SA loop with step counting.
    
    # Re-initialize from best_set found by greedy? Or just continue from current?
    # Current might have degraded. Best_set is the max found.
    # Let's restart SA from best_set to ensure we start from a good state.
    
    if time.time() < deadline - 5.0:
        current_set = set(best_set)
        in_set = [False] * (n + 1)
        for x in current_set:
            in_set[x] = True
        
        T = 1.0
        alpha = 0.99999 # Very slow cooling
        # Or fixed steps. Let's do fixed steps to ensure coverage.
        # Max steps ~ 50000
        for _ in range(50000):
            if time.time() > deadline - 2.0:
                break
            
            # Move
            if len(current_set) > 0 and random.random() < 0.3:
                # Remove
                to_remove = random.choice(list(current_set))
                current_set.remove(to_remove)
                in_set[to_remove] = False
                # Accept with prob exp(1/T) ? No, exp(-delta_cost/T)
                # delta_cost = 1 (cost increases because size decreases)
                # Accept if random < exp(-1/T)
                if random.random() < math.exp(-1.0 / T):
                    pass
                else:
                    current_set.add(to_remove)
                    in_set[to_remove] = True
            else:
                # Add
                candidates = [i for i in range(1, n + 1) if not in_set[i]]
                if not candidates:
                    break
                to_add = random.choice(candidates)
                if can_add(to_add, in_set):
                    current_set.add(to_add)
                    in_set[to_add] = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
            
            T *= alpha
            
            # If T is very small, we are effectively greedy. 
            # We can break early if T < T_end and no improvements in last X steps?
            # For now, just run.
            
    # Final check: ensure best_set is valid (it should be, as we only add valid elements and keep best)
    # Return sorted list
    return sorted(best_set)

# EVOLVE-BLOCK-END
