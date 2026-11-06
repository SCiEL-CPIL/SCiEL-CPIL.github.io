import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 150s kill
    
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
    
    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True
        
    def try_add(x, S_set):
        """Check if x can be added to S_set while keeping it sum-free. Returns True if valid."""
        if x + x in S_set:
            return False
        for a in S_set:
            if x + a in S_set:
                return False
            b = x - a
            if b in S_set:
                return False
        return True

    def local_search(S, initial, time_limit):
        """Local search to improve sum-free subset size."""
        current = list(initial)
        current_set = set(current)
        best = list(current)
        best_size = len(best)
        
        # Precompute list of all elements for quick access
        S_list = S
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try to add any element
            # Sort candidates by some heuristic? Random is fine for exploration
            candidates = [x for x in S_list if x not in current_set]
            random.shuffle(candidates)
            for add in candidates:
                if try_add(add, current_set):
                    current.append(add)
                    current_set.add(add)
                    if len(current) > best_size:
                        best = list(current)
                        best_size = len(current)
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Try swap: remove one, add one
            # Iterate over elements in current set
            # To make it faster, we can sample
            for i in range(len(current)):
                if time.time() > time_limit:
                    break
                
                rem = current[i]
                # Create a temporary set without rem
                # Instead of creating a new set every time, we can just "temporarily remove"
                # But try_add checks membership.
                # We can pass a set and a "blocked" element?
                # Or just create the set. n is small (<= 240).
                trial_set = current_set.copy()
                trial_set.remove(rem)
                
                # Try to add any candidate
                candidates = [x for x in S_list if x not in trial_set and x != rem]
                random.shuffle(candidates)
                
                added_any = False
                for add in candidates:
                    if try_add(add, trial_set):
                        # Accept swap if size increases? 
                        # Swap keeps size same. We only want to accept if it leads to a larger set later?
                        # No, standard local search: accept if size increases.
                        # Since swap keeps size same, we should only accept if it allows a subsequent add.
                        # But we don't know that.
                        # However, sometimes a swap is necessary to break a deadlock.
                        # Let's accept swaps if they don't decrease size (which is never) 
                        # and we hope it helps.
                        # Actually, if we accept a swap that doesn't increase size, we might loop.
                        # We should only accept swaps if we can immediately add something?
                        # Or use simulated annealing?
                        # Let's try: accept swap if it allows an add in the next step?
                        # Too complex.
                        
                        # Alternative: only accept if we find an add in the same iteration.
                        # We are already in the "try add" loop for the trial_set.
                        # So if we find an add, we perform the swap AND the add.
                        # This increases size by 1.
                        new_current = [x for x in current if x != rem]
                        new_current.append(add)
                        best = new_current
                        best_size = len(new_current)
                        current = new_current
                        current_set = set(current)
                        improved = True
                        added_any = True
                        break
                
                if improved:
                    break
            
            if not improved:
                # Stuck. Try random restart or break?
                # If we can't improve, break to save time for other instances
                break
                
        return best

    def solve_instance(S, time_limit):
        n = len(S)
        best = []
        best_size = 0
        
        # Strategy 1: Odds
        odds = [x for x in S if x % 2 != 0]
        if len(odds) > best_size:
            best = odds
            best_size = len(odds)
            
        # Strategy 2: Positives upper half
        pos = sorted([x for x in S if x > 0])
        if pos:
            half = len(pos) // 2
            pos_large = pos[half:]
            if len(pos_large) > best_size:
                best = pos_large
                best_size = len(pos_large)
                
        # Strategy 3: Negatives lower half
        neg = sorted([x for x in S if x < 0])
        if neg:
            half = len(neg) // 2
            neg_small = neg[:half] # Most negative
            if len(neg_small) > best_size:
                best = neg_small
                best_size = len(neg_small)
                
        # Strategy 4: Random Greedy
        for _ in range(10):
            if time.time() > time_limit:
                break
            order = list(S)
            random.shuffle(order)
            current = []
            current_set = set()
            for x in order:
                if try_add(x, current_set):
                    current.append(x)
                    current_set.add(x)
            if len(current) > best_size:
                best = current
                best_size = len(current)
                
        # Improve best with local search
        if time.time() < time_limit - 2:
            improved = local_search(S, best, time_limit)
            if len(improved) > best_size:
                best = improved
                best_size = len(improved)
                
        return best

    # Solve all three instances
    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for size, seed in instances:
        S = instance(size, seed)
        time_remaining = DEADLINE - (time.time() - start_time)
        if time_remaining < 5:
            # Quick greedy
            order = list(S)
            random.shuffle(order)
            result = []
            result_set = set()
            for x in order:
                if try_add(x, result_set):
                    result.append(x)
                    result_set.add(x)
            results.append(result)
        else:
            # Allocate time based on size?
            # Larger sets need more time.
            # 60: 10%, 120: 30%, 240: 60%?
            # Or just split remaining time equally?
            # Let's split remaining time proportionally to size.
            # Total size = 60+120+240 = 420.
            # 60/420 = 1/7.
            # 120/420 = 2/7.
            # 240/420 = 4/7.
            pass
            
    # Better time allocation:
    # We have ~115s total.
    # Instance 1 (60): ~10s
    # Instance 2 (120): ~30s
    # Instance 3 (240): ~70s
    # Let's just use a fraction of the deadline.
    
    times = [10.0, 30.0, 70.0]
    for i, (size, seed) in enumerate(instances):
        S = instance(size, seed)
        # Adjust time limit to not exceed deadline
        t_limit = min(time.time() + times[i], start_time + DEADLINE - 1)
        if t_limit <= time.time():
            # Out of time, use best found so far or quick greedy
            result = [x for x in S if x % 2 != 0]
            results.append(result)
        else:
            result = solve_instance(S, t_limit)
            results.append(result)
            
    # Final verification
    for i, result in enumerate(results):
        if not is_sum_free(result):
            S = instance(instances[i][0], instances[i][1])
            results[i] = [x for x in S if x % 2 != 0]
            
    return results
