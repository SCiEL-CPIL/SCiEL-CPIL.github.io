import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    N = 1000
    max_sum = 2 * N
    
    # counts[s] = number of pairs (a,b) in S with a<=b and a+b=s
    counts = [0] * (max_sum + 1)
    in_set = [False] * (N + 1)
    S = []
    
    def can_add(y):
        # Check if adding y keeps all counts <= 2
        # Sums involving y: y+x for x in S, and 2y
        # Note: if y is in S, we don't add it. We assume y not in S.
        # We need to check if any of these sums would exceed 2.
        # Since we are adding, we just check current counts.
        # If counts[s] + 1 > 2, i.e., counts[s] == 2, then invalid.
        for x in S:
            s = x + y
            if counts[s] == 2:
                return False
        s_2y = 2 * y
        if counts[s_2y] == 2:
            return False
        return True

    def do_add(y):
        for x in S:
            counts[x + y] += 1
        counts[2 * y] += 1
        in_set[y] = True
        S.append(y)

    def do_remove(x):
        for y in S:
            if y != x:
                counts[x + y] -= 1
        counts[2 * x] -= 1
        in_set[x] = False
        S.remove(x)

    def get_congestion():
        # Sum of squares of counts
        return sum(c * c for c in counts)
        
    # 1. Greedy Initialization
    for x in range(1, N + 1):
        if can_add(x):
            do_add(x)
            
    best_size = len(S)
    best_S = list(S)
    
    # 2. Local Search
    # We will try to improve by swapping elements to reduce congestion,
    # hoping to allow more additions.
    
    current_congestion = get_congestion()
    
    while time.time() < deadline:
        # Phase 1: Try to add any element
        added = False
        # Try candidates in random order to avoid bias
        candidates = [i for i in range(1, N + 1) if not in_set[i]]
        random.shuffle(candidates)
        
        for y in candidates:
            if can_add(y):
                do_add(y)
                added = True
                break
        
        if added:
            if len(S) > best_size:
                best_size = len(S)
                best_S = list(S)
            current_congestion = get_congestion()
            continue
        
        # Phase 2: No addition possible. Try to swap to reduce congestion.
        # We look for a swap (x in S, y not in S) such that S' is valid
        # and congestion(S') < congestion(S).
        
        best_swap_x = -1
        best_swap_y = -1
        best_swap_congestion = current_congestion
        
        # To speed up, we might not check ALL pairs. 
        # But N=1000, |S|~700. Pairs ~ 700 * 300 = 210,000.
        # Each check is O(|S|) ~ 700. Total ~ 1.5 * 10^8 ops.
        # This might be too slow in Python if done every iteration.
        # Let's optimize the check or sample.
        
        # Optimization: 
        # Instead of full O(|S|) check for can_add after removal, 
        # we can be smarter.
        
        # Let's try a limited number of swaps per "stuck" state to keep it fast.
        # Or, we can pre-compute which sums are "full" (count 2).
        
        # Let's try a random sampling of swaps if the full search is too slow.
        # First, let's try a full search but with early termination or sampling.
        
        # Actually, let's just do a random search over swaps to find *any* valid swap 
        # that reduces congestion, or just *any* valid swap to perturb.
        
        found_improvement = False
        attempts = 0
        max_attempts = 1000 # Limit search per iteration
        
        while attempts < max_attempts and not found_improvement:
            attempts += 1
            x = random.choice(S)
            y = random.choice(candidates) # candidates is not in S
            
            # Check if swap x->y is valid
            # 1. Remove x
            # We need to simulate.
            # Sums affected by removing x: x+y' for y' in S\{x}, 2x.
            # Sums affected by adding y: y+x' for x' in S\{x}, 2y.
            
            # Let's just do it:
            # Save state? No, too expensive.
            # Let's check validity logically.
            
            # Condition for validity of S' = S - {x} + {y}:
            # For any sum s, count_{S'}(s) <= 2.
            
            # Let's just try it with a helper that does the add/remove 
            # but we need to be careful about time.
            
            # Given the constraints, let's stick to the simple 
            # "try all candidates for addition" and "try random swaps".
            
            # To make swaps effective, we should check if the swap 
            # actually reduces congestion.
            
            # Let's implement a fast validity check for swap.
            
            # We'll skip the complex fast check for now and rely on 
            # the fact that random swaps might find a way out.
            
            # However, random swaps that are valid but don't reduce congestion 
            # might lead to a dead end.
            
            # Let's try this: 
            # If we find a valid swap, we calculate the new congestion.
            # If it's better, we take it.
            
            # To speed up, we only check a few random swaps per "stuck" phase.
            
            # Simulate swap
            # Remove x
            for y_prime in S:
                if y_prime != x:
                    counts[x + y_prime] -= 1
            counts[2 * x] -= 1
            in_set[x] = False
            S.remove(x)
            
            # Check if y can be added to new S
            valid = True
            for x_prime in S:
                if counts[x_prime + y] == 2:
                    valid = False
                    break
            if valid and counts[2 * y] == 2:
                valid = False
                
            if valid:
                # Calculate new congestion (approximate or exact?)
                # Exact is expensive. Let's just add it and see if size increases later.
                # Or, we can check if this swap "frees up" a sum.
                
                # Let's just add y
                for x_prime in S:
                    counts[x_prime + y] += 1
                counts[2 * y] += 1
                in_set[y] = True
                S.append(y)
                
                # Check if this state allows any addition?
                # We'll let the main loop handle additions.
                # But we should only accept the swap if it's "promising".
                # Heuristic: If the new max count is lower, or if we can add something.
                
                # Let's check if we can add something immediately.
                added_after_swap = False
                candidates_after = [i for i in range(1, N + 1) if not in_set[i]]
                random.shuffle(candidates_after)
                for z in candidates_after:
                    if can_add(z):
                        do_add(z)
                        added_after_swap = True
                        break
                        
                if added_after_swap:
                    # Great, we escaped!
                    found_improvement = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = list(S)
                    current_congestion = get_congestion()
                else:
                    # Didn't help immediately. 
                    # We can revert the swap to keep searching for a better one,
                    # or keep it if it reduced congestion.
                    
                    # Let's revert for now to avoid drifting into bad states.
                    # Remove y
                    for x_prime in S:
                        if x_prime != y:
                            counts[x_prime + y] -= 1
                    counts[2 * y] -= 1
                    in_set[y] = False
                    S.remove(y)
                    
                    # Add x back
                    for y_prime in S:
                        counts[x + y_prime] += 1
                    counts[2 * x] += 1
                    in_set[x] = True
                    S.append(x)
                    
            else:
                # Invalid swap, revert removal of x
                for y_prime in S:
                    counts[x + y_prime] += 1
                counts[2 * x] += 1
                in_set[x] = True
                S.append(x)
                
        if not found_improvement:
            # If we tried many swaps and didn't find an escape, 
            # we might be stuck in a local optimum.
            # We can try to restart with a different random seed or order.
            # But restarting is expensive.
            # Let's just break if we've spent too much time.
            pass

    return best_S

# Test locally
if __name__ == "__main__":
    res = construct()
    print(len(res))
