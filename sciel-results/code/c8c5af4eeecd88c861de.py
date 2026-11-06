import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit
    
    N = 256
    universe = set(range(1, N+1))
    
    # State
    S = set()
    in_S = [False] * (N+1)
    count_3 = 0
    
    # Helper functions
    def delta_3_ap_add(x):
        # Returns number of new 3-APs if x is added to S
        new_count = 0
        # x is middle: (x-d, x, x+d)
        # x is left: (x, x+d, x+2d)
        # x is right: (x-2d, x-d, x)
        # We can just iterate d
        # Max d is N/2
        for d in range(1, N//2 + 1):
            # x as middle
            if x-d >= 1 and x+d <= N:
                if in_S[x-d] and in_S[x+d]:
                    new_count += 1
            # x as left
            if x+d <= N and x+2*d <= N:
                if in_S[x+d] and in_S[x+2*d]:
                    new_count += 1
            # x as right
            if x-2*d >= 1:
                if in_S[x-2*d] and in_S[x-d]:
                    new_count += 1
        return new_count

    def delta_3_ap_remove(x):
        # Returns number of 3-APs lost if x is removed from S
        # Same logic, but x is assumed to be in S
        lost_count = 0
        for d in range(1, N//2 + 1):
            if x-d >= 1 and x+d <= N:
                if in_S[x-d] and in_S[x+d]:
                    lost_count += 1
            if x+d <= N and x+2*d <= N:
                if in_S[x+d] and in_S[x+2*d]:
                    lost_count += 1
            if x-2*d >= 1:
                if in_S[x-2*d] and in_S[x-d]:
                    lost_count += 1
        return lost_count

    def creates_4ap(x):
        # Check if adding x to S creates a 4-AP
        # x can be a, a+d, a+2d, a+3d
        for d in range(1, N//3 + 1):
            # x is 1st term: x, x+d, x+2d, x+3d
            if x+3*d <= N:
                if in_S[x+d] and in_S[x+2*d] and in_S[x+3*d]:
                    return True
            # x is 2nd term: x-d, x, x+d, x+2d
            if x-d >= 1 and x+2*d <= N:
                if in_S[x-d] and in_S[x+d] and in_S[x+2*d]:
                    return True
            # x is 3rd term: x-2d, x-d, x, x+d
            if x-2*d >= 1 and x+d <= N:
                if in_S[x-2*d] and in_S[x-d] and in_S[x+d]:
                    return True
            # x is 4th term: x-3d, x-2d, x-d, x
            if x-3*d >= 1:
                if in_S[x-3*d] and in_S[x-2*d] and in_S[x-d]:
                    return True
        return False

    def add(x):
        nonlocal count_3
        S.add(x)
        in_S[x] = True
        count_3 += delta_3_ap_add(x)

    def remove(x):
        nonlocal count_3
        S.remove(x)
        in_S[x] = False
        count_3 -= delta_3_ap_remove(x)

    # 1. Greedy Initialization
    # Randomize order to break symmetry
    candidates = list(universe)
    random.shuffle(candidates)
    for x in candidates:
        if not creates_4ap(x):
            add(x)
    
    best_set = set(S)
    best_score = count_3

    # 2. Simulated Annealing
    T = 50.0
    T_min = 0.1
    cooling = 0.99
    
    while time.time() < deadline and T > T_min:
        T *= cooling
        
        # Pick move: Add, Remove, Swap
        move_type = random.random()
        
        if move_type < 0.4: # Add
            # Pick a candidate not in S
            # Optimization: Pre-compute valid candidates? No, dynamic.
            # Just pick random from universe
            candidates_to_try = [x for x in universe if x not in S]
            if not candidates_to_try:
                continue
            x = random.choice(candidates_to_try)
            if creates_4ap(x):
                continue # Invalid move
            delta = delta_3_ap_add(x)
            # Always accept if delta > 0? Or use Metropolis?
            # Since we maximize, and constraint is hard, we accept if delta > 0.
            # But sometimes we might need to remove to add more later?
            # SA allows worse moves to escape.
            # But "worse" here means fewer 3-APs.
            if delta > 0 or random.random() < pow(2.71828, delta/T):
                add(x)
                if count_3 > best_score:
                    best_score = count_3
                    best_set = set(S)
                    
        elif move_type < 0.7: # Remove
            if not S:
                continue
            x = random.choice(list(S))
            delta = -delta_3_ap_remove(x)
            # Removing decreases score (usually).
            # We do this to allow future additions.
            if random.random() < pow(2.71828, delta/T):
                remove(x)
                
        else: # Swap
            if not S:
                continue
            x_in = random.choice(list(S))
            candidates_out = [x for x in universe if x not in S]
            if not candidates_out:
                continue
            x_out = random.choice(candidates_out)
            
            # Check if swap is valid
            # Remove x_in, Add x_out
            # Must check if x_out creates 4AP with S \ {x_in}
            # And x_in is removed.
            
            # Temporarily remove x_in
            remove(x_in)
            
            if not creates_4ap(x_out):
                delta_in = delta_3_ap_add(x_out)
                delta_out = -delta_3_ap_remove(x_in) # This was calculated before removal? No, we removed it.
                # We need delta of the swap.
                # Score_new = Score_old - lost(x_in) + gained(x_out)
                # But lost(x_in) depends on S with x_in.
                # We already removed x_in.
                # Let's re-calculate properly.
                
                # Actually, simpler:
                # Save state
                # Try swap
                # If valid and (score improves or accept), keep
                # Else revert
                
                # This is getting complex with incremental updates during swap.
                # Let's just do Add and Remove moves. They are sufficient for SA.
                # Swap is just Remove then Add.
                
                # Revert if not accepted
                # We need to know if we should accept.
                # We don't have the delta yet.
                # Let's stick to Add/Remove for simplicity and robustness.
                # Swap can be simulated by Remove then Add in next step.
                
                # Revert
                add(x_in)
            else:
                add(x_in)

    return sorted(best_set)
