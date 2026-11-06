import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    N = 256

    # Precompute 3-APs and 4-APs
    three_aps = []
    four_aps = []
    
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))
            
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))

    # Map element to indices of APs it belongs to
    # Also store the "other" elements for fast check
    elem_to_3aps = [[] for _ in range(N + 1)]
    for idx, (a, b, c) in enumerate(three_aps):
        elem_to_3aps[a].append((idx, b, c))
        elem_to_3aps[b].append((idx, a, c))
        elem_to_3aps[c].append((idx, a, b))

    elem_to_4aps = [[] for _ in range(N + 1)]
    for idx, (a, b, c, d) in enumerate(four_aps):
        elem_to_4aps[a].append((idx, b, c, d))
        elem_to_4aps[b].append((idx, a, c, d))
        elem_to_4aps[c].append((idx, a, b, d))
        elem_to_4aps[d].append((idx, a, b, c))

    def count_3aps(S):
        """Count 3-APs in bitmask S"""
        count = 0
        for a, b, c in three_aps:
            if (S >> (a-1)) & 1 and (S >> (b-1)) & 1 and (S >> (c-1)) & 1:
                count += 1
        return count

    def is_safe_add(S, x):
        """Check if adding x to S creates any 4-AP"""
        for _, u, v, w in elem_to_4aps[x]:
            if ((S >> (u-1)) & 1) and ((S >> (v-1)) & 1) and ((S >> (w-1)) & 1):
                return False
        return True

    def gain_3aps(S, x):
        """Calculate number of new 3-APs formed by adding x to S"""
        gain = 0
        for _, u, v in elem_to_3aps[x]:
            if ((S >> (u-1)) & 1) and ((S >> (v-1)) & 1):
                gain += 1
        return gain

    def loss_3aps(S, x):
        """Calculate number of 3-APs lost by removing x from S"""
        # x is currently in S
        loss = 0
        for _, u, v in elem_to_3aps[x]:
            if ((S >> (u-1)) & 1) and ((S >> (v-1)) & 1):
                loss += 1
        return loss

    # Initialize with randomized greedy
    def generate_initial_set():
        S = 0
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        for x in elements:
            if is_safe_add(S, x):
                S |= (1 << (x-1))
        return S

    # Local Search: Remove-Fill strategy
    def local_search(S, time_limit):
        current_S = S
        current_score = count_3aps(current_S)
        best_S = current_S
        best_score = current_score
        
        T = 10.0
        alpha = 0.995
        
        while time.time() < time_limit:
            # Find a random element to remove
            # We want to remove an element that is "cheap" to lose (low loss_3aps) 
            # or just random to explore.
            # Let's pick a random element from the current set.
            bits = [i+1 for i in range(N) if (current_S >> i) & 1]
            if not bits:
                break
            
            x = random.choice(bits)
            
            # Remove x
            new_S = current_S & ~(1 << (x-1))
            loss = loss_3aps(current_S, x)
            new_score = current_score - loss
            
            # Greedily add safe elements to new_S
            # Try adding elements not in new_S, in random order
            candidates = [i for i in range(1, N+1) if not ((new_S >> (i-1)) & 1)]
            random.shuffle(candidates)
            
            added_count = 0
            for y in candidates:
                if is_safe_add(new_S, y):
                    gain = gain_3aps(new_S, y)
                    new_S |= (1 << (y-1))
                    new_score += gain
                    added_count += 1
                    # Early stop if we've added many and score is already good?
                    # No, we want maximal set.
            
            # Compare
            if new_score > best_score:
                best_S = new_S
                best_score = new_score
                current_S = new_S
                current_score = new_score
            else:
                # Simulated Annealing acceptance
                delta = new_score - current_score
                if delta < 0:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_S = new_S
                        current_score = new_score
                    # else revert (do nothing)
                else:
                    current_S = new_S
                    current_score = new_score
            
            T *= alpha
            if T < 0.1:
                T = 0.1

        return best_S, best_score

    # Run multiple restarts
    best_global_S = 0
    best_global_score = 0
    
    num_restarts = 50 # Max potential, limited by time
    
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        S_init = generate_initial_set()
        score_init = count_3aps(S_init)
        
        # Quick local search for this restart
        # Allocate time based on remaining budget
        time_left = deadline - time.time()
        if time_left < 2.0:
            break
            
        # Give each restart a slice of time, or just run until deadline for the last few?
        # Better: run local search for a fixed number of steps or time slice.
        # Let's run local search with a deadline of start + 10s per restart? 
        # No, total time is 50s.
        # Let's do 10 restarts with 4s each, then 1 final long run.
        
        if i < 10:
            slice_deadline = start_time + (i+1)*4.0
            if slice_deadline > deadline:
                slice_deadline = deadline
            S_opt, score_opt = local_search(S_init, slice_deadline)
        else:
            S_opt, score_opt = local_search(S_init, deadline)
            
        if score_opt > best_global_score:
            best_global_score = score_opt
            best_global_S = S_opt

    # Convert bitmask to list
    result = [i for i in range(1, N+1) if (best_global_S >> (i-1)) & 1]
    
    # Final safety check: Ensure no 4-APs
    # If somehow invalid, fallback to a simple greedy
    valid = True
    for a, b, c, d in four_aps:
        if (best_global_S >> (a-1)) & 1 and (best_global_S >> (b-1)) & 1 and \
           (best_global_S >> (c-1)) & 1 and (best_global_S >> (d-1)) & 1:
            valid = False
            break
            
    if not valid or not result:
        # Fallback: Simple greedy
        S = 0
        for x in range(1, N+1):
            if is_safe_add(S, x):
                S |= (1 << (x-1))
        result = [i for i in range(1, N+1) if (S >> (i-1)) & 1]

    return result
