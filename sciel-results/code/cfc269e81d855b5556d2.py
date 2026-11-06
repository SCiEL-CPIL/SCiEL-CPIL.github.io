import time
import random
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    # 90s limit, use 55s for safety
    deadline = start_time + 55.0
    N = 4096
    
    def check_and_get_new_sums(S, sums_set, x):
        """
        Checks if x can be added to S without violating unique triple sums.
        Returns (True, new_sums_set) if valid, (False, None) otherwise.
        """
        n = len(S)
        new_sums = set()
        
        # 1. (x, x, x)
        s3 = 3 * x
        if s3 in sums_set:
            return False, None
        new_sums.add(s3)
        
        # 2. (x, x, s_i)
        for s_i in S:
            s_val = 2 * x + s_i
            if s_val in sums_set or s_val in new_sums:
                return False, None
            new_sums.add(s_val)
            
        # 3. (x, s_i, s_j) with i <= j
        for i in range(n):
            s_i = S[i]
            for j in range(i, n):
                s_j = S[j]
                s_val = x + s_i + s_j
                if s_val in sums_set or s_val in new_sums:
                    return False, None
                new_sums.add(s_val)
                
        return True, new_sums

    def greedy_search(order):
        """
        Performs a greedy search given a specific order of candidates.
        Returns the selected set.
        """
        S = []
        sums_set = set()
        
        for x in order:
            if time.time() > deadline:
                break
            valid, new_sums = check_and_get_new_sums(S, sums_set, x)
            if valid:
                S.append(x)
                sums_set.update(new_sums)
                
        return S

    def compute_all_sums(S):
        """Recomputes the full set of triple sums for a given set S."""
        sums = set()
        n = len(S)
        # Using nested loops for speed in Python for small n
        for i in range(n):
            si = S[i]
            for j in range(i, n):
                sj = S[j]
                for k in range(j, n):
                    sk = S[k]
                    sums.add(si + sj + sk)
        return sums

    # Initialize with a natural order greedy pass (often finds a decent baseline quickly)
    # This is O(N * K^2). N=4096, K~30-50.
    # 4096 * 2500 ~ 10^7 ops. Should be < 1s in optimized loop? Maybe 2-3s.
    # Let's just run it.
    
    best_set = []
    best_len = 0
    
    # Strategy 1: Natural Order (Smallest to Largest)
    # This is a standard heuristic for B_h sets as it keeps sums small.
    order_natural = list(range(1, N + 1))
    S_nat = greedy_search(order_natural)
    if len(S_nat) > best_len:
        best_set = S_nat
        best_len = len(S_nat)
        
    if time.time() > deadline - 5.0:
        return best_set

    # Strategy 2: Randomized Greedy Restarts
    # We try to find larger sets by varying the order.
    # Smaller numbers are generally "better" for packing, but random order might avoid local traps.
    random.seed(42)
    
    # We can afford roughly 50-100 restarts depending on speed.
    # Each restart is O(N * K^2).
    # Let's loop until time is up.
    
    while time.time() < deadline - 10.0: # Leave 10s for local improvement
        # Create a random order
        # Bias: sometimes prefer smaller numbers?
        # If we shuffle uniformly, we might pick large numbers early, which might block many small ones later?
        # Actually, picking large numbers early creates large sums, which might not collide with small sums.
        # But we want to maximize count.
        
        # Let's try a mix:
        # 1. Uniform shuffle
        # 2. Shuffle but sort by value % some_prime (to decorrelate)
        # 3. Just uniform shuffle is usually fine for diverse starts.
        
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S_try = greedy_search(order)
        
        if len(S_try) > best_len:
            best_set = S_try
            best_len = len(S_try)
            
        # Optional: Occasionally try reverse order or biased order
        if random.random() < 0.1:
             order_rev = list(range(N, 0, -1))
             random.shuffle(order_rev) # Shuffle reverse
             S_try_rev = greedy_search(order_rev)
             if len(S_try_rev) > best_len:
                 best_set = S_try_rev
                 best_len = len(S_try_rev)

    # Strategy 3: Local Improvement
    # Try to add any remaining elements to the best set found.
    if best_len > 0 and time.time() < deadline - 2.0:
        S_cur = list(best_set)
        sums_cur = compute_all_sums(S_cur)
        S_cur_set = set(S_cur)
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Try to add any number from 1 to N
            for x in range(1, N + 1):
                if x in S_cur_set:
                    continue
                valid, new_sums = check_and_get_new_sums(S_cur, sums_cur, x)
                if valid:
                    S_cur.append(x)
                    S_cur_set.add(x)
                    sums_cur.update(new_sums)
                    best_len = len(S_cur)
                    best_set = S_cur
                    improved = True
                    break # Restart the loop as the set changed
                
        # If no simple addition worked, try a "kick" (remove one, add one)
        # This is more expensive. We only do this if we have time.
        if time.time() < deadline - 5.0:
            # Try a few random kicks
            for _ in range(5): # Limit kicks
                if time.time() > deadline - 5.0:
                    break
                S_kick = list(best_set)
                sums_kick = compute_all_sums(S_kick)
                S_kick_set = set(S_kick)
                
                # Remove a random element
                if not S_kick:
                    break
                idx_to_remove = random.randint(0, len(S_kick) - 1)
                removed_elem = S_kick.pop(idx_to_remove)
                S_kick_set.discard(removed_elem)
                
                # Try to add any element
                found_addition = False
                for x in range(1, N + 1):
                    if x in S_kick_set:
                        continue
                    valid, new_sums = check_and_get_new_sums(S_kick, sums_kick, x)
                    if valid:
                        S_kick.append(x)
                        S_kick_set.add(x)
                        sums_kick.update(new_sums)
                        found_addition = True
                        break
                        
                if found_addition:
                    # We have a new set of the same size (or potentially larger if we can add more)
                    # Let's see if we can add MORE to this new set
                    improved2 = True
                    while improved2 and time.time() < deadline - 5.0:
                        improved2 = False
                        for x in range(1, N + 1):
                            if x in S_kick_set:
                                continue
                            valid, new_sums = check_and_get_new_sums(S_kick, sums_kick, x)
                            if valid:
                                S_kick.append(x)
                                S_kick_set.add(x)
                                sums_kick.update(new_sums)
                                improved2 = True
                                break
                                
                    if len(S_kick) > best_len:
                        best_set = S_kick
                        best_len = len(S_kick)

    return best_set
