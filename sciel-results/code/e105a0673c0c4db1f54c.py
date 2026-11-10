import time
import random
from fractions import Fraction

def construct():
    N = 256
    start_time = time.time()
    time_limit = 50.0  # Safety margin
    
    # Bitmask for membership
    # x is in [1, N], so we use bits 0..N-1 representing 1..N
    # bit i represents number i+1
    
    def can_add(x, mask):
        # x is 1-based index
        # Check if x is already in mask
        if (mask >> (x - 1)) & 1:
            return False
        
        # 1. x is middle: a + c = 2x => c = 2x - a
        # We need to check if there exists a in S such that c = 2x - a is in S and c != a.
        # Note: if c == a, then 2x = 2a => x=a, but we already checked x not in S.
        # Actually if c=a, then a is in S and 2x-a=a => 2x=2a => x=a. Contradiction.
        # So we just check if c is in mask.
        
        # 2. x is end:
        # Case A: a, b, x => b = (a+x)/2 => a = 2b - x.
        # Check if exists b in S such that a = 2b - x is in S.
        # If a == b, then 2b-x=b => b=x. Contradiction.
        # So we just check if a is in mask.
        
        # We can combine these checks by iterating over all elements in S?
        # Or iterate over all possible a?
        # Since N=256, we can just iterate a from 1 to N.
        # But we can be smarter.
        
        # Let's iterate over a in 1..N. If a is in mask:
        #   c1 = 2*x - a. If c1 in mask and c1 != a: invalid. (x is middle)
        #   c2 = 2*a - x. If c2 in mask and c2 != a: invalid. (x is end, a is middle)
        
        # Wait, the condition "c1 in mask and c1 != a" covers x as middle.
        # The condition "c2 in mask and c2 != a" covers x as end (where a is middle).
        
        # Is it possible that c1 == a?
        # 2x - a = a => 2x = 2a => x = a. But x not in mask. So c1 != a always if x not in mask?
        # No, if a is in mask, and x not in mask, then x != a. So c1 != a is automatically true?
        # If c1 = a, then 2x - a = a => x = a. Since x not in mask, this cannot happen for any a in mask.
        # So we can just check `if c1 in mask`.
        
        # Similarly for c2:
        # c2 = 2a - x. If c2 = a, then 2a - x = a => a = x. Since x not in mask, this cannot happen for any a in mask.
        # So we can just check `if c2 in mask`.
        
        # So the check simplifies to:
        # For any a in S:
        #   if (2x - a) in S: return False
        #   if (2a - x) in S: return False
        
        # We can iterate over a from 1 to N.
        # To speed up, we can iterate only over set bits in mask?
        # Python integers don't have a fast "iterate set bits" in standard lib without tricks.
        # But N=256 is small. 256 iterations is very fast.
        
        # Actually, we can just check all a in 1..N.
        # But we can prune:
        # For 2x - a to be in [1, N], a must be in [2x-N, 2x-1].
        # For 2a - x to be in [1, N], a must be in [x+1, x+N] -> [x+1, N] (since a<=N).
        
        # Let's just iterate a from 1 to N for simplicity and correctness, it's fast enough.
        # 256 iterations * 2 checks * 2 bit ops = ~1000 ops per can_add.
        # With ~100 elements in S, and ~100 candidates, 10k swaps * 1000 ops = 10M ops.
        # Python can do ~10M-50M simple ops/sec. This might be tight for 50s if we do many passes.
        # Let's optimize by iterating only over a in S.
        
        # How to iterate over set bits efficiently?
        # m = mask
        # while m:
        #   lsb = m & -m
        #   a = lsb.bit_length() # 1-based index
        #   m ^= lsb
        
        m = mask
        while m:
            lsb = m & -m
            a = lsb.bit_length() # This gives 1-based index directly?
            # bit_length of 1<<0 is 1. So a=1. Correct.
            # bit_length of 1<<1 is 2. So a=2. Correct.
            
            # Check 1: x is middle. a, x, c. c = 2x - a.
            c1 = 2 * x - a
            if 1 <= c1 <= N:
                if (mask >> (c1 - 1)) & 1:
                    return False
            
            # Check 2: x is end. c, a, x. c = 2a - x.
            c2 = 2 * a - x
            if 1 <= c2 <= N:
                if (mask >> (c2 - 1)) & 1:
                    return False
            
            m ^= lsb
            
        return True

    def get_score(mask):
        # Sum of reciprocals
        s = 0.0
        m = mask
        while m:
            lsb = m & -m
            x = lsb.bit_length()
            s += 1.0 / x
            m ^= lsb
        return s

    # 1. Greedy Initialization
    # Add elements in increasing order (smallest first) to maximize reciprocal sum.
    mask_greedy = 0
    for x in range(1, N + 1):
        if can_add(x, mask_greedy):
            mask_greedy |= (1 << (x - 1))
    
    current_mask = mask_greedy
    current_score = get_score(current_mask)
    best_mask = current_mask
    best_score = current_score
    
    # 2. Local Search
    # Strategy: Systematic Swaps
    # Try to swap every x in S with every y not in S where y < x.
    # If valid and score improves, accept.
    # Repeat until no improvement or time limit.
    
    # To make it stochastic/restartable, we can shuffle the order of checks or restart.
    # But systematic is better for finding the specific small y.
    
    # We will run passes. In each pass, we try all valid swaps.
    # If any swap is accepted, we restart the pass (or continue? Restarting is safer for hill climbing).
    
    improved = True
    while time.time() - start_time < time_limit and improved:
        improved = False
        # Get list of elements in S and not in S
        # S = [x for x in 1..N if in mask]
        # NotS = [y for y in 1..N if not in mask]
        
        # Sort S descending (large x first, low value)
        # Sort NotS ascending (small y first, high value)
        S_list = []
        NotS_list = []
        m = current_mask
        while m:
            lsb = m & -m
            x = lsb.bit_length()
            S_list.append(x)
            m ^= lsb
        S_list.sort(reverse=True)
        
        for y in range(1, N + 1):
            if not ((current_mask >> (y - 1)) & 1):
                NotS_list.append(y)
        
        # Try swaps
        # For each x in S_list (large to small)
        # For each y in NotS_list (small to large)
        # If y >= x, break inner loop (since y is ascending, further y are larger, 1/y <= 1/x, no gain)
        
        for x in S_list:
            # If time limit approaching, break
            if time.time() - start_time > time_limit - 5:
                break
                
            for y in NotS_list:
                if y >= x:
                    break
                
                # Try swap x with y
                # Remove x, add y
                new_mask = current_mask
                new_mask &= ~(1 << (x - 1))
                
                if can_add(y, new_mask):
                    new_mask |= (1 << (y - 1))
                    new_score = current_score - (1.0 / x) + (1.0 / y)
                    
                    if new_score > current_score + 1e-12:
                        current_mask = new_mask
                        current_score = new_score
                        improved = True
                        
                        if current_score > best_score:
                            best_mask = current_mask
                            best_score = current_score
                        
                        # Restart inner/outer loops? 
                        # Since we changed the set, S_list and NotS_list are stale.
                        # We should restart the pass to re-evaluate.
                        break
            
            if improved:
                break
        
        # If no improvement in this pass, we might be stuck.
        # We can try a random perturbation (remove a random element, add random small ones)
        # Or just stop if we want pure greedy hill climbing.
        # Given the time budget, let's add a simple random restart if stuck for a while.
        # But systematic swap is quite powerful.
        
        # If we didn't improve in the last pass, let's try a few random swaps to escape local optima
        if not improved:
             # Random perturbation
             if S_list:
                 x_to_remove = random.choice(S_list)
                 new_mask = current_mask & ~(1 << (x_to_remove - 1))
                 # Try to add a random small y
                 candidates = [y for y in range(1, N+1) if not ((new_mask >> (y-1)) & 1) and y < x_to_remove]
                 if candidates:
                     y_to_add = random.choice(candidates)
                     if can_add(y_to_add, new_mask):
                         new_mask |= (1 << (y_to_add - 1))
                         new_score = current_score - (1.0/x_to_remove) + (1.0/y_to_add)
                         if new_score > current_score - 0.01: # Accept slightly worse to escape?
                             # Only accept if it leads to improvement later?
                             # For now, just accept if better.
                             pass
             # If still stuck, we might need to restart from greedy?
             # Or just stop.
             pass

    # Return best set found
    result_list = []
    m = best_mask
    while m:
        lsb = m & -m
        x = lsb.bit_length()
        result_list.append(x)
        m ^= lsb
        
    return result_list
