import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    N = 1000
    MAX_SUM = 2 * N  # 2000
    
    best_size = 0
    best_chosen = []
    
    # We will run multiple restarts
    # Each restart: Greedy + Local Search
    
    rng = random.Random(42)
    
    def run_one_search(seed_offset=0):
        """Runs one instance of Greedy + Local Search."""
        nonlocal best_size, best_chosen
        
        # Local state
        chosen_bits = 0
        chosen = set()
        sum_counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        blocked_bits = 0
        
        # Pre-compute bit masks for elements? No, just use 1 << x
        
        def can_add(x):
            # Check if x can be added
            # Sums: x+y for y in chosen, and 2x
            # (chosen_bits << x) has bits at x+y
            # If any of these overlap with blocked_bits, it's invalid.
            # Also check 2x separately if not covered? 
            # If y=x, then 2x is in chosen_bits << x? 
            # chosen_bits has bit x set only if x is in chosen. But we are checking if x *can* be added, so x is NOT in chosen yet.
            # So 2x is NOT covered by (chosen_bits << x) unless x is in chosen.
            # So we must check 2x explicitly.
            if (chosen_bits << x) & blocked_bits:
                return False
            if (blocked_bits >> (2*x)) & 1:
                return False
            return True

        def add_element(x):
            nonlocal chosen_bits, blocked_bits
            for y in chosen:
                s = x + y
                sum_counts[s] += 1
                if sum_counts[s] == 2:
                    blocked_bits |= (1 << s)
            s2 = 2 * x
            sum_counts[s2] += 1
            if sum_counts[s2] == 2:
                blocked_bits |= (1 << s2)
            chosen_bits |= (1 << x)
            chosen.add(x)

        def remove_element(x):
            nonlocal chosen_bits, blocked_bits
            for y in chosen:
                if y == x:
                    continue
                s = x + y
                sum_counts[s] -= 1
                if sum_counts[s] == 1:
                    blocked_bits &= ~(1 << s)
            s2 = 2 * x
            sum_counts[s2] -= 1
            if sum_counts[s2] == 1:
                blocked_bits &= ~(1 << s2)
            chosen_bits &= ~(1 << x)
            chosen.discard(x)

        # Phase 1: Greedy Construction
        # Use a random permutation to diversify restarts
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        for x in order:
            if time.time() > deadline:
                return
            if can_add(x):
                add_element(x)
        
        current_size = len(chosen)
        
        # Phase 2: Local Search
        # Try to improve by adding elements or swapping
        # We will iterate until no improvement or time runs out
        # To avoid infinite loops, we limit the number of "stagnant" steps
        stagnant_count = 0
        max_stagnant = 200
        temp = 1.0 # Not really using SA, just a counter
        
        while time.time() < deadline - 5 and stagnant_count < max_stagnant:
            improved = False
            
            # 1. Try to add any valid element
            # We can scan from 1 to N, or pick random ones
            # Scanning is deterministic and fast if we break early? No, we need to find ANY.
            # Let's try a few random candidates first, then scan if needed?
            # Actually, if we just scan 1..N, it's O(N) per check.
            # Better: maintain a list of candidates? 
            # For simplicity, let's just pick random candidates from the unchosen set.
            
            unchosen = [x for x in range(1, N + 1) if x not in chosen]
            if not unchosen:
                break
            
            # Try a few random adds
            found_add = False
            for _ in range(10):
                x = rng.choice(unchosen)
                if can_add(x):
                    add_element(x)
                    improved = True
                    found_add = True
                    # Update unchosen list? It's expensive. 
                    # We can just let it be stale for a bit, or rebuild it periodically.
                    # Rebuilding unchosen is O(N). Let's do it every 100 iterations or if it gets small.
                    break
            
            if found_add:
                stagnant_count = 0
                continue
            
            # 2. Try a swap: Remove one, Add one (size same, but might help?)
            # Or Remove one, Add two (size +1).
            
            # Strategy: Pick a random element to remove.
            x_out = rng.choice(list(chosen))
            remove_element(x_out)
            
            # Now try to add elements to the new set
            # We want to add at least 1 element to not lose size, preferably 2 to gain.
            # Let's try to add random elements.
            unchosen_now = [x for x in range(1, N + 1) if x not in chosen]
            
            added_any = False
            # Try to add up to 2 elements
            for _ in range(20):
                if not unchosen_now:
                    break
                x_in = rng.choice(unchosen_now)
                if can_add(x_in):
                    add_element(x_in)
                    unchosen_now.remove(x_in) # Expensive, but list is small-ish?
                    added_any = True
            
            if added_any:
                # We removed 1, added k. 
                # If k >= 1, size is same or better.
                # If k >= 2, size is better.
                # We accept if size >= previous size? 
                # If size < previous size (k=0), we should revert.
                # But we only add if can_add, so k>=1 if added_any is True.
                # So size is at least same.
                stagnant_count = 0
                improved = True
            else:
                # Revert the removal
                add_element(x_out)
                stagnant_count += 1
        
        # End of local search for this restart
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_chosen = list(chosen)
        
        return len(chosen)

    # Run multiple restarts
    # How many? Depends on time.
    # Greedy is O(N^2) roughly? No, O(N * cost_of_add). cost_of_add is O(|S|).
    # So Greedy is O(N^2) ~ 10^6 ops. Very fast.
    # Local search is the main cost.
    
    restarts = 0
    while time.time() < deadline - 10:
        run_one_search(restarts)
        restarts += 1
        # If we've found a very good set, maybe stop?
        # Max possible is ~1.5 * 1000 = 1500? No, B2[2] density is ~1/sqrt(2) * N? 
        # Actually, for B2[2], max size is approx N / sqrt(2) ? 
        # Wait, Sidon (B2[1]) is ~sqrt(N). 
        # B2[2] allows 2 representations. The density is higher.
        # Known bounds: |S| ~ c * N. c is around 1? Or higher?
        # The prompt says "asymptotic constant is at least 3/2". 
        # So we expect size around 1500? But N=1000. So max size is 1000.
        # Wait, "Large subsets of {1, ..., 1000}".
        # If the constant is 3/2, it means |S| ~ 1.5 * sqrt(N)? No.
        # Let's check the asymptotic.
        # For B_h[g] sets, the maximum size in [1, N] is ~ c_{h,g} N^{1/h}.
        # Here h=2. So size ~ c * sqrt(1000) ~ c * 31.6.
        # That seems small.
        # Let's re-read: "every sum has at most two representations".
        # This is a B_2[2] set.
        # The number of pairs is ~ |S|^2 / 2.
        # The sums are in [2, 2000].
        # If each sum has at most 2 representations, total representations <= 2 * 2000 = 4000.
        # So |S|^2 / 2 <= 4000 => |S|^2 <= 8000 => |S| <= 89.
        # This contradicts the "constant is at least 3/2" if it refers to linear size.
        # Ah, the prompt says "Large subsets of {1, ..., 1000}".
        # If the size is ~90, the score would be small.
        # Let's check the seed program score 1.0.
        # The seed program returns a list. The evaluator measures the raw value (size).
        # If the seed program gets size ~90, and score is 1.0, then the baseline is ~90.
        # If I can get 95, score is 1.05.
        
        # Let's verify the bound.
        # Sum s has r_s representations. sum r_s = number of pairs (a,b) with a<=b.
        # Number of pairs = |S|*(|S|+1)/2.
        # Max sum is 2000. Min sum is 2.
        # Range of sums is 1999.
        # If r_s <= 2 for all s, then |S|*(|S|+1)/2 <= 2 * 1999 = 3998.
        # |S|^2 <= 7996.
        # |S| <= 89.4.
        
        # So the maximum size is around 89 or 90.
        # The seed program likely finds something close to this.
        # My local search should try to push it to the theoretical max.
        
        # If we are at 89, we are near optimal.
        # The "3/2" constant might refer to something else, or I am misinterpreting.
        # "asymptotic constant is at least 3/2" for the extremal function.
        # Maybe it means |S| >= 1.5 * sqrt(N)?
        # 1.5 * sqrt(1000) = 47.
        # 89 is much larger.
        # Maybe the "constant" is for the density in a different regime?
        # Regardless, the upper bound of ~90 is hard.
        
        # So the goal is to get as close to 89-90 as possible.
        # The local search is crucial to escape local minima of size 80-85.
        
        # Optimization:
        # The `unchosen` list construction is O(N). Doing it inside the loop is expensive.
        # We can maintain a set of unchosen elements.
        
        # Let's rewrite `run_one_search` to be more efficient.

    return best_chosen
