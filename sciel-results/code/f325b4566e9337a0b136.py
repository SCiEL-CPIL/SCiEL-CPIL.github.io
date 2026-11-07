import time
import random

P = {'n': 128}

def construct():
    n = P['n']
    deadline = time.time() + 85.0
    
    # Precompute bit masks for each value in 1..n
    # bit i represents value i+1
    # We need to check if a set S is valid.
    # Condition: No distinct a,b,c in S such that a+b, a+c, b+c are all in S.
    
    # Bitset representation:
    # s is an integer where bit i is 1 if (i+1) is in S.
    
    def is_valid_bitset(s):
        # Extract elements to list for triple iteration
        # Optimization: If s is empty or small, handle quickly
        # We can iterate over set bits.
        # To speed up, we can precompute pairs? 
        # Actually, for N=128, iterating triples is O(N^3) ~ 2M ops worst case.
        # But we only need to check if ANY triple fails.
        # A faster check:
        # For each pair (a, b) in S, let s_ab = a+b.
        # If s_ab is in S, then for any c in S such that c > b (to ensure distinctness and avoid duplicates),
        # we need to check if a+c and b+c are in S.
        # If both are in S, invalid.
        
        # Let's use a list of elements for the triple loop, but use bitset for membership.
        # This is still O(N^3) in Python loops, which is ~1M-2M iterations.
        # 2M iterations * ~100ns = 0.2s per check. 
        # We can do ~100-500 checks per second. That might be too slow for extensive search.
        
        # Better: Vectorize the check?
        # Let's try to optimize the triple check.
        # Iterate a < b < c.
        # Check (a+b in S) AND (a+c in S) AND (b+c in S).
        
        # We can precompute for each pair (a,b) if a+b is in S.
        # Then for each a, b, if a+b in S, we need to find c > b such that a+c in S and b+c in S.
        # This is equivalent to: (S shifted by a) & (S shifted by b) & (S shifted by ?) ... no.
        
        # Let's stick to the bitset membership but optimize the loop.
        # Extract elements:
        elems = []
        temp = s
        while temp:
            if temp & 1:
                elems.append(len(elems) + 1) # 1-based index
            temp >>= 1
        
        L = len(elems)
        if L < 3:
            return True
        
        # Quick check: if min*2 > max, valid
        if 2 * elems[0] > elems[-1]:
            return True
            
        # Precompute sums existence?
        # For each a, b, check if a+b in S.
        # We can do this with bit shifts.
        # Let's just loop.
        # To speed up, we can break early.
        
        # Optimization: 
        # For each a, b (a < b):
        #   if (a+b) not in S: continue
        #   For each c > b:
        #      if (a+c) in S and (b+c) in S: return False
        
        # This is still O(N^3) worst case but often breaks early.
        # Let's try to use numpy? No, standard lib is safer for bit ops if done right.
        
        # Alternative: 
        # For each c (largest), check if there exist a, b < c such that a+b, a+c, b+c in S.
        # This doesn't simplify much.
        
        # Let's assume the loop is the bottleneck. 
        # We can use a "forbidden pairs" logic?
        # If a+b is in S, then for any c, we cannot have both a+c and b+c in S.
        # So for fixed a,b with a+b in S, the set of valid c's is those where NOT (a+c in S AND b+c in S).
        
        # Let's implement the loop with early breaks.
        # To make it faster, we can iterate over the actual elements present.
        
        for i in range(L):
            a = elems[i]
            for j in range(i+1, L):
                b = elems[j]
                sum_ab = a + b
                if not (s >> (sum_ab - 1)) & 1:
                    continue
                # Now check c > b
                for k in range(j+1, L):
                    c = elems[k]
                    if ((s >> (a + c - 1)) & 1) and ((s >> (b + c - 1)) & 1):
                        return False
        return True

    # Initial state: {65, ..., 128}
    # Size 64.
    # Bitset: bits 64 to 127 set (0-indexed 64..127 correspond to values 65..128)
    # Value v corresponds to bit v-1.
    # So 65 is bit 64, 128 is bit 127.
    mask = (1 << 64) - 1
    initial_s = mask << 64
    current_s = initial_s
    best_s = initial_s
    best_size = 64
    
    # Randomized Local Search
    # We will try to add elements or swap elements.
    # Adding an element is good if it increases size and keeps validity.
    # Swapping an element (remove one, add one) keeps size same but might allow further additions.
    
    # Since we want to maximize size, we primarily try to ADD.
    # If we get stuck, we try SWAPS to escape local optima.
    
    # Precompute list of all values 1..128
    all_vals = list(range(1, 129))
    
    # Temperature for simulated annealing? 
    # Or just greedy with random restarts?
    # Given the time limit, a simple hill climbing with random restarts might work.
    # But 64 is already quite high. Can we go higher?
    # Theoretical max?
    # If we take all even numbers, size 64. 
    # If we take 1..64, size 64.
    # Can we mix?
    # Let's try to add small numbers to the upper half set.
    
    # Strategy:
    # 1. Start with {65..128}.
    # 2. Try to add any number from 1..64.
    # 3. If adding fails, try to remove a "blocking" number from the current set and add the new one.
    # 4. Iterate.
    
    # To make this efficient, we need a fast validity check.
    # The O(N^3) check in Python is ~10ms for N=64? 
    # 64^3 / 6 ~ 43k triples. 43k * 10 ops = 430k ops. ~0.01s.
    # So we can do ~100 checks per second.
    # 85 seconds -> 8500 checks.
    # This is a small search space.
    
    # Let's optimize the check further.
    # We can precompute for the CURRENT set which pairs (a,b) have a+b in S.
    # But S changes.
    
    # Let's use the bitset check as defined above. It's reasonably fast.
    
    # Improvement: 
    # Instead of checking all triples, we can check "conflicts" more efficiently.
    # For a new element x to be added to S:
    # 1. Check triples involving x: (x, a, b).
    #    Sums: x+a, x+b, a+b.
    #    If a+b in S, and x+a in S, and x+b in S -> Invalid.
    # 2. Check triples NOT involving x, but where x completes a sum.
    #    (a, b, c) in S.
    #    If a+b = x, and a+c in S, and b+c in S -> Invalid.
    #    Since a+b=x, c must be such that a+c and b+c are in S.
    #    Note: x is not in S_old, so a+b was NOT in S_old.
    #    Now x is added.
    
    # This "incremental check" is O(N^2) instead of O(N^3).
    # Let's implement this for the search.
    
    def can_add(s, x):
        # s is bitset of current set
        # x is the value to add
        # x must not be in s
        if (s >> (x - 1)) & 1:
            return False
            
        # 1. Check triples (x, a, b) with a < b in S
        #    Need a+b in S, x+a in S, x+b in S
        #    Iterate a, b in S.
        #    Optimization: a+b must be in S.
        #    So iterate pairs (a,b) in S such that a+b in S.
        #    For each such pair, check if x+a in S and x+b in S.
        
        # 2. Check triples (a, b, c) in S such that a+b = x.
        #    Then we need a+c in S and b+c in S.
        #    Since a+b=x, c is fixed? No, c is any element in S.
        #    Wait, the triple is (a,b,c). The sums are a+b, a+c, b+c.
        #    If a+b = x (new element), then for the triple to be invalid, 
        #    a+c must be in S_new (which is S_old U {x}) and b+c must be in S_new.
        #    Since a+c > x (if c > b? not necessarily), 
        #    Actually, if a+c = x, then c = x-a = b. But c must be distinct from a,b.
        #    So a+c cannot be x. Thus a+c must be in S_old.
        #    Similarly b+c must be in S_old.
        #    So we just need to check if there exists c in S_old such that a+c in S_old and b+c in S_old.
        
        # Let's implement this O(N^2) check.
        
        # Get elements of S
        elems = []
        temp = s
        while temp:
            if temp & 1:
                elems.append(len(elems) + 1)
            temp >>= 1
            
        L = len(elems)
        
        # Precompute pairs (a,b) with a+b in S
        # This is O(N^2)
        # And for each such pair, we need to check condition 1.
        
        # Condition 1:
        # For any a, b in S (a<b):
        #   if a+b in S:
        #      if (x+a in S) and (x+b in S): return False
        
        # Condition 2:
        # For any a, b in S (a<b) such that a+b == x:
        #   For any c in S (c != a, c != b):
        #      if (a+c in S) and (b+c in S): return False
        #   Note: Since a+b=x, and x not in S_old, this pair (a,b) was NOT a "sum-in-S" pair before.
        
        # Let's combine loops.
        for i in range(L):
            a = elems[i]
            for j in range(i+1, L):
                b = elems[j]
                sum_ab = a + b
                
                # Check if sum_ab is in S
                if (s >> (sum_ab - 1)) & 1:
                    # Condition 1 check
                    if ((s >> (x + a - 1)) & 1) and ((s >> (x + b - 1)) & 1):
                        return False
                else:
                    # sum_ab not in S
                    # Check if sum_ab == x (Condition 2)
                    if sum_ab == x:
                        # Check if there exists c in S such that a+c in S and b+c in S
                        # Iterate c in elems
                        for k in range(L):
                            if k == i or k == j:
                                continue
                            c = elems[k]
                            if ((s >> (a + c - 1)) & 1) and ((s >> (b + c - 1)) & 1):
                                return False
        return True

    # Search Loop
    # Try to add elements.
    # If no element can be added, try to swap.
    
    # To avoid getting stuck, we can use a "kick" move: remove a random element and try to add 2 new ones?
    # Or just random restarts.
    
    # Let's try greedy addition first.
    # Then random swaps.
    
    # Sort candidates by value? 
    # Smaller values are more likely to cause conflicts? Or less?
    # Upper half set {65..128} is "safe" because 2*65 > 128.
    # Adding small numbers breaks this property.
    
    # Let's iterate through candidates 1..64.
    
    # We will maintain a list of "blocked" candidates to avoid re-checking immediately?
    # No, just re-check.
    
    # To speed up, we can cache the result of can_add? 
    # It depends on s, so no.
    
    # Let's run a loop:
    # While time remains:
    #   1. Try to add any element from 1..64 (random order).
    #   2. If added, continue.
    #   3. If no element can be added, try a random swap:
    #      - Pick random element r in S to remove.
    #      - Pick random element c in 1..128 not in S to add.
    #      - If S - {r} + {c} is valid, accept.
    #      - (We can check validity of the new set directly, or use can_add on S-{r} for c)
    #      - Actually, checking validity of S' = S - {r} + {c} is O(N^2) with the can_add logic?
    #      - No, can_add checks if c can be added to S.
    #      - We need to check if c can be added to S - {r}.
    #      - This is effectively the same complexity.
        
    # Let's implement this.
    
    # Randomize candidate order to avoid bias
    candidates = list(range(1, 65))
    
    # We'll use a "stagnation" counter.
    stagnation = 0
    
    while time.time() < deadline:
        # Try to add
        added = False
        random.shuffle(candidates)
        for c in candidates:
            if can_add(current_s, c):
                current_s |= (1 << (c - 1))
                best_s = current_s
                best_size = len(bin(current_s).replace('0b',''))
                added = True
                stagnation = 0
                break
            else:
                # Check time frequently?
                if time.time() > deadline:
                    break
        
        if not added:
            stagnation += 1
            # Try swap
            # Pick random element to remove
            elems = []
            temp = current_s
            while temp:
                if temp & 1:
                    elems.append(len(elems) + 1)
                temp >>= 1
            
            if elems:
                r = random.choice(elems)
                # Try to add some c
                # We want to maintain or increase size.
                # Since we removed r, size is same if we add c.
                # But maybe we can add c and then add another?
                # For now, just swap.
                
                # To make swap useful, we should try to add a c that was previously blocked by r.
                # But finding that is hard.
                # Just random c.
                c = random.randint(1, 128)
                if c == r:
                    continue
                
                # Check if c can be added to S - {r}
                s_temp = current_s & ~(1 << (r - 1))
                if can_add(s_temp, c):
                    current_s = s_temp | (1 << (c - 1))
                    # Check if this new state allows further additions?
                    # We'll handle that in the next iteration's "Try to add" phase.
                    # But we should update best if size increased? 
                    # Size is same. But maybe it unlocks additions.
                    # We don't update best_s here because size didn't increase.
                    # However, if we get stuck, we might want to keep this state.
                    # Let's just keep current_s as the working state.
                    # If we find a better size later, we update best_s.
                    # If we don't, we return best_s.
                    # So we should only update best_s if size > best_size.
                    # Here size == best_size (or less if we removed and didn't add? No, we added).
                    # So size is same.
                    # But wait, if we remove r and add c, and c allows adding d, then size increases.
                    # That will be caught in the next "Try to add" loop.
                    # So we don't need to update best_s here unless we somehow increased size directly? 
                    # No, swap keeps size same.
                    # So we just continue.
                    # BUT, we should be careful: if we swap and the new set is valid, we keep it.
                    # If it's invalid, we revert? 
                    # can_add ensures validity of the addition to s_temp.
                    # So current_s is valid.
                    pass
            else:
                # Set is empty? Unlikely.
                break

        # If we've been stuck for a while, maybe restart?
        # Or just keep swapping.
        
    # Final check: ensure best_s is valid and return its elements
    # Extract elements from best_s
    result = []
    temp = best_s
    while temp:
        if temp & 1:
            result.append(len(result) + 1)
        temp >>= 1
        
    return result

