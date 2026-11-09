import time
import random
import sys

def construct():
    """
    Returns a list of up to 48 distinct positive integers <= 512 such that no element 
    is the exact arithmetic mean of any subset of at least two other elements.
    
    Strategy:
    1. Highly optimized validity check using integer bitsets.
    2. Fast greedy construction to get a large initial set.
    3. Local search with "repair" moves: try to add elements, and if blocked, 
       remove a conflicting element to make room, attempting to maintain or increase size.
    """
    deadline = time.time() + 50  # Safety margin for 60s limit
    N = 512
    target_size = 48
    
    # Precompute powers of 2? No, just use bit operations.
    
    def is_valid_global(S):
        """
        Check if set S is valid.
        Condition: No element a in S is the mean of any subset T of S\{a} with |T| >= 2.
        i.e., sum(T) != a * |T| for any T subset of S\{a}, |T| >= 2.
        """
        m = len(S)
        if m < 3:
            return True
        
        S_sorted = sorted(S)
        
        for idx, a in enumerate(S_sorted):
            # Build DP for S without a
            # dp[k] is a bitset of possible sums using exactly k elements
            max_k = m - 1
            dp = [0] * (max_k + 1)
            dp[0] = 1  # 0 elements, sum 0
            
            count = 0
            for v in S_sorted:
                if v == a and count == 0:
                    count = 1
                    continue
                
                # Update DP in reverse order of k
                # We only need to update up to the number of elements processed so far
                for k in range(min(count, max_k - 1), -1, -1):
                    if dp[k]:
                        dp[k + 1] |= dp[k] << v
                count += 1
            
            # Now check if any a * k is achievable
            for k in range(2, max_k + 1):
                target_sum = a * k
                if target_sum > 512 * k:
                    continue
                if target_sum > 24576:
                    break
                if (dp[k] >> target_sum) & 1:
                    return False
        
        return True

    def greedy_build(order):
        """
        Greedily add elements from 'order' to an empty set if valid.
        """
        current_set = []
        current_set_set = set()
        
        for c in order:
            if len(current_set) >= target_size:
                break
            if c in current_set_set:
                continue
            
            candidate_list = current_set + [c]
            if is_valid_global(candidate_list):
                current_set.append(c)
                current_set_set.add(c)
                
        return current_set

    best_set = []
    best_len = 0
    
    # 1. Try deterministic orders to find a good baseline quickly
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        list(range(1, N + 1, 2)), # Odds
        list(range(2, N + 1, 2)), # Evens
        list(range(1, N + 1, 3)),
        list(range(N, 0, -3)),
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        s = greedy_build(order)
        if len(s) > best_len:
            best_len = len(s)
            best_set = s[:]
            if best_len == target_size:
                return best_set

    # 2. Randomized Greedy Restarts
    rng = random.Random(42)
    for i in range(50):
        if time.time() > deadline:
            break
        p = list(range(1, N + 1))
        rng.shuffle(p)
        s = greedy_build(p)
        if len(s) > best_len:
            best_len = len(s)
            best_set = s[:]
            if best_len == target_size:
                return best_set

    # 3. Local Search / Repair Heuristic
    # If we are close to 48, try to improve by swapping.
    # Strategy: Pick a random element x not in best_set.
    # Check if adding x violates validity.
    # If it does, identify a "conflict" element a in best_set.
    # Remove a, add x. If size stays same, try to add another y.
    # This is complex to implement correctly with the global check.
    # Simpler Local Search: 
    # Take best_set. 
    # Try swapping a random element a in best_set with a random element b not in best_set.
    # If the new set is valid and we can then greedily add more, accept.
    
    if best_len < target_size:
        for _ in range(200):
            if time.time() > deadline:
                break
            
            current_set = best_set[:]
            current_set_set = set(current_set)
            
            # Try to add a random element
            candidates = [x for x in range(1, N + 1) if x not in current_set_set]
            if not candidates:
                break
                
            x = rng.choice(candidates)
            
            # Check if x can be added
            test_set = current_set + [x]
            if is_valid_global(test_set):
                # If it can be added, we should have done this in greedy? 
                # Maybe greedy didn't try this x at this step.
                current_set = test_set
                current_set_set.add(x)
                # Try to greedily fill the rest
                remaining = [y for y in range(1, N + 1) if y not in current_set_set]
                rng.shuffle(remaining)
                for y in remaining:
                    if len(current_set) >= target_size:
                        break
                    test_set2 = current_set + [y]
                    if is_valid_global(test_set2):
                        current_set.append(y)
                        current_set_set.add(y)
                        
                if len(current_set) > best_len:
                    best_len = len(current_set)
                    best_set = current_set[:]
                    if best_len == target_size:
                        return best_set
            else:
                # x conflicts. Try to swap.
                # Pick a random element a from current_set to remove.
                if current_set:
                    a_idx = rng.randrange(len(current_set))
                    a = current_set[a_idx]
                    
                    # New set: (current_set - {a}) + {x}
                    new_set = [v for v in current_set if v != a]
                    new_set.append(x)
                    new_set_set = set(new_set)
                    
                    if is_valid_global(new_set):
                        # Swap accepted. Now try to fill up.
                        remaining = [y for y in range(1, N + 1) if y not in new_set_set]
                        rng.shuffle(remaining)
                        for y in remaining:
                            if len(new_set) >= target_size:
                                break
                            test_set3 = new_set + [y]
                            if is_valid_global(test_set3):
                                new_set.append(y)
                                new_set_set.add(y)
                                
                        if len(new_set) > best_len:
                            best_len = len(new_set)
                            best_set = new_set[:]
                            if best_len == target_size:
                                return best_set

    return best_set

# EVOLVE-BLOCK-START
# The code above is the full implementation.
# EVOLVE-BLOCK-END
