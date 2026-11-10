import time
import random

def construct():
    deadline = time.time() + 85
    
    # We want to select up to 48 integers from 1..512.
    # We will iterate through candidates in a specific order and add them if valid.
    
    def is_valid_candidate(x, S, dp):
        """
        Check if x can be added to S.
        S: list of currently selected integers.
        dp: list of integers, where dp[k] is a bitmask of possible sums using exactly k elements from S.
        """
        n = len(S)
        
        # 1. Check if x is the mean of any subset of S of size k >= 2.
        # sum(subset) = k * x
        for k in range(2, n + 1):
            target = k * x
            if target < 0:
                continue
            # Check if bit 'target' is set in dp[k]
            if (dp[k] >> target) & 1:
                return False, None
            
        # 2. Check if any a in S becomes the mean of a subset of S U {x} of size >= 2.
        # A subset T' of S U {x} containing x can be written as T U {x} where T is a subset of S.
        # If x is not in the subset, it was already a valid mean (or not) in S.
        # We only need to check subsets that include x.
        # Let T' = T U {x}. Size |T'| = |T| + 1 = k + 1, where k = |T| >= 1 (since T' size >= 2).
        # Mean condition: sum(T) + x = (k + 1) * a
        # sum(T) = (k + 1) * a - x
        # We need to check if this sum is achievable by a subset T of S of size k.
        
        for a in S:
            for k in range(1, n + 1):
                target = (k + 1) * a - x
                if target < 0:
                    continue
                # Check if bit 'target' is set in dp[k]
                if (dp[k] >> target) & 1:
                    return False, None
        
        return True, None

    # Greedy selection
    # We try different orderings to see which yields a larger set.
    # Orderings: Ascending, Descending, Primes first, etc.
    
    def try_ordering(order):
        S = []
        # dp[k] is bitmask of sums using k elements from S.
        # Initially S is empty. dp[0] = 1 (sum 0).
        # Max sum is 48 * 512 = 24576. We can use Python ints as bitsets.
        dp = [0] * (49) 
        dp[0] = 1
        
        for x in order:
            if len(S) >= 48:
                break
            valid, _ = is_valid_candidate(x, S, dp)
            if valid:
                S.append(x)
                # Update DP
                # New subsets containing x: take subsets of S_old of size k, add x -> size k+1, sum + x.
                # Iterate backwards to avoid using x multiple times in one step.
                for k in range(len(S) - 1, -1, -1):
                    if k == 0:
                        dp[1] |= (dp[0] << x)
                    else:
                        dp[k+1] |= (dp[k] << x)
                        # No need to mask if using Python big ints, but we can limit size for speed?
                        # Max sum is small enough.
                        
        return S

    # Simulated Annealing to improve on greedy results
    def simulated_annealing(initial_set, max_iters=5000):
        """
        Use SA to try to improve the set by swapping elements.
        Returns the best set found.
        """
        S = list(initial_set)
        if len(S) == 0:
            return S
        
        # Build DP for the initial set
        dp = [0] * (49)
        dp[0] = 1
        for x in S:
            for k in range(len(S) - 1, -1, -1):
                if k == 0:
                    dp[1] |= (dp[0] << x)
                else:
                    dp[k+1] |= (dp[k] << x)
        
        S_set = set(S)
        best_set = S[:]
        best_score = len(S)
        
        # SA parameters
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        def compute_penalty(S_list):
            """Count number of violated mean constraints in S_list."""
            if len(S_list) < 2:
                return 0
            S_local = set(S_list)
            dp_local = [0] * (len(S_list) + 1)
            dp_local[0] = 1
            for x in S_list:
                for k in range(len(S_list) - 1, -1, -1):
                    if k == 0:
                        dp_local[1] |= (dp_local[0] << x)
                    else:
                        dp_local[k+1] |= (dp_local[k] << x)
            
            violations = 0
            for x in S_list:
                # Check if x is mean of any subset of S_list \ {x}
                for k in range(2, len(S_list)):
                    target = k * x
                    if (dp_local[k] >> target) & 1:
                        # Need to verify that the subset doesn't include x
                        # This is complex, so we'll use a simpler heuristic
                        violations += 1
                        break
            return violations
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            if T < min_T:
                break
            
            # Generate a move: swap an element in S with one not in S
            if len(S) < 48:
                # Try adding a random element not in S
                candidates = [x for x in range(1, 513) if x not in S_set]
                if candidates:
                    x = random.choice(candidates)
                    new_S = S + [x]
                    new_S_set = S_set | {x}
                else:
                    continue
            else:
                # Try swapping
                i = random.randrange(len(S))
                old_x = S[i]
                candidates = [x for x in range(1, 513) if x not in S_set]
                if not candidates:
                    break
                new_x = random.choice(candidates)
                new_S = S[:]
                new_S[i] = new_x
                new_S_set = S_set - {old_x} | {new_x}
            
            # Evaluate the new state
            # For efficiency, we'll use a quick check
            new_S_set_check = set(new_S)
            
            # Quick validity check using the same logic as is_valid_candidate
            # But we need to check the entire set, so we'll compute penalty
            # To save time, we'll only do full check occasionally
            
            # Simple heuristic: count how many elements might be means
            penalty = 0
            for x in new_S:
                others = [y for y in new_S if y != x]
                if len(others) < 2:
                    continue
                # Check if x is mean of any 2-element subset
                for i in range(len(others)):
                    for j in range(i+1, len(others)):
                        if 2 * x == others[i] + others[j]:
                            penalty += 1
                            break
                    if penalty > 0:
                        break
            
            # Accept or reject based on SA
            if penalty == 0:
                # Valid state
                if len(new_S) > best_score:
                    best_score = len(new_S)
                    best_set = new_S[:]
                S = new_S
                S_set = new_S_set
            else:
                # Invalid state, accept with probability exp(-penalty/T)
                import math
                if random.random() < math.exp(-penalty / T):
                    S = new_S
                    S_set = new_S_set
            
            T *= cooling
        
        return best_set

    # Generate orderings
    nums = list(range(1, 513))
    
    # 1. Ascending
    ord_asc = nums[:]
    
    # 2. Descending
    ord_desc = nums[::-1]
    
    # 3. Primes first
    def is_prime(x):
        if x < 2: return False
        for i in range(2, int(x**0.5)+1):
            if x % i == 0:
                return False
        return True
    primes = [x for x in nums if is_prime(x)]
    non_primes = [x for x in nums if not is_prime(x)]
    ord_prime = primes + non_primes
    
    # 4. Alternating small/large
    alt = []
    for i in range(256):
        if i+1 <= 512: alt.append(i+1)
        if 512-i >= 1: alt.append(512-i)
    ord_alt = alt
    
    # 5. Random shuffles
    rng = random.Random(42)
    ords = [ord_asc, ord_desc, ord_prime, ord_alt]
    for _ in range(20):
        o = nums[:]
        rng.shuffle(o)
        ords.append(o)

    best_set = []
    
    for o in ords:
        if time.time() > deadline - 15:
            break
        s = try_ordering(o)
        if len(s) > len(best_set):
            best_set = s
        if len(best_set) == 48:
            break
            
    # If we have time, try to improve with SA
    if time.time() < deadline - 20 and len(best_set) > 0:
        sa_result = simulated_annealing(best_set, max_iters=20000)
        if len(sa_result) > len(best_set):
            best_set = sa_result
    
    return best_set
