import random
import time
import sys

def construct():
    """
    Constructs a set of up to 48 distinct positive integers <= 512.
    No element is the mean of any subset of at least two other elements.
    Uses Local Search with Repair to maximize size.
    """
    N = 512
    MAX_SIZE = 48
    
    # Fast DP verification
    def is_valid_and_count(S):
        """
        Returns (is_valid, violation_count).
        Uses integer bitset DP.
        """
        S_list = list(S)
        m = len(S_list)
        if m < 3:
            return True, 0
        
        total_violations = 0
        
        for a in S_list:
            others = [x for x in S_list if x != a]
            m_others = len(others)
            if m_others < 2:
                continue
            
            # DP for others
            # dp[k] is a bitset where bit s is 1 if sum s is possible with k elements
            max_sum = sum(others)
            dp = [0] * (m_others + 1)
            dp[0] = 1
            
            for x in others:
                for k in range(m_others, 0, -1):
                    dp[k] |= (dp[k-1] << x)
            
            # Check violations for a
            for k in range(2, m_others + 1):
                target = k * a
                if target > max_sum:
                    break
                if (dp[k] >> target) & 1:
                    total_violations += 1
                    # We could break here if we just want existence, 
                    # but counting helps gradient? 
                    # For speed, let's just break and return 1 violation.
                    # Actually, just returning 1 is enough for "Is it valid?".
                    # Let's optimize: just return False, 1 if found.
                    return False, 1
        
        return True, 0

    def get_violation(S):
        """
        Returns (a, T) if a is the mean of subset T of S\{a}, else (None, None).
        T is a list of elements.
        """
        S_list = list(S)
        for a in S_list:
            others = [x for x in S_list if x != a]
            m_others = len(others)
            if m_others < 2:
                continue
            
            max_sum = sum(others)
            # We need to reconstruct the subset if we find a violation.
            # This requires storing the path or backtracking.
            # For simplicity in the repair step, we can just return 'a' and let the repair remove 'a' 
            # or a random 'other'.
            # To be more effective, we should find the subset T.
            
            # DP with parent tracking? 
            # Or just check k=2,3,4 explicitly with sets?
            # Given N=512 and Size=48, k=2,3 are most common.
            
            # Check k=2
            for i in range(len(others)):
                for j in range(i+1, len(others)):
                    if others[i] + others[j] == 2 * a:
                        return a, [others[i], others[j]]
            
            # Check k=3
            for i in range(len(others)):
                for j in range(i+1, len(others)):
                    rem = 3 * a - others[i] - others[j]
                    if rem > others[j] and rem in others:
                        # Ensure distinct indices? 
                        # others is a list of unique values.
                        # If rem == others[i] or rem == others[j], it's not a distinct subset of size 3 
                        # unless we have duplicates (we don't).
                        # But we need 3 DISTINCT elements.
                        # If rem is in others, and rem != others[i] and rem != others[j], 
                        # we need to ensure we pick the correct index.
                        # Since values are unique, we can just check membership.
                        if rem != others[i] and rem != others[j]:
                            return a, [others[i], others[j], rem]
            
            # Check k=4
            # O(m^3) might be too slow if called often. 
            # But repair is called less often than validity check.
            # Let's stick to k=2,3 for repair guidance. 
            # If k>=4 violation exists, we'll just remove 'a' or random other.
            
            # Fallback: Use DP to check existence for k>=4
            dp = [0] * (m_others + 1)
            dp[0] = 1
            for x in others:
                for k in range(m_others, 0, -1):
                    dp[k] |= (dp[k-1] << x)
            
            for k in range(4, m_others + 1):
                target = k * a
                if target > max_sum:
                    break
                if (dp[k] >> target) & 1:
                    # Violation exists, but we don't have the subset T.
                    # Return a and a random sample from others?
                    # Or just return a and None, meaning "remove a".
                    return a, None
        
        return None, None

    start_time = time.time()
    time_limit = 85.0 # Safety margin
    
    candidates = list(range(1, N + 1))
    
    best_set = set()
    best_size = 0
    
    # Strategy:
    # 1. Try to build a valid set of size 48.
    # 2. Start with a random set of size 48.
    # 3. Repair it until valid.
    # 4. Try to add more elements (if size < 48).
    
    for restart in range(50):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize S with 48 random candidates
        S = set(random.sample(candidates, MAX_SIZE))
        
        # Phase 1: Repair S to make it valid
        # Max 1000 repair steps
        for step in range(1000):
            if time.time() - start_time > time_limit:
                break
            
            is_valid, _ = is_valid_and_count(S)
            if is_valid:
                break
            
            # Find a violation
            a, T = get_violation(S)
            
            if a is None:
                # Should not happen if is_valid is False, but safety
                break
            
            # Repair Strategy:
            # 1. Remove 'a'
            # 2. Remove a random element from T (if T is not None)
            # 3. Remove a random element from S\{a}
            
            r = random.random()
            if T is not None and len(T) > 0 and r < 0.5:
                # Remove an element from the violating subset
                victim = random.choice(T)
                S.remove(victim)
            elif r < 0.8:
                # Remove the mean element
                S.remove(a)
            else:
                # Remove a random other element
                others = [x for x in S if x != a]
                if others:
                    S.remove(random.choice(others))
                else:
                    S.remove(a)
        
        # If S is valid, try to maximize size
        if len(S) > best_size:
            best_set = S.copy()
            best_size = len(S)
        
        if len(S) == MAX_SIZE:
            # Verify one last time before returning
            is_valid, _ = is_valid_and_count(S)
            if is_valid:
                return list(S)
        
        # If not full, try to add elements
        if len(S) < MAX_SIZE:
            # Get remaining candidates
            remaining = [x for x in candidates if x not in S]
            random.shuffle(remaining)
            
            for c in remaining:
                if len(S) >= MAX_SIZE:
                    break
                if time.time() - start_time > time_limit:
                    break
                
                S_trial = S.copy()
                S_trial.add(c)
                
                # Check if adding c makes it invalid
                # If invalid, we can try to "fix" by removing an element from the new violation
                # But for simplicity, if invalid, skip c.
                # To be more aggressive, we could swap c with an element in S.
                
                is_valid, _ = is_valid_and_count(S_trial)
                if is_valid:
                    S = S_trial
                    if len(S) > best_size:
                        best_set = S.copy()
                        best_size = len(S)
        
        # Local Search on Size:
        # Try to swap elements to increase size?
        # If we have a valid set of size 45, can we find a valid set of size 46?
        # We can try to add a random element, and if it fails, remove a "bad" element.
        
        for _ in range(100):
            if len(S) >= MAX_SIZE or time.time() - start_time > time_limit:
                break
            
            remaining = [x for x in candidates if x not in S]
            if not remaining:
                break
                
            c = random.choice(remaining)
            S_trial = S.copy()
            S_trial.add(c)
            
            is_valid, _ = is_valid_and_count(S_trial)
            
            if is_valid:
                S = S_trial
                if len(S) > best_size:
                    best_set = S.copy()
                    best_size = len(S)
            else:
                # Try to repair S_trial
                # Find violation in S_trial
                a, T = get_violation(S_trial)
                if a is not None:
                    # Remove a or element from T from S_trial
                    # Then S_trial is smaller than MAX_SIZE, but valid?
                    # We want to increase size. 
                    # This "add then remove" effectively swaps an element.
                    # If we remove an element from S (not c), size stays same.
                    # If we remove c, size decreases.
                    # So this is a "Swap" move.
                    
                    if T is not None and len(T) > 0:
                        # Remove an element from T (which is in S)
                        # This keeps c in the set, and removes a conflict source.
                        # Size remains len(S) + 1 - 1 = len(S).
                        # Wait, we added c, so size was len(S)+1.
                        # Removing one element from T (subset of S) makes size len(S).
                        # So it's a swap.
                        victim = random.choice(T)
                        S_trial.remove(victim)
                        
                        # Check if S_trial is valid
                        is_valid2, _ = is_valid_and_count(S_trial)
                        if is_valid2:
                            # Accept swap if it doesn't decrease size? 
                            # It keeps size same. 
                            # But it might allow future additions.
                            # Let's accept it with some probability to escape local minima.
                            if random.random() < 0.5:
                                S = S_trial
                    else:
                        # Remove a element from S (not c)
                        # This is also a swap.
                        others_in_S = [x for x in S if x != a] # a is in S or c?
                        # a is the mean. a could be c or in S.
                        # If a == c, removing c reverts to S.
                        # If a in S, removing a swaps c for a.
                        if a != c and a in S:
                            S_trial.remove(a)
                            is_valid2, _ = is_valid_and_count(S_trial)
                            if is_valid2 and random.random() < 0.5:
                                S = S_trial

    # Return the best valid set found
    if best_set:
        # Final verification
        is_valid, _ = is_valid_and_count(best_set)
        if is_valid:
            return list(best_set)
    
    # Fallback
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# EVOLVE-BLOCK-END
