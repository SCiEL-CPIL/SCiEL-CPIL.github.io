import random
import time
import sys

def construct():
    """
    Constructs a set of up to 48 distinct positive integers <= 512.
    No element is the mean of any subset of at least two other elements.
    
    Strategy:
    1. Fast verification using integer bitset DP.
    2. Greedy initialization with multiple diverse orders (descending, ascending, random, weighted).
    3. Local search (swap and add) to expand the set.
    4. Repair mechanism if the set becomes invalid.
    """
    N = 512
    MAX_SIZE = 48
    
    # Precompute candidate list
    candidates = list(range(1, N + 1))
    
    # Fast DP verification
    def is_valid(S):
        """
        Returns True if S is valid, False otherwise.
        Uses integer bitset DP.
        """
        S_list = list(S)
        m = len(S_list)
        if m < 3:
            return True
        
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
                    return False
        
        return True

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
                        if rem != others[i] and rem != others[j]:
                            return a, [others[i], others[j], rem]
            
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
                    return a, None
        
        return None, None

    def greedy_construct(order):
        """
        Builds a set by adding elements in the given order if valid.
        """
        S = set()
        for x in order:
            if len(S) >= MAX_SIZE:
                break
            S_trial = S.copy()
            S_trial.add(x)
            if is_valid(S_trial):
                S = S_trial
        return S

    start_time = time.time()
    time_limit = 85.0 # Safety margin
    
    best_set = set()
    best_size = 0
    
    # Phase 1: Greedy Initialization with multiple orders
    # 1. Descending
    S_desc = greedy_construct(range(N, 0, -1))
    if len(S_desc) > best_size:
        best_set = S_desc
        best_size = len(S_desc)
        
    # 2. Ascending
    S_asc = greedy_construct(range(1, N + 1))
    if len(S_asc) > best_size:
        best_set = S_asc
        best_size = len(S_asc)
        
    # 3. Random Orders (a few)
    for _ in range(10):
        if time.time() - start_time > time_limit:
            break
        random.shuffle(candidates)
        S_rand = greedy_construct(candidates)
        if len(S_rand) > best_size:
            best_set = S_rand
            best_size = len(S_rand)
            
    # 4. Weighted random (favor larger numbers)
    weights = [i**2 for i in range(1, N+1)]
    for _ in range(10):
        if time.time() - start_time > time_limit:
            break
        order = random.choices(candidates, weights=weights, k=N)
        # Ensure unique
        order = list(dict.fromkeys(order))
        S_weighted = greedy_construct(order)
        if len(S_weighted) > best_size:
            best_set = S_weighted
            best_size = len(S_weighted)

    # Phase 2: Local Search / Repair / Expansion
    # Start with the best greedy set
    S = best_set.copy()
    
    for restart in range(50):
        if time.time() - start_time > time_limit:
            break
        
        # Check if current S is valid
        if not is_valid(S):
            # Repair S
            for step in range(1000):
                if time.time() - start_time > time_limit:
                    break
                if is_valid(S):
                    break
                a, T = get_violation(S)
                if a is None:
                    break
                r = random.random()
                if T is not None and len(T) > 0 and r < 0.5:
                    victim = random.choice(T)
                    S.remove(victim)
                elif r < 0.8:
                    S.remove(a)
                else:
                    others = [x for x in S if x != a]
                    if others:
                        S.remove(random.choice(others))
                    else:
                        S.remove(a)
        
        # Now S is valid (or we gave up on repair)
        if len(S) > best_size:
            best_set = S.copy()
            best_size = len(S)
            
        if len(S) == MAX_SIZE:
            if is_valid(S):
                return list(S)
        
        # Try to expand/swap to reach MAX_SIZE
        for step in range(200):
            if len(S) >= MAX_SIZE or time.time() - start_time > time_limit:
                break
            
            remaining = [x for x in candidates if x not in S]
            if not remaining:
                break
                
            # Pick a random candidate to add
            c = random.choice(remaining)
            S_trial = S.copy()
            S_trial.add(c)
            
            if is_valid(S_trial):
                S = S_trial
                if len(S) > best_size:
                    best_set = S.copy()
                    best_size = len(S)
                if len(S) == MAX_SIZE:
                    if is_valid(S):
                        return list(S)
            else:
                # Try to fix S_trial by removing an element from S (swap)
                a, T = get_violation(S_trial)
                if a is not None:
                    candidates_to_remove = []
                    if T:
                        candidates_to_remove.extend([x for x in T if x in S])
                    if a in S and a != c:
                        candidates_to_remove.append(a)
                    
                    if candidates_to_remove:
                        victim = random.choice(candidates_to_remove)
                        S_trial.remove(victim)
                        if is_valid(S_trial):
                            if random.random() < 0.3:
                                S = S_trial
                                if len(S) > best_size:
                                    best_set = S.copy()
                                    best_size = len(S)

    # Return the best valid set found
    if best_set:
        if is_valid(best_set):
            return list(best_set)
    
    # Fallback
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# EVOLVE-BLOCK-END
