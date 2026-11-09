import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    n = 256
    
    def is_valid_add(S, x):
        # Check if adding x to S creates a 3-AP
        # S is a set
        for a in S:
            # a, x, b => b = 2x - a
            b = 2 * x - a
            if b in S and b != x:
                return False
            # a, b, x => b = 2a - x
            b = 2 * a - x
            if b in S and b != x:
                return False
        return True

    def get_score(S):
        return sum(1.0 / x for x in S)

    def greedy_construction(random_skip_prob=0.0):
        S = set()
        for x in range(1, n + 1):
            if random.random() < random_skip_prob:
                continue
            if is_valid_add(S, x):
                S.add(x)
        return S

    def local_search(S):
        S = set(S)
        current_score = get_score(S)
        improved = True
        while improved:
            improved = False
            # We want to swap i in S with j not in S
            # such that j < i (to increase 1/j - 1/i)
            # Iterate j from 1 to n
            for j in range(1, n + 1):
                if j in S:
                    continue
                # Try to find an i in S such that i > j and swap is valid
                # Sort S elements greater than j
                candidates = [i for i in S if i > j]
                # Try swapping with the largest i first? Or smallest?
                # Swapping with larger i gives bigger gain?
                # Gain = 1/j - 1/i. Larger i -> smaller 1/i -> bigger gain.
                # So try larger i first.
                for i in sorted(candidates, reverse=True):
                    # Check validity of (S - {i}) + {j}
                    new_S = set(S)
                    new_S.remove(i)
                    new_S.add(j)
                    if is_valid_add(new_S, j): # Note: is_valid_add checks against new_S which already has j? No, it checks if adding j to a set is valid.
                        # Wait, is_valid_add(S, x) checks if x can be added to S.
                        # Here we removed i and added j.
                        # We need to check if j is valid in S \ {i}.
                        # The function is_valid_add(S_minus_i, j) does exactly that.
                        if is_valid_add(S - {i}, j):
                            S = new_S
                            current_score = get_score(S)
                            improved = True
                            break
                if improved:
                    break
        return S, current_score

    best_set = set()
    best_score = 0.0
    
    # Run a few random restarts
    restarts = 50
    for r in range(restarts):
        if time.time() > deadline:
            break
        
        # Vary the greedy strategy slightly
        # r=0: deterministic
        # r>0: small random skip probability
        skip_prob = 0.0 if r == 0 else random.uniform(0.0, 0.05)
        
        S = greedy_construction(skip_prob)
        S, score = local_search(S)
        
        if score > best_score:
            best_score = score
            best_set = S

    # If we still have time, do more restarts with higher skip prob?
    # Or just return the best found.
    
    if not best_set:
        # Fallback to simple greedy if something went wrong
        S = greedy_construction(0.0)
        best_set = S
        best_score = get_score(S)

    return list(best_set)
