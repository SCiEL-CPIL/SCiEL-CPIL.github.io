import math
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    N = 512
    MAX_A = 511
    # Safety margin: evaluator kills at 150s. We want to be well under.
    # Let's aim to finish in ~100s to be safe, or use a dynamic deadline.
    deadline = start_time + 110.0
    
    # Precompute prime factors for each number 1..511
    # factors[i] is a set of prime factors of i
    factors = [set() for _ in range(MAX_A + 1)]
    factors[1] = set()
    for i in range(2, MAX_A + 1):
        if not factors[i]:
            # i is prime
            for j in range(i, MAX_A + 1, i):
                factors[j].add(i)
    
    # Weights: w[a] = 1/(512-a). We want to maximize sum of w[a].
    # w[a] is strictly increasing with a.
    # So we prefer larger a.
    weights = [0.0] * (MAX_A + 1)
    for a in range(1, MAX_A + 1):
        weights[a] = 1.0 / (N - a)
    
    # Helper: check if a is coprime to all elements in set S
    def is_coprime_to_set(a, S):
        for b in S:
            if factors[a] & factors[b]:
                return False
        return True
    
    # Greedy initialization
    # Iterate from largest a to smallest.
    # If a is coprime to all already chosen, add it.
    # This maximizes the count of large numbers first.
    chosen_set = set()
    for a in range(MAX_A, 0, -1):
        if is_coprime_to_set(a, chosen_set):
            chosen_set.add(a)
    
    best_score = sum(weights[a] for a in chosen_set)
    best_set = set(chosen_set)
    
    # Local Search
    # We try to improve the set by swapping elements.
    # A swap (x, y) where x in S, y not in S is valid if:
    # 1. y is coprime to all z in S \ {x}
    # 2. weights[y] > weights[x] (i.e., y > x)
    # Since we iterate y from high to low, the first valid y for a given x
    # is the best y for that x.
    # However, we need to find the global best swap.
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        best_gain = 0.0
        best_swap = None
        
        # To make this efficient, we can precompute for each y not in S,
        # which elements in S block it.
        # But a simpler approach: iterate x in S, then y not in S (y > x).
        
        # Optimization:
        # For a fixed x, the best y is the largest y not in S such that
        # y is coprime to S \ {x}.
        # Since weights are monotonic, we just need to find the largest such y.
        
        # Let's iterate x in chosen_set.
        for x in list(chosen_set):
            # We want to find y > x such that y is coprime to all z in chosen_set \ {x}
            # Iterate y from MAX_A down to x+1
            # If we find one, it's the best y for this x.
            # Then we compare gain with other x's.
            
            # To speed up, we can break early if we find a y such that
            # weights[y] - weights[x] <= best_gain?
            # No, because a different x might allow a much larger y.
            
            # Let's just find the best y for this x.
            best_y_for_x = -1
            for y in range(MAX_A, x, -1):
                if y in chosen_set:
                    continue
                # Check if y is coprime to all z in chosen_set \ {x}
                valid = True
                for z in chosen_set:
                    if z == x:
                        continue
                    if factors[y] & factors[z]:
                        valid = False
                        break
                if valid:
                    best_y_for_x = y
                    break # Found the largest valid y for this x
            
            if best_y_for_x != -1:
                gain = weights[best_y_for_x] - weights[x]
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (x, best_y_for_x)
        
        if best_swap:
            x, y = best_swap
            chosen_set.remove(x)
            chosen_set.add(y)
            best_score += best_gain
            best_set = set(chosen_set)
            improved = True
    
    # One more pass with tighter time limit for fine-tuning?
    # The above loop stops when no single swap improves the score.
    # This is a local optimum.
    
    # Return the best set found
    return list(best_set)
