import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 90s. We use 85s to be safe.
    deadline = start_time + 85.0
    
    n = 192
    best_set = []
    best_size = 0
    candidates = list(range(1, n + 1))
    
    # Helper to check if x can be added to S
    def can_add(x, S):
        k = len(S)
        
        # 1. x does not divide sum of any 3 distinct others in S
        if k >= 3:
            # We can precompute sums or just iterate. Iteration is fine for k<50.
            for a, b, c in combinations(S, 3):
                if (a + b + c) % x == 0:
                    return False
        
        # 2. No a in S divides x + b + c for distinct b, c in S \ {a}
        if k >= 2:
            for i, a in enumerate(S):
                # We need pairs b, c from S excluding a
                # To avoid creating new lists, we can iterate indices
                # But creating a list of others is simpler and k is small
                others = [S[j] for j in range(k) if j != i]
                if len(others) < 2:
                    continue
                for b, c in combinations(others, 2):
                    if (x + b + c) % a == 0:
                        return False
        return True

    # Greedy construction given an order
    def greedy_construct(order):
        S = []
        S_set = set()
        for x in order:
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            if can_add(x, S):
                S.append(x)
                S_set.add(x)
                if len(S) > best_size:
                    # Update best size globally? No, just return S and let caller compare.
                    pass
        return S

    # Strategies for ordering
    def get_order(strategy, rng):
        if strategy == 'desc':
            return list(range(n, 0, -1))
        elif strategy == 'asc':
            return list(range(1, n + 1))
        elif strategy == 'random':
            order = candidates[:]
            rng.shuffle(order)
            return order
        elif strategy == 'large_first_random':
            large = [x for x in candidates if x > n // 2]
            small = [x for x in candidates if x <= n // 2]
            rng.shuffle(large)
            rng.shuffle(small)
            return large + small
        elif strategy == 'small_first_random':
            large = [x for x in candidates if x > n // 2]
            small = [x for x in candidates if x <= n // 2]
            rng.shuffle(large)
            rng.shuffle(small)
            return small + large
        elif strategy == 'modular_bias':
            # Bias towards numbers with fewer divisors or prime-like properties?
            # Or just random with a slight preference for numbers > 100
            order = candidates[:]
            # Sort by value descending, then shuffle within buckets?
            # Let's try sorting by number of divisors (fewer divisors first)
            # This might be expensive to compute every time, so we precompute or use a simple heuristic
            # For now, let's stick to value-based heuristics which are faster
            rng.shuffle(order)
            return order
        else:
            return candidates[:]

    rng = random.Random(42)
    
    # Run multiple attempts
    # We'll interleave different strategies
    strategies = ['desc', 'asc', 'random', 'large_first_random', 'small_first_random']
    
    attempts = 0
    # We can run many attempts if they are fast.
    # Each attempt is O(n * k^3) roughly. n=192, k~48.
    # 192 * 48^3 is approx 2 million ops. Very fast.
    # So we can run hundreds or thousands of attempts.
    
    while time.time() < deadline - 1.0:
        attempts += 1
        # Cycle through strategies, with random being more frequent
        if attempts % 10 == 1:
            strat = 'desc'
        elif attempts % 10 == 2:
            strat = 'asc'
        elif attempts % 10 == 3:
            strat = 'large_first_random'
        elif attempts % 10 == 4:
            strat = 'small_first_random'
        else:
            strat = 'random'
            
        order = get_order(strat, rng)
        S = greedy_construct(order)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
            # If we hit the theoretical max, we can stop
            if best_size == 48:
                break
        
        # If time is running out, do one last quick check with the best strategy found
        if time.time() > deadline - 2.0:
            break

    return best_set
