import random
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 58.0
    
    N = 128
    # Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    def is_valid_add(S, x):
        """Check if x can be added to S without forming an equal-gcd triple."""
        for i in range(len(S)):
            a = S[i]
            g_ax = gcd_table[a][x]
            for j in range(i+1, len(S)):
                b = S[j]
                g_bx = gcd_table[b][x]
                g_ab = gcd_table[a][b]
                if g_ax == g_bx == g_ab:
                    return False
        return True

    def greedy_construct(order):
        S = []
        for x in order:
            if is_valid_add(S, x):
                S.append(x)
        return S

    def local_search(S):
        S = list(S)
        # Try to improve
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # 1. Try to add any element
            candidates = [x for x in range(1, N+1) if x not in S]
            random.shuffle(candidates)
            for x in candidates:
                if is_valid_add(S, x):
                    S.append(x)
                    improved = True
                    break
            if improved:
                continue
                
            # 2. Try 1-1 swaps
            if S and candidates:
                # Pick random u in S
                u_idx = random.randint(0, len(S)-1)
                u = S[u_idx]
                # Pick random v not in S
                v = random.choice(candidates)
                
                # Check validity of swap
                S_temp = S[:u_idx] + S[u_idx+1:]
                if is_valid_add(S_temp, v):
                    S = S_temp + [v]
                    # Don't mark improved=True to avoid infinite swap loops without progress?
                    # Actually, if we swap, we should try to add again.
                    # To prevent cycling, we can limit swaps or just rely on restarts.
                    # Let's mark improved=True to continue the loop and try adds.
                    # But if no add is possible, we might cycle.
                    # Let's only accept swap if it's "better" in some heuristic?
                    # For now, just do a few swaps and break if no add found for a while.
                    # Simpler: Just do random restarts. Local search with swaps is tricky without a cost function.
                    # Let's stick to Greedy + Restart.
                    pass

        return S

    best_size = 0
    best_set = []
    
    # Exclude 1 from consideration in greedy order? 
    # Let's try including it sometimes, but bias against it.
    # Actually, let's just run greedy on 2..128.
    
    items_base = list(range(2, N+1))
    
    def get_order(strategy):
        items = list(items_base)
        if strategy == 0:
            random.shuffle(items)
        elif strategy == 1:
            # Primes first
            primes = [x for x in items if all(x % i != 0 for i in range(2, int(x**0.5)+1))]
            rest = [x for x in items if x not in primes]
            random.shuffle(rest)
            items = primes + rest
        elif strategy == 2:
            # High numbers first?
            items.sort(reverse=True)
            random.shuffle(items)
        elif strategy == 3:
            # Multiples of large primes first
            # Just random for now
            random.shuffle(items)
        return items

    # Run many restarts
    while time.time() < deadline:
        strategy = random.randint(0, 3)
        order = get_order(strategy)
        S = greedy_construct(order)
        
        # Optional: Local Search (Add only)
        # Greedy already adds if possible in that order. 
        # But maybe a different order of the *remaining* candidates would allow more?
        # We can try to add remaining candidates in random order.
        remaining = [x for x in items_base if x not in S]
        random.shuffle(remaining)
        for x in remaining:
            if is_valid_add(S, x):
                S.append(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            if best_size == N - 1: # 1 excluded
                break

    # Final check: if 1 can be added to best_set, add it?
    # If 1 is added, it must not form a triple with any a,b in best_set.
    # i.e., for all a,b in best_set, gcd(a,b) != 1.
    # This is a very strong condition. Unlikely to be satisfied for large sets.
    # So best to exclude 1.
    
    return best_set
