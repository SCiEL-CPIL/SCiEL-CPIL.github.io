import random
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill
    
    N = 128
    # Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def is_bad_triple(a, b, c):
        """Check if triple (a,b,c) has all three pairwise GCDs equal."""
        return gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]
    
    def can_add(S_set, S_list, x):
        """Check if x can be added to S without forming an equal-gcd triple."""
        for i in range(len(S_list)):
            a = S_list[i]
            g_ax = gcd_table[a][x]
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                g_bx = gcd_table[b][x]
                g_ab = gcd_table[a][b]
                if g_ax == g_bx == g_ab:
                    return False
        return True
    
    def greedy_construct(order):
        """Greedy: add elements in given order if valid."""
        S = []
        S_set = set()
        for x in order:
            if can_add(S_set, S, x):
                S.append(x)
                S_set.add(x)
        return S
    
    def greedy_removal(full_set, order):
        """Start with full set, remove elements in order if they form a bad triple."""
        S = list(full_set)
        S_set = set(S)
        # Precompute which triples are bad
        for x in order:
            if x not in S_set:
                continue
            # Check if x is in any bad triple with other elements in S
            should_remove = False
            for i in range(len(S)):
                a = S[i]
                if a == x:
                    continue
                g_ax = gcd_table[a][x]
                for j in range(i+1, len(S)):
                    b = S[j]
                    if b == x or b == a:
                        continue
                    g_bx = gcd_table[b][x]
                    g_ab = gcd_table[a][b]
                    if g_ax == g_bx == g_ab:
                        should_remove = True
                        break
                if should_remove:
                    break
            if should_remove:
                S.remove(x)
                S_set.discard(x)
        return S
    
    def local_search_add(S):
        """Try to add any remaining elements to S."""
        S = list(S)
        S_set = set(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            candidates = [x for x in range(1, N+1) if x not in S_set]
            random.shuffle(candidates)
            for x in candidates:
                if can_add(S_set, S, x):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    break
        return S
    
    def local_search_swap(S):
        """Try 1-1 swaps to potentially enable more additions later."""
        S = list(S)
        S_set = set(S)
        # Try a few random swaps
        for _ in range(10):
            if time.time() >= deadline:
                break
            if not S:
                break
            candidates = [x for x in range(1, N+1) if x not in S_set]
            if not candidates:
                break
            u_idx = random.randint(0, len(S)-1)
            u = S[u_idx]
            v = random.choice(candidates)
            
            # Check if removing u and adding v is valid
            S_temp = S[:u_idx] + S[u_idx+1:]
            if can_add(set(S_temp), S_temp, v):
                S = S_temp + [v]
                S_set = set(S)
        return S
    
    best_size = 0
    best_set = []
    
    items_base = list(range(1, N+1))
    full_set = list(range(1, N+1))
    
    def get_order(strategy):
        items = list(items_base)
        if strategy == 0:
            random.shuffle(items)
        elif strategy == 1:
            # Primes first
            primes = [x for x in items if x > 1 and all(x % i != 0 for i in range(2, int(x**0.5)+1))]
            rest = [x for x in items if x not in primes]
            random.shuffle(rest)
            items = primes + rest
        elif strategy == 2:
            # High numbers first
            items.sort(reverse=True)
            random.shuffle(items)
        elif strategy == 3:
            # Numbers with few divisors first (more likely to be coprime to many)
            def num_divisors(n):
                count = 0
                for i in range(1, int(n**0.5)+1):
                    if n % i == 0:
                        count += 2
                        if i != n//i:
                            count -= 1
                return count
            items.sort(key=lambda x: num_divisors(x))
            random.shuffle(items)
        elif strategy == 4:
            # Random
            random.shuffle(items)
        return items
    
    # Phase 1: Greedy construction with multiple strategies
    while time.time() < deadline - 5:
        strategy = random.randint(0, 4)
        order = get_order(strategy)
        S = greedy_construct(order)
        S = local_search_add(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            if best_size >= N - 5:  # Close to maximum
                break
    
    # Phase 2: Greedy removal from full set
    while time.time() < deadline - 2:
        strategy = random.randint(0, 4)
        order = get_order(strategy)
        S = greedy_removal(full_set, order)
        S = local_search_add(S)
        S = local_search_swap(S)
        S = local_search_add(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Phase 3: Local search on best set
    if time.time() < deadline:
        S = local_search_add(best_set)
        S = local_search_swap(S)
        S = local_search_add(S)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    return best_set
