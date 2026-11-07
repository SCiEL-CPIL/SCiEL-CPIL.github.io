"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

sys.setrecursionlimit(10000)

P = {'n': 40, 'maximum_size': 18}

CANDIDATES = list(range(2, P['n'] + 1))
N_CANDS = len(CANDIDATES)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    def is_prime(x):
        if x < 2:
            return False
        for d in range(2, int(x**0.5) + 1):
            if x % d == 0:
                return False
        return True

    def can_add(v, products):
        """Check if adding value v to products creates no collision."""
        for p in products:
            if (p * v) in products:
                return False
        return True

    def greedy_solve(order):
        S = set()
        products = {1}
        for idx in order:
            val = CANDIDATES[idx]
            if can_add(val, products):
                S.add(idx)
                products = products | {p * val for p in products}
        return S, products

    # Try multiple orderings for greedy
    best_size = 0
    best_set = set()
    
    primes = [i for i in range(N_CANDS) if is_prime(CANDIDATES[i])]
    composites = [i for i in range(N_CANDS) if not is_prime(CANDIDATES[i])]
    
    orderings = [primes + composites]
    
    def num_prime_factors(x):
        count = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                count += 1
                x //= d
            d += 1
        if x > 1:
            count += 1
        return count
    
    orderings.append(sorted(range(N_CANDS), key=lambda i: num_prime_factors(CANDIDATES[i])))
    
    for _ in range(10):
        orderings.append(sorted(range(N_CANDS), key=lambda i: random.random()))
    
    for order in orderings:
        S, products = greedy_solve(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Branch and bound with pruning
    def bb_search(order, current_set, products, pos):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
        
        if pos == len(order):
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
            return
        
        ub = len(current_set) + (len(order) - pos)
        if ub <= best_size:
            return
        
        idx = order[pos]
        val = CANDIDATES[idx]
        
        # Branch 1: include idx
        if can_add(val, products):
            new_set = current_set | {idx}
            new_products = products | {p * val for p in products}
            bb_search(order, new_set, new_products, pos + 1)
        
        # Branch 2: exclude idx
        bb_search(order, current_set, products, pos + 1)

    if best_size < P['maximum_size']:
        bb_search(primes + composites, set(), {1}, 0)
    
    # Local search refinement
    if best_size < P['maximum_size']:
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            
            S = best_set.copy()
            products = {1}
            for idx in S:
                val = CANDIDATES[idx]
                products = products | {p * val for p in products}
            
            # Try to add elements greedily
            added = True
            while added and time.time() < deadline - 5:
                added = False
                for idx in range(N_CANDS):
                    if idx not in S:
                        val = CANDIDATES[idx]
                        if can_add(val, products):
                            S.add(idx)
                            products = products | {p * val for p in products}
                            added = True
            
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
            
            # Try 1-for-2 swaps
            if len(S) < P['maximum_size']:
                for remove_idx in list(S):
                    if time.time() > deadline - 5:
                        break
                    S2 = S - {remove_idx}
                    products2 = {1}
                    for idx in S2:
                        val = CANDIDATES[idx]
                        products2 = products2 | {p * val for p in products2}
                    
                    for add1 in range(N_CANDS):
                        if add1 not in S2:
                            val1 = CANDIDATES[add1]
                            if can_add(val1, products2):
                                S3 = S2 | {add1}
                                products3 = products2 | {p * val1 for p in products2}
                                for add2 in range(N_CANDS):
                                    if add2 not in S3 and add2 != add1:
                                        val2 = CANDIDATES[add2]
                                        if can_add(val2, products3):
                                            S4 = S3 | {add2}
                                            if len(S4) > best_size:
                                                best_size = len(S4)
                                                best_set = S4.copy()

    return [CANDIDATES[i] for i in best_set]

# EVOLVE-BLOCK-END
