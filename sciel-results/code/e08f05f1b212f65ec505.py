import time
import random
from math import isqrt

def _is_prime(a):
    if a < 2: return False
    if a < 4: return True
    if a % 2 == 0 or a % 3 == 0: return False
    d = 5
    while d * d <= a:
        if a % d == 0 or a % (d + 2) == 0: return False
        d += 6
    return True

def construct():
    candidates = list(range(2, 41))
    primes = [a for a in candidates if _is_prime(a)]
    composites = [a for a in candidates if not _is_prime(a)]
    
    best_size = 0
    best_set = []
    deadline = time.time() + 55 # 55s margin
    
    def all_subset_products(S):
        products = {1}
        for x in S:
            new_products = set()
            for p in products:
                new_products.add(p * x)
            products = products.union(new_products)
        return products
    
    def can_add(c, S, products):
        # products is a set of all subset products of S
        new_products = set()
        for p in products:
            new_products.add(p * c)
        # Check intersection
        if products.isdisjoint(new_products):
            return True, products.union(new_products)
        return False, None

    # Strategy 1: Greedy from primes
    for _ in range(500): # Adjust based on time
        if time.time() > deadline: break
        S = primes.copy()
        products = {1}
        # Add primes to products
        for p in S:
            new_p = set()
            for x in products:
                new_p.add(x * p)
            products = products.union(new_p)
        
        # Shuffle composites
        c_list = composites.copy()
        random.shuffle(c_list)
        
        for c in c_list:
            ok, new_products = can_add(c, S, products)
            if ok:
                S.append(c)
                products = new_products
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S.copy()
        
        # Local search on S
        # Try to swap or add more?
        # If we hit a local max, try swapping a composite with another composite?
        # Or remove a prime to add 2 composites?
        # Let's do a simple swap loop
        for _ in range(50):
            if time.time() > deadline: break
            # Pick random s in S, c not in S
            if not S: continue
            s_idx = random.randint(0, len(S)-1)
            s_val = S[s_idx]
            # Pick c from remaining
            remaining = [x for x in candidates if x not in S]
            if not remaining: break
            c_val = random.choice(remaining)
            
            # Test S' = S - {s_val} + {c_val}
            # We need to recompute products for S - {s_val}
            # This is expensive. 
            # Optimization: Only try swaps if we are stuck at < 18?
            # Or just rely on greedy restarts.
            S_new = [x for x in S if x != s_val]
            S_new.append(c_val)
            # Check if S_new is valid
            prods_new = all_subset_products(S_new)
            if len(prods_new) == 2 ** len(S_new):
                # Valid swap. Keep it if it allows adding more later? 
                # For now, just accept if size is same or larger.
                # Actually, size is same. We accept to explore.
                S = S_new
                products = prods_new
                # Try to add more
                c_list2 = [x for x in candidates if x not in S]
                random.shuffle(c_list2)
                for c in c_list2:
                    ok, new_products = can_add(c, S, products)
                    if ok:
                        S.append(c)
                        products = new_products
                        if len(S) > best_size:
                            best_size = len(S)
                            best_set = S.copy()
            else:
                # Invalid swap, revert (S was not modified in place yet, we used S_new)
                pass

    return best_set
