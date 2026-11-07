import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Primes in 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verify primes are valid
    def verify(S):
        products = {1}
        for n in S:
            new_products = set()
            for p in products:
                np = p * n
                if np in products or np in new_products:
                    return False
                new_products.add(np)
            products = products | new_products
        return True

    if verify(primes):
        return primes

    # Fallback: Randomized search (just in case primes are invalid for some reason)
    # But they are valid.
    import random
    candidates = list(range(2, 41))
    
    def greedy_complete(S, deadline):
        S = set(S)
        added = True
        while added and time.time() < deadline:
            added = False
            candidates_remaining = [c for c in candidates if c not in S]
            random.shuffle(candidates_remaining)
            for x in candidates_remaining:
                if verify(S | {x}):
                    S.add(x)
                    added = True
                    break
        return S

    best_set = set()
    for _ in range(100):
        if time.time() > deadline: break
        S = greedy_complete(set(random.sample(candidates, 5)), deadline)
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) == 12:
                break
                
    return [int(x) for x in best_set]
