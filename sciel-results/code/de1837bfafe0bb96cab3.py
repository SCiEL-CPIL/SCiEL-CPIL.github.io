# EVOLVE-BLOCK-START
import time
import random

P = {'n': 40, 'maximum_size': 18}

CANDIDATES = list(range(2, P['n'] + 1))
N_CANDS = len(CANDIDATES)

def get_products(indices):
    """Generate set of all subset products for the given indices."""
    prods = {1}
    for idx in indices:
        val = CANDIDATES[idx]
        # Create new products by multiplying existing ones by val
        # We must use a list comprehension or similar to avoid modifying while iterating
        # Actually, we can just do:
        new_prods = {p * val for p in prods}
        prods.update(new_prods)
    return prods

def is_valid(indices):
    """Check if all subset products are distinct."""
    prods = {1}
    for idx in indices:
        val = CANDIDATES[idx]
        new_prods = {p * val for p in prods}
        if new_prods & prods:
            return False
        prods.update(new_prods)
    return True

def greedy_fill(S, prods, deadline):
    """Greedily add valid elements."""
    added = True
    while added and time.time() < deadline:
        added = False
        # Sort candidates by value or random? Random is better for diversity.
        candidates = [c for c in range(N_CANDS) if c not in S]
        random.shuffle(candidates)
        for c in candidates:
            val = CANDIDATES[c]
            new_prods = {p * val for p in prods}
            if not (new_prods & prods):
                S.add(c)
                prods.update(new_prods)
                added = True
                break
    return S, prods

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_set = set()
    best_size = 0
    best_prods = {1}

    # Initial set: Primes
    primes = [i for i, x in enumerate(CANDIDATES) if all(x % d for d in range(2, int(x**0.5)+1))]
    
    for restart in range(500):
        if time.time() > deadline:
            break
            
        # Vary starting set
        if restart % 5 == 0:
            S = set(primes)
        elif restart % 5 == 1:
            S = set(random.sample(primes, max(1, len(primes) - random.randint(0, 2))))
        else:
            # Random subset of all candidates, biased towards small numbers?
            # Or just random
            S = set(random.sample(range(N_CANDS), random.randint(5, 10)))
            
        # Ensure validity of start
        if not is_valid(S):
            continue
            
        prods = get_products(S)
        
        # Greedy Fill
        S, prods = greedy_fill(S, prods, deadline)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            best_prods = set(prods)
            if best_size == P['maximum_size']:
                break

        # Local Search: 1-Swap
        if len(S) < P['maximum_size'] and time.time() < deadline - 2:
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                # Try removing one and adding one
                for to_remove in list(S):
                    if time.time() > deadline: break
                    S.remove(to_remove)
                    # Recalculate prods for S \ {to_remove}
                    # This is O(2^k). k is ~16-17. 2^17 = 131k. Fast.
                    S_list = list(S)
                    prods_temp = get_products(S_list)
                    
                    # Try adding any candidate
                    candidates = [c for c in range(N_CANDS) if c not in S]
                    random.shuffle(candidates)
                    for to_add in candidates:
                        val = CANDIDATES[to_add]
                        new_prods = {p * val for p in prods_temp}
                        if not (new_prods & prods_temp):
                            S.add(to_add)
                            prods_temp.update(new_prods)
                            # Greedy fill again
                            S, prods_temp = greedy_fill(S, prods_temp, deadline)
                            
                            if len(S) > best_size:
                                best_size = len(S)
                                best_set = set(S)
                                best_prods = set(prods_temp)
                                if best_size == P['maximum_size']:
                                    return [CANDIDATES[i] for i in best_set]
                            
                            improved = True
                            break
                    # If we didn't break, we need to restore S to original state for next to_remove
                    # Wait, the logic above modifies S.
                    # If we added something, S is changed.
                    # If we didn't add, S is just missing to_remove.
                    # We should probably restart the inner loop or continue.
                    # Actually, if we found an improvement, we break out of to_add loop.
                    # But we still need to handle the case where we didn't find an improvement.
                    # If no improvement, we should restore to_remove?
                    # No, the standard hill climbing is: try move, if better, keep. If not, revert.
                    # Here, if we add to_add, we keep it.
                    # If we don't find any to_add, we should put to_remove back.
                    if to_remove not in S:
                        S.add(to_remove)
                        prods_temp = get_products(list(S))

    return [CANDIDATES[i] for i in best_set]
# EVOLVE-BLOCK-END
