import time
import random

def _check_insert(seq, pos, val):
    """Check if inserting val at pos in seq keeps all block products distinct."""
    n = len(seq)
    if n == 0:
        return True
    
    # left[i] = product of seq[i:pos] for i in 0..pos
    left = [1] * (pos + 1)
    p = 1
    for i in range(pos - 1, -1, -1):
        p *= seq[i]
        left[i] = p
    # left[pos] = 1
    
    # right[j] = product of seq[pos:pos+j] for j in 0..n-pos
    right = [1] * (n - pos + 1)
    p = 1
    for j in range(1, n - pos + 1):
        p *= seq[pos + j - 1]
        right[j] = p
    
    seen = set()
    for i in range(pos + 1):
        for j in range(n - pos + 1):
            prod = left[i] * val * right[j]
            if prod in seen:
                return False
            seen.add(prod)
    return True

def _greedy():
    values = []
    products = set()
    for x in range(2, 129):
        tails = []
        prod = x
        tails.append(prod)
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values

def _is_valid(seq):
    prods = set()
    for i in range(len(seq)):
        p = 1
        for j in range(i, len(seq)):
            p *= seq[j]
            if p in prods:
                return False
            prods.add(p)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Baseline: greedy
    greedy_result = _greedy()
    
    # Start with all primes (always valid by unique factorization)
    primes = [x for x in range(2, 129) if all(x % i != 0 for i in range(2, int(x**0.5) + 1))]
    composites = [x for x in range(2, 129) if x not in set(primes)]
    
    best = list(primes)
    best_len = len(best)
    
    if len(greedy_result) > best_len:
        best = greedy_result
        best_len = best_len
    
    # Try to insert composites
    candidates = list(composites)
    rng = random.Random(42)
    rng.shuffle(candidates)
    
    stuck_count = 0
    max_stuck = 200
    
    for it in range(10000):
        if time.time() > deadline:
            break
        if not candidates:
            break
        
        val = candidates[0]
        n = len(best)
        inserted = False
        for pos in range(n + 1):
            if _check_insert(best, pos, val):
                best = best[:pos] + [val] + best[pos:]
                candidates.pop(0)
                inserted = True
                stuck_count = 0
                break
        
        if not inserted:
            stuck_count += 1
            if stuck_count > max_stuck:
                break
            candidates.append(candidates.pop(0))
    
    # Local search
    if time.time() < deadline - 5:
        in_set = set(best)
        candidates2 = [x for x in range(2, 129) if x not in in_set]
        rng2 = random.Random(123)
        
        for it in range(5000):
            if time.time() > deadline:
                break
            if not candidates2:
                break
            
            idx = rng2.randrange(len(candidates2))
            val = candidates2[idx]
            n = len(best)
            inserted = False
            for pos in range(n + 1):
                if _check_insert(best, pos, val):
                    best = best[:pos] + [val] + best[pos:]
                    candidates2.pop(idx)
                    inserted = True
                    break
            
            if not inserted and rng2.random() < 0.3:
                if len(best) > 1:
                    rem_idx = rng2.randrange(len(best))
                    rem_val = best[rem_idx]
                    temp = best[:rem_idx] + best[rem_idx+1:]
                    for pos in range(len(temp) + 1):
                        if pos == rem_idx:
                            continue
                        if _check_insert(temp, pos, rem_val):
                            best = temp[:pos] + [rem_val] + temp[pos:]
                            break
                    if candidates2:
                        idx2 = rng2.randrange(len(candidates2))
                        val2 = candidates2[idx2]
                        for pos in range(len(best) + 1):
                            if _check_insert(best, pos, val2):
                                best = best[:pos] + [val2] + best[pos:]
                                candidates2.pop(idx2)
                                break
    
    # Final verification
    if not _is_valid(best):
        return greedy_result
    
    return best

