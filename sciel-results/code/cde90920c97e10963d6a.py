"""Stronger algorithm for dense increasing sequences with unique block products."""
import time
import random
from functools import lru_cache

# EVOLVE-BLOCK-START

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _all_block_products(seq):
    """Return the set of all nonempty consecutive block products."""
    prods = set()
    n = len(seq)
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            prods.add(p)
    return prods


def _validate(seq):
    """Check if seq is strictly increasing in [2,128] and all block products distinct."""
    if not seq:
        return True
    if seq[0] < 2 or seq[-1] > 128:
        return False
    for i in range(1, len(seq)):
        if seq[i] <= seq[i - 1]:
            return False
    prods = _all_block_products(seq)
    return len(prods) == len(seq) * (len(seq) + 1) // 2


def _compute_products(seq):
    """Compute all block products for a sequence. Returns set of products."""
    prods = set()
    n = len(seq)
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            prods.add(p)
    return prods


def _check_insertion(seq, pos, val, existing_prods):
    """
    Check if inserting val at position pos in seq keeps all block products distinct.
    Only checks new products that involve val.
    Returns (is_valid, new_prods_set) if valid, else (False, None).
    """
    n = len(seq)
    new_prods = set()
    
    # Products that start at pos (val * seq[pos] * ... * seq[pos+k])
    p = val
    for k in range(n - pos):
        new_prods.add(p)
        if k + 1 < n - pos:
            p *= seq[pos + k + 1]
    
    # Products that end at pos (seq[pos-k] * ... * seq[pos-1] * val)
    p = val
    for k in range(pos):
        new_prods.add(p)
        p *= seq[pos - k - 1]
    
    # Check against existing products
    if new_prods & existing_prods:
        return False, None
    
    return True, new_prods


def _greedy_with_order(order):
    """Greedy: try each x in given order, add if all new tail products are fresh."""
    values, products = [], set()
    for x in order:
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values


def _primes_first_greedy():
    """Start with all primes, then greedily insert composites."""
    primes = [x for x in range(2, 129) if _is_prime(x)]
    composites = [x for x in range(2, 129) if not _is_prime(x)]
    
    # Start with all primes
    seq = list(primes)
    prods = _compute_products(seq)
    
    # Try to insert each composite
    for c in composites:
        # Find valid insertion positions
        for pos in range(len(seq) + 1):
            if pos > 0 and seq[pos - 1] >= c:
                continue
            if pos < len(seq) and seq[pos] <= c:
                continue
            
            valid, new_prods = _check_insertion(seq, pos, c, prods)
            if valid:
                seq = seq[:pos] + [c] + seq[pos:]
                prods = set(prods) | new_prods
                break
    
    return seq


def _randomized_composite_insertion(seed, max_rounds=50):
    """
    Start with all primes, then randomly try to insert composites.
    Multiple rounds of random insertion attempts.
    """
    rng = random.Random(seed)
    primes = [x for x in range(2, 129) if _is_prime(x)]
    composites = [x for x in range(2, 129) if not _is_prime(x)]
    
    # Start with all primes
    seq = list(primes)
    prods = _compute_products(seq)
    
    for _round in range(max_rounds):
        used = set(seq)
        unused_composites = [c for c in composites if c not in used]
        if not unused_composites:
            break
        
        rng.shuffle(unused_composites)
        inserted = False
        
        for c in unused_composites:
            # Try random positions
            positions = list(range(len(seq) + 1))
            rng.shuffle(positions)
            
            for pos in positions:
                if pos > 0 and seq[pos - 1] >= c:
                    continue
                if pos < len(seq) and seq[pos] <= c:
                    continue
                
                valid, new_prods = _check_insertion(seq, pos, c, prods)
                if valid:
                    seq = seq[:pos] + [c] + seq[pos:]
                    prods = set(prods) | new_prods
                    inserted = True
                    break
            
            if inserted:
                break
        
        if not inserted:
            break
    
    return seq


def _swap_local_search(seq, deadline, max_iterations=200):
    """
    Local search using swaps: try swapping adjacent elements to find better configurations.
    Also try inserting unused elements.
    """
    seq = list(seq)
    prods = _compute_products(seq)
    best = list(seq)
    best_len = len(seq)
    
    all_nums = set(range(2, 129))
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        improved = False
        used = set(seq)
        
        # Try inserting unused numbers at valid positions
        unused = [x for x in all_nums if x not in used]
        random.shuffle(unused)
        
        for c in unused:
            if time.time() > deadline:
                break
            for pos in range(len(seq) + 1):
                if pos > 0 and seq[pos - 1] >= c:
                    continue
                if pos < len(seq) and seq[pos] <= c:
                    continue
                
                valid, new_prods = _check_insertion(seq, pos, c, prods)
                if valid:
                    new_seq = seq[:pos] + [c] + seq[pos:]
                    new_prods_full = set(prods) | new_prods
                    if len(new_seq) > best_len:
                        best = list(new_seq)
                        best_len = len(new_seq)
                    seq = new_seq
                    prods = new_prods_full
                    improved = True
                    break
            if improved:
                break
        
        # Try swapping adjacent elements to potentially unlock insertions
        if not improved:
            for i in range(len(seq) - 1):
                if time.time() > deadline:
                    break
                
                # Swap seq[i] and seq[i+1]
                seq[i], seq[i+1] = seq[i+1], seq[i]
                
                # Check if still valid (products might change)
                new_prods = _compute_products(seq)
                if len(new_prods) == len(seq) * (len(seq) + 1) // 2:
                    # Valid, try inserting
                    used = set(seq)
                    unused = [x for x in all_nums if x not in used]
                    random.shuffle(unused)
                    
                    for c in unused:
                        if time.time() > deadline:
                            break
                        for pos in range(len(seq) + 1):
                            if pos > 0 and seq[pos - 1] >= c:
                                continue
                            if pos < len(seq) and seq[pos] <= c:
                                continue
                            
                            valid, new_prods_insert = _check_insertion(seq, pos, c, new_prods)
                            if valid:
                                new_seq = seq[:pos] + [c] + seq[pos:]
                                new_prods_full = set(new_prods) | new_prods_insert
                                if len(new_seq) > best_len:
                                    best = list(new_seq)
                                    best_len = len(new_seq)
                                seq = new_seq
                                prods = new_prods_full
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                else:
                    # Invalid, swap back
                    seq[i], seq[i+1] = seq[i+1], seq[i]
        
        if not improved:
            break
    
    return best


def _greedy_forward():
    """Greedy: try each x in [2,128], add if all new tail products are fresh."""
    return _greedy_with_order(range(2, 129))


def _greedy_prime_first():
    """Greedy but prefer primes first, then composites."""
    primes = [x for x in range(2, 129) if _is_prime(x)]
    composites = [x for x in range(2, 129) if not _is_prime(x)]
    return _greedy_with_order(primes + composites)


def _greedy_composite_first():
    """Greedy but prefer composites first, then primes."""
    primes = [x for x in range(2, 129) if _is_prime(x)]
    composites = [x for x in range(2, 129) if not _is_prime(x)]
    return _greedy_with_order(composites + primes)


def _greedy_random(seed):
    """Greedy with random ordering."""
    rng = random.Random(seed)
    order = list(range(2, 129))
    rng.shuffle(order)
    return _greedy_with_order(order)


def construct():
    deadline = time.time() + 50.0  # safety margin before 60s limit
    candidates = []
    
    # Strategy 1: Primes-first greedy (strong baseline)
    g1 = _primes_first_greedy()
    candidates.append(g1)
    
    # Strategy 2: Multiple randomized composite insertions
    for seed in range(30):
        if time.time() > deadline - 15:
            break
        gr = _randomized_composite_insertion(seed, max_rounds=30)
        candidates.append(gr)
    
    # Strategy 3: Simple greedy variants
    g2 = _greedy_forward()
    candidates.append(g2)
    g3 = _greedy_prime_first()
    candidates.append(g3)
    
    # Strategy 4: Random greedy
    for seed in range(10):
        if time.time() > deadline - 10:
            break
        gr = _greedy_random(seed)
        candidates.append(gr)
    
    best_start = max(candidates, key=len)
    
    # Strategy 5: Swap-based local search from best candidate
    if time.time() < deadline - 5:
        ls_result = _swap_local_search(best_start, deadline, max_iterations=300)
        candidates.append(ls_result)
    
    # Pick the longest valid result
    best = max(candidates, key=len)
    
    # Final validation
    if not _validate(best):
        # Fallback to primes-first greedy
        best = _primes_first_greedy()
        if not _validate(best):
            best = _greedy_forward()
    
    return best

# EVOLVE-BLOCK-END
