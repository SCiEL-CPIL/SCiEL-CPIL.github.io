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


def _insert_and_update(seq, pos, val, existing_prods):
    """Insert val at pos, return new seq and updated product set."""
    new_seq = seq[:pos] + [val] + seq[pos:]
    new_prods = _compute_products(new_seq)
    return new_seq, new_prods


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


def _local_search(seq, deadline, max_iterations=500):
    """
    Local search: try inserting unused elements at valid positions.
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
            # Find valid insertion positions
            for pos in range(len(seq) + 1):
                if pos > 0 and seq[pos - 1] >= c:
                    continue
                if pos < len(seq) and seq[pos] <= c:
                    continue
                
                # Check if insertion is valid
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


def _dfs_extend(start_seq, deadline, max_depth=8):
    """Bounded DFS to extend a sequence by trying insertions at various positions."""
    best = list(start_seq)
    best_len = len(best)
    best_prods = _compute_products(best)
    
    def _search(seq, prods, depth):
        nonlocal best, best_len, best_prods
        if time.time() > deadline:
            return
        if len(seq) > best_len:
            best = list(seq)
            best_len = len(seq)
            best_prods = set(prods)
        if depth >= max_depth:
            return
        
        # Try inserting each unused number at each valid position
        used = set(seq)
        candidates = [x for x in range(2, 129) if x not in used]
        # Sort candidates: prefer primes, then smaller numbers
        candidates.sort(key=lambda x: (0 if _is_prime(x) else 1, x))
        
        for c in candidates:
            if time.time() > deadline:
                return
            for i in range(len(seq) + 1):
                if i > 0 and seq[i - 1] >= c:
                    continue
                if i < len(seq) and seq[i] <= c:
                    continue
                
                valid, new_prods = _check_insertion(seq, i, c, prods)
                if valid:
                    new_seq = seq[:i] + [c] + seq[i:]
                    new_prods_full = set(prods) | new_prods
                    _search(new_seq, new_prods_full, depth + 1)
    
    _search(start_seq, best_prods, 0)
    return best


def construct():
    deadline = time.time() + 55.0  # safety margin before 60s limit
    candidates = []
    
    # Strategy 1: Simple greedy
    g1 = _greedy_forward()
    candidates.append(g1)
    
    # Strategy 2: Prime-first greedy
    g2 = _greedy_prime_first()
    candidates.append(g2)
    
    # Strategy 3: Composite-first greedy
    g3 = _greedy_composite_first()
    candidates.append(g3)
    
    # Strategy 4: Multiple random greedy
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        gr = _greedy_random(seed)
        candidates.append(gr)
    
    best_start = max(candidates, key=len)
    
    # Strategy 5: DFS extension from best greedy (shallow)
    if time.time() < deadline - 8:
        dfs_result = _dfs_extend(best_start, deadline, max_depth=4)
        candidates.append(dfs_result)
    
    # Strategy 6: Local search from best candidate
    if time.time() < deadline - 5:
        best_current = max(candidates, key=len)
        ls_result = _local_search(best_current, deadline, max_iterations=500)
        candidates.append(ls_result)
    
    # Pick the longest valid result
    best = max(candidates, key=len)
    
    # Final validation
    if not _validate(best):
        # Fallback to simple greedy
        best = _greedy_forward()
    
    return best

# EVOLVE-BLOCK-END
