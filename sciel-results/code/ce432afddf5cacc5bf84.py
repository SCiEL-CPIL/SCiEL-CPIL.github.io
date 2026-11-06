"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _greedy_construct():
    """Deterministic greedy: iterate 2..128, add if no product conflict."""
    values, products = [], set()
    for x in range(2, 129):
        tails = []
        prod = x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        tails.append(x)  # singleton
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values


def _get_all_products(seq):
    """Return set of all nonempty consecutive block products."""
    products = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            products.add(prod)
    return products


def _check_insertion(seq, x):
    """Check if x can be inserted into seq at its sorted position without product conflicts.
    Only checks new products that include x (those crossing the insertion point).
    Returns (pos, True) if valid, (None, False) otherwise."""
    # Find position to insert x
    pos = -1
    for i in range(len(seq)):
        if seq[i] > x:
            pos = i
            break
    if pos == -1:
        pos = len(seq)
    
    # Check x not already in seq
    if pos < len(seq) and seq[pos] == x:
        return None, False
    if pos > 0 and seq[pos-1] == x:
        return None, False
    
    # Get all existing products
    existing_products = _get_all_products(seq)
    
    # New products are all consecutive blocks that include x
    # These are: [x], [x, seq[pos]], [x, seq[pos], seq[pos+1]], ...
    # and: [seq[pos-1], x], [seq[pos-2], seq[pos-1], x], ...
    # and: [seq[pos-1], x, seq[pos]], etc.
    
    # Products ending at x (going left): x, seq[pos-1]*x, seq[pos-2]*seq[pos-1]*x, ...
    prod = x
    if prod in existing_products:
        return None, False
    for i in range(pos - 1, -1, -1):
        prod *= seq[i]
        if prod in existing_products:
            return None, False
    
    # Products starting at x (going right): x, x*seq[pos], x*seq[pos]*seq[pos+1], ...
    prod = x
    for j in range(pos, len(seq)):
        prod *= seq[j]
        if prod in existing_products:
            return None, False
    
    # Products that span across x: seq[i..pos-1] * x * seq[pos..j]
    # for i in [0..pos-1], j in [pos..n-1]
    # We need to check all such products
    left_prods = [x]  # left_prods[k] = product of seq[pos-1-k .. pos-1] * x, or just x if k=-1
    # Actually let's compute more carefully
    # A product spanning x is: (product of seq[i..pos-1]) * x * (product of seq[pos..j])
    # for i from max(0, pos-1) down to 0, and j from pos to n-1
    
    # Let's compute left suffix products ending at pos-1
    left_sufs = []
    prod = 1
    for i in range(pos - 1, -1, -1):
        prod *= seq[i]
        left_sufs.append(prod)
    # left_sufs[k] = product of seq[pos-1-k .. pos-1]
    # So left_sufs[0] = seq[pos-1], left_sufs[1] = seq[pos-2]*seq[pos-1], etc.
    
    # Right prefix products starting at pos
    right_prefs = []
    prod = 1
    for j in range(pos, len(seq)):
        prod *= seq[j]
        right_prefs.append(prod)
    # right_prefs[k] = product of seq[pos .. pos+k]
    
    # Now check all spanning products
    for ls in left_sufs:
        for rp in right_prefs:
            p = ls * x * rp
            if p in existing_products:
                return None, False
    
    return pos, True


def _try_insert_fast(seq, x):
    """Try to insert x into seq. Returns (new_seq, True) if successful."""
    pos, ok = _check_insertion(seq, x)
    if not ok:
        return None, False
    new_seq = seq[:pos] + [x] + seq[pos:]
    return new_seq, True


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Get initial solution via greedy
    best_seq = _greedy_construct()
    best_len = len(best_seq)
    
    rng = random.Random(42)
    
    all_numbers = set(range(2, 129))
    
    def _greedy_fill(seq, available, time_limit):
        """Greedily add elements from available to seq, trying each in order."""
        avail_list = list(available)
        # Try smaller numbers first (they're more likely to fit)
        avail_list.sort()
        for x in avail_list:
            if time.time() >= time_limit:
                break
            new_seq, ok = _try_insert_fast(seq, x)
            if ok:
                seq = new_seq
        return seq
    
    while time.time() < deadline:
        # Start from best_seq
        seq = best_seq[:]
        current_len = len(seq)
        
        # Strategy 1: Random removal then greedy refill
        if current_len > 3:
            num_removals = rng.randint(1, 3)
            for _ in range(num_removals):
                if len(seq) > 2:
                    idx = rng.randint(0, len(seq) - 1)
                    seq.pop(idx)
        
        # Try to add back removed or new elements
        available = all_numbers - set(seq)
        seq = _greedy_fill(seq, available, deadline)
        
        # Strategy 2: Single swap - remove one, try to replace with better
        if time.time() < deadline:
            if len(seq) > 2:
                idx = rng.randint(0, len(seq) - 1)
                removed = seq.pop(idx)
                available = all_numbers - set(seq)
                avail_list = list(available)
                rng.shuffle(avail_list)
                added = False
                for x in avail_list[:30]:
                    new_seq, ok = _try_insert_fast(seq, x)
                    if ok:
                        seq = new_seq
                        added = True
                        break
                if not added:
                    # Put back the removed element
                    pos = 0
                    for i in range(len(seq)):
                        if seq[i] > removed:
                            pos = i
                            break
                    seq.insert(pos, removed)
        
        # Strategy 3: Remove two adjacent elements and refill
        if time.time() < deadline and len(seq) > 4:
            idx = rng.randint(0, len(seq) - 2)
            r1 = seq.pop(idx)
            r2 = seq.pop(idx)
            available = all_numbers - set(seq)
            seq = _greedy_fill(seq, available, deadline)
        
        if len(seq) > best_len:
            best_seq = seq[:]
            best_len = len(seq)
    
    return best_seq

# EVOLVE-BLOCK-END
