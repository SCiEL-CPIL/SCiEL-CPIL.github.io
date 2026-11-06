"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _all_block_products(seq):
    """Return set of all nonempty consecutive block products."""
    prods = set()
    n = len(seq)
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            prods.add(p)
    return prods

def _can_append(seq, x, products):
    """Check if x can be appended to seq without creating duplicate block products."""
    new_prods = set()
    p = x
    new_prods.add(p)
    for a in reversed(seq):
        p *= a
        new_prods.add(p)
    return not products.intersection(new_prods), new_prods

def _can_insert(seq, pos, x, products):
    """Check if x can be inserted at position pos in seq without creating duplicate block products.
    Returns (ok, new_prods, removed_prods)."""
    new_prods = set()
    # Blocks starting at x (position pos) going right
    p = x
    new_prods.add(p)
    for j in range(pos, len(seq)):
        p *= seq[j]
        new_prods.add(p)
    # Blocks ending at x (position pos) going left
    p = x
    for i in range(pos - 1, -1, -1):
        p *= seq[i]
        new_prods.add(p)
    
    # Compute old blocks that cross position pos
    # Old blocks [i..j] where i < pos <= j (inserting before index pos)
    # i.e., start index i in [0, pos-1], end index j in [pos, len(seq)-1]
    removed_prods = set()
    for i in range(pos):
        p = 1
        for j in range(pos, len(seq)):
            p *= seq[j]
            removed_prods.add(p)
    
    # Check for duplicates in new_prods (shouldn't happen for distinct integers > 1, but safe)
    # And check intersection with remaining old products
    remaining = products - removed_prods
    if new_prods & remaining:
        return False, None, None
    
    return True, new_prods, removed_prods

def _try_insertion_pass(seq, candidates, products, time_limit):
    """Try to insert excluded candidates into the sequence."""
    start = time.time()
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        random.shuffle(candidates)
        for x in candidates[:]:
            if x in seq:
                continue
            # Try all insertion positions
            for pos in range(len(seq) + 1):
                ok, new_prods = _can_insert(seq, pos, x, products)
                if ok:
                    new_seq = seq[:pos] + [x] + seq[pos:]
                    new_products = products - set()  # we need to recompute removed
                    # Recompute removed products for this insertion
                    removed_prods = set()
                    for i in range(pos):
                        p = 1
                        for j in range(pos, len(seq)):
                            p *= seq[j]
                            removed_prods.add(p)
                    new_products = (products - removed_prods) | new_prods
                    seq = new_seq
                    products = new_products
                    candidates.remove(x)
                    improved = True
                    break
    return seq, products, candidates

def _try_swap_pass(seq, products, time_limit):
    """Try swapping adjacent elements to enable further insertions."""
    start = time.time()
    n = len(seq)
    for _ in range(100):  # bounded
        if time.time() - start > time_limit:
            break
        for i in range(n - 1):
            # Try swapping seq[i] and seq[i+1]
            new_seq = seq[:]
            new_seq[i], new_seq[i+1] = new_seq[i+1], new_seq[i]
            new_products = _all_block_products(new_seq)
            if len(new_products) == sum(len(new_seq) * (len(new_seq) - i) for i in range(len(new_seq))):
                # Valid
                seq = new_seq
                products = new_products
                break
    return seq, products

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Phase 1: Greedy construction (fast)
    values, products = [], set()
    for x in range(2, 129):
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    
    if time.time() > deadline:
        return values
    
    # Phase 2: Randomized insertion of excluded numbers
    excluded = [x for x in range(2, 129) if x not in values]
    random.seed(42)
    
    seq = list(values)
    products_set = set(products)
    
    # Multiple insertion passes with different random orders
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        trial_seq = list(seq)
        trial_products = set(products_set)
        trial_excluded = list(excluded)
        random.shuffle(trial_excluded)
        
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            for x in trial_excluded[:]:
                for pos in range(len(trial_seq) + 1):
                    # Enforce strictly increasing order
                    if pos > 0 and trial_seq[pos-1] >= x:
                        continue
                    if pos < len(trial_seq) and trial_seq[pos] <= x:
                        continue
                        
                    ok, new_prods, removed_prods = _can_insert(trial_seq, pos, x, trial_products)
                    if ok:
                        new_trial_seq = trial_seq[:pos] + [x] + trial_seq[pos:]
                        new_trial_products = (trial_products - removed_prods) | new_prods
                        
                        trial_seq = new_trial_seq
                        trial_products = new_trial_products
                        trial_excluded.remove(x)
                        improved = True
                        break
                if time.time() > deadline - 5:
                    break
        
        if len(trial_seq) > len(seq):
            seq = trial_seq
            products_set = trial_products
            excluded = [x for x in range(2, 129) if x not in seq]
    
    if time.time() > deadline:
        return seq
    
    # Phase 3: Bounded local search - try to swap adjacent pairs to enable more insertions
    for trial in range(20):
        if time.time() > deadline - 3:
            break
        trial_seq = list(seq)
        n = len(trial_seq)
        
        # Try a few random adjacent swaps
        for _ in range(10):
            if time.time() > deadline - 3:
                break
            if n < 2:
                break
            i = random.randint(0, n - 2)
            new_seq = trial_seq[:]
            new_seq[i], new_seq[i+1] = new_seq[i+1], new_seq[i]
            
            new_products = _all_block_products(new_seq)
            expected_count = sum(len(new_seq) * (len(new_seq) - i) for i in range(len(new_seq)))
            if len(new_products) == expected_count:
                # Valid swap - try to insert more
                trial_seq = new_seq
                trial_products = new_products
                
                # Try insertions
                trial_excluded = [x for x in range(2, 129) if x not in trial_seq]
                random.shuffle(trial_excluded)
                for x in trial_excluded:
                    if time.time() > deadline - 3:
                        break
                    for pos in range(len(trial_seq) + 1):
                        # Enforce strictly increasing order
                        if pos > 0 and trial_seq[pos-1] >= x:
                            continue
                        if pos < len(trial_seq) and trial_seq[pos] <= x:
                            continue

                        ok, new_prods, removed_prods = _can_insert(trial_seq, pos, x, trial_products)
                        if ok:
                            trial_seq = trial_seq[:pos] + [x] + trial_seq[pos:]
                            trial_products = (trial_products - removed_prods) | new_prods
                            break
        
        if len(trial_seq) > len(seq):
            seq = trial_seq
            products_set = trial_products
    
    return seq
# EVOLVE-BLOCK-END
