"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _greedy_construct():
    """Deterministic greedy: iterate 2..128, add if no product conflict."""
    values = []
    products = set()
    for x in range(2, 129):
        # Compute new products if x is appended
        # New products are x * (product of suffix of values)
        # Suffix products: 1, values[-1], values[-1]*values[-2], ...
        suffix_prod = 1
        new_products = [x] # singleton
        for a in reversed(values):
            suffix_prod *= a
            new_products.append(x * suffix_prod)
        
        # Check for conflicts
        conflict = False
        for p in new_products:
            if p in products:
                conflict = True
                break
        
        if not conflict:
            values.append(x)
            products.update(new_products)
            
    return values, products


def _check_insertion_fast(seq, products, x):
    """
    Check if x can be inserted into seq (which is strictly increasing) 
    maintaining the distinct block products property.
    Returns (new_seq, new_products) if valid, else (None, None).
    """
    if not seq:
        return [x], {x}
    
    # Find insertion position
    pos = 0
    for i in range(len(seq)):
        if x < seq[i]:
            pos = i
            break
        else:
            pos = i + 1
            
    # Check if x is already in seq (should be handled by caller, but safety check)
    if pos < len(seq) and seq[pos] == x:
        return None, None
    if pos > 0 and seq[pos-1] == x:
        return None, None

    left_seq = seq[:pos]
    right_seq = seq[pos:]
    
    # Compute suffix products of left_seq
    # left_prods[i] = prod(left_seq[i:])
    # We need all suffixes: prod(left_seq[j:]) for j in 0..pos-1
    # And the empty suffix (1)
    left_prods = [1]
    prod = 1
    for i in range(len(left_seq) - 1, -1, -1):
        prod *= left_seq[i]
        left_prods.append(prod)
        
    # Compute prefix products of right_seq
    # right_prods[i] = prod(right_seq[:i+1])
    # We need all prefixes: prod(right_seq[:j]) for j in 1..len(right_seq)
    # And the empty prefix (1)
    right_prods = [1]
    prod = 1
    for i in range(len(right_seq)):
        prod *= right_seq[i]
        right_prods.append(prod)
        
    # Generate new products: l * x * r for l in left_prods, r in right_prods
    # Check for conflicts with existing products and duplicates among new products
    new_products_set = set()
    for l in left_prods:
        base = l * x
        if base in products:
            # Collision with existing product (case where r=1 is included in right_prods)
            # Wait, if r=1, product is l*x. If l=1, product is x.
            # So checking base in products covers r=1 case.
            return None, None
        for r in right_prods:
            p = base * r
            if p in products:
                return None, None
            if p in new_products_set:
                return None, None
            new_products_set.add(p)
            
    # If we get here, it's valid
    new_seq = left_seq + [x] + right_seq
    new_products = products.union(new_products_set)
    return new_seq, new_products


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Get initial solution via greedy
    best_seq, best_products = _greedy_construct()
    best_len = len(best_seq)
    
    rng = random.Random(42)
    all_numbers = set(range(2, 129))
    
    while time.time() < deadline:
        seq = best_seq[:]
        products = best_products.copy()
        
        # Try to improve by inserting missing elements
        improved = True
        while improved and time.time() < deadline:
            improved = False
            available = all_numbers - set(seq)
            avail_list = list(available)
            rng.shuffle(avail_list)
            
            for x in avail_list:
                new_seq, new_products = _check_insertion_fast(seq, products, x)
                if new_seq is not None:
                    seq = new_seq
                    products = new_products
                    improved = True
                    if len(seq) > best_len:
                        best_len = len(seq)
                        best_seq = seq[:]
                        best_products = products
                    break # Restart loop with updated seq
        
        # If no improvement, perform a perturbation (kick) to escape local optimum
        if time.time() >= deadline:
            break
            
        # Strategy: Remove a random element, then try to re-fill
        # Or: Swap an existing element with a missing one?
        # Let's try removing 1-2 random elements and then re-running the greedy fill
        
        if len(seq) > 2:
            num_removals = rng.randint(1, 2)
            for _ in range(num_removals):
                if len(seq) > 1:
                    idx = rng.randint(0, len(seq) - 1)
                    removed_val = seq[idx]
                    seq.pop(idx)
                    # We need to rebuild products set? 
                    # Removing an element invalidates all products involving it.
                    # Recalculating products from scratch is O(N^2).
                    # N is ~40. 1600 ops. Acceptable.
                    # Let's just rebuild products to be safe and simple.
                    products = set()
                    n = len(seq)
                    for i in range(n):
                        prod = 1
                        for j in range(i, n):
                            prod *= seq[j]
                            products.add(prod)
            
            # Now try to fill again
            fill_improved = True
            while fill_improved and time.time() < deadline:
                fill_improved = False
                available = all_numbers - set(seq)
                avail_list = list(available)
                rng.shuffle(avail_list)
                
                for x in avail_list:
                    new_seq, new_products = _check_insertion_fast(seq, products, x)
                    if new_seq is not None:
                        seq = new_seq
                        products = new_products
                        fill_improved = True
                        if len(seq) > best_len:
                            best_len = len(seq)
                            best_seq = seq[:]
                            best_products = products
                        break
        
    return best_seq

# EVOLVE-BLOCK-END
