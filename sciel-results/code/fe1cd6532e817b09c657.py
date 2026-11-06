"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

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


def _is_valid(seq):
    """Check if seq is strictly increasing in [2,128] with distinct block products."""
    if not seq:
        return True
    for i in range(1, len(seq)):
        if seq[i] <= seq[i-1]:
            return False
    if any(x < 2 or x > 128 for x in seq):
        return False
    products = _get_all_products(seq)
    return len(products) == len(seq) * (len(seq) + 1) // 2


def _greedy_construct():
    """Deterministic greedy: iterate 2..128, add if no product conflict."""
    values = []
    products = set()
    for x in range(2, 129):
        # Compute all new products ending at x (blocks that include x)
        new_prods = []
        prod = x
        new_prods.append(prod)
        for a in reversed(values):
            prod *= a
            new_prods.append(prod)
        # Check no conflict with existing products
        if not products.intersection(new_prods):
            values.append(x)
            products.update(new_prods)
    return values, products


def _try_insert_fast(seq, products, x):
    """Try to insert x into seq at the correct position.
    Returns (new_seq, new_products, True) if successful, (None, None, False) otherwise."""
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
        return None, None, False
    if pos > 0 and seq[pos-1] == x:
        return None, None, False
    
    # Compute all new products involving x
    new_prods = set()
    
    # Products ending at x: x, x*seq[pos-1], x*seq[pos-1]*seq[pos-2], ...
    prod = x
    new_prods.add(prod)
    for i in range(pos - 1, -1, -1):
        prod *= seq[i]
        new_prods.add(prod)
        if prod > 10**18:  # overflow guard
            break
    
    # Products starting at x: x*seq[pos], x*seq[pos]*seq[pos+1], ...
    if pos < len(seq):
        prod = x * seq[pos]
        new_prods.add(prod)
        for i in range(pos + 1, len(seq)):
            prod *= seq[i]
            new_prods.add(prod)
            if prod > 10**18:
                break
    
    # Check no conflict
    if products.intersection(new_prods):
        return None, None, False
    
    new_seq = seq[:pos] + [x] + seq[pos:]
    new_products = products | new_prods
    return new_seq, new_products, True


def _try_remove(seq, products, idx):
    """Remove element at idx from seq. Returns (new_seq, new_products, True) or (None, None, False)."""
    removed = seq[idx]
    new_seq = seq[:idx] + seq[idx+1:]
    
    # Remove all products involving removed
    # Products that included removed need to be recomputed
    # Easier: just recompute all products
    new_products = _get_all_products(new_seq)
    return new_seq, new_products, True


def _local_search(seq, products, deadline, rng):
    """Perform local search: try swaps, removals+reinsertions."""
    best_seq = seq[:]
    best_products = products
    best_len = len(seq)
    
    while time.time() < deadline:
        improved = False
        n = len(seq)
        
        # Try single removals and reinsertions
        for i in range(n):
            if time.time() >= deadline:
                break
            removed = seq[i]
            new_seq = seq[:i] + seq[i+1:]
            new_products = _get_all_products(new_seq)
            
            # Try to reinsert removed or another element
            available = set(range(2, 129)) - set(new_seq)
            avail_list = list(available)
            rng.shuffle(avail_list)
            
            for x in avail_list:
                new_seq2, new_products2, ok = _try_insert_fast(new_seq, new_products, x)
                if ok and len(new_seq2) > len(seq):
                    seq = new_seq2
                    products = new_products2
                    best_seq = seq[:]
                    best_products = products
                    best_len = len(seq)
                    improved = True
                    break
                elif ok and len(new_seq2) == len(seq):
                    # Same length, might help later - accept with some probability
                    if rng.random() < 0.1:
                        seq = new_seq2
                        products = new_products2
                        improved = True
                        break
            if improved:
                break
        
        # Try adjacent swaps
        if not improved:
            for i in range(n - 1):
                if time.time() >= deadline:
                    break
                # Swap seq[i] and seq[i+1] - but must maintain increasing order
                # Actually, swapping breaks increasing order unless we check
                # Let's try: remove seq[i], insert it after seq[i+1]
                # This is equivalent to moving element i to position i+1
                # But we need to maintain increasing order
                # Actually, let's try: remove element at i, try inserting it at different positions
                removed = seq[i]
                new_seq = seq[:i] + seq[i+1:]
                new_products = _get_all_products(new_seq)
                
                # Try inserting removed at various positions
                for pos in range(len(new_seq) + 1):
                    if pos < len(new_seq) and new_seq[pos] <= removed:
                        continue
                    if pos > 0 and new_seq[pos-1] >= removed:
                        continue
                    new_seq2, new_products2, ok = _try_insert_fast(new_seq, new_products, removed)
                    if ok:
                        # Check if this is different from original
                        if new_seq2 != seq:
                            # Try to add more elements
                            available = set(range(2, 129)) - set(new_seq2)
                            avail_list = list(available)
                            rng.shuffle(avail_list)
                            for x in avail_list[:10]:
                                new_seq3, new_products3, ok2 = _try_insert_fast(new_seq2, new_products2, x)
                                if ok2 and len(new_seq3) > len(seq):
                                    seq = new_seq3
                                    products = new_products3
                                    best_seq = seq[:]
                                    best_products = products
                                    best_len = len(seq)
                                    improved = True
                                    break
                            if improved:
                                break
        if not improved:
            break
    
    return best_seq, best_products, best_len


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Get initial solution via greedy
    best_seq, best_products = _greedy_construct()
    best_len = len(best_seq)
    
    rng = random.Random(42)
    
    # Phase 1: Local search from greedy
    best_seq, best_products, best_len = _local_search(best_seq, best_products, deadline, rng)
    
    # Phase 2: Multiple random restarts with different strategies
    all_numbers = set(range(2, 129))
    
    while time.time() < deadline:
        # Start from a random subset
        seq = []
        products = set()
        
        # Randomly select some elements
        num_initial = rng.randint(5, 15)
        initial_candidates = rng.sample(range(2, 129), min(num_initial, 127))
        rng.shuffle(initial_candidates)
        
        for x in initial_candidates:
            new_seq, new_products, ok = _try_insert_fast(seq, products, x)
            if ok:
                seq = new_seq
                products = new_products
        
        # Try to fill in remaining
        available = all_numbers - set(seq)
        avail_list = list(available)
        rng.shuffle(avail_list)
        
        for x in avail_list:
            new_seq, new_products, ok = _try_insert_fast(seq, products, x)
            if ok:
                seq = new_seq
                products = new_products
        
        # Local search
        seq, products, cur_len = _local_search(seq, products, deadline, rng)
        
        if cur_len > best_len:
            best_seq = seq[:]
            best_products = products
            best_len = cur_len
    
    return best_seq

# EVOLVE-BLOCK-END

