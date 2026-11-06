"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # First, get a good initial solution via greedy
    greedy_seq = _greedy_construct()
    best_seq = greedy_seq[:]
    best_len = len(greedy_seq)
    
    # Try multiple random seeds for stochastic search
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Randomly perturb the current best sequence
        seq = best_seq[:]
        
        # Randomly remove 1-3 elements
        num_removals = rng.randint(1, 3)
        for _ in range(num_removals):
            if len(seq) > 1:
                idx = rng.randint(0, len(seq) - 1)
                seq.pop(idx)
        
        # Try to add elements back in random order
        candidates = [x for x in range(2, 129) if x not in seq]
        rng.shuffle(candidates)
        
        for x in candidates:
            if _can_add(seq, x):
                seq.append(x)
                # Keep the sequence sorted
                seq.sort()
        
        # Try local improvements: swap elements
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(seq)):
                for j in range(i + 1, len(seq)):
                    # Try swapping seq[i] and seq[j]
                    old_i, old_j = seq[i], seq[j]
                    seq[i], seq[j] = old_j, old_i
                    
                    if _is_valid(seq):
                        if len(seq) > best_len:
                            best_len = len(seq)
                            best_seq = seq[:]
                        improved = True
                    else:
                        seq[i], seq[j] = old_i, old_j
        
        # Try adding more elements
        for x in range(2, 129):
            if x not in seq and _can_add(seq, x):
                seq.append(x)
                seq.sort()
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
    
    return best_seq


def _greedy_construct():
    values, products = [], set()
    for x in range(2, 129):
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values


def _get_products(seq):
    """Get all nonempty consecutive block products for a sequence."""
    products = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            products.add(prod)
    return products


def _is_valid(seq):
    """Check if all nonempty consecutive block products are distinct."""
    products = _get_products(seq)
    return len(products) == len(seq) * (len(seq) + 1) // 2


def _can_add(seq, x):
    """Check if x can be appended to seq without creating duplicate products."""
    if not seq:
        return True
    
    # Compute products that would be created by appending x
    new_products = set()
    prod = x
    new_products.add(prod)
    for a in reversed(seq):
        prod *= a
        new_products.add(prod)
    
    # Check against existing products
    existing = _get_products(seq)
    return not existing.intersection(new_products)
# EVOLVE-BLOCK-END
