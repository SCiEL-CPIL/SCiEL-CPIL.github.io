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
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Start from best_seq, but allow perturbation to escape local optima
        seq = best_seq[:]
        
        # Randomly remove 1-2 elements to allow rearrangement/addition of others
        if len(seq) > 2:
            num_removals = rng.randint(1, 2)
            for _ in range(num_removals):
                if len(seq) > 1:
                    idx = rng.randint(0, len(seq) - 1)
                    seq.pop(idx)
        
        # Local improvement: try to insert any missing element
        improved = True
        while improved and time.time() < deadline:
            improved = False
            candidates = [x for x in range(2, 129) if x not in seq]
            rng.shuffle(candidates)
            
            for x in candidates:
                new_seq = _try_insert(seq, x)
                if new_seq is not None:
                    seq = new_seq
                    improved = True
                    break # Restart candidate loop with updated seq
        
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


def _try_insert(seq, x):
    """
    Try to insert x into the sorted sequence seq.
    Returns the new sequence if valid, else None.
    """
    if not seq:
        return [x]
        
    # Find insertion point to keep seq sorted
    # Since seq is strictly increasing, x can be inserted at any position k 
    # such that seq[k-1] < x < seq[k]. 
    # Actually, for the product constraint, the position matters.
    # We must try all valid positions? 
    # No, the sequence must be strictly increasing. 
    # So x can only be placed at ONE specific position to maintain strict increase:
    # Between seq[i] and seq[i+1] where seq[i] < x < seq[i+1].
    # If x < seq[0], insert at 0.
    # If x > seq[-1], insert at end.
    # Otherwise, there is exactly one valid index `pos` such that seq[pos-1] < x < seq[pos].
    
    pos = 0
    for i in range(len(seq)):
        if x < seq[i]:
            pos = i
            break
        else:
            pos = i + 1
            
    # Construct new sequence
    new_seq = seq[:pos] + [x] + seq[pos:]
    
    # Check validity
    if _is_valid(new_seq):
        return new_seq
    return None

# EVOLVE-BLOCK-END
