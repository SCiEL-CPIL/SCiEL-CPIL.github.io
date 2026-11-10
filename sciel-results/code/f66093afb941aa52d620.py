"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

def _prime_factorization(n):
    """Return prime factorization as a tuple of (prime, exponent) pairs, sorted."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return tuple(sorted(factors.items()))

# Precompute factorizations for all numbers in [2, 128]
_FACTS = {}
for _x in range(2, 129):
    _FACTS[_x] = _prime_factorization(_x)

def _vec_add(a, b):
    """Add two factorization tuples, returning a new tuple."""
    if not a:
        return b
    if not b:
        return a
    # Convert to dict for addition
    da = dict(a)
    db = dict(b)
    for k, v in db.items():
        da[k] = da.get(k, 0) + v
    return tuple(sorted(da.items()))

def _compute_block_vecs(seq):
    """Compute all consecutive block product factorization vectors for a sequence.
    Returns a set of frozensets (for hashability)."""
    vecs = set()
    n = len(seq)
    for i in range(n):
        prod_vec = ()
        for j in range(i, n):
            prod_vec = _vec_add(prod_vec, _FACTS[seq[j]])
            vecs.add(frozenset(prod_vec))
    return vecs

def _is_valid(seq):
    """Check if sequence is strictly increasing and all block products are distinct."""
    if not seq:
        return True
    for i in range(1, len(seq)):
        if seq[i] <= seq[i-1]:
            return False
    n = len(seq)
    expected = n * (n + 1) // 2
    vecs = _compute_block_vecs(seq)
    return len(vecs) == expected

def _try_insert_fast(seq, x):
    """Try to insert x into seq. Returns (new_seq, new_vecs) or (None, None).
    Uses incremental computation where possible."""
    n = len(seq)
    # Find position
    pos = 0
    while pos < n and seq[pos] < x:
        pos += 1
    if pos < n and seq[pos] == x:
        return None, None
    
    new_seq = seq[:pos] + [x] + seq[pos:]
    # Quick check: recompute all vecs (fast enough for our sizes)
    vecs = _compute_block_vecs(new_seq)
    expected = len(new_seq) * (len(new_seq) + 1) // 2
    if len(vecs) == expected:
        return new_seq, vecs
    return None, None

def _simulated_annealing(seq, deadline, rng):
    """Simulated annealing to improve sequence length."""
    best = list(seq)
    best_len = len(best)
    current = list(seq)
    current_vecs = _compute_block_vecs(current)
    available = [x for x in range(2, 129) if x not in current]
    
    start_temp = 10.0
    end_temp = 0.1
    steps = 5000
    
    for step in range(steps):
        if time.time() >= deadline:
            break
        
        # Temperature schedule
        progress = step / steps
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Choose a random move
        move_type = rng.random()
        
        if move_type < 0.6:
            # Insertion: add a random available number
            if not available:
                continue
            x = rng.choice(available)
            new_seq, new_vecs = _try_insert_fast(current, x)
            if new_seq is not None:
                # Accept (it's valid and longer)
                current = new_seq
                current_vecs = new_vecs
                available.remove(x)
                if len(current) > best_len:
                    best = list(current)
                    best_len = len(current)
        elif move_type < 0.85:
            # Swap: remove one, add one
            if not current or not available:
                continue
            i = rng.randrange(len(current))
            removed = current[i]
            base = current[:i] + current[i+1:]
            # Try to insert a random available element
            x = rng.choice(available)
            if x == removed:
                continue
            new_seq, new_vecs = _try_insert_fast(base, x)
            if new_seq is not None:
                # Accept swap if it maintains validity
                # For length maximization, we accept valid swaps to explore
                current = new_seq
                current_vecs = new_vecs
                available.remove(x)
                available.append(removed)
                if len(current) > best_len:
                    best = list(current)
                    best_len = len(current)
        else:
            # Deletion: remove a random element (to escape local optima)
            if not current:
                continue
            i = rng.randrange(len(current))
            removed = current[i]
            current = current[:i] + current[i+1:]
            current_vecs = _compute_block_vecs(current)
            available.append(removed)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    rng = random.Random(42)
    
    # Phase 1: Greedy construction with prime-factor-based validation
    values = []
    for x in range(2, 129):
        if time.time() >= deadline - 10.0:
            break
        new_seq, _ = _try_insert_fast(values, x)
        if new_seq is not None:
            values = new_seq
    
    greedy_len = len(values)
    
    # Phase 2: Simulated annealing to escape local optima
    if time.time() < deadline - 3.0:
        sa_result = _simulated_annealing(values, deadline - 2.0, rng)
        if len(sa_result) > greedy_len:
            values = sa_result
            greedy_len = len(values)
    
    # Phase 3: Try different greedy orderings if time permits
    if time.time() < deadline - 5.0:
        # Try ordering by number of prime factors (fewer first)
        alt_order = sorted(range(2, 129), key=lambda x: sum(1 for _ in _FACTS[x]))
        trial = []
        for x in alt_order:
            if time.time() >= deadline - 3.0:
                break
            new_seq, _ = _try_insert_fast(trial, x)
            if new_seq is not None:
                trial = new_seq
        if len(trial) > greedy_len:
            values = trial
            greedy_len = len(values)
            # Apply SA to improved result
            if time.time() < deadline - 3.0:
                sa_result = _simulated_annealing(values, deadline - 2.0, rng)
                if len(sa_result) > greedy_len:
                    values = sa_result
                    greedy_len = len(values)
        
        # Try random orderings
        for restart in range(3):
            if time.time() >= deadline - 3.0:
                break
            rem = list(range(2, 129))
            rng.shuffle(rem)
            trial = []
            for x in rem:
                if time.time() >= deadline - 3.0:
                    break
                new_seq, _ = _try_insert_fast(trial, x)
                if new_seq is not None:
                    trial = new_seq
            if len(trial) > greedy_len:
                values = trial
                greedy_len = len(values)
                if time.time() < deadline - 3.0:
                    sa_result = _simulated_annealing(values, deadline - 2.0, rng)
                    if len(sa_result) > greedy_len:
                        values = sa_result
                        greedy_len = len(values)
    
    return values
# EVOLVE-BLOCK-END
