"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

_DEADLINE = 85.0  # seconds, safety margin before 90s kill
_START = None

def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START

def _time_left():
    return _DEADLINE - _now()

def _valid_products(seq):
    """Return set of all nonempty consecutive block products, or None if any product exceeds limit."""
    n = len(seq)
    products = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return None  # collision
            products.add(prod)
    return products

def _is_valid(seq):
    """Check if sequence has all distinct consecutive block products."""
    n = len(seq)
    products = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True

def _greedy_build(order, used_set=None):
    """Build a sequence greedily following the given order of candidates."""
    values = []
    products = set()
    for x in order:
        # Compute all new block products ending at x
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

def _local_improve(seq, max_iter=200):
    """Try to improve sequence by swapping or reordering elements."""
    best = list(seq)
    best_len = len(best)
    for _ in range(max_iter):
        if _time_left() < 5.0:
            break
        improved = False
        n = len(best)
        # Try moving an element to a different position
        for i in range(n):
            if _time_left() < 5.0:
                break
            for j in range(i+1, n):
                # Try swapping i and j
                trial = list(best)
                trial[i], trial[j] = trial[j], trial[i]
                if _is_valid(trial):
                    best = trial
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best

def _try_insert(seq, x, products_cache=None):
    """Try to insert x into seq at some position, maintaining validity."""
    n = len(seq)
    # Try all positions
    for pos in range(n + 1):
        trial = seq[:pos] + [x] + seq[pos:]
        if _is_valid(trial):
            return trial
    return None

def construct():
    _now()
    candidates = list(range(2, 129))
    
    best_seq = []
    best_len = 0
    
    # First, get the greedy solution as a baseline
    greedy_seq = _greedy_build(candidates)
    if len(greedy_seq) > best_len:
        best_seq = greedy_seq
        best_len = len(greedy_seq)
    
    # Multiple randomized restarts with different orderings
    rng = random.Random(42)
    
    for restart in range(500):
        if _time_left() < 10.0:
            break
        
        # Different random strategies for ordering
        strategy = rng.randint(0, 3)
        if strategy == 0:
            order = candidates[:]
            rng.shuffle(order)
        elif strategy == 1:
            # Prefer larger numbers first (they create more distinct products)
            order = candidates[:]
            rng.shuffle(order)
            order.sort(reverse=True)
            # Add some randomness
            for i in range(len(order)-1, 0, -1):
                if rng.random() < 0.3:
                    j = rng.randint(0, i)
                    order[i], order[j] = order[j], order[i]
        elif strategy == 2:
            # Prefer numbers with more distinct prime factors
            def prime_factor_count(n):
                count = 0
                d = 2
                while d * d <= n:
                    if n % d == 0:
                        count += 1
                        while n % d == 0:
                            n //= d
                    d += 1
                if n > 1:
                    count += 1
                return count
            order = candidates[:]
            rng.shuffle(order)
            order.sort(key=lambda x: prime_factor_count(x), reverse=True)
            # Add randomness
            for i in range(len(order)-1, 0, -1):
                if rng.random() < 0.4:
                    j = rng.randint(0, i)
                    order[i], order[j] = order[j], order[i]
        else:
            # Weighted random: prefer numbers that are "harder" to place
            order = candidates[:]
            weights = [1.0 / x for x in order]  # smaller numbers get higher weight
            total = sum(weights)
            shuffled = []
            remaining = order[:]
            remaining_weights = weights[:]
            for _ in range(len(remaining)):
                r = rng.random() * sum(remaining_weights)
                cum = 0
                for idx, w in enumerate(remaining_weights):
                    cum += w
                    if cum >= r:
                        shuffled.append(remaining[idx])
                        remaining.pop(idx)
                        remaining_weights.pop(idx)
                        break
            order = shuffled
        
        seq = _greedy_build(order)
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
        
        # Try to improve by inserting missing numbers
        if len(seq) < 127:
            missing = set(candidates) - set(seq)
            # Try to insert missing numbers one by one
            for x in sorted(missing, key=lambda v: rng.random()):
                if _time_left() < 10.0:
                    break
                new_seq = _try_insert(seq, x)
                if new_seq is not None:
                    seq = new_seq
                    if len(seq) > best_len:
                        best_len = len(seq)
                        best_seq = seq
        
        # Local improvement via swaps
        if _time_left() > 15.0:
            improved_seq = _local_improve(seq, max_iter=50)
            if len(improved_seq) > best_len:
                best_len = len(improved_seq)
                best_seq = improved_seq
    
    # Final attempt: try to add more elements to best_seq
    if _time_left() > 5.0:
        missing = set(candidates) - set(best_seq)
        for x in sorted(missing):
            if _time_left() < 5.0:
                break
            new_seq = _try_insert(best_seq, x)
            if new_seq is not None:
                best_seq = new_seq
    
    # Ensure strictly increasing
    best_seq.sort()
    
    return best_seq
# EVOLVE-BLOCK-END
