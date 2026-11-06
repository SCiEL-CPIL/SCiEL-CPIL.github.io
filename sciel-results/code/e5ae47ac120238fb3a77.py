"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 85.0  # safety margin before 90s kill

def _all_block_sums(seq):
    """Return set of all non-empty consecutive block sums for seq."""
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _is_valid(seq):
    """Check strictly increasing and distinct consecutive block sums."""
    if not seq:
        return False
    # strictly increasing
    for i in range(1, len(seq)):
        if seq[i] <= seq[i-1]:
            return False
    # all in [1, 256]
    for v in seq:
        if v < 1 or v > 256:
            return False
    # distinct block sums
    n = len(seq)
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_build():
    """Original greedy strategy as a baseline."""
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _random_restart_sequence():
    """Generate a random strictly increasing sequence in [1,256] of random length."""
    # Try random lengths from 5 to 20
    length = random.randint(5, 18)
    # Pick 'length' distinct values from 1..256, sort them
    candidates = random.sample(range(1, 257), length)
    candidates.sort()
    return candidates

def _local_search(seq, max_iter=200):
    """Improve a sequence via random perturbations. Returns best valid seq found."""
    best = seq[:]
    best_len = len(best)
    cur = seq[:]
    
    for _ in range(max_iter):
        if time.time() > _DEADLINE:
            break
        
        # Random perturbation: swap, insert, remove, or change a value
        op = random.choice(['swap', 'insert', 'remove', 'change'])
        cand = cur[:]
        
        if op == 'swap' and len(cand) >= 2:
            i, j = random.sample(range(len(cand)), 2)
            cand[i], cand[j] = cand[j], cand[i]
            # Re-sort to maintain strictly increasing
            cand.sort()
            # Check for duplicates after sort
            if len(set(cand)) != len(cand):
                continue
        elif op == 'insert':
            # Insert a new random value
            v = random.randint(1, 256)
            if v not in cand:
                cand.append(v)
                cand.sort()
        elif op == 'remove' and len(cand) > 3:
            i = random.randint(0, len(cand) - 1)
            cand.pop(i)
        elif op == 'change' and len(cand) > 0:
            i = random.randint(0, len(cand) - 1)
            v = random.randint(1, 256)
            if v not in cand:
                cand[i] = v
                cand.sort()
                if len(set(cand)) != len(cand):
                    continue
        
        # Validate
        if _is_valid(cand):
            if len(cand) > best_len:
                best = cand[:]
                best_len = len(best)
            cur = cand[:]
        else:
            # With some probability, accept invalid to escape local optima (annealing-like)
            if random.random() < 0.1:
                cur = cand[:]
    
    return best

def construct():
    start_time = time.time()
    
    # Start with greedy baseline
    best_seq = _greedy_build()
    best_len = len(best_seq)
    
    # If greedy already gives a good result, still try to improve
    # Randomized restarts
    random.seed(42)  # Reproducibility
    
    while time.time() < _DEADLINE:
        # Generate a new random starting point
        rand_seq = _random_restart_sequence()
        
        # Local search from this start
        improved = _local_search(rand_seq, max_iter=50)
        
        if len(improved) > best_len:
            best_seq = improved
            best_len = len(best_seq)
        
        # Also try local search from the current best
        if time.time() < _DEADLINE:
            improved2 = _local_search(best_seq, max_iter=50)
            if len(improved2) > best_len:
                best_seq = improved2
                best_len = len(best_seq)
    
    # Final safety check
    if not _is_valid(best_seq):
        # Fallback to greedy
        best_seq = _greedy_build()
    
    return best_seq
# EVOLVE-BLOCK-END
