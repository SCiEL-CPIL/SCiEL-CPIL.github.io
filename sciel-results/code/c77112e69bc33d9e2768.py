import time
import math
import random
import numpy as np

def compute_score(U):
    """Compute the score for a set U."""
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def compute_score_fast(U):
    """Faster score computation using numpy for larger sets."""
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    # For small sets, use the set-based approach
    if len(U) <= 50:
        sum_set = set()
        diff_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
                diff_set.add(a - b)
        ratio = len(diff_set) / len(sum_set)
        if ratio <= 0:
            return 0
        log_ratio = math.log(ratio)
        log_denom = math.log(2 * maxU + 1)
        if log_denom <= 0:
            return 0
        return 1 + log_ratio / log_denom
    else:
        # For larger sets, still use set-based but it's O(n^2)
        sum_set = set()
        diff_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
                diff_set.add(a - b)
        ratio = len(diff_set) / len(sum_set)
        if ratio <= 0:
            return 0
        log_ratio = math.log(ratio)
        log_denom = math.log(2 * maxU + 1)
        if log_denom <= 0:
            return 0
        return 1 + log_ratio / log_denom

def local_optimize(U, deadline, best_score, best_U):
    """Perform local optimization on the set U."""
    U = list(U)
    random.seed(42)
    
    while time.time() < deadline - 3:
        # Try different operations
        op = random.choice(['add', 'remove', 'modify', 'swap', 'perturb'])
        U_new = U[:]
        
        if op == 'add' and len(U_new) < 40:
            # Add a new element
            max_val = max(U_new) if U_new else 1
            new_val = random.randint(1, max(2 * max_val + 10, 100))
            U_new.append(new_val)
            U_new = sorted(set(U_new))
        elif op == 'remove' and len(U_new) > 2:
            # Remove a random element (not 0)
            idx = random.randint(1, len(U_new) - 1)
            U_new.pop(idx)
        elif op == 'modify' and len(U_new) > 1:
            # Modify a random element
            idx = random.randint(1, len(U_new) - 1)
            max_val = max(U_new)
            U_new[idx] = random.randint(1, max(2 * max_val + 10, 100))
            U_new = sorted(set(U_new))
        elif op == 'swap' and len(U_new) > 2:
            # Swap two elements
            i, j = random.sample(range(1, len(U_new)), 2)
            U_new[i], U_new[j] = U_new[j], U_new[i]
        elif op == 'perturb' and len(U_new) > 1:
            # Perturb a random element by a small amount
            idx = random.randint(1, len(U_new) - 1)
            delta = random.randint(-5, 5)
            U_new[idx] = max(1, U_new[idx] + delta)
            U_new = sorted(set(U_new))
        
        if len(U_new) >= 2:
            s = compute_score_fast(U_new)
            if s > best_score:
                best_score = s
                best_U = U_new
                U = U_new[:]
    
    return best_score, best_U

def construct():
    start = time.time()
    deadline = start + 140  # Leave 10s safety margin
    
    best_score = compute_score([0, 1, 3])
    best_U = [0, 1, 3]
    
    # Try structured sets: powers of 2, Mersenne, etc.
    candidates = []
    
    # Powers of 2
    for n in range(2, 25):
        U = [0] + [2**i for i in range(1, n)]
        candidates.append(U)
    
    # Mersenne numbers
    for n in range(2, 25):
        U = [0] + [2**i - 1 for i in range(1, n)]
        candidates.append(U)
    
    # Fibonacci-like
    fib = [0, 1]
    for i in range(2, 25):
        fib.append(fib[-1] + fib[-2])
    for n in range(3, 25):
        candidates.append(fib[:n])
    
    # Geometric progressions with different bases
    for base in [2, 3, 4, 5, 10]:
        for n in range(2, 15):
            U = [0] + [base**i for i in range(1, n)]
            if max(U) < 10**6:
                candidates.append(U)
    
    # Random sets with various sizes
    random.seed(42)
    for trial in range(3000):
        if time.time() > deadline - 30:
            break
        n = random.randint(3, 25)
        # Use different ranges
        max_val = random.choice([100, 500, 1000, 5000, 10000, 50000])
        U = set([0])
        for _ in range(n - 1):
            U.add(random.randint(1, max_val))
        candidates.append(sorted(U))
    
    # Evaluate all candidates
    for U in candidates:
        if time.time() > deadline - 30:
            break
        s = compute_score_fast(U)
        if s > best_score:
            best_score = s
            best_U = U
    
    # Local search from best
    best_score, best_U = local_optimize(best_U, deadline, best_score, best_U)
    
    # Try multiple random restarts if time permits
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        # Create a new random starting point
        n = random.randint(3, 20)
        max_val = random.choice([100, 500, 1000, 5000])
        U_restart = set([0])
        for _ in range(n - 1):
            U_restart.add(random.randint(1, max_val))
        U_restart = sorted(U_restart)
        
        s = compute_score_fast(U_restart)
        if s > best_score:
            best_score = s
            best_U = U_restart
        
        best_score, best_U = local_optimize(best_U, deadline - 5, best_score, best_U)
    
    return best_U
