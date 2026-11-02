"""Model-and-optimize approach for sumset/difference-set exponent lower bound."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    # Compute sumset and difference set efficiently using sets
    sumset = set()
    for a in U_set:
        for b in U_set:
            sumset.add(a + b)
    
    diffset = set()
    for a in U_set:
        for b in U_set:
            diffset.add(a - b)
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _score_set_fast(U):
    """Faster scoring using numpy where beneficial for moderate sizes."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    arr = np.array(sorted(U_set), dtype=np.int64)
    
    # Sumset via broadcasting for small n
    if n <= 300:
        sums = arr[:, None] + arr[None, :]
        sumset_size = len(np.unique(sums))
        diffs = arr[:, None] - arr[None, :]
        diffset_size = len(np.unique(diffs))
    else:
        # Fall back to set-based for large n
        sumset = set()
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
        sumset_size = len(sumset)
        
        diffset = set()
        for a in U_set:
            for b in U_set:
                diffset.add(a - b)
        diffset_size = len(diffset)
    
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _construct_sidon_sequence(n, max_val=None):
    """Construct a Sidon (B_2) sequence of length n using greedy method."""
    if max_val is None:
        max_val = n * n * 4
    
    seq = [0]
    used_sums = {0}
    
    for i in range(1, n):
        pos = 1
        while pos <= max_val:
            valid = True
            for j, a in enumerate(seq):
                s = a + pos
                if s in used_sums:
                    valid = False
                    break
            if valid:
                # Check all pairs including new element
                all_vals = seq + [pos]
                sums = set()
                for x in range(len(all_vals)):
                    for y in range(x, len(all_vals)):
                        s = all_vals[x] + all_vals[y]
                        if s in sums:
                            valid = False
                            break
                        sums.add(s)
                    if not valid:
                        break
                if valid:
                    seq.append(pos)
                    break
            pos += 1
        if len(seq) < i + 1:
            break
    
    return seq

def _construct_sidon_miller(n):
    """Construct a Sidon sequence using Miller's construction."""
    if n <= 1:
        return [0]
    
    # Find a prime p >= n
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(math.isqrt(x)) + 1):
            if x % i == 0:
                return False
        return True
    
    p = n
    while not is_prime(p):
        p += 1
    
    # Miller's construction: {i * p + i^2 mod p : i in 0..n-1}
    seq = set()
    for i in range(n):
        val = i * p + (i * i) % p
        seq.add(val)
    
    return sorted(seq)

def _local_search(U, max_iters=3000, deadline=None, max_size=100):
    """Improve set U via local search with add/remove/modify operations."""
    U = sorted(set(U))
    best_score = _score_set_fast(U)
    best_U = U[:]
    max_u = max(U) if U else 0
    upper_bound = max_u + 200
    
    # Use fast scorer for sets up to 300 elements
    use_fast = len(U) <= 300
    scorer = _score_set_fast if use_fast else _score_set
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        U_work = U[:]
        random.shuffle(U_work)
        
        # Try modifying one element
        for i in range(len(U_work)):
            old_val = U_work[i]
            # Try a few candidate replacements
            candidates = []
            # Try values near current
            for delta in [-10, -5, -3, -2, -1, 1, 2, 3, 5, 10, 20, 50]:
                nv = old_val + delta
                if 0 <= nv <= upper_bound and nv != old_val:
                    candidates.append(nv)
            # Try random values
            for _ in range(5):
                nv = random.randint(0, upper_bound)
                if nv != old_val:
                    candidates.append(nv)
            
            for new_val in candidates:
                new_U = U_work[:]
                new_U[i] = new_val
                new_U_set = set(new_U)
                if len(new_U_set) != len(new_U):
                    continue
                new_U_sorted = sorted(new_U_set)
                if new_U_sorted[0] != 0:
                    continue
                s = scorer(new_U_sorted)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U_sorted
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try adding an element
            if len(U) < max_size:
                # Try random positions first, then systematic
                tried = set()
                for _ in range(20):
                    new_val = random.randint(0, upper_bound)
                    if new_val in U or new_val in tried:
                        continue
                    tried.add(new_val)
                    new_U = U + [new_val]
                    s = scorer(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = sorted(new_U)
                        improved = True
                        break
        
        if not improved:
            # Try removing an element (if size > 2)
            if len(U) > 2:
                for i in range(len(U)):
                    if U[i] == 0:
                        continue
                    new_U = U[:i] + U[i+1:]
                    s = scorer(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U
                        improved = True
                        break
        
        if not improved:
            # Random restart: mutate the whole set
            if random.random() < 0.15:
                mutated = U[:]
                # Add 1-3 random elements
                for _ in range(random.randint(1, 3)):
                    nv = random.randint(1, upper_bound)
                    if nv not in mutated:
                        mutated.append(nv)
                # Remove 0-2 random elements (not 0)
                removable = [x for x in mutated if x != 0]
                for _ in range(random.randint(0, min(2, len(removable)))):
                    if removable:
                        idx = random.randint(0, len(removable) - 1)
                        val = removable.pop(idx)
                        mutated.remove(val)
                if 0 in mutated:
                    mutated = sorted(set(mutated))
                    s = scorer(mutated)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = mutated
                        improved = True
        
        if not improved and it > 50:
            # Try expanding the search space
            upper_bound += 100
            U = best_U[:]
    
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # Phase 1: Try various Sidon sequence constructions
    # Miller's construction for various sizes
    for n in range(3, 50):
        if time.time() > deadline:
            break
        ss = _construct_sidon_miller(n)
        if len(ss) >= 2:
            s = _score_set(ss)
            if s > best_score:
                best_score = s
                best_U = ss[:]
    
    # Greedy Sidon sequences
    for n in range(3, 30):
        if time.time() > deadline:
            break
        ss = _construct_sidon_sequence(n, max_val=n * n * 6)
        if len(ss) >= n:
            s = _score_set(ss)
            if s > best_score:
                best_score = s
                best_U = ss[:]
    
    # Phase 2: Local search refinement on best candidate
    if time.time() < deadline:
        improved_U, improved_score = _local_search(best_U, max_iters=5000, deadline=deadline, max_size=80)
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U
    
    # Phase 3: Try local search on other good candidates
    if time.time() < deadline:
        candidates = []
        
        # More Miller constructions
        for n in [10, 15, 20, 25, 30, 40]:
            if time.time() > deadline:
                break
            ss = _construct_sidon_miller(n)
            if len(ss) >= 5:
                candidates.append(ss)
        
        # Random sets
        random.seed(42)
        for _ in range(15):
            if time.time() > deadline:
                break
            n = random.randint(5, 25)
            max_val = random.randint(30, 150)
            U = [0] + random.sample(range(1, max_val + 1), n - 1)
            U = sorted(U)
            candidates.append(U)
        
        for U in candidates:
            if time.time() > deadline:
                break
            s = _score_set(U)
            if s > best_score - 0.05:  # Only if close to best
                improved_U, improved_score = _local_search(U, max_iters=2000, deadline=deadline, max_size=60)
                if improved_score > best_score:
                    best_score = improved_score
                    best_U = improved_U
    
    # Phase 4: Final polish with more iterations if time remains
    if time.time() < deadline - 10:
        improved_U, improved_score = _local_search(best_U, max_iters=3000, deadline=deadline, max_size=80)
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U
    
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
