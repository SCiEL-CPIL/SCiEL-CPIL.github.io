import time
import random
import math
import numpy as np

def _score(A):
    """Compute max(|A+A|, |A*A|) using numpy for speed."""
    arr = np.array(A, dtype=np.int64)
    sums = np.add.outer(arr, arr)
    prods = np.multiply.outer(arr, arr)
    n_sums = len(np.unique(sums))
    n_prods = len(np.unique(prods))
    return max(n_sums, n_prods)

def _valid(A):
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    if any(x <= 0 or x > 10000 for x in A):
        return False
    return True

def _local_search(A, max_iter=3000, deadline=None):
    """Local search with structured perturbations."""
    current_A = list(A)
    current_score = _score(current_A)
    best_A = list(current_A)
    best_score = current_score
    rng = random.Random(99)
    n = len(current_A)
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        # Try modifying one element
        k = rng.randrange(n)
        old_val = current_A[k]
        
        # Generate candidate replacements: nearby values and structured alternatives
        candidates = []
        # Nearby values
        for delta in [-5, -3, -2, -1, 1, 2, 3, 5]:
            nv = old_val + delta
            if 1 <= nv <= 10000:
                candidates.append(nv)
        # Geometric neighbors
        if old_val > 2:
            candidates.append(max(1, old_val // 2))
            candidates.append(min(10000, old_val * 2))
        # Random
        candidates.append(rng.randint(1, 10000))
        candidates.append(rng.randint(1, max(2, old_val + 10)))
        
        rng.shuffle(candidates)
        
        for new_val in candidates:
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in current_A and new_val != old_val:
                continue
            
            current_A[k] = new_val
            new_score = _score(current_A)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_A = list(current_A)
                break
            else:
                current_A[k] = old_val
        
        if not improved:
            # Try a more aggressive random move
            k = rng.randrange(n)
            old_val = current_A[k]
            new_val = rng.randint(1, 10000)
            if new_val in current_A and new_val != old_val:
                continue
            current_A[k] = new_val
            new_score = _score(current_A)
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_A = list(current_A)
            else:
                current_A[k] = old_val
    
    return best_A, best_score

def _sa_search(A, steps=20000, start_temp=8.0, end_temp=0.1, seed=42, deadline=None):
    """Simulated annealing with structured moves."""
    rng = random.Random(seed)
    current_A = list(A)
    current_score = _score(current_A)
    best_A = list(current_A)
    best_score = current_score
    n = len(current_A)
    
    for step in range(steps):
        if deadline and time.time() > deadline:
            break
        
        T = start_temp * (end_temp / start_temp) ** (step / max(1, steps))
        
        k = rng.randrange(n)
        old_val = current_A[k]
        
        # Choose move type
        r = rng.random()
        if r < 0.5:
            # Small local move
            delta = rng.randint(-15, 15)
            new_val = old_val + delta
        elif r < 0.75:
            # Medium geometric move
            if old_val > 3:
                new_val = max(1, int(old_val * rng.uniform(0.5, 1.5)))
            else:
                new_val = rng.randint(1, 50)
        else:
            # Large random move
            new_val = rng.randint(1, 10000)
        
        if new_val < 1 or new_val > 10000:
            continue
        if new_val in current_A and new_val != old_val:
            continue
        
        current_A[k] = new_val
        new_score = _score(current_A)
        
        delta_score = new_score - current_score
        if delta_score < 0 or rng.random() < math.exp(-delta_score / max(T, 0.01)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = list(current_A)
        else:
            current_A[k] = old_val
    
    return best_A, best_score

def _generate_structured_candidates(deadline):
    """Generate diverse structured candidates."""
    candidates = []
    rng = random.Random(123)
    
    # 1. Consecutive small integers
    candidates.append(list(range(1, 21)))
    
    # 2. Geometric progressions with various parameters
    for start in range(1, 8):
        for log_ratio in np.linspace(0.05, 0.8, 30):
            if time.time() > deadline:
                break
            ratio = math.exp(log_ratio)
            A = []
            val = float(start)
            for _ in range(20):
                iv = max(1, int(val + 0.5))
                if iv > 10000:
                    break
                if iv not in A:
                    A.append(iv)
                else:
                    iv += 1
                    if iv <= 10000 and iv not in A:
                        A.append(iv)
                val *= ratio
            if len(A) == 20 and _valid(A):
                candidates.append(A)
    
    # 3. Log-spaced
    for base_low in [1.2, 1.5, 2.0, 2.5]:
        for base_high in [base_low + 0.3, base_low + 0.5, base_low + 1.0]:
            A = []
            for i in range(20):
                t = i / 19.0
                val = int(base_low ** (t * 19))
                if val < 1:
                    val = 1
                if val > 10000:
                    val = 10000
                if val not in A:
                    A.append(val)
                else:
                    val += 1
                    if val <= 10000 and val not in A:
                        A.append(val)
            if len(A) == 20 and _valid(A):
                candidates.append(A)
    
    # 4. Powers
    for base in [2, 3, 4, 5]:
        A = [base ** i for i in range(20)]
        if max(A) <= 10000 and len(set(A)) == 20:
            candidates.append(A)
    
    # 5. Mixed: small consecutive + geometric tail
    for n_small in range(5, 16):
        small = list(range(1, n_small + 1))
        remaining = 20 - n_small
        if remaining <= 0:
            continue
        # Geometric tail starting after small
        start = n_small + 1
        for ratio in [1.5, 2.0, 2.5, 3.0]:
            tail = []
            val = float(start)
            for _ in range(remaining):
                iv = max(start + len(tail), int(val + 0.5))
                if iv > 10000:
                    break
                if iv not in small and iv not in tail:
                    tail.append(iv)
                val *= ratio
            if len(tail) == remaining:
                A = sorted(small + tail)
                if _valid(A):
                    candidates.append(A)
    
    # 6. Polynomial-like sequences
    for c in range(1, 10):
        A = [c * i * i + i + 1 for i in range(20)]
        if max(A) <= 10000 and len(set(A)) == 20:
            candidates.append(A)
    
    # 7. Random structured: pick from log-spaced grid
    for _ in range(50):
        if time.time() > deadline:
            break
        grid = [int(10000 ** (i / 19.0)) for i in range(20)]
        # Perturb
        A = []
        for i, g in enumerate(grid):
            jitter = rng.randint(-g // 5, g // 5)
            val = max(1, g + jitter)
            if val not in A:
                A.append(val)
            else:
                val += 1
                if val <= 10000 and val not in A:
                    A.append(val)
        if len(A) == 20 and _valid(A):
            candidates.append(A)
    
    # 8. Pure random samples
    for _ in range(30):
        if time.time() > deadline:
            break
        A = sorted(rng.sample(range(1, 10001), 20))
        candidates.append(A)
    
    # 9. Bimodal: cluster of small + cluster of large
    for _ in range(20):
        if time.time() > deadline:
            break
        n1 = rng.randint(5, 15)
        n2 = 20 - n1
        small = sorted(rng.sample(range(1, 50), n1))
        large = sorted(rng.sample(range(500, 10001), n2))
        A = sorted(small + large)
        if len(set(A)) == 20 and _valid(A):
            candidates.append(A)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s
    
    # Generate structured candidates
    candidates = _generate_structured_candidates(deadline)
    
    # Evaluate all candidates
    best_A = None
    best_score = float('inf')
    
    for cand in candidates:
        if not _valid(cand):
            continue
        s = _score(cand)
        if s < best_score:
            best_score = s
            best_A = list(cand)
    
    if best_A is None:
        best_A = list(range(1, 21))
        best_score = _score(best_A)
    
    # Local search on best
    if time.time() < deadline - 5:
        A_ls, s_ls = _local_search(best_A, max_iter=2000, deadline=deadline)
        if s_ls < best_score:
            best_score = s_ls
            best_A = A_ls
    
    # Try local search on top diverse candidates
    scored_cands = []
    for cand in candidates:
        if not _valid(cand):
            continue
        s = _score(cand)
        scored_cands.append((s, cand))
    scored_cands.sort(key=lambda x: x[0])
    
    for s_c, cand in scored_cands[:5]:
        if time.time() > deadline - 10:
            break
        if s_c >= best_score:
            break
        A_ls, s_ls = _local_search(cand, max_iter=1000, deadline=deadline)
        if s_ls < best_score:
            best_score = s_ls
            best_A = A_ls
    
    # Simulated annealing from best
    if time.time() < deadline - 10:
        A_sa, s_sa = _sa_search(best_A, steps=15000, start_temp=5.0, end_temp=0.05, seed=7, deadline=deadline)
        if s_sa < best_score:
            best_score = s_sa
            best_A = A_sa
    
    # Second SA from a different starting point
    if time.time() < deadline - 5 and len(scored_cands) > 1:
        alt_cand = scored_cands[1][1]
        A_sa2, s_sa2 = _sa_search(alt_cand, steps=10000, start_temp=8.0, end_temp=0.1, seed=42, deadline=deadline)
        if s_sa2 < best_score:
            best_score = s_sa2
            best_A = A_sa2
    
    # Final local polish
    if time.time() < deadline - 2:
        A_polish, s_polish = _local_search(best_A, max_iter=2000, deadline=deadline)
        if s_polish < best_score:
            best_score = s_polish
            best_A = A_polish
    
    return best_A
