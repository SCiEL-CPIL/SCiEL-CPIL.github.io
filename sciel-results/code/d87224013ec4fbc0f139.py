import time
import random
import numpy as np

def set_sizes(A):
    """Compute sizes of A+A and A*A with equal summands/factors allowed."""
    A = np.array(A, dtype=np.int64)
    n = len(A)
    # Upper triangle including diagonal
    idx_i, idx_j = np.triu_indices(n)
    sums = np.unique(A[idx_i] + A[idx_j])
    prods = np.unique(A[idx_i] * A[idx_j])
    return len(sums), len(prods)

def objective(A):
    s, p = set_sizes(A)
    return max(s, p)

def is_7_smooth(x, limit=10000):
    """Check if x is a 7-smooth number (only prime factors 2,3,5,7) and <= limit."""
    if x < 1 or x > limit:
        return False
    for p in [2, 3, 5, 7]:
        while x % p == 0:
            x //= p
    return x == 1

def generate_7_smooths(max_val=10000):
    """Generate all 7-smooth numbers up to max_val."""
    smooths = set()
    for a in range(0, 15):  # 2^a
        for b in range(0, 10):  # 3^b
            for c in range(0, 7):  # 5^c
                for d in range(0, 5):  # 7^d
                    v = (2**a) * (3**b) * (5**c) * (7**d)
                    if 1 <= v <= max_val:
                        smooths.add(v)
    return sorted(smooths)

def construct():
    start = time.time()
    deadline = start + 50.0  # safety margin under 60s
    
    best_set = None
    best_obj = float('inf')
    
    # Precompute 7-smooth numbers
    smooths = generate_7_smooths(10000)
    smooth_set = set(smooths)
    
    def try_candidate(A):
        nonlocal best_set, best_obj
        if len(A) != 20:
            return
        if len(set(A)) != 20:
            return
        if any(x < 1 or x > 10000 for x in A):
            return
        obj = objective(A)
        if obj < best_obj:
            best_obj = obj
            best_set = A[:]
    
    # Family 0: 7-smooth number selections (most promising based on knowledge)
    # Try subsets of 7-smooth numbers that are well-spread
    for i in range(0, len(smooths) - 19, 3):
        A = smooths[i:i+20]
        if len(set(A)) == 20 and max(A) <= 10000:
            try_candidate(A)
    
    # Also try geometric-like subsets of 7-smooth numbers
    for log_step in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        for log_start in [0.0, 0.5, 1.0, 1.5]:
            A = []
            target_log = log_start
            for k in range(20):
                # Find closest 7-smooth to exp(target_log)
                target_val = int(np.exp(target_log))
                best_smooth = min(smooths, key=lambda s: abs(np.log(s) - target_log))
                A.append(best_smooth)
                target_log += log_step
            # Deduplicate
            seen = set()
            uniq = []
            for x in A:
                if x not in seen and 1 <= x <= 10000:
                    seen.add(x)
                    uniq.append(x)
            if len(uniq) == 20:
                try_candidate(uniq)
    
    # Family 1: Pure geometric sequences
    for base in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        A = []
        v = base
        for i in range(20):
            A.append(v)
            v *= base
        if max(A) <= 10000:
            try_candidate(A)
        else:
            for start_val in [1, 2, 3]:
                A = []
                v = start_val
                for i in range(20):
                    A.append(v)
                    v *= base
                if max(A) <= 10000 and len(set(A)) == 20:
                    try_candidate(A)
                    break
    
    # Family 2: Geometric with ratio close to 1 (dense in log space)
    for ratio in [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.5, 3.0]:
        for start_val in [1, 2, 3, 4, 5, 10]:
            A = []
            v = float(start_val)
            for i in range(20):
                val = int(round(v))
                if val < 1:
                    val = 1
                A.append(val)
                v *= ratio
            seen = set()
            uniq = []
            for x in A:
                if x not in seen and 1 <= x <= 10000:
                    seen.add(x)
                    uniq.append(x)
            if len(uniq) == 20:
                try_candidate(uniq)
    
    # Family 3: Powers of 2 times small odd numbers
    for offset in [1, 2, 3, 5]:
        for shift in [0, 1, 2]:
            A = []
            for i in range(20):
                val = (2**i) * (1 + (i + shift) * offset)
                if val > 10000:
                    val = 10000
                A.append(val)
            seen = set()
            uniq = []
            for x in A:
                if x not in seen and 1 <= x <= 10000:
                    seen.add(x)
                    uniq.append(x)
            if len(uniq) == 20:
                try_candidate(uniq)
    
    # Family 4: Sparse geometric
    for base in [2, 3, 4, 5]:
        for step in [1, 2, 3]:
            A = [base**(i*step) for i in range(20)]
            if max(A) <= 10000 and len(set(A)) == 20:
                try_candidate(A)
    
    # Family 5: Mixed - first half geometric, second half perturbed
    for base in [2, 3, 4]:
        A = [base**i for i in range(10)]
        last = base**9
        for i in range(10):
            A.append(last * (i + 2))
        seen = set()
        uniq = []
        for x in A:
            if x not in seen and 1 <= x <= 10000:
                seen.add(x)
                uniq.append(x)
        if len(uniq) == 20:
            try_candidate(uniq)
    
    # Family 6: Log-uniform with rounding - spread evenly in log space
    for log_min in [0, 0.5, 1.0]:
        for log_max in [8.0, 9.0, 9.5, 10.0]:
            if log_max <= log_min:
                continue
            A = []
            for i in range(20):
                t = i / 19.0
                log_val = log_min + t * (log_max - log_min)
                val = int(round(np.exp(log_val)))
                if val < 1:
                    val = 1
                if val > 10000:
                    val = 10000
                A.append(val)
            seen = set()
            uniq = []
            for x in A:
                if x not in seen and 1 <= x <= 10000:
                    seen.add(x)
                    uniq.append(x)
            if len(uniq) == 20:
                try_candidate(uniq)
    
    # Family 7: Two interleaved geometric sequences
    for base1 in [2, 3, 4]:
        for base2 in [3, 5, 7]:
            if base1 == base2:
                continue
            A1 = [base1**i for i in range(10)]
            A2 = [base2**i for i in range(10)]
            A = A1 + A2
            seen = set()
            uniq = []
            for x in A:
                if x not in seen and 1 <= x <= 10000:
                    seen.add(x)
                    uniq.append(x)
            if len(uniq) == 20:
                try_candidate(uniq)
    
    # Family 8: Fibonacci-like sequences
    for a0, a1 in [(1, 1), (1, 2), (1, 3), (2, 3), (1, 4)]:
        A = [a0, a1]
        while len(A) < 20:
            A.append(A[-1] + A[-2])
        if max(A) <= 10000 and len(set(A)) == 20:
            try_candidate(A)
    
    # Now do local search from the best candidate found
    if best_set is None:
        best_set = list(range(1, 21))
        best_obj = objective(best_set)
    
    random.seed(42)
    
    def local_search(start_set, max_iters=50000, prefer_smooth=False):
        nonlocal best_set, best_obj
        A = list(start_set)
        cur_obj = objective(A)
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_set = A[:]
        it = 0
        no_improve = 0
        while it < max_iters:
            if time.time() > deadline:
                return
            it += 1
            i = random.randrange(20)
            x = A[i]
            cands = set()
            if prefer_smooth:
                # Prefer 7-smooth neighbors
                for factor in [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0]:
                    nv = int(round(x * factor))
                    if 1 <= nv <= 10000 and nv in smooth_set:
                        cands.add(nv)
                # Also add some smooths near x
                for delta in [-20, -10, -5, -2, -1, 1, 2, 5, 10, 20, 50]:
                    nv = x + delta
                    if 1 <= nv <= 10000 and nv in smooth_set:
                        cands.add(nv)
                # Fallback to non-smooth if too few
                if len(cands) < 3:
                    for factor in [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0]:
                        nv = int(round(x * factor))
                        if 1 <= nv <= 10000:
                            cands.add(nv)
                    for delta in [-10, -5, -2, -1, 1, 2, 5, 10, 50, 100]:
                        nv = x + delta
                        if 1 <= nv <= 10000:
                            cands.add(nv)
            else:
                # Neighbors in log space
                for factor in [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0]:
                    nv = int(round(x * factor))
                    if 1 <= nv <= 10000:
                        cands.add(nv)
                # Additive neighbors
                for delta in [-10, -5, -2, -1, 1, 2, 5, 10, 50, 100]:
                    nv = x + delta
                    if 1 <= nv <= 10000:
                        cands.add(nv)
            # Random
            if prefer_smooth and len(smooths) > 0:
                cands.add(random.choice(smooths))
            else:
                cands.add(random.randrange(1, 10001))
            cands.discard(x)
            cands = [c for c in cands if c not in A]
            if not cands:
                continue
            nv = random.choice(cands)
            old = A[i]
            A[i] = nv
            new_obj = objective(A)
            if new_obj <= cur_obj:
                cur_obj = new_obj
                no_improve = 0
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_set = A[:]
            else:
                A[i] = old
                no_improve += 1
            if no_improve > 3000:
                break
    
    # Local search from best candidate - prefer smooth moves
    local_search(best_set, max_iters=40000, prefer_smooth=True)
    
    # Final polish: tighter local search
    if best_set is not None and time.time() < deadline - 5:
        local_search(best_set, max_iters=20000, prefer_smooth=True)
    
    # Also try local search from a few other structured starts
    other_starts = [
        list(range(1, 21)),
    ]
    # Add 7-smooth based start
    if len(smooths) >= 20:
        other_starts.append(smooths[:20])
    for base in [2, 3]:
        A = []
        v = 1
        for i in range(20):
            A.append(v)
            v *= base
        if max(A) <= 10000 and len(set(A)) == 20:
            other_starts.append(A)
    
    for sp in other_starts:
        if time.time() > deadline - 3:
            break
        if sp == best_set:
            continue
        local_search(sp, max_iters=10000, prefer_smooth=True)
    
    return best_set
