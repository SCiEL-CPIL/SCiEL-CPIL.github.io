import time
import random
import numpy as np

def set_sizes(A):
    """Compute sizes of A+A and A*A with equal summands/factors allowed."""
    A = np.array(A, dtype=np.int64)
    n = len(A)
    # Use upper triangle including diagonal for sums and products
    i_idx, j_idx = np.triu_indices(n)
    sums = A[i_idx] + A[j_idx]
    prods = A[i_idx] * A[j_idx]
    return len(np.unique(sums)), len(np.unique(prods))

def objective(A):
    s, p = set_sizes(A)
    return max(s, p)

def construct():
    start = time.time()
    deadline = start + 50.0  # safety margin under 60s
    
    best_set = None
    best_obj = float('inf')
    
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
    
    # Family 1: Pure geometric sequences with integer ratios
    for base in range(2, 11):
        for start_val in [1, 2, 3, 4, 5]:
            A = []
            v = start_val
            for i in range(20):
                A.append(v)
                v *= base
            if max(A) <= 10000 and len(set(A)) == 20:
                try_candidate(A)
    
    # Family 2: Geometric with ratio close to 1 (dense in log space)
    for ratio in [1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.5, 3.0]:
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
    
    # Family 9: Prime-based multiplicative sets
    primes = [2, 3, 5, 7, 11, 13]
    for p in primes:
        for q in primes:
            if p == q:
                continue
            A = []
            for i in range(20):
                val = (p ** min(i, 10)) * (q ** min(20-i, 10))
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
    
    # Family 10: Focused geometric search - optimize base and start for minimal sumset
    # For a geometric sequence a*r^i, the product set has size 2n-1 = 39
    # The sumset size depends on collisions in a*r^i + a*r^j
    # Try many (base, start) combinations systematically
    for base in range(2, 8):
        for start_val in range(1, 50):
            A = []
            v = start_val
            for i in range(20):
                A.append(v)
                v *= base
            if max(A) <= 10000 and len(set(A)) == 20:
                try_candidate(A)
    
    # Now do local search from the best candidate found
    if best_set is None:
        best_set = list(range(1, 21))
        best_obj = objective(best_set)
    
    random.seed(42)
    
    def local_search(start_set, max_iters=50000):
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
            if no_improve > 5000:
                break
    
    # Local search from best candidate
    local_search(best_set, max_iters=30000)
    
    # Also try local search from a few other structured starts
    other_starts = [
        list(range(1, 21)),
        [2**i for i in range(20)] if 2**19 <= 10000 else list(range(1, 21)),
    ]
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
        if sp is best_set:
            continue
        local_search(sp, max_iters=15000)
    
    # Final polish phase: fine-tune each element with small perturbations
    if time.time() < deadline - 5:
        A = list(best_set)
        cur_obj = best_obj
        for _ in range(100):  # multiple passes
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(20):
                if time.time() > deadline - 2:
                    break
                x = A[i]
                # Try small additive and multiplicative perturbations
                for delta in [-3, -2, -1, 1, 2, 3, 5, 10]:
                    nv = x + delta
                    if 1 <= nv <= 10000 and nv not in A:
                        old = A[i]
                        A[i] = nv
                        new_obj = objective(A)
                        if new_obj < cur_obj:
                            cur_obj = new_obj
                            improved = True
                            if new_obj < best_obj:
                                best_obj = new_obj
                                best_set = A[:]
                            break
                        else:
                            A[i] = old
                # Also try multiplicative
                for factor in [0.9, 0.95, 1.05, 1.1]:
                    nv = int(round(x * factor))
                    if 1 <= nv <= 10000 and nv not in A and nv != x:
                        old = A[i]
                        A[i] = nv
                        new_obj = objective(A)
                        if new_obj < cur_obj:
                            cur_obj = new_obj
                            improved = True
                            if new_obj < best_obj:
                                best_obj = new_obj
                                best_set = A[:]
                            break
                        else:
                            A[i] = old
            if not improved:
                break
    
    return best_set
