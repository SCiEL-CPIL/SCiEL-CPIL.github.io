import time
import random
import numpy as np

def set_sizes(A):
    """Compute sizes of A+A and A*A with equal summands/factors allowed using NumPy."""
    A = np.array(A, dtype=np.int64)
    n = len(A)
    # Upper triangle including diagonal for sums and products
    i_idx, j_idx = np.triu_indices(n, k=0)
    sums = A[i_idx] + A[j_idx]
    prods = A[i_idx] * A[j_idx]
    return len(np.unique(sums)), len(np.unique(prods))

def objective(A):
    s, p = set_sizes(A)
    return max(s, p)

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin under 60s
    
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
    for base in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        for start_val in [1, 2, 3, 4, 5]:
            A = []
            v = start_val
            for i in range(20):
                A.append(v)
                v *= base
            if max(A) <= 10000 and len(set(A)) == 20:
                try_candidate(A)
    
    # Family 2: Geometric with rational ratios (p/q)
    for p in [3, 4, 5, 6, 7, 8, 9, 10, 12, 15]:
        for q in [2, 3, 4, 5]:
            if p <= q:
                continue
            ratio = p / q
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
    
    # Family 3: Sparse geometric with larger steps
    for base in [2, 3, 4, 5]:
        for step in [1, 2, 3, 4, 5]:
            A = [base**(i*step) for i in range(20)]
            if max(A) <= 10000 and len(set(A)) == 20:
                try_candidate(A)
    
    # Family 4: Mixed geometric + arithmetic
    for base in [2, 3, 4]:
        for n_geo in [8, 10, 12]:
            A = [base**i for i in range(n_geo)]
            last = base**(n_geo-1)
            for i in range(20 - n_geo):
                A.append(last * (i + 2))
            seen = set()
            uniq = []
            for x in A:
                if x not in seen and 1 <= x <= 10000:
                    seen.add(x)
                    uniq.append(x)
            if len(uniq) == 20:
                try_candidate(uniq)
    
    if best_set is None:
        best_set = list(range(1, 21))
        best_obj = objective(best_set)
    
    random.seed(42)
    
    def local_search(start_set, max_iters=100000):
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
            # Wider neighborhood for exploration
            for factor in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0]:
                nv = int(round(x * factor))
                if 1 <= nv <= 10000:
                    cands.add(nv)
            for delta in [-20, -10, -5, -2, -1, 1, 2, 5, 10, 20, 50, 100]:
                nv = x + delta
                if 1 <= nv <= 10000:
                    cands.add(nv)
            # Add a few random candidates
            for _ in range(3):
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
            if no_improve > 8000:
                break
    
    # Run local search from best candidate
    local_search(best_set, max_iters=80000)
    
    # Try additional starting points
    other_starts = [
        list(range(1, 21)),
    ]
    # Add geometric starts
    for base in [2, 3]:
        A = []
        v = 1
        for i in range(20):
            A.append(v)
            v *= base
        if max(A) <= 10000 and len(set(A)) == 20:
            other_starts.append(A)
    
    for sp in other_starts:
        if time.time() > deadline - 5:
            break
        if sp is best_set:
            continue
        local_search(sp, max_iters=40000)
    
    # Final polish: tighter neighborhood for refinement
    if time.time() < deadline - 2:
        A = list(best_set)
        cur_obj = objective(A)
        no_improve = 0
        it = 0
        while it < 100000 and no_improve < 3000 and time.time() < deadline - 1:
            it += 1
            i = random.randrange(20)
            x = A[i]
            cands = set()
            # Very tight neighborhood for final polish
            for factor in [0.9, 0.95, 1.05, 1.1]:
                nv = int(round(x * factor))
                if 1 <= nv <= 10000:
                    cands.add(nv)
            for delta in [-2, -1, 1, 2]:
                nv = x + delta
                if 1 <= nv <= 10000:
                    cands.add(nv)
            cands.discard(x)
            cands = [c for c in cands if c not in A]
            if not cands:
                no_improve += 1
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
    
    return best_set
