import time
import random
import numpy as np

def set_sizes(A):
    """Compute sizes of A+A and A*A with equal summands/factors allowed."""
    A = np.array(A, dtype=np.int64)
    n = len(A)
    i_idx, j_idx = np.triu_indices(n)
    sums = A[i_idx] + A[j_idx]
    prods = A[i_idx] * A[j_idx]
    return len(np.unique(sums)), len(np.unique(prods))

def objective(A):
    s, p = set_sizes(A)
    return max(s, p)

def construct():
    start = time.time()
    deadline = start + 45.0  # safety margin under 60s
    
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
            best_set = list(A)
    
    # === Phase 1: Structured candidates (fast) ===
    
    # Pure geometric sequences
    for base in range(2, 12):
        for start_val in range(1, 20):
            A = []
            v = start_val
            for i in range(20):
                A.append(v)
                v *= base
            if max(A) <= 10000 and len(set(A)) == 20:
                try_candidate(A)
    
    # Log-uniform spreads
    for log_min in [0, 0.3, 0.5, 0.7]:
        for log_max in [7.0, 8.0, 8.5, 9.0, 9.5, 10.0]:
            if log_max <= log_min:
                continue
            A = []
            for i in range(20):
                t = i / 19.0
                log_val = log_min + t * (log_max - log_min)
                val = int(round(np.exp(log_val)))
                val = max(1, min(10000, val))
                A.append(val)
            if len(set(A)) == 20:
                try_candidate(A)
    
    # Geometric with ratio close to 1
    for ratio in [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0]:
        for start_val in [1, 2, 3, 5, 10]:
            A = []
            v = float(start_val)
            for i in range(20):
                val = int(round(v))
                val = max(1, min(10000, val))
                A.append(val)
                v *= ratio
            seen = set()
            uniq = []
            for x in A:
                if x not in seen:
                    seen.add(x)
                    uniq.append(x)
            if len(uniq) == 20:
                try_candidate(uniq)
    
    # Mixed geometric: powers of small base times small multipliers
    for base in [2, 3, 4]:
        for mult_set in [[1,2,3,5,7,11], [1,2,4,8,16,32], [1,3,5,7,9,11]]:
            A = []
            for i in range(20):
                p = base ** min(i, 12)
                m = mult_set[i % len(mult_set)]
                val = p * m
                if val > 10000:
                    val = 10000
                A.append(val)
            seen = set()
            uniq = []
            for x in A:
                if x not in seen:
                    seen.add(x)
                    uniq.append(x)
            if len(uniq) == 20:
                try_candidate(uniq)
    
    # Two interleaved geometric
    for b1 in [2, 3, 4, 5]:
        for b2 in [3, 4, 5, 7]:
            if b1 == b2:
                continue
            A1 = [b1**i for i in range(10)]
            A2 = [b2**i for i in range(10)]
            A = A1 + A2
            if max(A) <= 10000 and len(set(A)) == 20:
                try_candidate(A)
    
    # === Phase 2: Greedy constructive with multiple strategies ===
    
    random.seed(123)
    
    def greedy_construct(strategy, seed_val=0):
        nonlocal best_set, best_obj
        rng = random.Random(seed_val)
        used = set()
        sums = set()
        prods = set()
        A = []
        
        # Choose initial elements based on strategy
        if strategy == 0:  # Start with small integers
            init_vals = list(range(1, 11))
            rng.shuffle(init_vals)
        elif strategy == 1:  # Start with geometric
            base = rng.choice([2, 3, 4])
            init_vals = [base**i for i in range(10) if base**i <= 10000]
            if len(init_vals) < 10:
                for v in range(1, 100):
                    if v not in init_vals:
                        init_vals.append(v)
                    if len(init_vals) >= 10:
                        break
        elif strategy == 2:  # Start with log-uniform
            log_min, log_max = 0.5, 9.0
            init_vals = []
            for i in range(10):
                t = i / 9.0
                val = int(round(np.exp(log_min + t * (log_max - log_min))))
                val = max(1, min(10000, val))
                init_vals.append(val)
        else:  # Random start
            init_vals = rng.sample(range(1, 10001), 10)
        
        for v in init_vals:
            if v in used:
                continue
            used.add(v)
            A.append(v)
            sums.add(v + v)
            prods.add(v * v)
        
        A.sort()
        
        # Greedily add remaining 10 elements
        for step in range(10):
            if time.time() > deadline - 10:
                break
            
            best_val = None
            best_score = float('inf')
            
            # Generate candidates
            cands = set()
            # Multiplicative neighbors of existing elements
            for a in A:
                for f in [2, 3, 4, 5, 0.5, 0.7, 1.5]:
                    nv = int(round(a * f))
                    if 1 <= nv <= 10000:
                        cands.add(nv)
            # Additive neighbors
            for a in A:
                for d in [-10, -5, -2, -1, 1, 2, 5, 10, 50]:
                    nv = a + d
                    if 1 <= nv <= 10000:
                        cands.add(nv)
            # Random candidates
            for _ in range(30):
                cands.add(rng.randint(1, 10000))
            # Small integers
            for v in range(1, 50):
                cands.add(v)
            
            cands -= used
            
            # Score each candidate
            scored_cands = []
            for c in list(cands)[:200]:  # Limit evaluation
                new_sums = len(sums) + sum(1 for a in A if (a + c) not in sums and (a + c) != c + c) + (1 if (c+c) not in sums else 0)
                new_prods = len(prods) + sum(1 for a in A if (a * c) not in prods and (a * c) != c * c) + (1 if (c*c) not in prods else 0)
                score = max(new_sums, new_prods)
                scored_cands.append((score, c))
            
            if not scored_cands:
                break
            
            scored_cands.sort()
            # Pick from top candidates with some randomness
            top_n = min(5, len(scored_cands))
            chosen_score, chosen_val = scored_cands[rng.randint(0, top_n - 1)]
            
            A.append(chosen_val)
            used.add(chosen_val)
            for a in A[:-1]:
                sums.add(a + chosen_val)
                prods.add(a * chosen_val)
            sums.add(chosen_val + chosen_val)
            prods.add(chosen_val * chosen_val)
            A.sort()
        
        if len(A) == 20:
            try_candidate(A)
    
    # Run multiple greedy constructions
    for strat in range(4):
        for i in range(15):
            if time.time() > deadline - 15:
                break
            greedy_construct(strat, seed_val=1000 + strat * 100 + i)
    
    # === Phase 3: Local search ===
    
    def local_search(start_set, max_iters=20000):
        nonlocal best_set, best_obj
        A = list(start_set)
        cur_obj = objective(A)
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_set = list(A)
        no_improve = 0
        it = 0
        while it < max_iters:
            if time.time() > deadline - 3:
                return
            it += 1
            i = random.randrange(20)
            x = A[i]
            cands = set()
            for factor in [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0]:
                nv = int(round(x * factor))
                if 1 <= nv <= 10000:
                    cands.add(nv)
            for delta in [-10, -5, -2, -1, 1, 2, 5, 10, 50, 100]:
                nv = x + delta
                if 1 <= nv <= 10000:
                    cands.add(nv)
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
                    best_set = list(A)
            else:
                A[i] = old
                no_improve += 1
            if no_improve > 8000:
                break
    
    if best_set is not None:
        local_search(best_set, max_iters=30000)
    
    # Local search from structured starts
    for base in [2, 3, 4]:
        A = []
        v = 1
        for i in range(20):
            A.append(v)
            v *= base
        if max(A) <= 10000 and len(set(A)) == 20:
            if time.time() < deadline - 5:
                local_search(A, max_iters=10000)
    
    # === Phase 4: Final polish ===
    if best_set is not None and time.time() < deadline - 3:
        A = list(best_set)
        cur_obj = best_obj
        for _ in range(200):
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(20):
                if time.time() > deadline - 2:
                    break
                x = A[i]
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
                                best_set = list(A)
                            break
                        else:
                            A[i] = old
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
                                best_set = list(A)
                            break
                        else:
                            A[i] = old
            if not improved:
                break
    
    if best_set is None:
        best_set = list(range(1, 21))
    
    return best_set
