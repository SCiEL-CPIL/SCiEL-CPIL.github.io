import time
import random
import sys

def _check_sidon(S):
    seen = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _diffs(S):
    d = set()
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(S[i] - S[j])
            if diff > 0:
                d.add(diff)
    return d

def _validate(A, B, max_n=256):
    if not A or not B:
        return False, 0
    if len(A) > 64 or len(B) > 64:
        return False, 0
    for x in A:
        if x < 1 or x > max_n:
            return False, 0
    for x in B:
        if x < 1 or x > max_n:
            return False, 0
    if not _check_sidon(A):
        return False, 0
    if not _check_sidon(B):
        return False, 0
    dA = _diffs(A)
    dB = _diffs(B)
    if dA & dB:
        return False, 0
    return True, len(A) + len(B)

def _greedy_seeded(seed_A, seed_B, max_n=256, max_size=64, rng=None):
    """
    Greedy build starting from seeds. Try to add remaining numbers to A or B
    to maximize combined size.
    """
    if rng is None:
        rng = random.Random(0)
    
    A = list(seed_A)
    B = list(seed_B)
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()
    
    for a in A:
        sums_A.add(2 * a)
    for a in A:
        for b in A:
            if a != b:
                sums_A.add(a + b)
            diffs_A.add(abs(a - b))
    for b in B:
        sums_B.add(2 * b)
    for b in B:
        for c in B:
            if b != c:
                sums_B.add(b + c)
            diffs_B.add(abs(b - c))
    
    # Remove any invalid seed (shouldn't happen if seeds are valid)
    if not _check_sidon(A) or not _check_sidon(B) or (diffs_A & diffs_B):
        return A, B
    
    used = set(A) | set(B)
    candidates = [x for x in range(1, max_n + 1) if x not in used]
    rng.shuffle(candidates)
    
    for x in candidates:
        # Try A
        if len(A) < max_size:
            ok = True
            for a in A:
                if (x + a) in sums_A:
                    ok = False
                    break
            if ok and (2 * x) in sums_A:
                ok = False
            if ok:
                for a in A:
                    d = abs(x - a)
                    if d > 0 and d in diffs_B:
                        ok = False
                        break
            if ok:
                for a in A:
                    sums_A.add(x + a)
                sums_A.add(2 * x)
                for a in A:
                    d = abs(x - a)
                    if d > 0:
                        diffs_A.add(d)
                A.append(x)
                continue
        
        # Try B
        if len(B) < max_size:
            ok = True
            for b in B:
                if (x + b) in sums_B:
                    ok = False
                    break
            if ok and (2 * x) in sums_B:
                ok = False
            if ok:
                for b in B:
                    d = abs(x - b)
                    if d > 0 and d in diffs_A:
                        ok = False
                        break
            if ok:
                for b in B:
                    sums_B.add(x + b)
                sums_B.add(2 * x)
                for b in B:
                    d = abs(x - b)
                    if d > 0:
                        diffs_B.add(d)
                B.append(x)
    
    return A, B

def _backtrack_construct(max_n=256, max_size=64, time_budget=30.0, rng=None):
    """
    Joint backtracking: assign each x in 1..max_n to A, B, or skip.
    Use constraint propagation and freedom-maximizing branching.
    """
    if rng is None:
        rng = random.Random(42)
    
    start_time = time.time()
    
    # State
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()
    assigned = [None] * (max_n + 1)  # None, 'A', 'B'
    
    best = [0, 0, list(A), list(B)]  # [score, time, A, B]
    
    # Precompute for each x, the potential conflicts
    # We'll do DFS with backtracking
    
    def can_add_A(x):
        if len(A) >= max_size:
            return False
        if (2 * x) in sums_A:
            return False
        for a in A:
            if (x + a) in sums_A:
                return False
            d = abs(x - a)
            if d > 0 and d in diffs_B:
                return False
        return True
    
    def can_add_B(x):
        if len(B) >= max_size:
            return False
        if (2 * x) in sums_B:
            return False
        for b in B:
            if (x + b) in sums_B:
                return False
            d = abs(x - b)
            if d > 0 and d in diffs_A:
                return False
        return True
    
    def add_A(x):
        A.append(x)
        assigned[x] = 'A'
        for a in A[:-1]:
            sums_A.add(x + a)
            d = abs(x - a)
            if d > 0:
                diffs_A.add(d)
        sums_A.add(2 * x)
    
    def remove_A(x):
        A.remove(x)
        assigned[x] = None
        # Rebuild sums_A and diffs_A (expensive but simple)
        sums_A.clear()
        diffs_A.clear()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums_A.add(A[i] + A[j])
            for j in range(len(A)):
                if i != j:
                    d = abs(A[i] - A[j])
                    if d > 0:
                        diffs_A.add(d)
    
    def add_B(x):
        B.append(x)
        assigned[x] = 'B'
        for b in B[:-1]:
            sums_B.add(x + b)
            d = abs(x - b)
            if d > 0:
                diffs_B.add(d)
        sums_B.add(2 * x)
    
    def remove_B(x):
        B.remove(x)
        assigned[x] = None
        sums_B.clear()
        diffs_B.clear()
        for i in range(len(B)):
            for j in range(i, len(B)):
                sums_B.add(B[i] + B[j])
            for j in range(len(B)):
                if i != j:
                    d = abs(B[i] - B[j])
                    if d > 0:
                        diffs_B.add(d)
    
    # DFS
    def dfs(x):
        if time.time() - start_time > time_budget:
            return
        
        if x > max_n:
            score = len(A) + len(B)
            if score > best[0]:
                best[0] = score
                best[1] = time.time()
                best[2] = list(A)
                best[3] = list(B)
            return
        
        # Try adding to A
        if can_add_A(x):
            add_A(x)
            dfs(x + 1)
            remove_A(x)
            if time.time() - start_time > time_budget:
                return
        
        # Try adding to B
        if can_add_B(x):
            add_B(x)
            dfs(x + 1)
            remove_B(x)
            if time.time() - start_time > time_budget:
                return
        
        # Skip
        dfs(x + 1)
    
    # This pure DFS is too slow for 256 elements. Use a hybrid:
    # Greedy to get a good initial solution, then local search.
    
    # Initial greedy
    A0, B0 = _greedy_seeded([], [], max_n, max_size, rng)
    score0 = len(A0) + len(B0)
    
    if score0 > best[0]:
        best[0] = score0
        best[2] = list(A0)
        best[3] = list(B0)
    
    # Local search: try to improve by swapping or inserting
    A_ls = list(A0)
    B_ls = list(B0)
    
    for iteration in range(200):
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        
        # Try inserting unassigned elements
        used = set(A_ls) | set(B_ls)
        candidates = [x for x in range(1, max_n + 1) if x not in used]
        rng.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > time_budget:
                break
            
            # Try A
            if len(A_ls) < max_size:
                ok = True
                sums_A_temp = set()
                diffs_A_temp = set()
                for a in A_ls:
                    sums_A_temp.add(a + a)
                    for b in A_ls:
                        if a != b:
                            sums_A_temp.add(a + b)
                            d = abs(a - b)
                            if d > 0:
                                diffs_A_temp.add(d)
                
                if (2 * x) not in sums_A_temp:
                    for a in A_ls:
                        if (x + a) in sums_A_temp:
                            ok = False
                            break
                        d = abs(x - a)
                        if d > 0 and d in _diffs(B_ls):
                            ok = False
                            break
                
                if ok:
                    A_ls.append(x)
                    improved = True
                    break
            
            # Try B
            if len(B_ls) < max_size:
                ok = True
                sums_B_temp = set()
                diffs_B_temp = set()
                for b in B_ls:
                    sums_B_temp.add(b + b)
                    for c in B_ls:
                        if b != c:
                            sums_B_temp.add(b + c)
                            d = abs(b - c)
                            if d > 0:
                                diffs_B_temp.add(d)
                
                if (2 * x) not in sums_B_temp:
                    for b in B_ls:
                        if (x + b) in sums_B_temp:
                            ok = False
                            break
                        d = abs(x - b)
                        if d > 0 and d in _diffs(A_ls):
                            ok = False
                            break
                
                if ok:
                    B_ls.append(x)
                    improved = True
                    break
        
        # Try swapping: replace an element in A with one in B or vice versa
        if not improved:
            # Try to find a swap that increases size
            pass
        
        if not improved:
            break
        
        score_ls = len(A_ls) + len(B_ls)
        if score_ls > best[0]:
            best[0] = score_ls
            best[2] = list(A_ls)
            best[3] = list(B_ls)
    
    return best[2], best[3]

def _local_search_improve(A, B, max_n=256, max_size=64, time_budget=20.0, rng=None):
    """
    Local search: try to insert unassigned elements, swap elements, or reassign.
    """
    if rng is None:
        rng = random.Random(99)
    
    start_time = time.time()
    A = list(A)
    B = list(B)
    
    for _ in range(500):
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        
        # Try inserting
        used = set(A) | set(B)
        candidates = [x for x in range(1, max_n + 1) if x not in used]
        rng.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > time_budget:
                break
            
            # Try A
            if len(A) < max_size:
                ok = True
                for a in A:
                    if (x + a) in _sums_set(A) or (2 * x) in _sums_set(A):
                        ok = False
                        break
                if ok:
                    for a in A:
                        d = abs(x - a)
                        if d > 0 and d in _diffs(B):
                            ok = False
                            break
                if ok:
                    A.append(x)
                    improved = True
                    break
            
            # Try B
            if len(B) < max_size:
                ok = True
                for b in B:
                    if (x + b) in _sums_set(B) or (2 * x) in _sums_set(B):
                        ok = False
                        break
                if ok:
                    for b in B:
                        d = abs(x - b)
                        if d > 0 and d in _diffs(A):
                            ok = False
                            break
                if ok:
                    B.append(x)
                    improved = True
                    break
        
        if not improved:
            break
    
    return A, B

def _sums_set(S):
    s = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            s.add(S[i] + S[j])
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_A = [1]
    best_B = [3]
    best_score = 2
    
    # Phase 1: Backtracking with greedy init
    if time.time() < deadline - 30.0:
        A, B = _backtrack_construct(256, 64, 25.0, random.Random(42))
        valid, score = _validate(A, B)
        if valid and score > best_score:
            best_score = score
            best_A, best_B = list(A), list(B)
    
    # Phase 2: Local search
    if time.time() < deadline - 10.0:
        A_ls, B_ls = _local_search_improve(best_A, best_B, 256, 64, 20.0, random.Random(77))
        valid, score = _validate(A_ls, B_ls)
        if valid and score > best_score:
            best_score = score
            best_A, best_B = list(A_ls), list(B_ls)
    
    # Final validation
    if not best_A or not best_B:
        best_A, best_B = [1], [3]
    
    return [best_A, best_B]
