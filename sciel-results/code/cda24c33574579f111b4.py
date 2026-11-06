import time
import random
import sys

def _check_sidon(S):
    """Check Sidon property: all pair sums (with replacement) are unique."""
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
    """Return set of absolute nonzero differences."""
    d = set()
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(S[i] - S[j])
            if diff > 0:
                d.add(diff)
    return d

def _validate(A, B, max_n=256):
    """Validate solution. Returns (is_valid, score)."""
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

def _backtrack_construct(max_n=256, max_size=64, time_limit=80.0):
    """
    Backtracking search that assigns each integer 1..max_n to A, B, or skip.
    Uses constraint propagation to prune branches.
    """
    start_time = time.time()
    
    # State
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()
    
    best_A = [1]
    best_B = [3]
    best_score = 2
    
    # Precompute for speed
    max_val = max_n
    
    def can_add_to_A(x):
        if len(A) >= max_size:
            return False
        # Check Sidon
        for a in A:
            s = x + a
            if s in sums_A:
                return False
        if x + x in sums_A:
            return False
        # Check diffs disjoint
        for a in A:
            d = abs(x - a)
            if d > 0 and d in diffs_B:
                return False
        return True

    def can_add_to_B(x):
        if len(B) >= max_size:
            return False
        # Check Sidon
        for b in B:
            s = x + b
            if s in sums_B:
                return False
        if x + x in sums_B:
            return False
        # Check diffs disjoint
        for b in B:
            d = abs(x - b)
            if d > 0 and d in diffs_A:
                return False
        return True

    def add_to_A(x):
        nonlocal sums_A, diffs_A
        new_sums = set()
        for a in A:
            new_sums.add(x + a)
        new_sums.add(x + x)
        new_diffs = set()
        for a in A:
            d = abs(x - a)
            if d > 0:
                new_diffs.add(d)
        A.append(x)
        sums_A = sums_A | new_sums
        diffs_A = diffs_A | new_diffs

    def remove_from_A(x):
        nonlocal sums_A, diffs_A
        # This is tricky to undo efficiently, so we'll use recursion with state copies
        # For simplicity in this implementation, we rely on the recursion stack
        pass

    def backtrack(x, depth=0):
        nonlocal best_A, best_B, best_score
        
        if time.time() - start_time > time_limit:
            return False
        
        # Update best if we have a valid solution
        if A and B:
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A = list(A)
                best_B = list(B)
        
        if x > max_val:
            return True
        
        # Option 1: Skip x
        if backtrack(x + 1, depth + 1):
            return True
        
        # Option 2: Add to A
        if can_add_to_A(x):
            # Save state
            old_A = list(A)
            old_sums_A = set(sums_A)
            old_diffs_A = set(diffs_A)
            
            add_to_A(x)
            
            if backtrack(x + 1, depth + 1):
                return True
            
            # Restore state
            A[:] = old_A
            sums_A = old_sums_A
            diffs_A = old_diffs_A
        
        # Option 3: Add to B
        if can_add_to_B(x):
            # Save state
            old_B = list(B)
            old_sums_B = set(sums_B)
            old_diffs_B = set(diffs_B)
            
            # Add to B
            new_sums = set()
            for b in B:
                new_sums.add(x + b)
            new_sums.add(x + x)
            new_diffs = set()
            for b in B:
                d = abs(x - b)
                if d > 0:
                    new_diffs.add(d)
            B.append(x)
            sums_B = sums_B | new_sums
            diffs_B = diffs_B | new_diffs
            
            if backtrack(x + 1, depth + 1):
                return True
            
            # Restore state
            B[:] = old_B
            sums_B = old_sums_B
            diffs_B = old_diffs_B
        
        return False

    # Run backtracking
    backtrack(1)
    
    return best_A, best_B, best_score

def _greedy_construct(max_n=256, max_size=64):
    """
    Greedy construction: iterate 1..max_n, add to A or B if possible.
    """
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()
    
    for x in range(1, max_n + 1):
        # Try A first
        if len(A) < max_size:
            ok = True
            for a in A:
                if (x + a) in sums_A:
                    ok = False
                    break
            if ok and (x + x) in sums_A:
                ok = False
            if ok:
                for a in A:
                    d = abs(x - a)
                    if d > 0 and d in diffs_B:
                        ok = False
                        break
            if ok:
                new_sums = set()
                for a in A:
                    new_sums.add(x + a)
                new_sums.add(x + x)
                new_diffs = set()
                for a in A:
                    d = abs(x - a)
                    if d > 0:
                        new_diffs.add(d)
                A.append(x)
                sums_A = sums_A | new_sums
                diffs_A = diffs_A | new_diffs
                continue
        
        # Try B
        if len(B) < max_size:
            ok = True
            for b in B:
                if (x + b) in sums_B:
                    ok = False
                    break
            if ok and (x + x) in sums_B:
                ok = False
            if ok:
                for b in B:
                    d = abs(x - b)
                    if d > 0 and d in diffs_A:
                        ok = False
                        break
            if ok:
                new_sums = set()
                for b in B:
                    new_sums.add(x + b)
                new_sums.add(x + x)
                new_diffs = set()
                for b in B:
                    d = abs(x - b)
                    if d > 0:
                        new_diffs.add(d)
                B.append(x)
                sums_B = sums_B | new_sums
                diffs_B = diffs_B | new_diffs
    
    return A, B

def _local_search_improve(A, B, max_n=256, max_size=64, time_budget=10.0, rng=None):
    """
    Local search: try swapping elements to improve total size.
    """
    if rng is None:
        rng = random.Random(0)
    
    start = time.time()
    A = list(A)
    B = list(B)
    best_A, best_B = list(A), list(B)
    best_score = len(A) + len(B)
    
    used = set(A) | set(B)
    available = [x for x in range(1, max_n + 1) if x not in used]
    
    while time.time() - start < time_budget:
        # Try adding to A
        if available and len(A) < max_size:
            for _ in range(10):
                if not available:
                    break
                idx = rng.randrange(len(available))
                x = available[idx]
                A_new = A + [x]
                A_new.sort()
                if _check_sidon(A_new):
                    dA_new = _diffs(A_new)
                    dB = _diffs(B)
                    if not (dA_new & dB):
                        A = A_new
                        best_A = list(A)
                        best_score = len(A) + len(B)
                        available.pop(idx)
                        used.add(x)
                        break
        
        # Try adding to B
        if available and len(B) < max_size:
            for _ in range(10):
                if not available:
                    break
                idx = rng.randrange(len(available))
                x = available[idx]
                B_new = B + [x]
                B_new.sort()
                if _check_sidon(B_new):
                    dB_new = _diffs(B_new)
                    dA = _diffs(A)
                    if not (dB_new & dA):
                        B = B_new
                        best_B = list(B)
                        best_score = len(A) + len(B)
                        available.pop(idx)
                        used.add(x)
                        break
        
        # Try swap in A
        if A and available:
            i = rng.randrange(len(A))
            old = A[i]
            A_tmp = A[:i] + A[i+1:]
            for _ in range(5):
                if not available:
                    break
                idx = rng.randrange(len(available))
                x = available[idx]
                if x == old:
                    continue
                A_new = A_tmp + [x]
                A_new.sort()
                if _check_sidon(A_new):
                    dA_new = _diffs(A_new)
                    dB = _diffs(B)
                    if not (dA_new & dB):
                        A = A_new
                        best_A = list(A)
                        best_score = len(A) + len(B)
                        available.pop(idx)
                        used.discard(old)
                        used.add(x)
                        available.append(old)
                        break
        
        # Try swap in B
        if B and available:
            i = rng.randrange(len(B))
            old = B[i]
            B_tmp = B[:i] + B[i+1:]
            for _ in range(5):
                if not available:
                    break
                idx = rng.randrange(len(available))
                x = available[idx]
                if x == old:
                    continue
                B_new = B_tmp + [x]
                B_new.sort()
                if _check_sidon(B_new):
                    dB_new = _diffs(B_new)
                    dA = _diffs(A)
                    if not (dB_new & dA):
                        B = B_new
                        best_B = list(B)
                        best_score = len(A) + len(B)
                        available.pop(idx)
                        used.discard(old)
                        used.add(x)
                        available.append(old)
                        break
    
    return best_A, best_B, best_score

def construct():
    """
    Main function: backtracking search with local search improvement.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Phase 1: Greedy construction for initial solution
    A_greedy, B_greedy = _greedy_construct(256, 64)
    valid, score_greedy = _validate(A_greedy, B_greedy)
    
    best_A, best_B = list(A_greedy), list(B_greedy)
    best_score = score_greedy if valid else 0
    
    # Phase 2: Backtracking search
    if time.time() < deadline - 20.0:
        bt_budget = min(30.0, deadline - 20.0 - time.time())
        A_bt, B_bt, score_bt = _backtrack_construct(256, 64, bt_budget)
        valid_bt, score_bt_val = _validate(A_bt, B_bt)
        if valid_bt and score_bt_val > best_score:
            best_score = score_bt_val
            best_A, best_B = list(A_bt), list(B_bt)
    
    # Phase 3: Local search improvement
    if time.time() < deadline - 5.0:
        for seed in range(3):
            if time.time() > deadline - 3.0:
                break
            rng = random.Random(1000 + seed)
            ls_budget = min(10.0, deadline - 3.0 - time.time())
            A_ls, B_ls, score_ls = _local_search_improve(
                best_A, best_B, max_n=256, max_size=64,
                time_budget=ls_budget, rng=rng
            )
            valid_ls, score_ls_val = _validate(A_ls, B_ls)
            if valid_ls and score_ls_val > best_score:
                best_score = score_ls_val
                best_A, best_B = list(A_ls), list(B_ls)
    
    # Final validation
    if not best_A or not best_B:
        best_A, best_B = [1], [3]
    
    return [best_A, best_B]
