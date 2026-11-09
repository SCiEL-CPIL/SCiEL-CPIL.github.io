import time
import random
import numpy as np

N = 256
MAX_EACH = 64

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill
    
    best_A = None
    best_B = None
    best_score = 0
    
    def is_sidon(lst):
        """Check if list is a Sidon set (pair sums including doubles are unique)."""
        if not lst:
            return True
        sums = set()
        for i in range(len(lst)):
            for j in range(i, len(lst)):
                s = lst[i] + lst[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def get_diffs(lst):
        """Get nonzero absolute differences."""
        diffs = set()
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                d = abs(lst[i] - lst[j])
                if d > 0:
                    diffs.add(d)
        return diffs
    
    def validate(A, B):
        """Validate both sets are Sidon and have disjoint differences."""
        if not A or not B:
            return False
        if len(A) > MAX_EACH or len(B) > MAX_EACH:
            return False
        if not is_sidon(A):
            return False
        if not is_sidon(B):
            return False
        diffs_A = get_diffs(A)
        diffs_B = get_diffs(B)
        if diffs_A & diffs_B:
            return False
        return True
    
    def update_best(A, B):
        nonlocal best_A, best_B, best_score
        if validate(A, B):
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A = A[:]
                best_B = B[:]
    
    def build_sidon_greedy(candidates, blocked_diffs, blocked_sums, max_size, rng):
        """Build a Sidon set greedily, avoiding blocked differences and sums."""
        result = []
        used_sums = set()
        used_diffs = set()
        
        for x in candidates:
            if len(result) >= max_size:
                break
            if time.time() > deadline:
                break
            
            # Check Sidon property
            valid = True
            for a in result:
                if (x + a) in used_sums:
                    valid = False
                    break
            if valid and (2 * x) in used_sums:
                valid = False
            
            if not valid:
                continue
            
            # Check difference constraints
            valid_diff = True
            for a in result:
                d = abs(x - a)
                if d > 0 and (d in blocked_diffs or d in used_diffs):
                    valid_diff = False
                    break
            
            if not valid_diff:
                continue
            
            # Accept
            result.append(x)
            for a in result:
                used_sums.add(x + a)
            used_sums.add(2 * x)
            for a in result:
                d = abs(x - a)
                if d > 0:
                    used_diffs.add(d)
        
        return result, used_sums, used_diffs
    
    def local_search(A, B, rng, max_iters=500):
        """Improve solution via local search: try removing/adding/swapping elements."""
        A = A[:]
        B = B[:]
        improved = True
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Try adding to A
            if len(A) < MAX_EACH:
                candidates = [x for x in range(1, N+1) if x not in A]
                rng.shuffle(candidates)
                for x in candidates[:20]:
                    new_A = A + [x]
                    if is_sidon(new_A):
                        d_new = get_diffs(new_A) - get_diffs(A)
                        if not (d_new & get_diffs(B)):
                        # Check full difference disjointness
                            if not (get_diffs(new_A) & get_diffs(B)):
                                A = new_A
                                improved = True
                                break
            
            # Try adding to B
            if len(B) < MAX_EACH:
                candidates = [x for x in range(1, N+1) if x not in B]
                rng.shuffle(candidates)
                for x in candidates[:20]:
                    new_B = B + [x]
                    if is_sidon(new_B):
                        if not (get_diffs(A) & get_diffs(new_B)):
                            B = new_B
                            improved = True
                            break
            
            # Try swapping: remove one from A, add one to A
            if len(A) > 1:
                idx = rng.randint(0, len(A)-1)
                old = A[idx]
                candidates = [x for x in range(1, N+1) if x not in A and x != old]
                rng.shuffle(candidates)
                for x in candidates[:15]:
                    new_A = [a for i, a in enumerate(A) if i != idx] + [x]
                    new_A.sort()
                    if is_sidon(new_A):
                        if not (get_diffs(new_A) & get_diffs(B)):
                            A = new_A
                            improved = True
                            break
            
            # Try swapping in B
            if len(B) > 1:
                idx = rng.randint(0, len(B)-1)
                old = B[idx]
                candidates = [x for x in range(1, N+1) if x not in B and x != old]
                rng.shuffle(candidates)
                for x in candidates[:15]:
                    new_B = [b for i, b in enumerate(B) if i != idx] + [x]
                    new_B.sort()
                    if is_sidon(new_B):
                        if not (get_diffs(A) & get_diffs(new_B)):
                            B = new_B
                            improved = True
                            break
        
        return A, B
    
    # Strategy 1: Structured construction using modular patterns
    def structured_build(seed):
        rng = random.Random(seed)
        
        # Build A using a base pattern
        A = []
        used_sums_A = set()
        used_diffs_A = set()
        
        # Start with sparse elements
        start = rng.randint(1, 5)
        step = rng.randint(3, 6)
        
        # Phase 1: Add elements with large gaps to minimize difference conflicts
        for x in range(start, N+1, step):
            if len(A) >= MAX_EACH:
                break
            if time.time() > deadline:
                break
            
            valid = True
            for a in A:
                if (x + a) in used_sums_A:
                    valid = False
                    break
            if valid and (2*x) in used_sums_A:
                valid = False
            
            if valid:
                valid_diff = True
                for a in A:
                    d = abs(x - a)
                    if d in used_diffs_A:
                        valid_diff = False
                        break
                if valid_diff:
                    A.append(x)
                    for a in A:
                        used_sums_A.add(x + a)
                    used_sums_A.add(2*x)
                    for a in A:
                        d = abs(x - a)
                        if d > 0:
                            used_diffs_A.add(d)
        
        # Phase 2: Fill in gaps
        candidates = [x for x in range(1, N+1) if x not in A]
        rng.shuffle(candidates)
        for x in candidates:
            if len(A) >= MAX_EACH:
                break
            if time.time() > deadline:
                break
            
            valid = True
            for a in A:
                if (x + a) in used_sums_A:
                    valid = False
                    break
            if valid and (2*x) in used_sums_A:
                valid = False
            
            if valid:
                valid_diff = True
                for a in A:
                    d = abs(x - a)
                    if d in used_diffs_A:
                        valid_diff = False
                        break
                if valid_diff:
                    A.append(x)
                    for a in A:
                        used_sums_A.add(x + a)
                    used_sums_A.add(2*x)
                    for a in A:
                        d = abs(x - a)
                        if d > 0:
                            used_diffs_A.add(d)
        
        # Build B avoiding A's differences
        B = []
        used_sums_B = set()
        used_diffs_B = set()
        
        candidates_B = [x for x in range(1, N+1)]
        rng.shuffle(candidates_B)
        
        for x in candidates_B:
            if len(B) >= MAX_EACH:
                break
            if time.time() > deadline:
                break
            
            valid = True
            for b in B:
                if (x + b) in used_sums_B:
                    valid = False
                    break
            if valid and (2*x) in used_sums_B:
                valid = False
            
            if valid:
                valid_diff = True
                for b in B:
                    d = abs(x - b)
                    if d > 0 and (d in used_diffs_A or d in used_diffs_B):
                        valid_diff = False
                        break
                if valid_diff:
                    B.append(x)
                    for b in B:
                        used_sums_B.add(x + b)
                    used_sums_B.add(2*x)
                    for b in B:
                        d = abs(x - b)
                        if d > 0:
                            used_diffs_B.add(d)
        
        return A, B
    
    # Strategy 2: Interleaved with difference tracking
    def interleaved_build(seed):
        rng = random.Random(seed)
        
        A = []
        B = []
        used_sums_A = set()
        used_sums_B = set()
        used_diffs_A = set()
        used_diffs_B = set()
        
        candidates = list(range(1, N+1))
        rng.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            # Decide which set to try
            if len(A) < len(B) - 2:
                order = [0]
            elif len(B) < len(A) - 2:
                order = [1]
            else:
                order = [0, 1] if rng.random() < 0.5 else [1, 0]
            
            for side in order:
                if side == 0 and len(A) < MAX_EACH:
                    valid = True
                    for a in A:
                        if (x + a) in used_sums_A:
                            valid = False
                            break
                    if valid and (2*x) in used_sums_A:
                        valid = False
                    
                    if valid:
                        valid_diff = True
                        for a in A:
                            d = abs(x - a)
                            if d > 0 and (d in used_diffs_A or d in used_diffs_B):
                                valid_diff = False
                                break
                        if valid_diff:
                            A.append(x)
                            for a in A:
                                used_sums_A.add(x + a)
                            used_sums_A.add(2*x)
                            for a in A:
                                d = abs(x - a)
                                if d > 0:
                                    used_diffs_A.add(d)
                            break
                elif side == 1 and len(B) < MAX_EACH:
                    valid = True
                    for b in B:
                        if (x + b) in used_sums_B:
                            valid = False
                            break
                    if valid and (2*x) in used_sums_B:
                        valid = False
                    
                    if valid:
                        valid_diff = True
                        for b in B:
                            d = abs(x - b)
                            if d > 0 and (d in used_diffs_A or d in used_diffs_B):
                                valid_diff = False
                                break
                        if valid_diff:
                            B.append(x)
                            for b in B:
                                used_sums_B.add(x + b)
                            used_sums_B.add(2*x)
                            for b in B:
                                d = abs(x - b)
                                if d > 0:
                                    used_diffs_B.add(d)
                            break
        
        return A, B
    
    # Run multiple strategies
    for seed in range(300):
        if time.time() > deadline:
            break
        
        # Alternate between strategies
        if seed % 2 == 0:
            A, B = structured_build(seed)
        else:
            A, B = interleaved_build(seed)
        
        if A and B:
            update_best(A, B)
        
        # Apply local search to promising solutions
        if A and B and len(A) + len(B) >= best_score - 5:
            A_ls, B_ls = local_search(A, B, random.Random(seed), max_iters=100)
            update_best(A_ls, B_ls)
    
    # Final local search on best solution
    if best_A and best_B and time.time() < deadline:
        A_final, B_final = local_search(best_A, best_B, random.Random(999), max_iters=300)
        update_best(A_final, B_final)
    
    # Fallback
    if best_A is None or best_B is None:
        best_A = [1]
        best_B = [4]
    
    return [best_A, best_B]
