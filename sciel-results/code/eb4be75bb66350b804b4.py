import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # Target: differences 1..30 must all appear
    target_mask = 0
    for d in range(1, 31):
        target_mask |= (1 << d)
    
    # We'll search for minimal D
    # For a Sidon set of size k, we have k*(k-1)//2 differences
    # Need at least 30, so k>=9 (9*8//2=36)
    # For k=9, D>=36 (30 diffs in 1..30, 6 diffs in 31..D, so D>=36)
    # For k=10, D>=45
    
    best_result = None
    best_D = float('inf')
    
    def try_dfs(D, max_k=12):
        nonlocal best_result, best_D
        if time.time() > deadline:
            return False
        
        # DFS: build set in increasing order, starting with 0
        # State: current list, diff_mask (tracks all diffs up to D), last element
        # Use bitmask for diffs 1..D (D <= ~50, so bitmask fits in int)
        
        current = [0]
        diff_mask = 0  # bit d set if difference d is used
        
        def dfs():
            nonlocal best_result, best_D
            if time.time() > deadline:
                return True  # timeout signal
            
            # Check if all target diffs covered
            if (diff_mask & target_mask) == target_mask:
                D_cur = current[-1]
                if D_cur < best_D:
                    best_D = D_cur
                    best_result = list(current)
                return True
            
            n = len(current)
            last = current[-1]
            
            # Pruning: if last >= best_D, no improvement possible
            if last >= best_D:
                return False
            
            # Pruning: can we cover remaining diffs?
            missing = target_mask & ~diff_mask
            missing_count = bin(missing).count('1')
            # Max new diffs from adding up to (max_k - n) more elements
            # Each new element x adds n diffs (x - a_i for i in current)
            # But some may exceed D or collide
            # Loose bound: (max_k - n) * n new diffs
            if (max_k - n) * n < missing_count:
                return False
            
            # Try next element x > last
            # x must be <= D
            for x in range(last + 1, D + 1):
                if time.time() > deadline:
                    return True
                
                # Check Sidon property: all (x - a_i) must be distinct and not in diff_mask
                ok = True
                new_diffs = 0
                for a_i in current:
                    d = x - a_i
                    if d > D:
                        ok = False
                        break
                    if diff_mask & (1 << d):
                        ok = False
                        break
                    new_diffs |= (1 << d)
                
                if not ok:
                    continue
                
                # Also need to check that new_diffs don't collide with each other
                # x - a_i == x - a_j => a_i == a_j, impossible since distinct
                # So new_diffs are automatically distinct from each other
                
                current.append(x)
                if dfs():
                    current.pop()
                    return True
                current.pop()
            
            return False
        
        # Try different sizes
        for k in range(9, max_k + 1):
            if time.time() > deadline:
                break
            # Lower bound on D for size k:
            # k*(k-1)//2 differences, 30 in 1..30, rest in 31..D
            # So D >= 30 + (k*(k-1)//2 - 30) = k*(k-1)//2
            # But also D >= 30
            min_D = max(30, k * (k - 1) // 2)
            if min_D > D:
                continue
            
            # Reset for this k
            current = [0]
            diff_mask = 0
            
            def dfs_k():
                nonlocal best_result, best_D
                if time.time() > deadline:
                    return True
                
                if (diff_mask & target_mask) == target_mask:
                    D_cur = current[-1]
                    if D_cur < best_D:
                        best_D = D_cur
                        best_result = list(current)
                    return True
                
                n = len(current)
                if n == k:
                    return False  # Can't add more
                
                last = current[-1]
                if last >= best_D:
                    return False
                
                missing = target_mask & ~diff_mask
                missing_count = bin(missing).count('1')
                if (k - n) * n < missing_count:
                    return False
                
                for x in range(last + 1, D + 1):
                    if time.time() > deadline:
                        return True
                    
                    ok = True
                    new_diffs = 0
                    for a_i in current:
                        d = x - a_i
                        if d > D:
                            ok = False
                            break
                        if diff_mask & (1 << d):
                            ok = False
                            break
                        new_diffs |= (1 << d)
                    
                    if not ok:
                        continue
                    
                    current.append(x)
                    if dfs_k():
                        current.pop()
                        return True
                    current.pop()
                
                return False
            
            dfs_k()
            if best_result is not None:
                return True
        
        return False
    
    # Search D from lower bound upward
    # For k=9, D>=36; for k=10, D>=45
    # Start with D=36 and go up
    for D in range(36, 80):
        if time.time() > deadline:
            break
        if best_result is not None and D >= best_D:
            break
        try_dfs(D)
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: greedy construction
    a = [0]
    covered = set()
    for d in range(1, 31):
        if d in covered:
            continue
        found = False
        for x in range(d, 200):
            if x in a:
                continue
            trial = sorted(a + [x])
            diffs = set()
            ok = True
            for i in range(len(trial)):
                for j in range(i + 1, len(trial)):
                    dd = trial[j] - trial[i]
                    if dd in diffs:
                        ok = False
                        break
                    diffs.add(dd)
                if not ok:
                    break
            if ok and d in diffs:
                a = trial
                covered = diffs
                found = True
                break
        if not found:
            # Add pair to cover d
            for y in range(1, 500):
                for z in range(y + 1, y + d + 1):
                    if z - y != d:
                        continue
                    if y in a or z in a:
                        continue
                    trial = sorted(a + [y, z])
                    diffs = set()
                    ok = True
                    for i in range(len(trial)):
                        for j in range(i + 1, len(trial)):
                            dd = trial[j] - trial[i]
                            if dd in diffs:
                                ok = False
                                break
                            diffs.add(dd)
                        if not ok:
                            break
                    if ok and d in diffs:
                        a = trial
                        covered = diffs
                        found = True
                        break
                if found:
                    break
        if not found:
            # Last resort: add large element
            y = 10 * max(a) + d + 100
            trial = sorted(a + [y])
            a = trial
            covered = set()
            for i in range(len(a)):
                for j in range(i + 1, len(a)):
                    covered.add(a[j] - a[i])
    
    # Verify and return
    diffs = set()
    ok = True
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            d = a[j] - a[i]
            if d in diffs:
                ok = False
            diffs.add(d)
    
    if not ok or not all(d in diffs for d in range(1, 31)):
        # Should not happen, but return something
        return a
    
    return a
