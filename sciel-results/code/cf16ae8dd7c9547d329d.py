import random
import time

def _validate(elements):
    n = len(elements)
    if n == 0:
        return True
    
    sums = set()
    num_subsets = 1 << n
    for mask in range(1, num_subsets):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        if s in sums:
            return False
        sums.add(s)
        
    sums_list = sorted(sums)
    for i in range(len(sums_list)):
        a = sums_list[i]
        for j in range(i + 1, len(sums_list)):
            b = sums_list[j]
            if b % a == 0:
                return False
    return True

def _check_addition_fast(current_sums, x):
    """Fast check if x can be added. Returns (valid, new_sums_set) or (False, None)."""
    if x in current_sums:
        return False, None
    
    new_sums = set()
    for s in current_sums:
        ns = s + x
        if ns in current_sums or ns in new_sums:
            return False, None
        new_sums.add(ns)
    new_sums.add(x)
    
    # Check divisibility only between existing sums and new sums (not within each group yet,
    # but we need to check all pairs)
    # Optimization: only check if a divides b where b > a
    all_sums = list(current_sums) + list(new_sums)
    for i in range(len(all_sums)):
        a = all_sums[i]
        for j in range(i + 1, len(all_sums)):
            b = all_sums[j]
            if b % a == 0:
                return False, None
    
    return True, new_sums

def _greedy_with_backtrack(start_val, deadline, max_backtrack_depth=3):
    """Greedy growth with limited backtracking."""
    current = [start_val]
    sums = {start_val}
    
    # Track history for backtracking
    history = [(current[:], sums.copy())]
    
    while len(current) < 30:
        if time.time() > deadline:
            break
        
        # Generate and score candidates
        candidates = [x for x in range(256, 0, -1) if x not in current]
        
        # Quick filter: remove candidates that are sums or would create obvious conflicts
        valid_candidates = []
        for x in candidates:
            if x in sums:
                continue
            # Quick check: x + any existing sum should not equal another existing sum
            conflict = False
            for s in sums:
                if s + x in sums:
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append(x)
        
        if not valid_candidates:
            # Try backtracking
            if len(history) > 1 and max_backtrack_depth > 0:
                history.pop()
                current, sums = history[-1]
                max_backtrack_depth -= 1
                continue
            else:
                break
        
        # Try candidates in order (larger first to reduce divisibility risk)
        added = False
        for x in valid_candidates[:30]:
            if time.time() > deadline:
                break
            
            valid, new_sums = _check_addition_fast(sums, x)
            if valid:
                # Save state before adding
                history.append((current[:], sums.copy()))
                current.append(x)
                sums = sums | new_sums
                added = True
                break
        
        if not added:
            # Backtrack
            if len(history) > 1 and max_backtrack_depth > 0:
                history.pop()
                current, sums = history[-1]
                max_backtrack_depth -= 1
            else:
                break
    
    return current

def construct():
    deadline = time.time() + 85
    best_set = [31, 32, 34]
    best_size = 3
    
    if _validate(best_set):
        best_size = len(best_set)
        
    random.seed(42)
    
    # Multiple strategies with different starting points
    start_strategies = [
        lambda: random.randint(120, 256),
        lambda: random.randint(150, 256),
        lambda: random.randint(80, 160),
        lambda: random.randint(50, 120),
        lambda: random.randint(20, 80),
    ]
    
    for restart in range(300):
        if time.time() > deadline:
            break
        
        strategy_idx = restart % len(start_strategies)
        start_val = start_strategies[strategy_idx]()
        
        # Add some randomness to start
        if random.random() < 0.3:
            start_val = random.randint(1, 256)
        
        current = _greedy_with_backtrack(start_val, deadline, max_backtrack_depth=2)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    # Final attempt: try to extend best set
    if best_size >= 4 and time.time() < deadline - 3:
        current = best_set[:]
        n = len(current)
        sums = set()
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += current[i]
            sums.add(s)
        
        for x in range(256, 0, -1):
            if time.time() > deadline:
                break
            if x in current:
                continue
            
            valid, new_sums = _check_addition_fast(sums, x)
            if valid:
                current.append(x)
                sums = sums | new_sums
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = current[:]
    
    return list(best_set)
