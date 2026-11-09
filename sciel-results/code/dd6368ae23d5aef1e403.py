import random
import time

N = 256

def get_sums(A):
    sums = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            sums.add(A[i] + A[j])
    return sums

def get_differences(A, sums):
    diffs = set()
    for s in sums:
        for a in A:
            diffs.add(s - a)
    return diffs

def is_maximal(A):
    # A must be Sidon. We assume it is in local search moves that preserve Sidon property or check it separately.
    # But for safety, we can check Sidon first if not guaranteed.
    # In our swaps, we check is_sidon. In removals, subset of Sidon is Sidon.
    
    S = get_sums(A)
    D = get_differences(A, S)
    A_set = set(A)
    
    for x in range(1, N + 1):
        if x in A_set:
            continue
        if 2 * x in S or x in D:
            continue
        return False
    return True

def is_sidon(A):
    sums = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def greedy_construct(order=None):
    if order is None:
        order = list(range(1, N + 1))
    
    selected = []
    sums = set()
    
    for x in order:
        ok = True
        new_sums = [2 * x] + [x + a for a in selected]
        
        if len(set(new_sums)) != len(new_sums):
            ok = False
            
        if ok:
            for s in new_sums:
                if s in sums:
                    ok = False
                    break
        
        if ok:
            selected.append(x)
            sums.update(new_sums)
            
    return selected

def local_search(initial_set, time_limit=1.0):
    best_set = list(initial_set)
    best_size = len(best_set)
    current_set = list(initial_set)
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Try removals
        # To speed up, we can try removing elements that are "least useful" for blocking?
        # Or just random.
        indices = list(range(len(current_set)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > time_limit:
                break
            y = current_set[idx]
            candidate = current_set[:idx] + current_set[idx+1:]
            
            # Candidate is Sidon
            if is_maximal(candidate):
                current_set = candidate
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                break # Restart loop with smaller set
        
        # Try swaps
        if time.time() - start_time > time_limit:
            break
            
        if not current_set:
            break
            
        # Pick random y to remove
        idx_y = random.randint(0, len(current_set) - 1)
        y = current_set[idx_y]
        
        # Pick random x to add
        # We can bias towards x that blocks more?
        # For now, random missing element
        A_set = set(current_set)
        missing = [x for x in range(1, N + 1) if x not in A_set]
        if not missing:
            break
        x = random.choice(missing)
        
        if x == y:
            continue
            
        candidate = current_set[:idx_y] + current_set[idx_y+1:] + [x]
        
        # Check Sidon
        if is_sidon(candidate):
            # Check Maximal
            if is_maximal(candidate):
                current_set = candidate
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)

    return best_set

def construct():
    # Start with deterministic greedy to have a baseline
    best_overall = greedy_construct()
    best_size = len(best_overall)
    
    start_time = time.time()
    deadline = start_time + 55
    
    # Run multiple randomized restarts
    while time.time() < deadline:
        # Randomized greedy
        order = list(range(1, N + 1))
        random.shuffle(order)
        init_set = greedy_construct(order)
        
        # Local search
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        if remaining < 1:
            break
        alloc = min(1.0, remaining * 0.5)
        
        improved_set = local_search(init_set, time_limit=alloc)
        
        if len(improved_set) < best_size:
            best_size = len(improved_set)
            best_overall = improved_set

    return best_overall
