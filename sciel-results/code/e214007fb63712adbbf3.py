import numpy as np
import time
import random

def _compute_metrics(A_set):
    """Compute peak multiplicity and validity. Returns (peak, valid)."""
    n = 129
    arr = np.zeros(n, dtype=np.int32)
    for a in A_set:
        arr[a] = 1
    conv = np.convolve(arr, arr, mode='full')
    targets = np.arange(128, 257)
    ordered = conv[targets]
    even_mask = (targets % 2 == 0)
    mids = targets // 2
    mid_in_A = np.zeros(len(targets), dtype=bool)
    valid_mid = (mids >= 0) & (mids < n)
    mid_in_A[valid_mid] = arr[mids[valid_mid]]
    correction = (even_mask & mid_in_A).astype(np.int32)
    unordered = (ordered + correction) // 2
    peak = int(np.max(unordered))
    valid = bool(np.min(unordered) > 0)
    return peak, valid

def _objective(A_set):
    """Return score to minimize."""
    if not A_set:
        return 999999
    peak, valid = _compute_metrics(A_set)
    if not valid:
        return 999999
    return peak * 1000 + len(A_set)

def _greedy_initial():
    """Build initial set via greedy removal from seed."""
    A = set(range(64, 129))
    current_obj = _objective(A)
    progress = True
    while progress:
        progress = False
        for a in sorted(A):
            trial = A - {a}
            if not trial:
                continue
            obj = _objective(trial)
            if obj < current_obj:
                A = trial
                current_obj = obj
                progress = True
                break
    return A

def _local_search(A, time_budget):
    """Local search with removals, additions, swaps."""
    start_time = time.time()
    A = set(A)
    current_obj = _objective(A)
    best_A = set(A)
    best_obj = current_obj
    stuck_count = 0
    
    while time.time() - start_time < time_budget:
        improved = False
        
        # Try removals
        for a in sorted(A):
            trial = A - {a}
            if not trial:
                continue
            obj = _objective(trial)
            if obj < current_obj:
                A = trial
                current_obj = obj
                best_A = set(A)
                best_obj = obj
                improved = True
                stuck_count = 0
                break
        if improved:
            continue
        
        # Try additions
        candidates = [x for x in range(129) if x not in A]
        random.shuffle(candidates)
        for a in candidates[:30]:
            trial = A | {a}
            obj = _objective(trial)
            if obj < current_obj:
                A = trial
                current_obj = obj
                best_A = set(A)
                best_obj = obj
                improved = True
                stuck_count = 0
                break
        if improved:
            continue
        
        # Try swaps
        if len(A) > 1:
            a = random.choice(list(A))
            b_candidates = [x for x in range(129) if x not in A]
            random.shuffle(b_candidates)
            found = False
            for b in b_candidates[:30]:
                trial = A - {a} | {b}
                if not trial:
                    continue
                obj = _objective(trial)
                if obj < current_obj:
                    A = trial
                    current_obj = obj
                    best_A = set(A)
                    best_obj = obj
                    found = True
                    stuck_count = 0
                    break
            if found:
                continue
        
        # Stuck - try perturbation
        stuck_count += 1
        if stuck_count > 5:
            new_A = set(A)
            n_perturb = random.randint(1, 4)
            to_remove = random.sample(list(new_A), min(n_perturb, len(new_A) - 1))
            for x in to_remove:
                new_A.discard(x)
            add_candidates = [x for x in range(129) if x not in new_A]
            if add_candidates:
                to_add = random.sample(add_candidates, min(n_perturb, len(add_candidates)))
                for x in to_add:
                    new_A.add(x)
            if new_A:
                obj = _objective(new_A)
                if obj < current_obj:
                    A = new_A
                    current_obj = obj
                    best_A = set(A)
                    best_obj = obj
                    stuck_count = 0
                    continue
            stuck_count = 0
        
        if stuck_count > 20:
            break
    
    return best_A, best_obj

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    random.seed(42)
    
    # Phase 1: Greedy initial
    A = _greedy_initial()
    best_A = set(A)
    best_obj = _objective(A)
    
    # Phase 2: Multiple local search runs from different starting points
    remaining = deadline - time.time()
    run_count = 0
    while time.time() < deadline - 3:
        run_count += 1
        if run_count == 1:
            start_set = set(best_A)
        else:
            start_set = set(best_A)
            n_perturb = random.randint(1, 5)
            to_remove = random.sample(list(start_set), min(n_perturb, len(start_set) - 1))
            for x in to_remove:
                start_set.discard(x)
            add_candidates = [x for x in range(129) if x not in start_set]
            if add_candidates:
                to_add = random.sample(add_candidates, min(n_perturb, len(add_candidates)))
                for x in to_add:
                    start_set.add(x)
        
        time_budget = min(8.0, deadline - time.time() - 2)
        if time_budget < 0.5:
            break
        
        candidate, obj = _local_search(start_set, time_budget)
        if obj < best_obj:
            best_A = candidate
            best_obj = obj
    
    # Phase 3: Final polish
    remaining = deadline - time.time()
    if remaining > 1:
        final_A, final_obj = _local_search(best_A, remaining - 0.5)
        if final_obj < best_obj:
            best_A = final_A
            best_obj = final_obj
    
    return sorted(best_A)
