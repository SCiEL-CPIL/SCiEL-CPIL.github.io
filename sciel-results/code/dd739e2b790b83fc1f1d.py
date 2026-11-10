import random
import time

N = 256

def _compute_sums(selected):
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            sums.add(a + selected[j])
    return sums

def _check_sidon(selected):
    if len(selected) == 0:
        return False
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, N):
    if not selected:
        return False
    selected_set = set(selected)
    sums = _compute_sums(selected)
    
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        if 2 * x in sums:
            continue
        blocked = False
        for a in selected:
            if x + a in sums:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _construct_maximal_greedy(rng, N):
    selected = []
    sums = set()
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        can_add = True
        for a in selected:
            s = x + a
            if s in sums:
                can_add = False
                break
        if can_add:
            if 2 * x in sums:
                can_add = False
        if can_add:
            selected.append(x)
            for a in selected[:-1]:
                sums.add(x + a)
            sums.add(2 * x)
            
    return selected

def _local_search(start_set, N, deadline, rng):
    current = list(start_set)
    current_size = len(current)
    best = list(current)
    best_size = current_size
    
    # Temperature for accepting size-preserving moves
    temp = 1.0
    cooling = 0.99
    
    while time.time() < deadline and temp > 0.01:
        if current_size < best_size:
            best = list(current)
            best_size = current_size
            
        if not current:
            break
            
        # Pick random y to remove
        idx = rng.randint(0, len(current) - 1)
        y = current[idx]
        
        # Remove y
        new_selected = [x for i, x in enumerate(current) if i != idx]
        new_sums = _compute_sums(new_selected)
        new_selected_set = set(new_selected)
        
        # Check if new_selected is maximal
        is_maximal = True
        for x in range(1, N + 1):
            if x in new_selected_set:
                continue
            if 2 * x in new_sums:
                continue
            blocked = False
            for a in new_selected:
                if x + a in new_sums:
                    blocked = True
                    break
            if not blocked:
                is_maximal = False
                break
        
        if is_maximal:
            # Accept removal
            current = new_selected
            current_size -= 1
        else:
            # Try to swap
            # Find candidates x to add
            candidates = []
            for x in range(1, N + 1):
                if x in new_selected_set:
                    continue
                # Check if x can be added to new_selected
                can_add = True
                if 2 * x in new_sums:
                    can_add = False
                else:
                    for a in new_selected:
                        if x + a in new_sums:
                            can_add = False
                            break
                if can_add:
                    candidates.append(x)
            
            if candidates:
                x = rng.choice(candidates)
                # Check if (new_selected + [x]) is maximal
                trial_selected = new_selected + [x]
                trial_sums = _compute_sums(trial_selected)
                trial_set = set(trial_selected)
                
                trial_maximal = True
                for z in range(1, N + 1):
                    if z in trial_set:
                        continue
                    if 2 * z in trial_sums:
                        continue
                    blocked = False
                    for a in trial_selected:
                        if z + a in trial_sums:
                            blocked = True
                            break
                    if not blocked:
                        trial_maximal = False
                        break
                
                if trial_maximal:
                    # Accept swap (size same)
                    # Probability based on temp
                    if rng.random() < temp:
                        current = trial_selected
                        # current_size unchanged
        
        temp *= cooling

    return best

def construct():
    deadline = time.time() + 55.0
    rng = random.Random(42)
    
    best_set = None
    best_size = N + 1
    
    # Generate multiple random maximal sets
    for i in range(100):
        if time.time() > deadline - 5.0:
            break
        candidate = _construct_maximal_greedy(rng, N)
        if _check_sidon(candidate) and _is_maximal(candidate, N):
            if len(candidate) < best_size:
                best_size = len(candidate)
                best_set = list(candidate)
                
    if best_set is None:
        best_set = _construct_maximal_greedy(rng, N)
        best_size = len(best_set)
        
    # Local search from best
    if time.time() < deadline - 1.0:
        improved = _local_search(best_set, N, deadline - 1.0, rng)
        if improved is not None and len(improved) < best_size:
            best_set = improved
            best_size = len(best_set)
            
    # Try local search from other good sets
    for i in range(50):
        if time.time() > deadline - 0.5:
            break
        candidate = _construct_maximal_greedy(rng, N)
        if len(candidate) <= best_size:
            sub_deadline = min(deadline - 0.5, time.time() + 1.0)
            improved = _local_search(candidate, N, sub_deadline, rng)
            if improved is not None and len(improved) < best_size:
                best_set = improved
                best_size = len(best_set)
                
    return best_set
