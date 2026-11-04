import time
import random
import math

def _score_set(U):
    """Compute the score for a set U (list of nonnegative ints containing 0)."""
    if not U or 0 not in U:
        return 0.0
    U_set = set(U)
    n = len(U_set)
    if n < 2:
        return 0.0
    max_u = max(U_set)
    # Compute sumset and diffset
    sums = set()
    diffs = set()
    for a in U_set:
        for b in U_set:
            sums.add(a + b)
            diffs.add(a - b)
    # Score: 1 + log(|U-U|/|U+U|)/log(2*max U + 1)
    if len(sums) == 0:
        return 0.0
    ratio = len(diffs) / len(sums)
    if ratio <= 1.0:
        return 1.0
    if 2 * max_u + 1 <= 1:
        return 1.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Initial seeds
    seeds = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 2, 5],
        [0, 1, 4, 6, 10],
        [0, 1, 4, 6, 9, 12],
    ]
    
    # Update best
    for seed in seeds:
        s = _score_set(seed)
        if s > best_score:
            best_score = s
            best_set = list(seed)
    
    # Local search: try adding, removing, or swapping elements
    def try_add(U, max_val):
        """Try adding a new element to U."""
        U_set = set(U)
        candidates = []
        for x in range(1, max_val + 1):
            if x not in U_set:
                candidates.append(x)
        best_add = None
        best_add_score = _score_set(U)
        for x in candidates:
            new_U = U_set | {x}
            s = _score_set(list(new_U))
            if s > best_add_score:
                best_add_score = s
                best_add = x
        if best_add is not None:
            return list(U_set | {best_add})
        return None
    
    def try_swap(U, max_val):
        """Try swapping an element in U with a new element."""
        U_set = set(U)
        best_swap = None
        best_swap_score = _score_set(U)
        for a in list(U_set):
            if a == 0:
                continue
            for b in range(1, max_val + 1):
                if b in U_set:
                    continue
                new_U = (U_set - {a}) | {b}
                s = _score_set(list(new_U))
                if s > best_swap_score:
                    best_swap_score = s
                    best_swap = (a, b)
        if best_swap is not None:
            a, b = best_swap
            return list((set(U) - {a}) | {b})
        return None
    
    def try_remove(U):
        """Try removing an element from U (keep 0)."""
        U_set = set(U)
        best_remove = None
        best_remove_score = _score_set(U)
        for a in list(U_set):
            if a == 0:
                continue
            new_U = U_set - {a}
            s = _score_set(list(new_U))
            if s > best_remove_score:
                best_remove_score = s
                best_remove = a
        if best_remove is not None:
            return list(set(U) - {best_remove})
        return None
    
    # Start local search from best seed
    current = list(best_set)
    current_score = best_score
    max_val = max(current) + 50
    
    while time.time() < deadline:
        improved = False
        
        # Try adding
        new_set = try_add(current, max_val)
        if new_set is not None:
            s = _score_set(new_set)
            if s > current_score + 1e-9:
                current = new_set
                current_score = s
                improved = True
                if s > best_score:
                    best_score = s
                    best_set = list(current)
                max_val = max(current) + 50
                continue
        
        # Try swapping
        new_set = try_swap(current, max_val)
        if new_set is not None:
            s = _score_set(new_set)
            if s > current_score + 1e-9:
                current = new_set
                current_score = s
                improved = True
                if s > best_score:
                    best_score = s
                    best_set = list(current)
                max_val = max(current) + 50
                continue
        
        # Try removing
        new_set = try_remove(current)
        if new_set is not None:
            s = _score_set(new_set)
            if s > current_score + 1e-9:
                current = new_set
                current_score = s
                improved = True
                if s > best_score:
                    best_score = s
                    best_set = list(current)
                continue
        
        if not improved:
            # Random restarts
            if random.random() < 0.3:
                # Try a random perturbation
                current_set = set(current)
                new_set = set(current_set)
                for _ in range(random.randint(1, 3)):
                    action = random.choice(['add', 'swap', 'remove'])
                    if action == 'add' and new_set:
                        x = random.randint(1, max_val)
                        if x not in new_set:
                            new_set.add(x)
                    elif action == 'swap' and len(new_set) > 1:
                        elems = [e for e in new_set if e != 0]
                        if elems:
                            a = random.choice(elems)
                            b = random.randint(1, max_val)
                            while b in new_set:
                                b = random.randint(1, max_val)
                            new_set.remove(a)
                            new_set.add(b)
                    elif action == 'remove' and len(new_set) > 1:
                        elems = [e for e in new_set if e != 0]
                        if elems:
                            new_set.discard(random.choice(elems))
                s = _score_set(list(new_set))
                if s > current_score:
                    current = list(new_set)
                    current_score = s
                    if s > best_score:
                        best_score = s
                        best_set = list(current)
                    max_val = max(current) + 50
                    continue
            
            # Try from another seed
            found = False
            for seed in seeds:
                if list(seed) == current:
                    continue
                s = _score_set(seed)
                if s > current_score:
                    current = list(seed)
                    current_score = s
                    max_val = max(current) + 50
                    found = True
                    break
            if not found:
                # No more improvements, break
                break
    
    return best_set
