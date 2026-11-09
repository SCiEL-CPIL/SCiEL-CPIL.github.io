import time
import random

N = 256

def construct():
    deadline = time.time() + 25.0
    
    def is_valid_sidon(s_list):
        sums = set()
        for i in range(len(s_list)):
            for j in range(i, len(s_list)):
                val = s_list[i] + s_list[j]
                if val in sums:
                    return False
                sums.add(val)
        return True

    def compute_sums(s_list):
        sums = set()
        for i in range(len(s_list)):
            for j in range(i, len(s_list)):
                sums.add(s_list[i] + s_list[j])
        return sums

    def is_maximal_fast(s_list, sums=None):
        if sums is None:
            sums = compute_sums(s_list)
        s_set = set(s_list)
        for x in range(1, N + 1):
            if x in s_set:
                continue
            # Check if x can be added
            if 2 * x in sums:
                continue
            can_add = True
            for a in s_list:
                if (x + a) in sums:
                    can_add = False
                    break
            if can_add:
                return False
        return True

    def greedy_with_blocking(seed):
        rng = random.Random(seed)
        perm = list(range(1, N + 1))
        rng.shuffle(perm)
        
        selected = []
        sums = set()
        candidates = set(perm)
        
        while candidates:
            # Find all addable candidates
            addable = []
            for x in candidates:
                if 2 * x in sums:
                    continue
                can_add = True
                for a in selected:
                    if (x + a) in sums:
                        can_add = False
                        break
                if can_add:
                    addable.append(x)
            
            if not addable:
                break
            
            # Compute blocking score: how many other candidates would be blocked
            # Use a sample of candidates for speed
            if len(candidates) > 50:
                sample = list(rng.sample(candidates, 50))
            else:
                sample = list(candidates)
            
            best_x = None
            best_score = -1
            
            for x in addable:
                # Compute new sums if x added
                new_sums = set(sums)
                for a in selected:
                    new_sums.add(x + a)
                new_sums.add(2 * x)
                
                # Count how many in sample get blocked
                blocked = 0
                for y in sample:
                    if y == x:
                        continue
                    # Check if y can still be added
                    if 2 * y in new_sums:
                        blocked += 1
                        continue
                    y_ok = True
                    for a in selected:
                        if (y + a) in new_sums:
                            y_ok = False
                            break
                    if not y_ok:
                        blocked += 1
                
                # Add random tiebreaker
                score = blocked + rng.random() * 0.1
                if score > best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                break
            
            # Add best_x
            selected.append(best_x)
            sums.add(2 * best_x)
            for a in selected[:-1]:
                sums.add(best_x + a)
            candidates.discard(best_x)
            
            # Remove candidates that can no longer be added
            to_remove = []
            for x in candidates:
                if 2 * x in sums:
                    to_remove.append(x)
                    continue
                for a in selected:
                    if (x + a) in sums:
                        to_remove.append(x)
                        break
            for x in to_remove:
                candidates.discard(x)
        
        return selected

    def peel_maximal(s_list):
        """Try to remove elements while keeping the set maximal."""
        current = list(s_list)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            indices = list(range(len(current)))
            random.shuffle(indices)
            
            for idx in indices:
                if idx >= len(current):
                    continue
                x = current[idx]
                candidate = current[:idx] + current[idx+1:]
                if is_maximal_fast(candidate):
                    current = candidate
                    changed = True
        return current

    # Start with known good set
    known_set = [1, 3, 7, 15, 31, 63, 127]
    if is_valid_sidon(known_set) and is_maximal_fast(known_set):
        best_set = list(known_set)
        best_size = len(best_set)
    else:
        best_set = greedy_with_blocking(0)
        best_set = peel_maximal(best_set)
        best_size = len(best_set)

    # Search loop with multiple strategies
    seed = 0
    while time.time() < deadline:
        # Strategy 1: Greedy with blocking + peel
        candidate = greedy_with_blocking(seed)
        candidate = peel_maximal(candidate)
        
        if len(candidate) < best_size:
            best_size = len(candidate)
            best_set = candidate
            if best_size <= 8:
                # Try more aggressive peeling
                candidate = peel_maximal(candidate)
                if len(candidate) < best_size:
                    best_size = len(candidate)
                    best_set = candidate
        
        seed += 1
    
    return best_set
