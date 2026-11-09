import time
import random

N = 256

def construct():
    deadline = time.time() + 25.0
    
    def compute_sums(s_list):
        sums = set()
        for i in range(len(s_list)):
            for j in range(i, len(s_list)):
                sums.add(s_list[i] + s_list[j])
        return sums
    
    def is_blocked(x, s_set, sums):
        """Check if x is blocked by the Sidon set s_set with sumset sums."""
        if x in s_set:
            return True
        if 2 * x in sums:
            return True
        for a in s_set:
            if (x + a) in sums:
                return True
        return False
    
    def is_maximal(s_list):
        s_set = set(s_list)
        sums = compute_sums(s_list)
        for x in range(1, N + 1):
            if x in s_set:
                continue
            if not is_blocked(x, s_set, sums):
                return False
        return True
    
    def greedy_maximal_smart(perm):
        """Greedy that picks elements to maximize blocking coverage."""
        selected = []
        s_set = set()
        sums = set()
        
        # Track which elements are currently unblocked
        unblocked = set(range(1, N + 1))
        
        for x in perm:
            if x in s_set:
                continue
            if x not in unblocked:
                continue
            
            # Check if x can be added to maintain Sidon property
            if 2 * x in sums:
                continue
            can_add = True
            for a in selected:
                if (x + a) in sums:
                    can_add = False
                    break
            
            if not can_add:
                continue
            
            # Compute how many currently unblocked elements this would block
            # New sums added: 2x and x+a for a in selected
            new_sums = {2 * x}
            for a in selected:
                new_sums.add(x + a)
            
            # Count how many unblocked elements become blocked
            newly_blocked = 0
            for y in unblocked:
                if y == x:
                    continue
                # y is newly blocked if it wasn't blocked before but is now
                # It's blocked if 2y in (sums ∪ new_sums) or y+a in (sums ∪ new_sums) for some a in selected∪{x}
                was_blocked = is_blocked(y, s_set, sums)
                if was_blocked:
                    continue
                # Check if y becomes blocked
                if 2 * y in new_sums:
                    newly_blocked += 1
                    continue
                blocked = False
                for a in selected:
                    if (y + a) in new_sums:
                        blocked = True
                        break
                if not blocked and (2 * y in sums or any((y + a) in sums for a in selected)):
                    pass  # was already blocked, shouldn't happen
                if not blocked:
                    newly_blocked += 1
            
            # Add x to selected
            selected.append(x)
            s_set.add(x)
            sums.update(new_sums)
            unblocked.discard(x)
            
            # Update unblocked: remove elements that are now blocked
            still_unblocked = set()
            for y in unblocked:
                if not is_blocked(y, s_set, sums):
                    still_unblocked.add(y)
            unblocked = still_unblocked
            
            if not unblocked:
                break
        
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
                candidate = current[:idx] + current[idx+1:]
                if len(candidate) == 0:
                    continue
                if is_maximal(candidate):
                    current = candidate
                    changed = True
        return current
    
    # Known good starting point
    known_set = [1, 3, 7, 15, 31, 63, 127]
    if is_maximal(known_set):
        best_set = list(known_set)
        best_size = len(best_set)
    else:
        perm = list(range(1, N + 1))
        best_set = greedy_maximal_smart(perm)
        best_set = peel_maximal(best_set)
        best_size = len(best_set)
    
    # Search for better solutions
    while time.time() < deadline:
        perm = list(range(1, N + 1))
        random.shuffle(perm)
        
        candidate = greedy_maximal_smart(perm)
        candidate = peel_maximal(candidate)
        
        if len(candidate) < best_size:
            best_size = len(candidate)
            best_set = candidate
            if best_size <= 5:
                break
    
    return best_set
