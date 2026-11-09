import random
import time

N = 256
MAX_SUM = 2 * N

def _is_sidon(S):
    """Check if S is a Sidon set."""
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(S):
    """Check if S is an inclusion-maximal Sidon set in [1, N]."""
    S_set = set(S)
    for x in range(1, N + 1):
        if x in S_set:
            continue
        # Check if S U {x} is Sidon
        candidate = list(S) + [x]
        if not _is_sidon(candidate):
            continue
        return False
    return True

def _greedy_maximal_sidon(start=1, end=N, reverse=False, seed=None):
    """Generate a maximal Sidon set using a greedy strategy."""
    if seed is not None:
        random.seed(seed)
    
    selected = []
    sums = set()
    
    # Create list of candidates
    candidates = list(range(start, end + 1))
    if reverse:
        candidates.reverse()
    else:
        random.shuffle(candidates)
        
    for x in candidates:
        # Check if x can be added
        added_sums = [x + a for a in selected] + [2 * x]
        # Check for duplicates within added_sums
        if len(set(added_sums)) != len(added_sums):
            continue
        # Check against existing sums
        if sums.intersection(added_sums):
            continue
        
        selected.append(x)
        sums.update(added_sums)
        
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin for 60s limit
    
    # Bitset-based helpers
    def _sums_to_mask(S):
        mask = 0
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                mask |= (1 << s)
        return mask

    def _can_add_mask(S, sums_mask, x):
        # New sums: 2*x and x+s for s in S
        new_mask = 1 << (2 * x)
        for s in S:
            new_mask |= 1 << (x + s)
        
        # Check if any new sum conflicts with existing sums
        if sums_mask & new_mask:
            return False
        
        # Check internal duplicates in new sums
        # 2*x == x+s => x=s (impossible as x not in S)
        # x+s1 == x+s2 => s1=s2 (impossible as S has distinct elements)
        # So internal duplicates are impossible for distinct s.
        return True

    def _is_maximal_mask(S, sums_mask):
        S_set = set(S)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            if _can_add_mask(S, sums_mask, x):
                return False
        return True

    def _find_redundant_elements(S, sums_mask):
        """Find elements whose removal still leaves S maximal."""
        redundant = []
        for i in range(len(S)):
            z = S[i]
            new_set = S[:i] + S[i+1:]
            if not new_set:
                continue
            new_mask = _sums_to_mask(new_set)
            if _is_maximal_mask(new_set, new_mask):
                redundant.append(i)
        return redundant

    def _remove_redundant(S, sums_mask):
        """Iteratively remove redundant elements."""
        changed = True
        while changed:
            changed = False
            redundant = _find_redundant_elements(S, sums_mask)
            if redundant:
                # Remove the first redundant element found
                idx = redundant[0]
                S = S[:idx] + S[idx+1:]
                sums_mask = _sums_to_mask(S)
                changed = True
        return S, sums_mask

    def _local_search(S, sums_mask, deadline):
        """Local search with removal and swap moves."""
        current_set = S[:]
        current_mask = sums_mask
        
        while time.time() < deadline:
            current_size = len(current_set)
            if current_size <= 1:
                break
            
            # Try removal first
            if current_size > 1:
                removals = []
                for i in range(current_size):
                    z = current_set[i]
                    new_set = current_set[:i] + current_set[i+1:]
                    if not new_set:
                        continue
                    new_mask = _sums_to_mask(new_set)
                    if _is_maximal_mask(new_set, new_mask):
                        removals.append(i)
                
                if removals:
                    idx = random.choice(removals)
                    current_set = current_set[:idx] + current_set[idx+1:]
                    current_mask = _sums_to_mask(current_set)
                    continue
            
            # Try swaps
            if current_size > 0:
                z_idx = random.randint(0, current_size - 1)
                z = current_set[z_idx]
                
                found_swap = False
                # Try a few random x values
                for _ in range(20):
                    x = random.randint(1, N)
                    if x in current_set:
                        continue
                    
                    # Remove z, add x
                    temp_set = current_set[:z_idx] + current_set[z_idx+1:]
                    temp_mask = _sums_to_mask(temp_set)
                    
                    if _can_add_mask(temp_set, temp_mask, x):
                        new_set = temp_set + [x]
                        new_mask = temp_mask | (1 << (2*x))
                        for s in temp_set:
                            new_mask |= 1 << (x+s)
                        
                        if _is_maximal_mask(new_set, new_mask):
                            current_set = new_set
                            current_mask = new_mask
                            found_swap = True
                            break
                
                if found_swap:
                    continue
            
            # If stuck, try adding an element to escape (increases size, but might allow future removals)
            # Only do this if we are not already large
            if current_size < 30:
                x = random.randint(1, N)
                if x not in current_set:
                    if _can_add_mask(current_set, current_mask, x):
                        new_set = current_set + [x]
                        new_mask = current_mask | (1 << (2*x))
                        for s in current_set:
                            new_mask |= 1 << (x+s)
                        current_set = new_set
                        current_mask = new_mask
                        continue
            
            break
        
        return current_set, current_mask

    best_size = N + 1
    best_set = []
    best_mask = 0
    
    # Phase 1: Try structured sets and greedy to get initial solutions
    structured_sets = [
        [1, 3, 7, 15, 31, 63, 127],
        [1, 3, 7, 15, 31, 63],
        [1, 3, 7, 15, 31],
        [1, 3, 7, 15],
        [1, 3, 7],
        [1, 3],
        [1],
    ]
    
    for s in structured_sets:
        if _is_sidon(s):
            s_mask = _sums_to_mask(s)
            if _is_maximal_mask(s, s_mask):
                if len(s) < best_size:
                    best_size = len(s)
                    best_set = s[:]
                    best_mask = s_mask
    
    # Try multiple greedy seeds
    for seed in range(10):
        if time.time() > deadline:
            break
        s = _greedy_maximal_sidon(seed=seed)
        if len(s) < best_size:
            best_size = len(s)
            best_set = s[:]
            best_mask = _sums_to_mask(s)
    
    # Phase 2: Local search from the best initial solution
    if best_set:
        # First, try to remove redundant elements
        best_set, best_mask = _remove_redundant(best_set, best_mask)
        best_size = len(best_set)
        
        # Then do local search
        deadline_local = min(deadline, start_time + 50.0)
        best_set, best_mask = _local_search(best_set, best_mask, deadline_local)
        best_size = len(best_set)
        
        # Try local search from a few other starting points
        for seed in range(5):
            if time.time() > deadline_local:
                break
            s = _greedy_maximal_sidon(seed=seed + 100)
            if len(s) < best_size + 2:  # Only if close to best
                s_mask = _sums_to_mask(s)
                s, s_mask = _remove_redundant(s, s_mask)
                s, s_mask = _local_search(s, s_mask, deadline_local)
                if len(s) < best_size:
                    best_size = len(s)
                    best_set = s[:]
                    best_mask = s_mask
    
    # Phase 3: Final aggressive search if time remains
    if time.time() < deadline - 5.0:
        # Try random restarts with local search
        for attempt in range(20):
            if time.time() > deadline - 2.0:
                break
            
            # Generate a random small Sidon set
            random.seed(attempt + 500)
            selected = []
            sums = set()
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            
            for x in candidates:
                added_sums = [x + a for a in selected] + [2 * x]
                if len(set(added_sums)) != len(added_sums):
                    continue
                if sums.intersection(added_sums):
                    continue
                selected.append(x)
                sums.update(added_sums)
                if len(selected) >= 5:  # Keep it small
                    break
            
            if not selected:
                continue
            
            s_mask = _sums_to_mask(selected)
            
            # If not maximal, try to repair by adding elements?
            # For now, skip if not maximal, but this is a weakness.
            # Let's try to make it maximal by adding necessary elements.
            if not _is_maximal_mask(selected, s_mask):
                # Try to add elements that are unblocked? No, we want to BLOCK them.
                # If x is unblocked, we must add it or change S.
                # Adding x increases size.
                # Let's just add all unblocked elements to make it maximal, then prune.
                # This might result in a large set, but pruning might help.
                changed = True
                while changed:
                    changed = False
                    for x in range(1, N+1):
                        if x in selected:
                            continue
                        if _can_add_mask(selected, s_mask, x):
                            # x is unblocked, so we MUST add it to be maximal?
                            # No, we can change S. But adding is a valid move to reach a maximal set.
                            selected.append(x)
                            s_mask |= (1 << (2*x))
                            for s in selected[:-1]:
                                s_mask |= 1 << (x+s)
                            changed = True
                            break
                
                # Now selected is maximal. Prune it.
                selected, s_mask = _remove_redundant(selected, s_mask)
                selected, s_mask = _local_search(selected, s_mask, deadline - 2.0)
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_set = selected[:]
                best_mask = s_mask
    
    return best_set
