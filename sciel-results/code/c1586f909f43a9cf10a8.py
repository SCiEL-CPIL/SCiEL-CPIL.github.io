import random
import time

N = 256

def _precompute_conflicts():
    """Precompute for each x, the set of y (y != x) such that {x, y} is not a Sidon pair
    when considered together with ANY other element z. Actually, we need: for a set S,
    x can be added if for all y in S, the sums x+y and 2x are distinct from all existing sums.
    
    Let's precompute: for each pair (x, y) with x <= y, the sum s = x+y.
    A set S is Sidon iff all pairwise sums (including x+x) are distinct.
    
    For maximality: x not in S is blocked if for every z not in S, S ∪ {x, z} is not Sidon,
    OR equivalently, S ∪ {x} is not Sidon (which means x is already blocked by S alone).
    
    Actually, the maximal condition is: for every x not in S, S ∪ {x} is NOT Sidon.
    This means: for every x not in S, there exists some y in S (or y=x) such that
    the sum x+y (or 2x) collides with an existing sum in S.
    
    Let's precompute for each x, which sums are "created" by adding x to a set:
    new_sums(x, S) = {2x} ∪ {x + y : y in S}
    x can be added to S iff new_sums(x, S) is disjoint from sums(S) and has no internal duplicates.
    """
    # Precompute all pair sums
    pair_sums = {}
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            s = a + b
            if s not in pair_sums:
                pair_sums[s] = []
            pair_sums[s].append((a, b))
    return pair_sums

_CONFLICTS = _precompute_conflicts()

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
    
    def _get_sums(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        return sums

    def _can_add_with_sums(S, sums_S, x):
        new_sums = [2*x]
        for s in S:
            new_sums.append(x + s)
        if len(set(new_sums)) != len(new_sums):
            return False
        if sums_S.intersection(new_sums):
            return False
        return True

    def _is_maximal_fast(S, sums_S):
        S_set = set(S)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            if _can_add_with_sums(S, sums_S, x):
                return False
        return True

    def _is_maximal_fast_vectorized(S, sums_S):
        """Faster maximality check using precomputed conflict info."""
        S_set = set(S)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            # Check if x can be added
            new_sums = {2 * x}
            for s in S:
                new_sums.add(x + s)
            if new_sums & sums_S:
                continue
            # Check internal duplicates in new_sums
            # 2x == x+s iff x == s, which would mean x in S, but x not in S
            # x+s1 == x+s2 iff s1 == s2, so no issue
            return False
        return True

    def _find_redundant_elements(S, sums_S):
        """Find ALL elements whose removal still leaves S maximal."""
        redundant = []
        S_set = set(S)
        for i in range(len(S)):
            z = S[i]
            new_set = S[:i] + S[i+1:]
            if not new_set:
                continue
            new_sums = _get_sums(new_set)
            if _is_maximal_fast(new_set, new_sums):
                redundant.append(i)
        return redundant

    def _remove_redundant(S, sums_S):
        """Iteratively remove redundant elements, removing all possible at each step."""
        changed = True
        while changed:
            changed = False
            redundant = _find_redundant_elements(S, sums_S)
            if redundant:
                # Remove ALL redundant elements found in this pass
                for idx in sorted(redundant, reverse=True):
                    S = S[:idx] + S[idx+1:]
                sums_S = _get_sums(S)
                changed = True
        return S, sums_S

    def _local_search(S, sums_S, deadline):
        """Local search with removal, swap, and 2-swap moves, tracking best solution."""
        current_set = S[:]
        current_sums = sums_S
        best_set = current_set[:]
        best_sums = current_sums
        best_size = len(current_set)
        
        # Precompute for fast maximality checks
        def _maximality_score(set_to_check, sums_to_check):
            """Count how many elements can still be added (lower = more maximal)."""
            S_set = set(set_to_check)
            count = 0
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                new_sums = {2 * x}
                for s in set_to_check:
                    new_sums.add(x + s)
                if not (new_sums & sums_to_check):
                    count += 1
            return count
        
        while time.time() < deadline:
            current_size = len(current_set)
            if current_size <= 1:
                break
            
            # Track best
            if current_size < best_size:
                best_size = current_size
                best_set = current_set[:]
                best_sums = current_sums
            
            # Try removal first (most likely to improve)
            if current_size > 1:
                # Try all removals, collect all valid ones
                removals = []
                for i in range(current_size):
                    z = current_set[i]
                    new_set = current_set[:i] + current_set[i+1:]
                    if not new_set:
                        continue
                    new_sums = _get_sums(new_set)
                    if _is_maximal_fast(new_set, new_sums):
                        removals.append(i)
                
                if removals:
                    # Remove ALL valid redundant elements at once
                    for idx in sorted(removals, reverse=True):
                        current_set = current_set[:idx] + current_set[idx+1:]
                    current_sums = _get_sums(current_set)
                    continue
            
            # Try swaps - more systematic
            if current_size > 0:
                found_swap = False
                # Try each element for removal, then try to add a replacement
                for z_idx in range(current_size):
                    if time.time() > deadline:
                        break
                    z = current_set[z_idx]
                    temp_set = current_set[:z_idx] + current_set[z_idx+1:]
                    temp_sums = _get_sums(temp_set)
                    
                    # Try to add random x values
                    for _ in range(15):
                        x = random.randint(1, N)
                        if x in current_set:
                            continue
                        
                        if _can_add_with_sums(temp_set, temp_sums, x):
                            new_set = temp_set + [x]
                            new_sums = temp_sums.union([x+s for s in temp_set] + [2*x])
                            
                            if _is_maximal_fast(new_set, new_sums):
                                current_set = new_set
                                current_sums = new_sums
                                found_swap = True
                                break
                    
                    if found_swap:
                        break
                
                if found_swap:
                    continue
            
            # Try 2-swap: remove two elements, add two different ones
            if current_size >= 2:
                found_2swap = False
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    i1 = random.randint(0, current_size - 1)
                    i2 = random.randint(0, current_size - 1)
                    if i1 == i2:
                        continue
                    
                    temp_set = current_set[:i1] + current_set[i1+1:]
                    # Adjust i2 if needed
                    if i2 > i1:
                        i2 -= 1
                    temp_set = temp_set[:i2] + temp_set[i2+1:]
                    temp_sums = _get_sums(temp_set)
                    
                    # Try to add two random x values
                    x1 = random.randint(1, N)
                    x2 = random.randint(1, N)
                    if x1 in current_set or x2 in current_set or x1 == x2:
                        continue
                    
                    if _can_add_with_sums(temp_set, temp_sums, x1):
                        temp_set2 = temp_set + [x1]
                        temp_sums2 = temp_sums.union([x1+s for s in temp_set] + [2*x1])
                        if _can_add_with_sums(temp_set2, temp_sums2, x2):
                            new_set = temp_set2 + [x2]
                            new_sums = temp_sums2.union([x2+s for s in temp_set2] + [2*x2])
                            if _is_maximal_fast(new_set, new_sums):
                                current_set = new_set
                                current_sums = new_sums
                                found_2swap = True
                                break
                
                if found_2swap:
                    continue
            
            # If stuck, try adding an element to escape
            x = random.randint(1, N)
            if x not in current_set:
                if _can_add_with_sums(current_set, current_sums, x):
                    new_set = current_set + [x]
                    new_sums = current_sums.union([x+s for s in current_set] + [2*x])
                    current_set = new_set
                    current_sums = new_sums
                    continue
            
            break
        
        return best_set, best_sums

    best_size = N + 1
    best_set = []
    
    # Phase 1: Try structured sets and greedy to get initial solutions
    structured_sets = [
        [1, 3, 7, 15, 31, 63, 127],
        [1, 3, 7, 15, 31, 63],
        [1, 3, 7, 15, 31],
        [1, 3, 7, 15],
        [1, 3, 7],
        [1, 3],
        [1],
        [2, 5, 11, 23, 47, 95, 191],
        [2, 5, 11, 23, 47, 95],
        [2, 5, 11, 23, 47],
        [2, 5, 11, 23],
        [2, 5, 11],
        [2, 5],
        [4, 9, 21, 43, 87, 175],
        [4, 9, 21, 43, 87],
        [4, 9, 21, 43],
        [4, 9, 21],
        [4, 9],
        [6, 13, 27, 55, 111, 223],
        [6, 13, 27, 55, 111],
        [6, 13, 27, 55],
        [6, 13, 27],
        [6, 13],
        [8, 17, 35, 71, 143],
        [8, 17, 35, 71],
        [8, 17, 35],
        [8, 17],
        [10, 21, 43, 87, 175],
        [10, 21, 43, 87],
        [10, 21, 43],
        [10, 21],
        [12, 25, 51, 103, 207],
        [12, 25, 51, 103],
        [12, 25, 51],
        [12, 25],
    ]
    
    for s in structured_sets:
        if _is_sidon(s):
            s_sums = _get_sums(s)
            if _is_maximal_fast(s, s_sums):
                if len(s) < best_size:
                    best_size = len(s)
                    best_set = s[:]
    
    # Try multiple greedy seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        s = _greedy_maximal_sidon(seed=seed)
        if len(s) < best_size:
            best_size = len(s)
            best_set = s[:]
        
        # Also try greedy with different orderings
        s2 = _greedy_maximal_sidon(seed=seed + 1000, reverse=(seed % 2 == 0))
        if len(s2) < best_size:
            best_size = len(s2)
            best_set = s2[:]
    
    # Phase 2: Local search from the best initial solution
    if best_set:
        best_sums = _get_sums(best_set)
        
        # First, try to remove redundant elements
        best_set, best_sums = _remove_redundant(best_set, best_sums)
        best_size = len(best_set)
        
        # Then do local search
        deadline_local = min(deadline, start_time + 52.0)
        best_set, best_sums = _local_search(best_set, best_sums, deadline_local)
        best_size = len(best_set)
        
        # Try local search from a few other starting points
        for seed in range(12):
            if time.time() > deadline_local:
                break
            s = _greedy_maximal_sidon(seed=seed + 100)
            if len(s) < best_size + 3:  # Only if close to best
                s_sums = _get_sums(s)
                s, s_sums = _remove_redundant(s, s_sums)
                s, s_sums = _local_search(s, s_sums, deadline_local)
                if len(s) < best_size:
                    best_size = len(s)
                    best_set = s[:]
                    best_sums = s_sums
    
    # Phase 3: Conflict-graph based construction
    # Model: we want a small set S such that every x not in S is "blocked" by S.
    # x is blocked by S if S ∪ {x} is not Sidon, i.e., there exists y in S (or y=x)
    # such that the sum x+y (or 2x) equals some sum a+b with a,b in S (a<=b).
    #
    # Strategy: Start with a small candidate set, then try to "cover" all other elements.
    # For each x not in S, find which elements in S block it.
    
    if time.time() < deadline - 3.0:
        # Try random restarts with local search
        for attempt in range(30):
            if time.time() > deadline - 1.0:
                break
            
            # Generate a random small Sidon set
            random.seed(attempt + 500)
            selected = []
            sums = set()
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            
            max_size = random.choice([4, 5, 6, 7, 8])
            for x in candidates:
                added_sums = [x + a for a in selected] + [2 * x]
                if len(set(added_sums)) != len(added_sums):
                    continue
                if sums.intersection(added_sums):
                    continue
                selected.append(x)
                sums.update(added_sums)
                if len(selected) >= max_size:
                    break
            
            if not selected:
                continue
            
            s_sums = _get_sums(selected)
            if not _is_maximal_fast(selected, s_sums):
                continue
            
            if len(selected) < best_size:
                selected, s_sums = _remove_redundant(selected, s_sums)
                selected, s_sums = _local_search(selected, s_sums, deadline - 1.0)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_set = selected[:]
                    best_sums = s_sums
    
    # Phase 4: Targeted construction - try to build minimal maximal sets directly
    if time.time() < deadline - 2.0:
        # For each small size k, try to find a maximal Sidon set of size k
        for target_size in range(2, min(best_size, 12)):
            if time.time() > deadline - 1.0:
                break
            
            # Try multiple random constructions targeting this size
            for trial in range(20):
                if time.time() > deadline - 1.0:
                    break
                
                random.seed(target_size * 1000 + trial)
                selected = []
                sums = set()
                
                # Use weighted selection: prefer elements that block many others
                candidates = list(range(1, N + 1))
                
                for _ in range(target_size):
                    # Score each candidate by how many elements it would block
                    best_candidate = None
                    best_score = -1
                    
                    # Sample a subset for speed
                    sample = random.sample(candidates, min(50, len(candidates)))
                    for x in sample:
                        added_sums = [x + a for a in selected] + [2 * x]
                        if len(set(added_sums)) != len(added_sums):
                            continue
                        if sums.intersection(added_sums):
                            continue
                        
                        # Score: how many new elements does this block?
                        # An element y is blocked if adding y creates a sum collision
                        new_sums_test = set(added_sums)
                        blocked = 0
                        for y in range(1, N + 1):
                            if y in selected or y == x:
                                continue
                            y_new_sums = {2 * y} | {y + s for s in selected} | {y + x}
                            if y_new_sums & (sums | new_sums_test):
                                blocked += 1
                        
                        if blocked > best_score:
                            best_score = blocked
                            best_candidate = x
                    
                    if best_candidate is None:
                        break
                    
                    added_sums = [best_candidate + a for a in selected] + [2 * best_candidate]
                    selected.append(best_candidate)
                    sums.update(added_sums)
                    candidates.remove(best_candidate)
                
                if len(selected) != target_size:
                    continue
                
                s_sums = _get_sums(selected)
                if _is_maximal_fast(selected, s_sums):
                    if len(selected) < best_size:
                        selected, s_sums = _remove_redundant(selected, s_sums)
                        selected, s_sums = _local_search(selected, s_sums, deadline - 1.0)
                        if len(selected) < best_size:
                            best_size = len(selected)
                            best_set = selected[:]
                            best_sums = s_sums
    
    return best_set
