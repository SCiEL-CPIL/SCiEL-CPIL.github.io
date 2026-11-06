"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random
import numpy as np

N = 256

# EVOLVE-BLOCK-START

def _is_valid_sidon(selected):
    """Check if selected is a Sidon set (all unordered sums distinct)."""
    sums = set()
    n = len(selected)
    for i in range(n):
        for j in range(i, n):
            s = selected[i] + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, N):
    """Check if selected is inclusion-maximal: adding any missing element breaks Sidon property."""
    selected_set = set(selected)
    sums = set()
    n = len(selected)
    for i in range(n):
        for j in range(i, n):
            sums.add(selected[i] + selected[j])
    
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        if 2 * x in sums:
            continue
        conflict = False
        for a in selected:
            if x + a in sums:
                conflict = True
                break
        if not conflict:
            return False
    return True

def _build_blocking_matrix(selected, N):
    """
    Build a matrix where blocking_matrix[i][j] = 1 if element i in selected
    blocks element j from being added (j not in selected).
    Used for constraint propagation.
    """
    selected_set = set(selected)
    sums = set()
    n = len(selected)
    for i in range(n):
        for j in range(i, n):
            sums.add(selected[i] + selected[j])
    
    # For each missing element x, determine which selected elements block it
    # x is blocked if 2x in sums, or x+a in sums for some a in selected
    # We want to find minimal subsets of selected that still block all missing elements
    
    missing = [x for x in range(1, N + 1) if x not in selected_set]
    if not missing:
        return None, None
    
    # For each missing x, find which pairs (a,b) in selected create the blocking sum
    # x is blocked if there exist a,b in selected with a<=b such that:
    #   - 2x = a+b (diagonal case), or
    #   - x+a = b+c for some distinct pairs... actually simpler:
    # x can't be added if {2x} ∪ {x+a : a in selected} intersects sums
    
    # blocking[x] = set of indices in selected that "contribute" to blocking x
    # More precisely: x is blocked if ∃ a in selected: x+a ∈ sums, or 2x ∈ sums
    # The elements contributing to the blocking are those a where x+a ∈ sums,
    # plus any pair (p,q) where p+q = 2x
    
    # For minimum vertex cover formulation:
    # Left side: missing elements (requirements)
    # Right side: selected elements (candidates to keep)
    # Edge (x, i) exists if removing selected[i] would unblock x
    # i.e., x is ONLY blocked by sums involving selected[i]
    
    # Simpler approach: for each missing x, find all selected elements a such that
    # x+a ∈ sums. If we remove a, x+a might no longer be in sums (if a was part of the pair).
    # This is complex. Let's use a simpler greedy + local search instead.
    
    return None, None

def _greedy_with_constraint_prop(N, deadline, seed_offset=0):
    """
    Greedy construction with constraint propagation:
    1. Start with a random ordering
    2. At each step, pick the element that maximally satisfies blocking requirements
    3. Use constraint propagation to prune invalid choices early
    """
    rng = random.Random(1000 + seed_offset)
    start_time = time.time()
    
    # Random permutation of candidates
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    selected = []
    selected_set = set()
    sums = set()
    
    # Track which missing elements are still "unblocked" (need to be blocked)
    # Initially all elements are potentially addable
    unblocked = set(range(1, N + 1))
    
    for x in candidates:
        if time.time() > deadline:
            break
        
        # Skip if already selected
        if x in selected_set:
            continue
        
        # Check if x can be added to current Sidon set
        added_sums = [2 * x]
        for a in selected:
            added_sums.append(x + a)
        
        if len(set(added_sums)) != len(added_sums):
            continue
        if sums.intersection(set(added_sums)):
            continue
        
        # Constraint propagation: if we add x, which elements become blocked?
        # An element y is blocked by adding x if:
        #   - 2y ∈ {2x} ∪ {x+a : a in selected} ∪ sums
        #   - or y+a ∈ {2x} ∪ {x+a : a in selected} ∪ sums for some a in selected
        
        new_sums_if_add = sums | set(added_sums)
        
        newly_blocked = set()
        for y in unblocked:
            if y == x:
                continue
            # Check if y would be blocked
            if 2 * y in new_sums_if_add:
                newly_blocked.add(y)
                continue
            for a in selected:
                if y + a in new_sums_if_add:
                    newly_blocked.add(y)
                    break
            # Also check y+x
            if y + x in new_sums_if_add:
                newly_blocked.add(y)
        
        # Score: how many unblocked elements does this block?
        # We want to block as many as possible to minimize final set size
        block_count = len(newly_blocked)
        
        # Only add if it blocks at least one element (or if it's a good choice)
        # For now, add it and track
        selected.append(x)
        selected_set.add(x)
        unblocked.discard(x)
        unblocked -= newly_blocked
        sums.update(added_sums)
        
        # If all elements are blocked, we're done
        if not unblocked:
            break
    
    return selected

def _local_search_reduce(selected, N, deadline):
    """
    Try to remove elements from selected while maintaining Sidon property and maximality.
    Greedy removal: try removing each element, check if still maximal.
    """
    start_time = time.time()
    current = list(selected)
    current_set = set(current)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(current)):
            if time.time() > deadline:
                break
            
            # Try removing current[i]
            candidate = [x for j, x in enumerate(current) if j != i]
            if not candidate:
                continue
            
            # Check Sidon property
            if not _is_valid_sidon(candidate):
                continue
            
            # Check maximality
            if _is_maximal(candidate, N):
                # Success! Remove this element
                current = candidate
                current_set = set(candidate)
                improved = True
                break  # Restart loop after modification
    
    return current

def _try_remove_to_minimize(selected, N, deadline):
    """
    Iteratively try to remove elements to minimize cardinality while maintaining
    Sidon property and maximality.
    """
    start_time = time.time()
    current = list(selected)
    
    # Multiple passes of local search
    for _ in range(5):
        if time.time() > deadline:
            break
        current = _local_search_reduce(current, N, deadline)
    
    return current

def _matching_based_cover(N, deadline, seed_offset=0):
    """
    Use maximum matching on a bipartite graph to find a minimum vertex cover
    that blocks all missing elements.
    
    Bipartite graph:
    - Left: missing elements (requirements that must be blocked)
    - Right: candidate Sidon elements (potential set members)
    - Edge (x, a): element a helps block missing element x
    
    A vertex cover in this graph corresponds to a set of candidates that
    collectively block all missing elements.
    """
    rng = random.Random(2000 + seed_offset)
    start_time = time.time()
    
    # First, build a valid Sidon set (greedy)
    selected = []
    selected_set = set()
    sums = set()
    for x in range(1, N + 1):
        added = [2 * x] + [x + a for a in selected]
        if len(set(added)) == len(added) and not sums.intersection(set(added)):
            selected.append(x)
            selected_set.add(x)
            sums.update(added)
    
    if not _is_maximal(selected, N):
        # Make it maximal by adding more elements
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            added = [2 * x] + [x + a for a in selected]
            if len(set(added)) == len(added) and not sums.intersection(set(added)):
                selected.append(x)
                selected_set.add(x)
                sums.update(added)
                if _is_maximal(selected, N):
                    break
    
    # Now try to reduce using matching-based approach
    # For each missing element, find which selected elements are "critical" for blocking it
    
    missing = [x for x in range(1, N + 1) if x not in selected_set]
    if not missing:
        return selected
    
    # Build conflict structure
    # For each missing x, determine which selected elements contribute to its blocking
    # x is blocked if ∃ a in selected: x+a ∈ sums, or 2x ∈ sums
    
    # Criticality: element a is critical for blocking x if removing a would unblock x
    # This is complex to compute exactly. Use heuristic:
    # a is "important" for x if x+a ∈ sums and a is part of a unique pair producing that sum
    
    # Simpler: just use local search
    return _try_remove_to_minimize(selected, N, deadline)

def _diverse_greedy(N, deadline, seed_offset=0):
    """
    Diverse greedy: start from different random cores and expand.
    """
    rng = random.Random(3000 + seed_offset)
    start_time = time.time()
    
    # Start with a small random core
    core_size = rng.randint(3, 8)
    core_candidates = rng.sample(range(1, N + 1), min(core_size, N))
    
    selected = []
    selected_set = set()
    sums = set()
    
    # Try to build a valid Sidon set from the core
    for x in core_candidates:
        added = [2 * x] + [x + a for a in selected]
        if len(set(added)) == len(added) and not sums.intersection(set(added)):
            selected.append(x)
            selected_set.add(x)
            sums.update(added)
    
    # Expand greedily to make it maximal
    remaining = [x for x in range(1, N + 1) if x not in selected_set]
    rng.shuffle(remaining)
    
    for x in remaining:
        if time.time() > deadline:
            break
        added = [2 * x] + [x + a for a in selected]
        if len(set(added)) == len(added) and not sums.intersection(set(added)):
            selected.append(x)
            selected_set.add(x)
            sums.update(added)
            if _is_maximal(selected, N):
                break
    
    # Check if maximal, if not add more
    if not _is_maximal(selected, N):
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            added = [2 * x] + [x + a for a in selected]
            if len(set(added)) == len(added) and not sums.intersection(set(added)):
                selected.append(x)
                selected_set.add(x)
                sums.update(added)
                if _is_maximal(selected, N):
                    break
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    best_result = None
    best_size = float('inf')
    
    # Strategy 1: Constraint propagation greedy
    for i in range(5):
        if time.time() > deadline:
            break
        result = _greedy_with_constraint_prop(N, deadline, i)
        if result and _is_valid_sidon(result) and _is_maximal(result, N):
            result = _try_remove_to_minimize(result, N, deadline)
            if _is_valid_sidon(result) and _is_maximal(result, N):
                if len(result) < best_size:
                    best_size = len(result)
                    best_result = result
    
    # Strategy 2: Matching-based cover
    for i in range(3):
        if time.time() > deadline:
            break
        result = _matching_based_cover(N, deadline, i)
        if result and _is_valid_sidon(result) and _is_maximal(result, N):
            if len(result) < best_size:
                best_size = len(result)
                best_result = result
    
    # Strategy 3: Diverse greedy with local search
    for i in range(10):
        if time.time() > deadline:
            break
        result = _diverse_greedy(N, deadline, i)
        if result and _is_valid_sidon(result) and _is_maximal(result, N):
            result = _try_remove_to_minimize(result, N, deadline)
            if _is_valid_sidon(result) and _is_maximal(result, N):
                if len(result) < best_size:
                    best_size = len(result)
                    best_result = result
    
    if best_result is None:
        # Fallback: simple ascending greedy
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected
    
    return best_result

# EVOLVE-BLOCK-END
