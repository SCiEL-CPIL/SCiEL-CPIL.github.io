"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256
DEADLINE_MARGIN = 0.5  # seconds before the 30s limit to stop searching


# EVOLVE-BLOCK-START
def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set (all unordered sums a+b with a<=b distinct)."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            s = selected[i] + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _can_add(selected, sums, x):
    """Check if x can be added to selected without breaking Sidon property."""
    new_sums = set()
    for a in selected:
        s = a + x
        if s in sums or s in new_sums:
            return False
        new_sums.add(s)
    s2 = 2 * x
    if s2 in sums or s2 in new_sums:
        return False
    return True


def _build_sums(selected):
    """Build the set of all unordered sums for a Sidon set."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            sums.add(selected[i] + selected[j])
    return sums


def _is_maximal(selected, sums):
    """Check if the Sidon set is inclusion-maximal in [1,N]."""
    for x in range(1, N + 1):
        if x not in selected and _can_add(selected, sums, x):
            return False
    return True


def _greedy_randomized(seed=None, deadline=None):
    """
    Greedy randomized construction: pick elements in random order,
    add if it doesn't break Sidon property. Then try to maximize by
    checking if it's maximal; if not, we just return what we have
    (the problem asks for maximal, so we need to ensure maximality).
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    # Random permutation of [1, N]
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    selected = []
    sums = set()
    
    for x in candidates:
        if _can_add(selected, sums, x):
            selected.append(x)
            # Update sums
            for a in selected[:-1]:  # all previously selected
                sums.add(a + x)
            sums.add(2 * x)
        
        if deadline is not None and time.time() > deadline:
            break
    
    # Check maximality; if not maximal, try to force maximality
    # by adding more elements (but we can't add if it breaks Sidon)
    # Actually, if it's not maximal, there exists some x we could add.
    # But our greedy already tried all candidates in random order.
    # If we couldn't add x, it's because it conflicts. So if we went
    # through all candidates, the set IS maximal (no element can be added).
    
    return selected


def _greedy_max_blocking(seed=None, deadline=None):
    """
    Greedy heuristic that selects the element which blocks the most other candidates.
    This is effective for finding small maximal Sidon sets.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    selected = []
    sums = set()
    remaining = set(range(1, N + 1))
    
    while remaining:
        if deadline is not None and time.time() > deadline:
            break
        
        best_x = None
        best_block_count = -1
        
        # Evaluate all remaining candidates
        # Since N=256, iterating over all remaining (max 256) and checking conflicts
        # with all other remaining (max 256) is ~65k operations per step.
        # With ~20-30 steps, this is very fast.
        
        for x in remaining:
            # First, check if x can even be added to current selected
            if not _can_add(selected, sums, x):
                continue
            
            # Count how many other elements y in remaining would be blocked by x
            # y is blocked if:
            # 1. x + y is in sums (existing)
            # 2. 2*x is in sums (impossible if x can be added, but 2*y might be)
            # 3. For any a in selected, a + y is in sums (y conflicts with selected)
            #    Wait, if y conflicts with selected, y is already blocked regardless of x.
            #    We want to count how many y that are *currently* addable become *unaddable* after adding x.
            
            block_count = 0
            for y in remaining:
                if y == x:
                    continue
                # Check if y is currently addable
                if not _can_add(selected, sums, y):
                    continue
                
                # Check if y would become unaddable if x is added
                # New sums added by x: {x + a for a in selected} U {2*x}
                # y is blocked if:
                # 1. y + x is in sums (old) -> but we checked x+y not in sums for x to be addable? No, _can_add checks x against sums.
                #    Actually, if x is addable, x+y is NOT in sums.
                #    But after adding x, x+y becomes a new sum.
                #    Does y conflict with this new sum?
                #    y conflicts if there exists b in selected U {x} such that y + b is already in sums (new or old).
                #    
                #    Let's simulate adding x to sums temporarily?
                #    Or just check conditions:
                #    y is blocked if:
                #    a) 2*y is in sums_old U {x+a, 2x}
                #    b) y + a is in sums_old U {x+b, 2x} for some b in selected
                #       Note: y+x is the new sum itself. y+x is not in sums_old.
                #       So we check if y+x equals some other existing sum? No, Sidon property is about distinct sums.
                #       y is blocked if adding y would create a duplicate sum.
                #       Sums created by y: {y + a for a in selected U {x}} U {2*y}
                #       Any of these must be in sums_old U {x + a for a in selected} U {2*x}
                
                # Simplified check:
                # 1. Check 2*y
                if 2 * y in sums:
                    continue # y was already blocked? No, _can_add(y) passed, so 2*y not in sums.
                
                # Check if y+x creates conflict
                # y+x is a new sum. It must not equal any existing sum in sums.
                # We know x is addable, so x+a not in sums for all a in selected, and 2x not in sums.
                # And x+y not in sums.
                # So x+y is a unique new sum.
                
                # However, y might conflict with x via other paths?
                # If we add x, the set of sums becomes S' = S U {x+a, 2x}.
                # Can we add y to S'?
                # We need {y+a' for a' in selected U {x}} U {2y} to be disjoint from S' and each other.
                
                # Since y was addable to S, {y+a for a in selected} U {2y} was disjoint from S.
                # Now we add x.
                # Potential conflicts:
                # 1. 2y == x + a (for some a in selected)
                # 2. 2y == 2x
                # 3. y + x == x + a (for some a in selected) -> y == a (impossible, y in remaining, a in selected)
                # 4. y + x == 2x -> y == x (impossible)
                # 5. y + a == x + b (for some a, b in selected)
                # 6. y + a == 2x (for some a in selected)
                
                # Let's check these specific conflicts
                blocked = False
                
                # 1. 2y == x + a
                for a in selected:
                    if 2 * y == x + a:
                        blocked = True
                        break
                if blocked:
                    continue
                    
                # 2. 2y == 2x
                if 2 * y == 2 * x:
                    blocked = True
                    continue
                    
                # 5 & 6. y + a == x + b or 2x
                for a in selected:
                    # y + a == x + b  => y - x = b - a
                    # y + a == 2x     => y - x = x - a
                    # We can just check if (y+a) is in the set of new sums from x?
                    # New sums from x: {x+a for a in selected} U {2x}
                    # If (y+a) is in {x+b for b in selected} U {2x}, then conflict.
                    # Note: y+a is NOT in S (since y was addable).
                    # So we just check if y+a equals any new sum from x.
                    
                    s_ya = y + a
                    # Check if s_ya is x + b for some b in selected
                    # s_ya == x + b => b = s_ya - x
                    # b must be in selected
                    if s_ya - x in selected:
                        blocked = True
                        break
                    # Check if s_ya == 2x
                    if s_ya == 2 * x:
                        blocked = True
                        break
                        
            if not blocked:
                # This logic is getting complicated and error-prone.
                # Let's use a simpler proxy:
                # Count how many y in remaining have x+y in sums OR 2x in sums (already checked) OR ...
                # Actually, the simplest correct way is to simulate the addition.
                pass
                
            # Let's restart the block counting with a robust simulation
            block_count = 0
            # Temporarily add x
            new_sums = set(sums)
            for a in selected:
                new_sums.add(x + a)
            new_sums.add(2 * x)
            
            for y in remaining:
                if y == x:
                    continue
                # Check if y can be added to (selected + [x]) with sums = new_sums
                # This is O(|selected|) per y.
                # Total O(|remaining| * |selected|) per x.
                # Total O(|remaining|^2 * |selected|) per step.
                # 256^2 * 20 ~ 1.3M operations. Might be too slow in Python for 30s?
                # 1.3M * 20 steps = 26M ops. Python does ~10-50M simple ops/sec.
                # This is borderline.
                
                # Let's optimize:
                # We only care if y was addable before and is not addable now.
                # If y was not addable before, it doesn't contribute to "blocking".
                
                # To speed up, we can precompute which y are addable.
                # But the set of addable y changes.
                
                # Alternative: Just count conflicts with existing sums for x.
                # If x creates a sum s, any y such that y+a=s or 2y=s is blocked.
                # This is faster.
                pass
                
        # Given the complexity, let's stick to a simpler heuristic that is known to work well:
        # Randomized greedy with a "look-ahead" of 1 or just pure random restarts with the existing `_greedy_randomized` but more of them.
        # Or, use the existing `_greedy_with_backtrack_heuristic` but fix the logic to be correct and fast.
        
        # Let's try a different angle:
        # The current score is 1.23. The seed is 1.0.
        # A better heuristic:
        # 1. Start with empty set.
        # 2. At each step, pick x that minimizes the number of "future options" left.
        #    This is the "Min-Remaining-Values" heuristic from CSPs.
        
        # Let's implement Min-Remaining-Values (MRV) properly.
        # For each x in remaining:
        #   If x is not addable, skip.
        #   Count how many y in remaining are STILL addable after adding x.
        #   Pick x that minimizes this count.
        
        # This is exactly what I started doing above. Let's optimize the inner loop.
        
        best_x = None
        min_remaining_options = float('inf')
        
        for x in remaining:
            if not _can_add(selected, sums, x):
                continue
            
            # Calculate new sums
            new_sums = set(sums)
            for a in selected:
                new_sums.add(x + a)
            new_sums.add(2 * x)
            
            # Count how many y in remaining are addable with new_sums
            addable_count = 0
            for y in remaining:
                if y == x:
                    continue
                # Check if y is addable with new_sums
                # Optimized check:
                # y is addable if:
                # 1. 2y not in new_sums
                # 2. y+a not in new_sums for all a in selected
                # 3. y+x not in new_sums (but y+x IS in new_sums, so this is always false? No, y+x is the sum itself.
                #    Wait, _can_add checks if NEW sums conflict with EXISTING sums.
                #    If we add y to selected+{x}, the new sums are {y+a for a in selected} U {y+x} U {2y}.
                #    These must be disjoint from new_sums (which includes x+a and 2x).
                #    And disjoint from each other.
                
                # Since y is distinct from x, y+x is unique.
                # We just need to check:
                # a) 2y not in new_sums
                # b) y+a not in new_sums for all a in selected
                
                ok = True
                if 2 * y in new_sums:
                    ok = False
                else:
                    for a in selected:
                        if (y + a) in new_sums:
                            ok = False
                            break
                
                if ok:
                    addable_count += 1
            
            if addable_count < min_remaining_options:
                min_remaining_options = addable_count
                best_x = x
                
            # Early exit if 0
            if min_remaining_options == 0:
                break
        
        if best_x is None:
            break
        
        # Add best_x
        selected.append(best_x)
        for a in selected[:-1]:
            sums.add(a + best_x)
        sums.add(2 * best_x)
        remaining.discard(best_x)
        
    return selected


def _local_search_improve(selected, sums, deadline=None):
    """
    Try to improve (reduce size) by removing elements and re-optimizing.
    Actually, we want to find a SMALLER maximal set, not a larger one.
    This is tricky. Instead, let's try different random seeds and keep the smallest maximal set.
    """
    return selected


def construct():
    start_time = time.time()
    deadline = start_time + 30 - DEADLINE_MARGIN
    
    best_set = None
    best_size = float('inf')
    
    # Approach 1: MRV Greedy (Min Remaining Values)
    # This tends to produce smaller maximal sets by forcing early decisions that block many options.
    for seed in range(200):
        if time.time() > deadline:
            break
        s = _greedy_max_blocking(seed=seed, deadline=deadline)
        if len(s) < best_size:
            sums = _build_sums(s)
            if _is_valid_sidon(s) and _is_maximal(s, sums):
                best_set = s
                best_size = len(s)
                
    # Approach 2: Random Greedy as backup/diversity
    for seed in range(100):
        if time.time() > deadline:
            break
        s = _greedy_randomized(seed=seed, deadline=deadline)
        if len(s) < best_size:
            sums = _build_sums(s)
            if _is_valid_sidon(s) and _is_maximal(s, sums):
                best_set = s
                best_size = len(s)
    
    # Fallback
    if best_set is None:
        selected = []
        sums = set()
        for x in range(1, N + 1):
            if _can_add(selected, sums, x):
                selected.append(x)
                for a in selected[:-1]:
                    sums.add(a + x)
                sums.add(2 * x)
        best_set = selected
    
    # Final verification
    sums = _build_sums(best_set)
    assert _is_valid_sidon(best_set), "Result is not a valid Sidon set"
    assert _is_maximal(best_set, sums), "Result is not maximal"
    
    return best_set
# EVOLVE-BLOCK-END
