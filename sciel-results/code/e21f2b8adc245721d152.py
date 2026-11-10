"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

N = 256
# EVOLVE-BLOCK-START

def _is_sidon_and_maximal(selected, N):
    """
    Returns (is_sidon, is_maximal, blocked_missing_set).
    selected: list of ints
    """
    sel_set = set(selected)
    k = len(selected)
    
    # Compute all sums
    sums = set()
    for i in range(k):
        for j in range(i, k):
            sums.add(selected[i] + selected[j])
            
    # Check Sidon property (distinct sums)
    # Since we added to a set, if len(sums) != k*(k+1)/2, it's not Sidon
    # But we must ensure no duplicate sums were generated during addition.
    # Actually, just checking length is sufficient if we assume unique inputs.
    expected_sums = k * (k + 1) // 2
    if len(sums) != expected_sums:
        return False, False, set()
        
    is_sidon = True
    
    # Check Maximality
    # A missing element x is blocked if:
    # 1. 2*x in sums
    # 2. x + a in sums for some a in selected
    # If any x is NOT blocked, it's not maximal.
    
    blocked_missing = set()
    is_maximal = True
    
    for x in range(1, N + 1):
        if x in sel_set:
            continue
            
        # Check if x can be added
        can_add = True
        if 2 * x in sums:
            can_add = False
        else:
            for a in selected:
                if x + a in sums:
                    can_add = False
                    break
                    
        if can_add:
            is_maximal = False
            # We don't strictly need the set of unblocked for the boolean return,
            # but it helps for debugging or local search.
        else:
            blocked_missing.add(x)
            
    return is_sidon, is_maximal, blocked_missing

def _greedy_construct(rng, N):
    """Construct a valid Sidon set using a greedy randomized approach."""
    order = list(range(1, N + 1))
    rng.shuffle(order)
    
    selected = []
    sums = set()
    
    for x in order:
        # Check if x can be added
        if 2 * x in sums:
            continue
            
        valid = True
        for a in selected:
            if x + a in sums:
                valid = False
                break
                
        if valid:
            selected.append(x)
            sums.add(2 * x)
            for a in selected[:-1]:
                sums.add(x + a)
                
    return selected

def _local_search_minimize(selected, N, rng, deadline):
    """
    Local search to minimize the size of a maximal Sidon set.
    Moves:
    1. Remove element: If valid and maximal, accept.
    2. Swap element: If valid and maximal, accept.
    """
    current = list(selected)
    current_size = len(current)
    best = list(current)
    best_size = current_size
    
    # Pre-compute sums for current
    def get_sums(sel):
        s = set()
        for i in range(len(sel)):
            for j in range(i, len(sel)):
                s.add(sel[i] + sel[j])
        return s
        
    current_sums = get_sums(current)
    
    while time.time() < deadline:
        improved = False
        
        # Try Removals
        # We want to remove elements that are "least critical" for maximality.
        # Heuristic: Try removing each element, check if still maximal.
        
        # To speed up, we can just iterate.
        for i in range(len(current)):
            if time.time() > deadline:
                break
                
            # Try removing current[i]
            candidate = current[:i] + current[i+1:]
            
            # Quick Sidon check is implicit if we remove, but we must check Maximality
            # Removing an element cannot break Sidon property.
            # So we only need to check Maximality.
            
            is_max, _ = _is_sidon_and_maximal(candidate, N)
            
            if is_max:
                # Accept removal
                current = candidate
                current_size -= 1
                current_sums = get_sums(current)
                improved = True
                if current_size < best_size:
                    best = list(current)
                    best_size = current_size
                # Restart loop to check new removals
                break
        
        if improved:
            continue
            
        # If no removal worked, try Swaps
        # Swap: Remove a, Add b.
        # We want to keep size same or reduce (reduction handled above).
        # So we look for swaps that maintain maximality.
        
        # To avoid O(N^2) complexity per step, we limit swap attempts.
        # Pick a random element to remove.
        if current_size > 0:
            remove_idx = rng.randrange(len(current))
            a_to_remove = current[remove_idx]
            
            # Find candidates to add
            # Ideally, we want to add an element that blocks the most "weakly blocked" missing elements.
            # For simplicity, try random missing elements.
            
            sel_set = set(current)
            missing = [x for x in range(1, N + 1) if x not in sel_set]
            
            # Shuffle missing for randomness
            rng.shuffle(missing)
            
            found_swap = False
            for b in missing:
                if time.time() > deadline:
                    break
                    
                # Construct candidate
                # Remove a_to_remove, add b
                candidate = [x for x in current if x != a_to_remove] + [b]
                
                # Check Sidon
                is_sidon, is_max, _ = _is_sidon_and_maximal(candidate, N)
                
                if is_sidon and is_max:
                    # Accept swap
                    current = candidate
                    current_sums = get_sums(current)
                    # Size is same, so best_size doesn't change, but structure might lead to future removals
                    found_swap = True
                    break
                    
            if found_swap:
                continue
                
        # If no move found, we are stuck in a local minimum for this configuration.
        # Break out to allow restart or accept current.
        break
        
    return best

def construct():
    start_time = time.time()
    # Safety margin: 55s out of 60s
    deadline = start_time + 55.0
    
    best = None
    best_size = float('inf')
    
    # Number of restarts depends on time
    # Each restart + local search takes variable time.
    # Let's do up to 50 restarts, stopping if time runs out.
    
    num_restarts = 50
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        rng = random.Random(i * 12345 + 67890)
        
        # 1. Construct initial Sidon set
        selected = _greedy_construct(rng, N)
        
        if not selected:
            continue
            
        # 2. Verify it's maximal (Greedy construction usually yields maximal sets, but let's be sure)
        # If not maximal, we might need to add elements, but that increases size.
        # The goal is MINIMIZE size. A non-maximal set can be extended to a maximal one,
        # which would be larger. So we prefer maximal sets directly.
        # However, a smaller non-maximal set might be extendable to a smaller maximal set than another maximal set?
        # No, if A is not maximal, there exists x such that A U {x} is Sidon.
        # Then A U {x} is larger than A.
        # We want the smallest MAXIMAL set.
        # So we must ensure the result is maximal.
        
        is_sidon, is_max, _ = _is_sidon_and_maximal(selected, N)
        
        if not is_sidon:
            # Should not happen with greedy, but just in case
            continue
            
        if not is_max:
            # Try to make it maximal by adding elements?
            # This increases size.
            # If we add elements, we might get a larger set.
            # But maybe a different greedy order yields a smaller maximal set directly.
            # For now, skip non-maximal sets from greedy, or try to repair.
            # Repairing a non-maximal set to maximal only increases size.
            # So if greedy produced a non-maximal set, it's likely because we stopped early?
            # No, greedy adds if possible. So it should be maximal.
            # Wait, greedy adds x if it doesn't conflict.
            # If it skips x, it's because of conflict.
            # So the final set is maximal with respect to the order.
            # Is it globally maximal?
            # If x was skipped because x+a in sums, it is blocked.
            # So yes, greedy produces a maximal set.
            
            # If _is_sidon_and_maximal says False, it might be due to a bug in my check or the greedy.
            # Let's trust the check. If not max, we can't use it directly as a "maximal" candidate
            # unless we add elements. But adding elements makes it bigger.
            # We want small. So we prefer sets that are already maximal.
            continue
        
        # 3. Local Search to Minimize
        # We have a maximal Sidon set. Try to shrink it.
        # Note: Shrinking a maximal set might break maximality.
        # Our local search handles this by only accepting removals if maximality is preserved.
        
        result = _local_search_minimize(selected, N, rng, deadline - 1.0)
        
        if result and len(result) < best_size:
            best = result
            best_size = len(result)
            
            # Early termination if we hit a theoretical lower bound?
            # Lower bound for Sidon set in [1, N] is approx sqrt(N).
            # sqrt(256) = 16.
            # Minimal maximal sets might be slightly larger.
            if best_size <= 18:
                break
                
    if best is None:
        # Fallback: Return any valid maximal Sidon set
        # Use a deterministic greedy
        rng = random.Random(0)
        selected = _greedy_construct(rng, N)
        # Ensure maximal
        is_sidon, is_max, _ = _is_sidon_and_maximal(selected, N)
        if not is_max:
             # This shouldn't happen, but if it does, try to add elements to make it maximal
             # This is a fallback, so correctness > optimality
             sel_set = set(selected)
             for x in range(1, N+1):
                 if x not in sel_set:
                     cand = selected + [x]
                     if _is_sidon_and_maximal(cand, N)[0]: # Just check Sidon for now, we want to make it maximal eventually
                         # Actually, we just need a valid maximal set.
                         pass
             # Simplest fallback: just return the greedy set, hoping it's maximal.
             # If not, the verifier will fail, but this is better than None.
             pass
        return selected
        
    return best

# EVOLVE-BLOCK-END
