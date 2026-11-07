import time
import heapq
import math

def _build_divisibility_table(max_val=256):
    """Precompute which values divide which others for fast lookup."""
    # divides[a] = set of values that a divides (a | b means a divides b)
    # For each a, b is a multiple of a
    divides = [set() for _ in range(max_val + 1)]
    for a in range(1, max_val + 1):
        for b in range(a, max_val + 1, a):
            if b != a:
                divides[a].add(b)
    return divides

def _can_add(sums, x, divides):
    """
    Check if x can be added to a set with current non-empty subset sums 'sums'.
    Returns True if valid, False otherwise.
    """
    # Generate new sums: x and s+x for all s in sums
    new_sums = set()
    new_sums.add(x)
    for s in sums:
        ns = s + x
        if ns in new_sums:
            return False  # duplicate sum
        new_sums.add(ns)
    
    # Check divisibility within new_sums
    new_list = sorted(new_sums)
    for i in range(len(new_list)):
        a = new_list[i]
        if a > 256:
            continue
        # Check if any other new_sum is a multiple of a
        for j in range(i + 1, len(new_list)):
            b = new_list[j]
            if b % a == 0:
                return False
    
    # Check divisibility between sums and new_sums
    for s in sums:
        for ns in new_sums:
            if s == ns:
                return False
            if s % ns == 0 or ns % s == 0:
                return False
    
    return True

def _add_element(sums, x):
    """Update sums to include new subset sums when adding x."""
    old_sums = set(sums)
    for s in old_sums:
        sums.add(s + x)
    sums.add(x)

def _validate_full(arr):
    """Full validation of a set of integers."""
    n = len(arr)
    all_sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += arr[i]
        if s in all_sums:
            return False
        all_sums.add(s)
    
    sums_list = sorted(all_sums)
    for i in range(len(sums_list)):
        a = sums_list[i]
        for j in range(i + 1, len(sums_list)):
            b = sums_list[j]
            if b % a == 0:
                return False
    return True

def construct():
    deadline = time.time() + 80  # Safety margin
    
    # Precompute divisibility table
    divides = _build_divisibility_table(256)
    
    best_size = 0
    best_set = []
    
    # Strategy: Try to build sets by starting with different "anchors"
    # and using a priority-based greedy that prefers "safer" additions
    
    # Phase 1: Systematic search over starting elements
    # Prefer primes and larger numbers as they are "safer"
    
    # Generate candidate ordering: primes first, then composites, larger preferred
    candidates_by_safety = []
    for x in range(256, 0, -1):  # Larger numbers first (less likely to be divisors)
        # Score: prefer primes, then numbers with fewer small factors
        is_prime = True
        for d in range(2, int(math.sqrt(x)) + 1):
            if x % d == 0:
                is_prime = False
                break
        safety_score = 100 if is_prime else 0
        # Bonus for being larger
        safety_score += x // 10
        candidates_by_safety.append((safety_score, x))
    
    candidates_by_safety.sort(key=lambda t: -t[0])  # Descending by safety
    
    # Try multiple starting points
    start_candidates = [x for _, x in candidates_by_safety[:50]]  # Top 50 safest starts
    
    for start_x in start_candidates:
        if time.time() > deadline:
            break
        
        # Start with this element
        current = [start_x]
        sums = set([start_x])
        
        # Build a priority queue of remaining candidates
        # Priority: how "safe" is adding this candidate?
        # We'll use a simple heuristic: candidates that don't share factors with existing sums
        
        remaining = [x for x in range(1, 257) if x != start_x]
        
        # Sort remaining by a heuristic: prefer numbers that are coprime to most existing sums
        def _safety_score(x):
            score = 0
            for s in sums:
                g = math.gcd(x, s)
                if g == 1:
                    score += 1
                elif g == 2:
                    score += 0.5
            # Bonus for being prime
            is_prime = True
            for d in range(2, int(math.sqrt(x)) + 1):
                if x % d == 0:
                    is_prime = False
                    break
            if is_prime:
                score += 2
            return score
        
        # Use a priority queue: higher score = try first
        # But we want to re-evaluate as sums grow, so we'll do iterative greedy
        
        for _iteration in range(30):  # Max 30 elements
            if time.time() > deadline:
                break
            
            if len(current) >= 30:
                break
            
            # Re-score remaining candidates based on current sums
            scored_remaining = []
            for x in remaining:
                if x in current:
                    continue
                # Quick check: x must not already be in sums
                if x in sums:
                    continue
                # Quick divisibility check with existing sums
                valid_quick = True
                for s in sums:
                    if s % x == 0 or x % s == 0:
                        valid_quick = False
                        break
                if not valid_quick:
                    continue
                
                score = _safety_score(x)
                scored_remaining.append((score, x))
            
            if not scored_remaining:
                break
            
            # Try candidates in order of decreasing safety score
            scored_remaining.sort(key=lambda t: -t[0])
            
            added = False
            for score, x in scored_remaining:
                if time.time() > deadline:
                    break
                
                # Full validation
                if _can_add(sums, x, divides):
                    current.append(x)
                    _add_element(sums, x)
                    remaining.remove(x)
                    added = True
                    break
            
            if not added:
                break
        
        # Update best
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    # Phase 2: Local improvement - try to replace elements to increase size
    if best_set and time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try swapping each element with a different one
            for i in range(len(best_set)):
                if time.time() > deadline - 5:
                    break
                old_val = best_set[i]
                for new_val in range(1, 257):
                    if new_val == old_val or new_val in best_set:
                        continue
                    # Try the swap
                    candidate = best_set[:]
                    candidate[i] = new_val
                    if _validate_full(candidate):
                        # Check if we can now add more elements
                        sums = set()
                        for mask in range(1, 1 << len(candidate)):
                            s = 0
                            for j in range(len(candidate)):
                                if mask & (1 << j):
                                    s += candidate[j]
                            sums.add(s)
                        
                        # Try to add more
                        for add_val in range(1, 257):
                            if add_val in candidate:
                                continue
                            if add_val in sums:
                                continue
                            if _can_add(sums, add_val, divides):
                                candidate.append(add_val)
                                break
                        
                        if len(candidate) > len(best_set):
                            best_set = candidate
                            improved = True
                            break
                if improved:
                    break
    
    # Final validation
    if best_set:
        if not _validate_full(best_set):
            # Fallback
            best_set = [31, 32, 34]
    
    return list(best_set)
