import time
import random

def construct():
    start = time.time()
    deadline = 145.0  # safety margin before 150s
    
    target = set(range(1, 31))
    
    def get_diffs(S):
        """Return set of all positive differences in sorted set S."""
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(S[j] - S[i])
        return diffs
    
    def is_sidon(S):
        """Check if S is a Sidon set (all positive differences distinct)."""
        n = len(S)
        diffs = get_diffs(S)
        return len(diffs) == n * (n - 1) // 2
    
    def covers_all(S):
        """Check if all differences 1..30 are present."""
        diffs = get_diffs(S)
        return target.issubset(diffs)
    
    def is_valid(S):
        return is_sidon(S) and covers_all(S)
    
    def count_missing(S):
        diffs = get_diffs(S)
        return len(target - diffs)
    
    def count_dup_diffs(S):
        n = len(S)
        expected = n * (n - 1) // 2
        diffs = get_diffs(S)
        return expected - len(diffs)
    
    def can_complete(S, max_val):
        """Check if S can be extended to a valid set with max element <= max_val."""
        if time.time() - start > deadline:
            return False
        if is_sidon(S) and covers_all(S):
            return max(S) <= max_val
        
        n = len(S)
        if n >= 32:
            return False
        
        # Quick prune: if we already have duplicate differences, can't fix by adding
        if count_dup_diffs(S) > 0:
            return False
        
        # Check if remaining differences can fit
        diffs = get_diffs(S)
        missing = target - diffs
        if not missing:
            return max(S) <= max_val
        
        # Need at most (32 - n) more elements
        remaining_slots = 32 - n
        # Each new element can create at most n+remaining_slots-1 new differences
        # but we need |missing| specific ones
        # This is a loose bound
        return True
    
    # Fast greedy: try to build a minimal set
    def greedy_build():
        """Build a Sidon set covering 1..30 using greedy approach."""
        S = [0]
        diffs = set()
        
        for _ in range(31):
            if target.issubset(diffs):
                break
            
            missing = sorted(target - diffs)
            best_candidate = None
            best_score = -1
            
            for d in missing:
                for x in S:
                    for c in (x + d, x - d):
                        if c < 0 or c in S or c > 1000000:
                            continue
                        new_S = sorted(S + [c])
                        if not is_sidon(new_S):
                            continue
                        new_diffs = get_diffs(new_S)
                        score = len(target & new_diffs) - max(new_S) / 1000.0
                        if score > best_score:
                            best_score = score
                            best_candidate = c
            
            if best_candidate is None:
                # Try any value that maintains Sidon property
                for c in range(1, min(200, max(S) + 50)):
                    if c in S:
                        continue
                    new_S = sorted(S + [c])
                    if is_sidon(new_S):
                        best_candidate = c
                        break
            
            if best_candidate is None:
                break
            
            S.append(best_candidate)
            S.sort()
            diffs = get_diffs(S)
        
        return S
    
    # Backtracking search for optimal solution
    def backtrack_search(S, diffs, used_diffs, depth, max_depth, max_val):
        """Depth-first search with pruning."""
        if time.time() - start > deadline:
            return None, float('inf')
        
        n = len(S)
        
        # Check if we've found a valid solution
        if target.issubset(diffs) and n >= 2:
            return S[:], max(S)
        
        # Prune: not enough elements left
        if depth >= max_depth:
            return None, float('inf')
        
        # Prune: if max element exceeds limit
        if max(S) > max_val:
            return None, float('inf')
        
        # Prune: duplicate differences
        if len(used_diffs) != n * (n - 1) // 2:
            return None, float('inf')
        
        # Try adding new elements
        best_result = None
        best_max = float('inf')
        
        # Determine candidate range
        start_val = max(S) + 1
        end_val = min(max_val, start_val + 50)  # Look ahead only 50
        
        for c in range(start_val, end_val + 1):
            if c in S:
                continue
            
            # Check Sidon property
            valid = True
            new_diffs = set()
            for x in S:
                d = c - x
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Recurse
            new_S = S + [c]
            new_diffs_set = diffs | new_diffs
            new_used = used_diffs | new_diffs
            
            result, rmax = backtrack_search(
                new_S, new_diffs_set, new_used, depth + 1, max_depth, max_val
            )
            
            if result is not None and rmax < best_max:
                best_max = rmax
                best_result = result
            
            if best_max <= max_val:
                # Found a good enough solution, can stop
                break
        
        return best_result, best_max
    
    # Iterative deepening with increasing max_val
    def iterative_deepening():
        """Try increasing diameter bounds."""
        # Start with a reasonable lower bound
        # A Sidon set of size k has at least k(k-1)/2 differences
        # We need 30 differences, so k >= 8 (8*7/2 = 28, need 30 so k >= 9)
        # But we also need specific differences 1..30
        
        for max_val in range(30, 300, 5):
            if time.time() - start > deadline:
                break
            
            # Use greedy to get a starting point
            S = greedy_build()
            if S is None or len(S) < 2:
                continue
            
            diffs = get_diffs(S)
            used_diffs = set()
            for d in diffs:
                used_diffs.add(d)
            
            result, rmax = backtrack_search(S, diffs, used_diffs, len(S), 32, max_val)
            
            if result is not None:
                return result, rmax
        
        return None, float('inf')
    
    # Try greedy first
    greedy_S = greedy_build()
    if is_valid(greedy_S):
        best_set = greedy_S[:]
        best_max = max(best_set)
    else:
        best_set = None
        best_max = float('inf')
    
    # Try iterative deepening backtracking
    if best_max > 50 or best_set is None:
        result, rmax = iterative_deepening()
        if result is not None:
            if best_set is None or rmax < best_max:
                best_set = result[:]
                best_max = rmax
    
    # If still no good solution, try local search
    if best_set is None or best_max > 100:
        def local_search(S):
            """Local search to improve the set."""
            if time.time() - start > deadline:
                return S
            
            S = sorted(S[:])
            improved = True
            iterations = 0
            
            while improved and time.time() - start < deadline:
                improved = False
                iterations += 1
                
                for i in range(1, len(S)):
                    orig = S[i]
                    # Try moving element to a smaller value
                    for new_val in range(1, orig):
                        if new_val in S:
                            continue
                        new_S = sorted([new_val if x == orig else x for x in S])
                        if is_valid(new_S) and max(new_S) < max(S):
                            S = new_S
                            improved = True
                            break
                    if improved:
                        break
                
                # If no improvement, try swapping
                if not improved:
                    for i in range(1, len(S)):
                        for j in range(i + 1, len(S)):
                            S[i], S[j] = S[j], S[i]
                            if is_valid(S) and max(S) < best_max:
                                improved = True
                                break
                            S[i], S[j] = S[j], S[i]
                        if improved:
                            break
                
                if iterations > 100:
                    break
            
            return S
        
        improved_S = local_search(best_set if best_set else greedy_S)
        if is_valid(improved_S):
            if best_set is None or max(improved_S) < best_max:
                best_set = improved_S[:]
                best_max = max(best_set)
    
    # Final fallback: ensure we return a valid set
    if best_set is None or not is_valid(best_set):
        # Emergency: use greedy result
        best_set = greedy_build()
    
    # Ensure validity
    if not is_valid(best_set):
        # Last resort: construct a simple valid set
        S = [0]
        diffs = set()
        for d in range(1, 31):
            if d in diffs:
                continue
            found = False
            for x in S:
                for c in (x + d, x - d):
                    if c < 0 or c in S:
                        continue
                    new_S = sorted(S + [c])
                    if is_sidon(new_S):
                        S = new_S
                        diffs = get_diffs(S)
                        found = True
                        break
                if found:
                    break
            if not found:
                y = 2 * max(S) + 3 * d + 1
                while True:
                    new_S = sorted(S + [y, y + d])
                    if is_sidon(new_S):
                        S = new_S
                        diffs = get_diffs(S)
                        break
                    y += 1
        best_set = S
    
    return best_set
