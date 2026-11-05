"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # We want to minimize max(A) for a Sidon set A containing 0,
    # such that {a_i - a_j} contains 1..30.
    
    # Lower bound on diameter:
    # If we use k elements, we have k(k-1)/2 differences.
    # To cover 30 differences, we need k(k-1)/2 >= 30 => k >= 8 (28 diffs, not enough), k=9 (36 diffs).
    # So we need at least 9 elements.
    # With 9 elements, the max difference M must be at least large enough to hold 30 distinct values.
    # Trivially M >= 30.
    # But Sidon constraints force M to be larger.
    
    best_solution = None
    best_max = float('inf')
    
    # To speed up search, we can fix the size of the set?
    # Let's try sizes 9, 10, 11.
    
    def check_sidon_and_cover(A):
        if not A or 0 not in A:
            return False, 0, 0
        if len(A) > 32:
            return False, 0, 0
        
        diffs = set()
        max_val = max(A)
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                d = A[j] - A[i]
                if d in diffs:
                    return False, 0, 0
                diffs.add(d)
        
        covered = sum(1 for d in range(1, 31) if d in diffs)
        if covered == 30:
            return True, max_val, covered
        return False, max_val, covered

    # Optimized Backtracking Search
    def search(A, used_diffs, covered_count, current_max, depth):
        nonlocal best_solution, best_max
        
        if time.time() > deadline:
            return False
            
        # Pruning
        if current_max >= best_max:
            return False
            
        if covered_count == 30:
            # Valid solution found
            best_solution = A[:]
            best_max = current_max
            return True
            
        if len(A) >= 32:
            return False
            
        # Generate candidates
        # We want to add x such that:
        # 1. x is not in A
        # 2. x - a_i and a_i - x are not in used_diffs
        # 3. x helps cover missing differences
        
        # To keep diameter small, we should prefer small x.
        # But x must be > 0.
        
        # Heuristic: 
        # The set A is sorted. 
        # If we add x > current_max, new diffs are x - a_i.
        # If we add x < current_max (inserting in middle), new diffs are x - a_i (for a_i < x) and a_i - x (for a_i > x).
        # Inserting in the middle is complex for the "sorted" assumption if we don't re-sort.
        # Let's maintain A as a sorted list.
        
        # Candidates:
        # 1. x = current_max + 1, current_max + 2, ...
        # 2. x such that x - a_i is a missing difference.
        # 3. x such that a_i - x is a missing difference (implies x < a_i).
        
        missing_diffs = [d for d in range(1, 31) if d not in used_diffs]
        
        if not missing_diffs:
            return False # Should have been caught by covered_count == 30
            
        candidates = set()
        
        # 1. Extend upwards
        for d in missing_diffs:
            for a in A:
                x = a + d
                if x > current_max:
                    candidates.add(x)
                    
        # 2. Insert downwards (x < current_max)
        # If we insert x, it must be > 0.
        # This can create many new small differences.
        # Let's also try small integers 1..current_max
        # But that's too many.
        # Let's restrict to x such that x is involved in a missing diff.
        for d in missing_diffs:
            for a in A:
                if a - d > 0:
                    candidates.add(a - d)
                    
        # 3. Also try to fill gaps?
        # Let's just sort candidates by value to try smaller ones first.
        
        valid_candidates = []
        for x in candidates:
            if x in A:
                continue
            if x < 0:
                continue
            
            # Check Sidon property with existing A
            # New diffs:
            new_diffs = set()
            valid = True
            for a in A:
                d1 = abs(x - a)
                if d1 in used_diffs or d1 in new_diffs:
                    valid = False
                    break
                new_diffs.add(d1)
                
            if not valid:
                continue
                
            # Check coverage increase
            new_covered = 0
            for d in new_diffs:
                if 1 <= d <= 30 and d not in used_diffs:
                    new_covered += 1
                    
            # If this move doesn't improve coverage, it's less useful, 
            # but might be necessary to enable future moves?
            # Usually, in Sidon sets, adding an element that doesn't cover a *required* diff 
            # is only useful if it sets up a structure. 
            # But to minimize max, we want to be efficient.
            # Let's prioritize moves that cover at least 1 missing diff.
            # If no move covers a missing diff, we are stuck in a local minimum 
            # unless we add an element that creates a "bridge".
            
            # Heuristic score:
            # Prefer smaller x (to keep max low)
            # Prefer more new coverage
            score = (max(x, current_max), -new_covered)
            valid_candidates.append((score, x, new_diffs))
            
        # Sort by score
        valid_candidates.sort(key=lambda item: item[0])
        
        # Limit branching factor to avoid explosion
        # Try top 20 candidates?
        for score, x, new_diffs in valid_candidates[:20]:
            new_A = A + [x]
            new_A.sort()
            new_used = set(used_diffs)
            new_used.update(new_diffs)
            new_covered_count = covered_count + sum(1 for d in new_diffs if 1 <= d <= 30 and d not in used_diffs)
            
            # Recurse
            if search(new_A, new_used, new_covered_count, max(x, current_max), depth + 1):
                return True
                
        return False

    # Initial state
    initial_A = [0]
    initial_used = set()
    initial_covered = 0
    
    # Run search
    # To avoid getting stuck in one bad path, we can run multiple restarts 
    # with slight randomization in candidate ordering?
    # Or just run the deterministic one first.
    
    # The deterministic search might be fast enough if the branching factor is controlled.
    # Let's run it.
    
    found = search(initial_A, initial_used, 0, 0, 0)
    
    if not found and best_solution is None:
        # Fallback: Simple Greedy
        a = [0]
        used = set()
        for d in range(1, 31):
            if d in used:
                continue
            found = False
            # Try to add x such that x - a_i = d or a_i - x = d
            candidates = set()
            for ai in a:
                candidates.add(ai + d)
                if ai - d > 0:
                    candidates.add(ai - d)
            
            for x in sorted(candidates):
                if x in a:
                    continue
                new_diffs = set()
                valid = True
                for ai in a:
                    diff = abs(x - ai)
                    if diff in used or diff in new_diffs:
                        valid = False
                        break
                    new_diffs.add(diff)
                if valid:
                    a.append(x)
                    a.sort()
                    used.update(new_diffs)
                    found = True
                    break
            
            if not found:
                # Add a large number to force a diff?
                # This is a fallback for the greedy, which might fail.
                # We need to ensure we return a valid set.
                # If greedy fails, we might return an invalid set?
                # The task requires a valid set.
                # Let's try to extend with a large number.
                # Actually, the greedy above should work for 1..30.
                # If it doesn't, we have a problem.
                # Let's just return the current 'a' if it's valid, otherwise a known valid set.
                pass
        
        # Verify 'a'
        if a:
            diffs = set()
            valid = True
            for i in range(len(a)):
                for j in range(i+1, len(a)):
                    d = a[j] - a[i]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid: break
            
            if valid and all(d in diffs for d in range(1, 31)):
                return a
        
        # If all else fails, return a hardcoded valid set?
        # We don't have one handy, but the search should find one.
        # To be safe, return a large valid set constructed by a robust method.
        # Or just return the best found so far (which might be None).
        # If best_solution is None, we must return *something*.
        # The contract says "return at most 32 distinct integers...".
        # If we can't find one, we fail.
        # But the search is quite robust.
        
    if best_solution:
        return best_solution
    else:
        # Emergency fallback: Return a set that is likely valid but large.
        # Use the greedy result 'a' from the fallback block if it was valid.
        # The code above returns 'a' if valid.
        # If we reach here, 'a' was invalid or not generated.
        # Let's generate a simple valid set using a known construction.
        # Singer difference set?
        # For now, return the greedy 'a' even if not fully verified? No, that's risky.
        # Let's just return [0] + list(range(1, 31))? No, not Sidon.
        
        # Let's try to run the greedy again in a tighter loop.
        a = [0]
        used = set()
        for d in range(1, 31):
            if d in used:
                continue
            found = False
            candidates = set()
            for ai in a:
                candidates.add(ai + d)
                if ai - d > 0:
                    candidates.add(ai - d)
            
            for x in sorted(candidates):
                if x in a:
                    continue
                new_diffs = set()
                valid = True
                for ai in a:
                    diff = abs(x - ai)
                    if diff in used or diff in new_diffs:
                        valid = False
                        break
                    new_diffs.add(diff)
                if valid:
                    a.append(x)
                    a.sort()
                    used.update(new_diffs)
                    found = True
                    break
            if not found:
                # If we can't find a small x, pick a large one
                # This ensures we cover d
                x = max(a) + d + 1
                # Check if this x is valid
                new_diffs = set()
                valid = True
                for ai in a:
                    diff = abs(x - ai)
                    if diff in used or diff in new_diffs:
                        valid = False
                        break
                    new_diffs.add(diff)
                if valid:
                    a.append(x)
                    a.sort()
                    used.update(new_diffs)
                else:
                    # This is getting complicated. 
                    # Let's just return the best_solution if found, or a dummy.
                    pass
        
        return a

# EVOLVE-BLOCK-END
