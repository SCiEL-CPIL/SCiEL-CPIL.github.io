import random
import time

N = 256

def construct():
    best_size = N + 1
    best_set = None
    
    # Random seed for reproducibility in case of ties, but we want variety in trials
    # We will run a few hundred iterations of the greedy heuristic
    
    for _ in range(2000):
        selected = []
        sums = set()
        candidates = set(range(1, N + 1))
        
        while candidates:
            # Identify valid candidates
            valid = []
            for x in candidates:
                ok = True
                for s in selected:
                    if x + s in sums:
                        ok = False
                        break
                if ok and 2 * x in sums:
                    ok = False
                if ok:
                    valid.append(x)
            
            if not valid:
                break
            
            # Heuristic: Choose x that maximizes the number of other valid candidates invalidated
            # This "aggressive" strategy tends to close the set faster.
            
            # Optimization: If valid set is small, just check all.
            # N=256, so |valid| <= 256.
            # Cost per step: |valid|^2 * |selected|.
            # |selected| ~ 10-20. |valid| starts at 256, decreases.
            # 256^2 * 20 ~ 1.3M ops per step. ~20 steps. ~26M ops per trial.
            # 2000 trials * 26M = 52B ops. Too slow in Python.
            
            # We need a faster heuristic or fewer trials.
            # Let's reduce trials to 100 and optimize the inner loop.
            # Or use a simpler heuristic: Pick the smallest valid element (fast) vs Pick element that blocks most (slow).
            # Let's try a mix: 50% smallest, 50% random, and see if we can find a good one quickly.
            # Actually, the "max destruction" is the key to *small* sets.
            # Let's optimize the "max destruction" check.
            
            # Instead of checking all pairs, we can just pick the element that has the most "conflicts" with the sum set?
            # No, the definition of invalidation depends on the NEW sums added.
            
            # Let's try a different approach:
            # Backtracking with pruning.
            # But the greedy "max destruction" is likely good enough if we run it ~100 times.
            # Let's optimize the inner loop by precomputing or breaking early.
            
            best_x = None
            max_invalidated = -1
            
            # To save time, we can sample if valid is large?
            # No, let's just run 100 trials.
            
            for x in valid:
                # Simulate adding x
                new_sums = {x + s for s in selected}
                new_sums.add(2 * x)
                
                invalidated = 0
                # Check other valid candidates
                for y in valid:
                    if y == x:
                        continue
                    # y is invalid if any y+s is in new_sums OR 2y is in new_sums
                    # Note: y+s for s in selected.
                    # If y+s is in new_sums, it means y+s = x+k for some k in selected U {x}.
                    # So y+k-x = s.
                    
                    # Simple check:
                    y_bad = False
                    for s in selected:
                        if y + s in new_sums:
                            y_bad = True
                            break
                    if not y_bad and 2 * y in new_sums:
                        y_bad = True
                            
                    if y_bad:
                        invalidated += 1
                        
                if invalidated > max_invalidated:
                    max_invalidated = invalidated
                    best_x = x
            
            # Commit best_x
            selected.append(best_x)
            for s in selected[:-1]:
                sums.add(best_x + s)
            sums.add(2 * best_x)
            
            candidates.remove(best_x)
            
            # Update candidates: remove those now invalid
            to_remove = []
            for y in candidates:
                y_bad = False
                for s in selected:
                    if y + s in sums:
                        y_bad = True
                        break
                if not y_bad and 2 * y in sums:
                    y_bad = True
                if y_bad:
                    to_remove.append(y)
            for y in to_remove:
                candidates.remove(y)
                
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected[:]
            if best_size <= 15: # Theoretical min is ~sqrt(256)=16, but maximal sets can be smaller? 
                # Actually, minimal maximal Sidon sets are around O(sqrt(N)).
                # For N=256, sqrt(256)=16. 
                # If we find <= 16, it's likely optimal or near-optimal.
                break

    return best_set
