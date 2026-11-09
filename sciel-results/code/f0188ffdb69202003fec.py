import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = 128
    
    # Precompute all nonconstant 4-term progressions
    # (a, a+d, a+2d, a+3d) with 0 <= a, a+3d < n
    progressions = []
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            idx3 = a + 3 * d
            if idx3 < n:
                progressions.append((a, a + d, a + 2 * d, idx3))
    
    # For fast lookup of violations, we can just iterate.
    # Number of APs is roughly sum_{d=1}^{42} (128 - 3d) ~ 42*128 - 3*42*43/2 ~ 5376 - 2709 ~ 2667.
    # This is small enough to check in O(1) per step if we update incrementally, 
    # or just check all in O(2667) per step which is ~2667 * 10000 steps = 26M ops, very fast in Python if optimized? 
    # Actually, Python loops are slow. 26M simple ops might take 1-2 seconds. 
    # Let's use a violation counter.
    
    def count_violations(colors):
        """Counts APs with < 3 distinct colors."""
        v = 0
        for (a, b, c, d) in progressions:
            # Check distinct colors
            # To be fast, avoid set() creation if possible?
            # c1, c2, c3, c4 = colors[a], colors[b], colors[c], colors[d]
            # distinct = len({c1, c2, c3, c4})
            # if distinct < 3: v += 1
            ca, cb, cc, cd = colors[a], colors[b], colors[c], colors[d]
            # Check if < 3 distinct
            # Case 1: All same
            if ca == cb == cc == cd:
                v += 1
            # Case 2: 2 distinct
            # e.g., A A A B, A A B A, A A B B, A B A A, A B A B, A B B A, A B B B...
            # Easier:
            elif ca == cb:
                if cc == ca or cd == ca:
                    if cc == cd or cc == ca or cd == ca: # This logic is getting messy
                        pass
            # Let's just use set. 2667 sets per evaluation.
            # If we do 10,000 evaluations, that's 26M set creations. Might be too slow in Python.
            # Let's optimize the check.
            
            # Check if distinct count < 3
            # This is equivalent to:
            # (ca == cb == cc == cd) OR
            # (ca == cb == cc) OR (ca == cb == cd) OR (ca == cc == cd) OR (cb == cc == cd) OR
            # (ca == cb and ca == cc and ... no)
            
            # Simpler check:
            # If any color appears 3 or 4 times, it's a violation?
            # No. A A B B has 2 colors. Violation.
            # A A B C has 3 colors. OK.
            # A B C D has 4 colors. OK.
            
            # So violation if:
            # 1. All 4 same.
            # 2. 3 same, 1 different.
            # 3. 2 same, 2 different (A A B B, A B A B, A B B A).
            
            # Let's count frequencies.
            # This is O(1).
            
            # Actually, set() is implemented in C and is quite fast.
            # Let's benchmark mentally: 2667 * 10000 = 26.67 million iterations.
            # In Python, a loop with a set() might take ~0.1-0.2 us? No, more like 0.5-1 us.
            # 26M * 1us = 26 seconds. Too slow for 10k iterations.
            # We need incremental updates or fewer iterations.
            
            # Alternative: 
            # We only change one position. 
            # We can maintain a violation count and update it based on the APs affected by the change.
            # Each position is in at most ~100 APs?
            # Let's precompute APs for each position.
            
            pass # Placeholder for optimized logic
            
        return v

    # Precompute APs per position for incremental update
    pos_aps = [[] for _ in range(n)]
    for i, ap in enumerate(progressions):
        for idx in ap:
            pos_aps[idx].append(i)

    def is_violation(colors, ap):
        ca, cb, cc, cd = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        # Check if < 3 distinct
        # Fast path:
        if ca == cb == cc == cd: return True
        if ca == cb:
            if cc == ca or cd == ca:
                # If cc==ca and cd==ca -> caught above
                # If cc==ca, cd!=ca -> A A A B (3 distinct? No, A and B. 2 distinct. Violation)
                # If cd==ca, cc!=ca -> A A B A (2 distinct. Violation)
                return True
            # ca == cb, cc != ca, cd != ca
            if cc == cd: return True # A A B B
            return False # A A B C
        if ca == cc:
            if cb == ca or cd == ca: return True # A B A A or A B A B (if cb!=cd? No, A B A A is 2 colors. A B A B is 2 colors)
            # Wait. A B A C. Distinct: A, B, C. 3 colors. OK.
            # So if ca == cc:
            #   if cb == ca: A A A C (2 colors) -> Violation
            #   if cd == ca: A B A A (2 colors) -> Violation
            #   if cb == cd: A B A B (2 colors) -> Violation
            #   else: A B A C (3 colors) -> OK
            if cb == ca or cd == ca or cb == cd: return True
            return False
        if ca == cd:
            if cb == ca or cc == ca: return True
            if cb == cc: return True # A B B A
            return False # A B C A
        if cb == cc:
            if ca == cb or cd == cb: return True
            if ca == cd: return True # A B B A
            return False
        if cb == cd:
            if ca == cb or cc == cb: return True
            if ca == cc: return True # A B C B? No, A B C B has colors A, B, C. 3 colors.
            # Wait. A B C B. Distinct: A, B, C. OK.
            # A A B B caught in ca==cb.
            # A B A B caught in ca==cc? No, ca==cd? No. cb==cd? Yes.
            # If cb == cd:
            #   if ca == cb: A B B B (2 colors) -> Violation
            #   if cc == cb: A B B B (2 colors) -> Violation
            #   if ca == cc: A B A B (2 colors) -> Violation
            #   else: A B C B (3 colors) -> OK
            if ca == cb or cc == cb or ca == cc: return True
            return False
        if cc == cd:
            if ca == cc or cb == cc: return True
            if ca == cb: return True # A A B B
            return False
        # All distinct or other cases?
        # If none of the pairs are equal, then all 4 are distinct?
        # Not necessarily. ca != cb, ca != cc, ca != cd, cb != cc, cb != cd, cc != cd.
        # Then all 4 distinct. OK.
        return False

    def local_search(k_colors, max_iters=20000):
        """
        Try to find a valid coloring using at most k_colors.
        Returns the coloring if found, else None.
        """
        # Initialize randomly
        colors = [random.randint(0, k_colors - 1) for _ in range(n)]
        
        # Count initial violations
        viol_count = 0
        for ap in progressions:
            if is_violation(colors, ap):
                viol_count += 1
        
        best_colors = colors[:]
        best_viol = viol_count
        
        if viol_count == 0:
            return best_colors
            
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            # Pick random position
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            
            # Pick new color
            new_color = random.randint(0, k_colors - 1)
            if new_color == old_color:
                continue
                
            # Calculate change in violations
            # Only APs containing pos are affected
            delta = 0
            for ap_idx in pos_aps[pos]:
                ap = progressions[ap_idx]
                was_viol = is_violation(colors, ap)
                # Apply change
                colors[pos] = new_color
                is_viol = is_violation(colors, ap)
                # Revert change for next AP in loop? 
                # No, we need to check all with the new color.
                # But we need to compare old state vs new state.
                # Let's store old state or just compute delta carefully.
                
                # Efficient way:
                # We already have colors[pos] = new_color.
                # We need to know if it WAS a violation.
                # We can't easily revert inside the loop without storing.
                # Let's just re-calculate for affected APs.
                
                # Actually, simpler:
                # 1. Store old_color.
                # 2. Set colors[pos] = new_color.
                # 3. For each AP in pos_aps[pos]:
                #      Check if it was violation (need old color).
                #      Check if it is violation (with new color).
                # 4. Update viol_count.
                
                # This requires knowing the old state.
                # Let's just do a full re-evaluation of affected APs.
                pass # Logic needs restructuring

            # Let's implement a cleaner incremental update.
            # Save old color.
            # For each AP containing pos:
            #   Check violation with old color.
            #   Check violation with new color.
            #   Update delta.
            
            # Re-implementing loop body properly:
            delta = 0
            for ap_idx in pos_aps[pos]:
                ap = progressions[ap_idx]
                # Check old
                colors[pos] = old_color
                was_viol = is_violation(colors, ap)
                # Check new
                colors[pos] = new_color
                is_viol = is_violation(colors, ap)
                
                if was_viol and not is_viol:
                    delta -= 1
                elif not was_viol and is_viol:
                    delta += 1
            # Final state is new_color
            viol_count += delta
            
            if viol_count < best_viol:
                best_viol = viol_count
                best_colors = colors[:]
            elif viol_count == 0:
                return colors[:]
            else:
                # Accept with some probability (Simulated Annealing) or just greedy?
                # Greedy is faster but might get stuck.
                # Let's try greedy first. If stuck, try SA.
                # For now, greedy: accept only if delta < 0.
                if delta >= 0:
                    colors[pos] = old_color # Revert
                    viol_count -= delta # Revert count
            
            # Optional: try to reduce colors used if valid?
            # If viol_count == 0, we can try to map to fewer colors.
            # But we are fixing k_colors.
            
        if best_viol == 0:
            return best_colors
        return None

    # Try k=3
    # Run multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        sol = local_search(3, max_iters=5000)
        if sol:
            # Verify
            if all(not is_violation(sol, ap) for ap in progressions):
                return sol
                
    # Try k=4
    for restart in range(50):
        if time.time() > deadline:
            break
        sol = local_search(4, max_iters=5000)
        if sol:
            if all(not is_violation(sol, ap) for ap in progressions):
                return sol

    # Fallback: Greedy construction with many colors
    # This ensures validity but might use many colors.
    colors = [0] * n
    for i in range(n):
        # Try colors 0..127
        for c in range(128):
            colors[i] = c
            valid = True
            # Check APs ending at i or containing i?
            # Just check all APs containing i.
            for ap_idx in pos_aps[i]:
                if is_violation(colors, progressions[ap_idx]):
                    valid = False
                    break
            if valid:
                break
        # If no color worked, we have a problem, but with 128 colors it should work.
        
    # Verify fallback
    # It might not be valid if greedy got stuck, but with 128 colors it's very likely valid.
    # Actually, greedy might fail if it paints a bridge.
    # But returning an invalid solution scores 0.
    # Let's ensure the fallback is valid.
    # A simple valid pattern: c_i = i. (128 colors).
    # Any 4-AP has 4 distinct indices, so 4 distinct colors. Valid.
    # So if all else fails, return list(range(128)).
    
    # Check if the greedy result is valid.
    if all(not is_violation(colors, ap) for ap in progressions):
        return colors
    else:
        return list(range(128))

# Quick check: The prompt says "Minimize the number of labels actually used".
# If we return list(range(128)), we use 128 labels.
# If we find a 3-color solution, we use 3 labels.
# The evaluator measures the raw value (lower is better).
# So 3 is much better than 128.

# One optimization: The local search for k=3 might be hard.
# Is 3 colors possible?
# Let's assume yes. If not, 4.

# Let's refine the local search to be more aggressive with restarts and SA if needed.
# But greedy with restarts is often sufficient for N=128.

# Let's double check the is_violation logic.
# A A A B -> Violation.
# My logic:
# ca==cb==cc==cd -> True
# ca==cb:
#   cc==ca or cd==ca -> True.
#   A A A B: ca=A, cb=A, cc=A, cd=B.
#   ca==cb (True).
#   cc==ca (True). Returns True. Correct.
# A A B B:
#   ca==cb (True).
#   cc==ca? No. cd==ca? No.
#   cc==cd? Yes. Returns True. Correct.
# A A B C:
#   ca==cb (True).
#   cc==ca? No. cd==ca? No.
#   cc==cd? No. Returns False. Correct.

# The logic seems sound.

# One potential issue: `local_search` uses `random.randint`.
# We should seed it for reproducibility? No, evaluator runs it once.
# But we can seed with time or fixed. Fixed is better for debugging, but random is fine.

# Let's add a check: if k=3 fails, try k=4.
# The loop structure handles this.

# Final check on constraints:
# "Every nonconstant four-term progression of positions must use AT LEAST three distinct colors."
# "Minimize the number of labels actually used"
# "labels need not be consecutive"

# The code returns a list of integers.
# The evaluator will count unique integers.

# One more thing: The `is_violation` function is called many times.
# It's pure Python.
# To speed up, we can inline it or use numpy?
# Numpy boolean logic on small arrays might be slower due to overhead.
# Pure Python with short-circuiting is probably best.

# Let's try to improve the local search by trying to eliminate colors if we have extra.
# But we start with k=3. If we find a valid 3-coloring, we stop.
# We don't need to eliminate colors if we started with 3.

# What if 3 is impossible?
# Then k=4.
# If 4 is impossible?
# Then k=5...
# Up to 128.

# Given N=128 is small, 3 or 4 should be findable.

# Let's wrap the main logic in `construct`.

# One detail: `pos_aps` is built once.
# `progressions` is built once.

# I will use a fixed seed for the first few restarts to ensure consistent behavior if possible,
# but generally random is fine.

# Let's ensure the fallback is always valid.
# `list(range(128))` is always valid.

# I'll add a simple heuristic: try mod 3, mod 4, etc. as initial states for local search.
# This might help converge faster.

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    progressions = []
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            idx3 = a + 3 * d
            if idx3 < n:
                progressions.append((a, a + d, a + 2 * d, idx3))
                
    pos_aps = [[] for _ in range(n)]
    for i, ap in enumerate(progressions):
        for idx in ap:
            pos_aps[idx].append(i)

    def is_violation(colors, ap):
        ca, cb, cc, cd = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        if ca == cb == cc == cd: return True
        if ca == cb:
            if cc == ca or cd == ca: return True
            if cc == cd: return True
            return False
        if ca == cc:
            if cb == ca or cd == ca or cb == cd: return True
            return False
        if ca == cd:
            if cb == ca or cc == ca or cb == cc: return True
            return False
        if cb == cc:
            if ca == cb or cd == cb or ca == cd: return True
            return False
        if cb == cd:
            if ca == cb or cc == cb or ca == cc: return True
            return False
        if cc == cd:
            if ca == cc or cb == cc or ca == cb: return True
            return False
        return False

    def local_search(k_colors, initial_colors=None, max_iters=10000):
        if initial_colors:
            colors = initial_colors[:]
        else:
            colors = [random.randint(0, k_colors - 1) for _ in range(n)]
            
        viol_count = 0
        for ap in progressions:
            if is_violation(colors, ap):
                viol_count += 1
        
        if viol_count == 0:
            return colors[:]
            
        best_colors = colors[:]
        best_viol = viol_count
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            new_color = random.randint(0, k_colors - 1)
            if new_color == old_color:
                continue
                
            delta = 0
            for ap_idx in pos_aps[pos]:
                ap = progressions[ap_idx]
                colors[pos] = old_color
                was_viol = is_violation(colors, ap)
                colors[pos] = new_color
                is_viol = is_violation(colors, ap)
                
                if was_viol and not is_viol:
                    delta -= 1
                elif not was_viol and is_viol:
                    delta += 1
            
            viol_count += delta
            
            if viol_count < best_viol:
                best_viol = viol_count
                best_colors = colors[:]
            elif viol_count == 0:
                return colors[:]
            else:
                if delta >= 0:
                    colors[pos] = old_color
                    viol_count -= delta
        
        if best_viol == 0:
            return best_colors
        return None

    # Try specific patterns first
    patterns = [
        [i % 3 for i in range(n)],
        [i % 4 for i in range(n)],
        [i % 5 for i in range(n)],
        [i % 7 for i in range(n)]
    ]
    
    for k in [3, 4, 5, 6]:
        for pat in patterns:
            # Map pattern to 0..k-1 if it uses fewer colors?
            # No, just use it as is if max color < k.
            if max(pat) < k:
                sol = local_search(k, pat, max_iters=2000)
                if sol:
                    return sol
                    
        # Random restarts
        for r in range(20):
            if time.time() > deadline:
                break
            sol = local_search(k, max_iters=5000)
            if sol:
                return sol
                
    # Fallback
    return list(range(n))

# Note: The patterns like i%3 might have violations. Local search will fix them if possible.
# If i%3 has violations, local search might move away from it.
# That's fine.

# One small optimization: In local_search, if we accept a move with delta > 0 (SA), we might escape local minima.
# But greedy is faster. Let's stick to greedy + restarts.

# Final Code Structure
