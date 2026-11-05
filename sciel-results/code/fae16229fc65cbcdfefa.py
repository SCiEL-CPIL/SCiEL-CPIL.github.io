import random
import time
import sys
import math
from collections import defaultdict

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 140.0  # Leave buffer for 150s limit

    # Helper to check time
    def time_up():
        return time.time() - start_time > time_limit

    best_result = None
    best_scope = float('inf')

    # Precompute all possible "frames" (rows of length k+1 starting at 0)
    # A frame is a tuple of k+1 integers (0, a1, ..., ak) with 0 < a1 < ... < ak
    # The differences are {aj - ai for i < j}
    
    # We'll work with a target scope and precompute valid frames
    # For efficiency, we represent used differences as a Python int bitmask
    # (since scope <= ~200, this fits in a Python int easily)
    
    def precompute_frames(target_scope):
        """
        Precompute all valid frames (rows) with max element <= target_scope.
        Returns a list of (frame_tuple, diff_mask, max_val).
        """
        frames = []
        
        # Generate all combinations of k elements from 1..target_scope
        # Row is (0, x1, x2, ..., xk)
        from itertools import combinations
        
        for combo in combinations(range(1, target_scope + 1), k):
            row = (0,) + combo
            max_val = row[-1]
            
            # Compute differences
            diffs = set()
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    d = row[j] - row[i]
                    diffs.add(d)
            
            # Check if all differences are <= target_scope (they should be)
            # Convert to bitmask
            mask = 0
            for d in diffs:
                if d > target_scope:
                    break
                mask |= (1 << d)
            else:
                # All differences fit
                frames.append((row, mask, max_val))
        
        return frames

    def solve_with_frames(target_scope, rng_seed, max_frames_to_use=None):
        """
        Use precomputed frames and backtracking with MRV heuristic.
        """
        nonlocal best_result, best_scope
        
        rng = random.Random(rng_seed)
        
        # Precompute frames
        frames = precompute_frames(target_scope)
        if not frames:
            return False
        
        # Sort frames by max_val (ascending) for better pruning
        frames.sort(key=lambda x: x[2])
        
        # If too many frames, subsample
        if max_frames_to_use and len(frames) > max_frames_to_use:
            # Keep a diverse subset
            frames = frames[:max_frames_to_use]
        
        # For efficiency, group frames by their max value
        frames_by_max = defaultdict(list)
        for row, mask, max_val in frames:
            frames_by_max[max_val].append((row, mask))
        
        max_vals_sorted = sorted(frames_by_max.keys())
        
        used_diffs = 0
        rows = []
        
        def _backtrack(row_idx):
            nonlocal used_diffs, rows, best_scope, best_result
            
            if time_up():
                return False
            
            if row_idx == n_rows:
                # All rows constructed
                actual_scope = max(r[-1] for r in rows)
                if actual_scope < best_scope:
                    best_scope = actual_scope
                    best_result = [list(r) for r in rows]
                return True
            
            # MRV heuristic: try frames with fewer remaining options
            # For simplicity, we'll iterate through possible max values
            
            # Collect all candidate frames that don't conflict
            candidates = []
            for max_val in max_vals_sorted:
                if max_val >= best_scope:
                    break
                for row, mask in frames_by_max[max_val]:
                    if used_diffs & mask:
                        continue
                    candidates.append((max_val, row, mask))
            
            if not candidates:
                return False
            
            # Sort by max_val (prefer smaller scopes)
            candidates.sort(key=lambda x: x[0])
            
            # Limit candidates for efficiency
            max_candidates = min(50, len(candidates))
            
            # Randomize order for diversity
            if rng_seed > 0:
                # Shuffle but keep smaller max_val frames more likely
                weighted = []
                for i, c in enumerate(candidates[:max_candidates]):
                    weight = 1.0 / (1.0 + i * 0.1)
                    weighted.append((weight, c))
                rng.shuffle(weighted)
                candidates = [c for _, c in weighted[:max_candidates]]
            else:
                candidates = candidates[:max_candidates]
            
            for max_val, row, mask in candidates:
                old_used = used_diffs
                used_diffs = used_diffs | mask
                rows.append(row)
                
                if _backtrack(row_idx + 1):
                    return True
                
                rows.pop()
                used_diffs = old_used
            
            return False
        
        return _backtrack(0)

    # Phase 1: Greedy construction for initial upper bound
    def _greedy_construct(target_scope=None):
        """
        Fast greedy construction.
        """
        if target_scope is None:
            target_scope = 200
        
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if x > target_scope:
                    return None
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        
        return rows

    # Try greedy first
    for ts in [110, 115, 120, 125, 130, 140, 150, 160, 180, 200]:
        rows = _greedy_construct(ts)
        if rows:
            scope = max(r[-1] for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_result = [r[:] for r in rows]
            if time_up():
                break

    # Phase 2: Backtracking with precomputed frames
    # Start from best known scope and try to improve
    if best_result is not None:
        current_target = best_scope
        while current_target >= 105 and not time_up():
            # Try to find a solution with scope <= current_target
            success = solve_with_frames(current_target, rng_seed=0, max_frames_to_use=2000)
            if success:
                if best_scope < current_target:
                    current_target = best_scope
                else:
                    current_target -= 1
            else:
                # Try with randomization
                for seed in range(1, 10):
                    if time_up():
                        break
                    success = solve_with_frames(current_target, rng_seed=seed, max_frames_to_use=1000)
                    if success:
                        if best_scope < current_target:
                            current_target = best_scope
                        else:
                            current_target -= 1
                        break
                else:
                    # Couldn't find with this target, try smaller
                    current_target -= 1
    else:
        # No initial solution, try larger scopes
        for ts in [150, 140, 130, 125, 120, 115, 112, 110]:
            if time_up():
                break
            success = solve_with_frames(ts, rng_seed=0, max_frames_to_use=3000)
            if success:
                break

    # Phase 3: Local search to polish
    def _local_search(rows, max_iter=100000):
        """
        Local search to reduce scope by moving elements within rows.
        """
        nonlocal best_result, best_scope
        rng = random.Random(42)
        
        if not rows:
            return
        
        current_rows = [r[:] for r in rows]
        current_scope = max(r[-1] for r in current_rows)
        
        # Build used differences set
        used_diffs = set()
        for row in current_rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])
        
        improved = True
        iters = 0
        while improved and iters < max_iter and not time_up():
            improved = False
            iters += 1
            
            # Try to reduce the maximum element
            max_row_idx = max(range(len(current_rows)), key=lambda i: current_rows[i][-1])
            row = current_rows[max_row_idx]
            
            # Try to move each element to a smaller valid position
            for pos in range(len(row) - 1, 0, -1):
                old_val = row[pos]
                # Try smaller values
                for new_val in range(row[pos - 1] + 1, old_val):
                    # Check if moving this element creates conflicts
                    old_diffs = set()
                    for i in range(len(row)):
                        if i != pos:
                            d = abs(row[pos] - row[i])
                            old_diffs.add(d)
                    
                    # Check if new value creates conflicts
                    new_val_valid = True
                    new_diffs = set()
                    for i in range(len(row)):
                        if i != pos:
                            d = abs(new_val - row[i])
                            if d in used_diffs and d not in old_diffs:
                                new_val_valid = False
                                break
                            new_diffs.add(d)
                    
                    if not new_val_valid:
                        continue
                    
                    if new_val <= row[pos - 1] or (pos < len(row) - 1 and new_val >= row[pos + 1]):
                        continue
                    
                    # Apply the change
                    row[pos] = new_val
                    used_diffs -= old_diffs
                    used_diffs |= new_diffs
                    
                    new_scope = max(r[-1] for r in current_rows)
                    if new_scope < current_scope:
                        current_scope = new_scope
                        improved = True
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_result = [r[:] for r in current_rows]
                    break
        
        final_scope = max(r[-1] for r in current_rows)
        if final_scope < best_scope:
            best_scope = final_scope
            best_result = [r[:] for r in current_rows]

    if best_result is not None:
        _local_search(best_result, max_iter=50000)

    # Final fallback
    if best_result is None:
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
        
    return best_result
