import math
import time
import random
import sys

# Use a fixed seed for reproducibility, but allow slight variation
random.seed(42)

def _score_set(U):
    """
    Compute the score for a set U.
    Score = 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    """
    if not U or 0 not in U:
        return 0.0
    
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Compute sumset and difference set
    # For efficiency, we can use sets
    sums = set()
    diffs = set()
    
    # Optimization: if len(U) is small, double loop is fine.
    # If len(U) is large, we might need convolution, but for this problem sizes are likely small (< 50)
    # because the search space grows rapidly.
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    
    size_sums = len(sums)
    size_diffs = len(diffs)
    
    if size_sums == 0:
        return 0.0
        
    ratio = size_diffs / size_sums
    if ratio <= 1.0:
        # log(ratio) <= 0, so score <= 1.
        # We want to maximize, so returning the actual value is safer for comparison,
        # but since we look for > 1, this effectively caps it.
        return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
        
    return 1.0 + math.log(ratio) / denom

def _greedy_golomb(n):
    """
    Construct a Golomb ruler of length n using a greedy heuristic.
    A Golomb ruler has distinct differences, which maximizes |U-U| for a given size.
    """
    if n <= 1:
        return [0]
    
    ruler = [0]
    for i in range(1, n):
        best_pos = -1
        best_score = -1
        
        # Try positions from 1 upwards
        # Upper bound estimate: n^2 is safe, but we can be tighter.
        # For n=10, length is ~ 100.
        max_pos = i * i + 10 
        
        for pos in range(1, max_pos + 1):
            # Check if pos creates a duplicate difference
            valid = True
            new_diffs = set()
            for x in ruler:
                d = abs(pos - x)
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                # Heuristic: prefer smaller positions to keep max(U) small?
                # Or prefer positions that are "far" to increase spread?
                # Actually, for Golomb rulers, we just want ANY valid position.
                # Greedy: pick the smallest valid position to minimize length?
                # Or pick the one that allows future extensions?
                # Let's stick to smallest valid position for determinism and compactness.
                best_pos = pos
                break # Found the smallest valid position
                
        if best_pos == -1:
            # Fallback: if no small position works, keep searching or give up?
            # For small n, we should find one.
            # If not, we might need to backtrack, but greedy is faster.
            # Let's extend the search range.
            for pos in range(max_pos + 1, max_pos + 100):
                valid = True
                new_diffs = set()
                for x in ruler:
                    d = abs(pos - x)
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    best_pos = pos
                    break
            
            if best_pos == -1:
                # Last resort: append a large number
                best_pos = ruler[-1] + 100
        
        ruler.append(best_pos)
        
    return ruler

def _local_search(initial_set, deadline, max_steps=5000):
    """
    Local search to improve the set.
    Moves:
    1. Move an element to a nearby position.
    2. Add a new element.
    3. Remove an element.
    4. Swap two elements (if we allow size change? No, swap keeps size).
    """
    current_set = initial_set[:]
    current_score = _score_set(current_set)
    best_local_score = current_score
    best_local_set = current_set[:]
    
    # Precompute some stats?
    # We'll just recompute score for each move.
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        if len(current_set) <= 1:
            break
            
        # Randomly select a move type
        # Bias towards moves that keep size similar or grow slightly
        r = random.random()
        
        if r < 0.6:
            # Move an existing element
            idx = random.randint(1, len(current_set) - 1) # Don't move 0
            val = current_set[idx]
            
            # Try a range of new values
            # Range: val +/- some spread
            spread = max(10, int(math.sqrt(val) * 2))
            candidates = []
            for delta in range(-spread, spread + 1):
                if delta == 0:
                    continue
                new_val = val + delta
                if new_val < 0:
                    continue
                if new_val in current_set:
                    continue
                candidates.append(new_val)
                
            if not candidates:
                continue
                
            # Shuffle candidates to avoid bias towards small deltas?
            # Or try them in order of proximity?
            random.shuffle(candidates)
            
            best_move = None
            best_move_score = current_score
            
            # Try a subset of candidates to save time
            for new_val in candidates[:20]:
                new_set = current_set[:]
                new_set[idx] = new_val
                new_set.sort()
                
                s = _score_set(new_set)
                if s > best_move_score:
                    best_move_score = s
                    best_move = new_val
                    
            if best_move is not None:
                current_set[idx] = best_move
                current_set.sort()
                current_score = best_move_score
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_set = current_set[:]
                    
        elif r < 0.8:
            # Add a new element
            if len(current_set) > 30: # Limit size to keep computation fast
                continue
                
            max_val = max(current_set)
            # Try adding near existing elements or at the end
            candidates = []
            for x in current_set:
                for delta in [1, 2, 3, 5, 10]:
                    candidates.append(x + delta)
                    candidates.append(x - delta)
            candidates.append(max_val + 1)
            candidates.append(max_val + 2)
            candidates.append(max_val + 5)
            
            # Filter invalids
            valid_candidates = [c for c in candidates if c >= 0 and c not in current_set]
            if not valid_candidates:
                continue
                
            random.shuffle(valid_candidates)
            
            best_move = None
            best_move_score = current_score
            
            for new_val in valid_candidates[:10]:
                new_set = current_set + [new_val]
                new_set.sort()
                s = _score_set(new_set)
                if s > best_move_score:
                    best_move_score = s
                    best_move = new_val
                    
            if best_move is not None:
                current_set.append(best_move)
                current_set.sort()
                current_score = best_move_score
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_set = current_set[:]
                    
        else:
            # Remove an element
            if len(current_set) <= 2:
                continue
                
            idx = random.randint(1, len(current_set) - 1)
            new_set = current_set[:]
            del new_set[idx]
            
            s = _score_set(new_set)
            if s > current_score:
                current_set = new_set
                current_score = s
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_set = current_set[:]
                    
    return best_local_set, best_local_score

def construct():
    start_time = time.time()
    # 120s limit, leave 20s margin
    deadline = start_time + 100
    
    best_score = 0
    best_set = [0, 1]
    
    # 1. Generate initial candidates using Greedy Golomb Rulers of various sizes
    # Golomb rulers maximize differences, which is good for the numerator.
    # We need to balance size and max(U).
    
    for n in range(2, 15):
        if time.time() > deadline - 10:
            break
            
        ruler = _greedy_golomb(n)
        if 0 in ruler:
            s = _score_set(ruler)
            if s > best_score:
                best_score = s
                best_set = ruler[:]
                
    # 2. Local search from the best candidate
    if best_set:
        best_set, best_score = _local_search(best_set, deadline, max_steps=2000)
        
    # 3. Try local search from other promising candidates
    # We might have missed a local optimum from a different starting point
    for n in range(2, 12):
        if time.time() > deadline - 5:
            break
            
        ruler = _greedy_golomb(n)
        if 0 in ruler:
            # Only search if this starting point is decent
            s_init = _score_set(ruler)
            if s_init > best_score - 0.02:
                candidate_set, candidate_score = _local_search(ruler, deadline, max_steps=1000)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_set = candidate_set
                    
    # 4. Final polish: Try to fine-tune the best set with smaller steps
    # Use a tighter deadline for the final pass
    final_deadline = min(deadline, time.time() + 10)
    if best_set:
        best_set, best_score = _local_search(best_set, final_deadline, max_steps=500)
        
    return best_set

# EVOLVE-BLOCK-END
