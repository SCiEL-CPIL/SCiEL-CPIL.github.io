import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _precompute_balls(q, n, radius):
    """Precompute balls for all words in {0..q-1}^n."""
    total = q ** n
    balls = []
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    for x in range(total):
        ball = [x]
        if radius == 1:
            for i in range(n):
                old_digit = (x // powers[i]) % q
                for new_digit in range(q):
                    if new_digit != old_digit:
                        y = x + (new_digit - old_digit) * powers[i]
                        ball.append(y)
        else:
            # Fallback for other radii (not expected for this task)
            # Simple enumeration for small n
            digits = []
            for i in range(n):
                digits.append((x // powers[i]) % q)
            from itertools import product
            # Generate all words within Hamming distance radius
            # This is expensive, but radius 1 is the main case
            pass 
            
        balls.append(ball)
    return balls, powers

def _int_to_digits(x, q, n, powers):
    digits = []
    for i in range(n):
        d = (x // powers[i]) % q
        digits.append(d)
    return digits

def _construct_covering_code(q, n, radius, deadline):
    total = q ** n
    start_time = time.time()
    
    # Precompute balls
    balls, powers = _precompute_balls(q, n, radius)
    
    # cover_count[y] = number of codewords in current set that cover y
    cover_count = np.zeros(total, dtype=np.int32)
    
    # Current set of codewords (as integers)
    current_set = set()
    # List of codewords for easy iteration
    current_list = []
    
    # 1. Greedy Initial Solution
    # uncovered[y] = 1 if not covered, 0 if covered
    uncovered = np.ones(total, dtype=bool)
    
    # Precompute for each word y, which codewords c can cover it?
    # inv[y] = list of c such that y in balls[c]
    # Note: y in balls[c] iff c in balls[y] (symmetry of Hamming distance)
    # So inv[y] is just balls[y].
    
    while np.any(uncovered):
        if time.time() > deadline - 20:
            break
            
        # Find the codeword c that covers the most uncovered points
        best_c = -1
        max_gain = 0
        
        # We don't need to check all 6561 candidates every time if we are smart,
        # but 6561 * 17 is small enough.
        # However, we can optimize by only considering c that covers at least one uncovered point.
        # Let's just iterate through all candidates to keep it simple and robust.
        for c in range(total):
            # Quick check: does c cover any uncovered point?
            # gain = count of y in balls[c] where uncovered[y] is True
            # Using numpy for speed
            ball_c = balls[c]
            gain = np.sum(uncovered[ball_c])
            if gain > max_gain:
                max_gain = gain
                best_c = c
                if max_gain == len(ball_c): # Perfect
                    break
        
        if best_c == -1:
            break
            
        # Add best_c
        current_set.add(best_c)
        current_list.append(best_c)
        
        # Update uncovered and cover_count
        ball_best = balls[best_c]
        cover_count[ball_best] += 1
        # Update uncovered: if cover_count[y] becomes 1, it's covered
        # Actually, we can just set uncovered[ball_best] = False where cover_count was 0?
        # No, cover_count increases. If it was 0, now it's 1, so covered.
        # We need to update uncovered array.
        for y in ball_best:
            if cover_count[y] == 1: # Newly covered
                uncovered[y] = False
        
        if not np.any(uncovered):
            break
            
    # 2. Local Search / Simulated Annealing
    best_size = len(current_set)
    best_set_snapshot = set(current_set)
    best_cover_snapshot = cover_count.copy()
    
    current_size = len(current_set)
    
    # Temperature setup
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    while time.time() < deadline - 5:
        improved = False
        
        # Try to remove redundant codewords
        # A codeword c is redundant if for all y in balls[c], cover_count[y] >= 2
        candidates = list(current_set)
        random.shuffle(candidates)
        
        for c in candidates:
            if time.time() > deadline - 5:
                break
            if c not in current_set:
                continue
            
            # Check redundancy
            ball_c = balls[c]
            # Check if any y in ball_c has cover_count[y] == 1
            # If yes, removing c makes it 0 (uncovered) -> not redundant
            # If no, removing c leaves >= 1 -> redundant
            # min_count = np.min(cover_count[ball_c])
            # if min_count >= 2: redundant
            
            # Faster check: any(cover_count[ball_c] == 1)
            is_redundant = not np.any(cover_count[ball_c] == 1)
            
            if is_redundant:
                # Remove c
                current_set.remove(c)
                current_list.remove(c) # O(n), acceptable for small lists? 
                                       # Actually list removal is O(n). 500 size is fine.
                cover_count[ball_c] -= 1
                current_size -= 1
                improved = True
                
                if current_size < best_size:
                    best_size = current_size
                    best_set_snapshot = set(current_set)
                    best_cover_snapshot = cover_count.copy()
                    
        if not improved:
            # Stuck. Perform a perturbation.
            # Strategy: Pick a random codeword c, remove it, then repair.
            if current_size > 0:
                c_to_remove = random.choice(list(current_set))
                
                # Remove it
                current_set.remove(c_to_remove)
                current_list.remove(c_to_remove)
                cover_count[balls[c_to_remove]] -= 1
                current_size -= 1
                
                # Now we have uncovered points. Repair them.
                # Find all y with cover_count[y] == 0
                uncovered_mask = (cover_count == 0)
                uncovered_idx = np.where(uncovered_mask)[0]
                
                if len(uncovered_idx) > 0:
                    # Greedy repair
                    # We need to cover these points.
                    # Pick a codeword that covers the most uncovered points.
                    # To speed up, we can sample a few uncovered points and find common neighbors?
                    # Or just do a standard greedy step on the uncovered set.
                    
                    while len(uncovered_idx) > 0 and time.time() < deadline - 5:
                        # Pick the best c to add
                        best_c = -1
                        max_gain = 0
                        # Sample a subset of candidates to speed up repair?
                        # Or just check all? 6561 checks * 17 ops is fast.
                        for c in range(total):
                            # Only consider c if it covers at least one uncovered point
                            # gain = sum(uncovered_mask[balls[c]])
                            # Optimization: 
                            # If c was just removed, maybe add it back?
                            # Let's just do full scan.
                            gain = np.sum(uncovered_mask[balls[c]])
                            if gain > max_gain:
                                max_gain = gain
                                best_c = c
                                if max_gain == len(balls[c]):
                                    break
                            
                        if best_c == -1:
                            break
                            
                        # Add best_c
                        current_set.add(best_c)
                        current_list.append(best_c)
                        cover_count[balls[best_c]] += 1
                        current_size += 1
                        
                        # Update uncovered_mask
                        # Points in balls[best_c] that were 0 are now 1
                        ball_best = balls[best_c]
                        # Uncovered_mask[ball_best] = False where cover_count was 0?
                        # Actually, cover_count[ball_best] is now >= 1.
                        # So they are covered.
                        uncovered_mask[ball_best] = False
                        uncovered_idx = np.where(uncovered_mask)[0]
                    
                    # After repair, we might be larger than before.
                    # Accept this state if we are cooling?
                    # For simplicity, let's just accept it and try to reduce again in next iteration.
                    # This is effectively a "kick".
                    
                # If no uncovered points (should not happen if we removed a redundant word, 
                # but we only remove non-redundant in perturbation? No, we remove random).
                # If we removed a non-redundant word, we HAVE uncovered points.
                # If we removed a redundant word (shouldn't happen as we checked above), 
                # then no repair needed.

    # Restore best found solution
    final_set = best_set_snapshot
    
    # Convert to digits
    codewords = []
    for idx in final_set:
        codewords.append(_int_to_digits(idx, q, n, powers))
        
    return codewords

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140 # Safety margin
    
    # For this specific task (q=3, n=8, r=1)
    # The known optimal is 486.
    
    try:
        result = _construct_covering_code(q, n, radius, deadline)
    except Exception as e:
        # Fallback to simple greedy if anything goes wrong
        # Just return a valid cover, even if large
        total = q ** n
        powers = [1] * n
        for i in range(1, n):
            powers[i] = powers[i-1] * q
        
        balls, _ = _precompute_balls(q, n, radius)
        covered = np.zeros(total, dtype=bool)
        selected = []
        
        while not np.all(covered):
            best_c = -1
            max_gain = 0
            for c in range(total):
                gain = np.sum(~covered[balls[c]])
                if gain > max_gain:
                    max_gain = gain
                    best_c = c
            selected.append(best_c)
            covered[balls[best_c]] = True
            if time.time() > deadline - 2:
                break
                
        codewords = []
        for idx in selected:
            codewords.append(_int_to_digits(idx, q, n, powers))
        result = codewords

    return result

# EVOLVE-BLOCK-END
