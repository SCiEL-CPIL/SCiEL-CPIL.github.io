"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time


def _encode_word(word):
    """Encode a tuple of digits as a base-q integer."""
    val = 0
    for d in word:
        val = val * 3 + d  # q is fixed to 3 in this task context, but let's make it generic if needed. 
                           # The task says q=3, n=6. We can hardcode base 3 for speed or pass q.
                           # Since construct(q, n, radius) takes q, we should use q.
                           # But _encode_word is internal. Let's use a helper that takes q.
    return val

def _decode_word(val, n, q):
    """Decode a base-q integer to a tuple of digits."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = val % q
        val //= q
    return tuple(word)

def construct(q, n, radius):
    """
    Greedy set-covering with simulated annealing refinement for ternary covering code.
    """
    start_time = time.time()
    time_limit = 115  # leave some buffer before 120s timeout
    
    # Total number of words in the space
    total_words = q ** n
    
    # Precompute all words as tuples and their encoded integer representation
    all_words = []
    word_to_idx = {}
    idx_to_word = []
    
    # Generate all words
    for i in range(total_words):
        # Decode i to get the word
        w = [0] * n
        val = i
        for j in range(n - 1, -1, -1):
            w[j] = val % q
            val //= q
        w = tuple(w)
        all_words.append(w)
        word_to_idx[w] = i
        idx_to_word.append(w)
        
    # Precompute balls for all possible centers
    # ball[i] is a set of indices of words covered by center i
    all_balls = [set() for _ in range(total_words)]
    
    for i, center in enumerate(all_words):
        # Generate all words within radius
        # For radius 1, this is center + 1 neighbor for each pos and each other val
        # For general radius, we can use BFS or recursive generation.
        # Given n=6, radius=1, it's small.
        # Let's implement a general ball generator for small n/radius.
        
        ball_set = {i}
        
        # If radius is 1:
        if radius == 1:
            for pos in range(n):
                orig = center[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    new_word = list(center)
                    new_word[pos] = new_val
                    new_idx = word_to_idx[tuple(new_word)]
                    ball_set.add(new_idx)
        elif radius == 2:
            # For radius 2, we can iterate over pairs of positions or use combinations
            # But n=6, radius=1 is the standard case for K_3(6,1).
            # If radius > 1, we need a more general method.
            # Let's assume radius is small.
            # For safety, let's use a BFS-like expansion or just iterate all words if total_words is small enough.
            # 729 * 729 = 531k operations. This is fine.
            for j, w in enumerate(all_words):
                # Calculate Hamming distance
                dist = 0
                for k in range(n):
                    if center[k] != w[k]:
                        dist += 1
                        if dist > radius:
                            break
                if dist <= radius:
                    ball_set.add(j)
        else:
            # General case
            for j, w in enumerate(all_words):
                dist = 0
                for k in range(n):
                    if center[k] != w[k]:
                        dist += 1
                        if dist > radius:
                            break
                if dist <= radius:
                    ball_set.add(j)
                    
        all_balls[i] = ball_set

    # 1. Greedy Initialization
    # Randomized greedy: at each step, pick a center that covers the most uncovered words.
    # To add diversity, we can pick from the top few.
    
    uncovered = set(range(total_words))
    centers = []
    
    # Precompute coverage counts? No, just scan.
    # For speed, we can maintain a count of how many times each word is covered? 
    # Greedy just needs the count of uncovered words covered by each candidate.
    
    # To speed up greedy, we can precompute the list of balls.
    # At each step, for each candidate center, count how many of its ball words are in 'uncovered'.
    # This is O(total_words * ball_size) per step. 729 * 13 ~ 9000 ops. 
    # Number of steps ~ 73. Total ~ 650k ops. Very fast.
    
    while uncovered:
        best_center_idx = -1
        best_count = -1
        # Random tie-breaking
        best_candidates = []
        
        for i in range(total_words):
            if i in centers: # Optimization: skip already chosen? Not strictly necessary but good.
                continue
            ball = all_balls[i]
            count = 0
            for w in ball:
                if w in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_candidates = [i]
            elif count == best_count and count > 0:
                best_candidates.append(i)
                
        if not best_candidates:
            break
            
        # Pick randomly from best candidates to add randomness
        chosen = random.choice(best_candidates)
        centers.append(chosen)
        uncovered -= all_balls[chosen]
        
    current_centers = set(centers)
    best_centers = set(centers)
    best_count = len(centers)
    
    # 2. Simulated Annealing
    # State: set of center indices
    # Moves:
    #   - Remove: pick random c in current. If valid, accept.
    #   - Swap: pick random c in current, random c_new not in current. If valid, accept.
    #   - Add: pick random c_new not in current. Add. Accept with prob exp(-1/T).
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.995
    
    # Helper to check validity of a set of centers
    # We need to know if union of balls covers all words.
    # Instead of recalculating from scratch, we can maintain coverage counts.
    
    # Initialize coverage counts
    coverage_counts = [0] * total_words
    for c in current_centers:
        for w in all_balls[c]:
            coverage_counts[w] += 1
    
    # Check if fully covered
    def is_fully_covered():
        # Check if any count is 0
        # Using min is O(N). 729 is small.
        return min(coverage_counts) > 0
        
    # Precompute list of all indices for random selection
    all_indices = list(range(total_words))
    
    while T > T_min and time.time() - start_time < time_limit:
        # Choose a move
        move_type = random.random()
        if move_type < 0.3:
            # Remove
            if len(current_centers) <= 1:
                T *= cooling_rate
                continue
            c = random.choice(list(current_centers))
            
            # Check if removing c keeps coverage
            # A word is still covered if coverage_counts[w] > 1 for all w in ball(c)
            valid = True
            for w in all_balls[c]:
                if coverage_counts[w] <= 1:
                    valid = False
                    break
            
            if valid:
                # Accept
                for w in all_balls[c]:
                    coverage_counts[w] -= 1
                current_centers.remove(c)
                new_count = len(current_centers)
                if new_count < best_count:
                    best_count = new_count
                    best_centers = set(current_centers)
            else:
                # Reject (or accept with prob 0 since it's invalid)
                pass
                
        elif move_type < 0.8:
            # Swap
            if len(current_centers) == 0:
                T *= cooling_rate
                continue
            c_old = random.choice(list(current_centers))
            # Pick a new center
            # To avoid picking one already in set, we can pick from all and check, or maintain a list of available.
            # Given 729 is small, just pick random from all and check if it's in set.
            c_new = random.choice(all_indices)
            if c_new in current_centers:
                T *= cooling_rate
                continue
                
            # Check if (current - {c_old} + {c_new}) is valid
            # Words uniquely covered by c_old must be covered by c_new
            # i.e., for all w in ball(c_old) where coverage_counts[w] == 1, w must be in ball(c_new)
            
            valid = True
            for w in all_balls[c_old]:
                if coverage_counts[w] == 1:
                    if w not in all_balls[c_new]:
                        valid = False
                        break
            
            if valid:
                # Accept
                # Update counts
                for w in all_balls[c_old]:
                    coverage_counts[w] -= 1
                for w in all_balls[c_new]:
                    coverage_counts[w] += 1
                current_centers.remove(c_old)
                current_centers.add(c_new)
                # Count doesn't change, so best_count might not update, but we explore.
                
        else:
            # Add
            c_new = random.choice(all_indices)
            if c_new in current_centers:
                T *= cooling_rate
                continue
                
            # Calculate delta cost: +1
            # Accept with probability exp(-1/T)
            if random.random() < np.exp(-1.0 / T):
                # Accept
                for w in all_balls[c_new]:
                    coverage_counts[w] += 1
                current_centers.add(c_new)
                # We don't update best here because count increased
            else:
                pass
                
        T *= cooling_rate
        
    # Final check: ensure best_centers is valid
    # It should be, as we only update best_centers when a valid removal occurred.
    
    # Convert indices back to words
    final_words = [idx_to_word[i] for i in best_centers]
    
    # Verify
    covered_set = set()
    for w in final_words:
        idx = word_to_idx[w]
        covered_set.update(all_balls[idx])
        
    if len(covered_set) != total_words:
        # Fallback to initial greedy
        final_words = [idx_to_word[i] for i in centers]
        
    return [list(w) for w in final_words]
# EVOLVE-BLOCK-END
