"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    deadline = time.time() + 110  # Safety margin before 120s limit

    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # Generic fallback: all words (valid but large)
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific case: q=3, n=7, radius=1
    total_words = 3 ** 7  # 2187

    # Precompute all words as tuples
    all_words = [tuple(w) for w in itertools.product(range(3), repeat=7)]
    word_index = {w: i for i, w in enumerate(all_words)}

    # Precompute balls as index arrays for fast numpy operations
    # Ball size: 1 + 7*2 = 15
    balls_indices = np.zeros((total_words, 15), dtype=np.int32)
    for i, w in enumerate(all_words):
        ball = [i]
        for pos in range(7):
            for val in range(3):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    ball.append(word_index[tuple(new_w)])
        balls_indices[i] = ball

    # Greedy set cover using numpy for speed
    covered = np.zeros(total_words, dtype=np.int32)
    chosen = []
    uncovered_count = total_words

    while uncovered_count > 0 and time.time() < deadline - 30:
        # Compute gain for each word: number of uncovered words in its ball
        gains = np.zeros(total_words, dtype=np.int32)
        for i in range(total_words):
            ball_indices = balls_indices[i]
            gains[i] = np.sum(covered[ball_indices] == 0)
        
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break
        
        chosen.append(best_idx)
        
        # Update coverage
        ball_indices = balls_indices[best_idx]
        for idx in ball_indices:
            if covered[idx] == 0:
                uncovered_count -= 1
            covered[idx] += 1

    # Local optimization
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        
        # 1. Try removing redundant codewords
        for i in range(len(chosen)):
            if time.time() > deadline - 10:
                break
            ball_i = balls_indices[chosen[i]]
            redundant = True
            for idx in ball_i:
                if covered[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for idx in ball_i:
                    covered[idx] -= 1
                chosen.pop(i)
                improved = True
                break

        # 2. Try 1-for-1 swaps
        if not improved:
            for i in range(len(chosen)):
                if time.time() > deadline - 10:
                    break
                ball_i = balls_indices[chosen[i]]
                lost = []
                for idx in ball_i:
                    if covered[idx] == 1:
                        lost.append(idx)
                
                if not lost:
                    # Redundant, remove
                    for idx in ball_i:
                        covered[idx] -= 1
                    chosen.pop(i)
                    improved = True
                    break
                
                # Find replacement
                best_replace = None
                best_new_gain = 0
                lost_set = set(lost)
                
                # Candidate replacements must cover all lost words
                # Intersection of balls of lost words
                candidates = set(balls_indices[lost[0]].tolist())
                for l in lost[1:]:
                    candidates &= set(balls_indices[l].tolist())
                
                for j in candidates:
                    if j in chosen:
                        continue
                    ball_j = balls_indices[j]
                    new_gain = np.sum(covered[ball_j] == 0)
                    # Ensure it covers all lost (already by construction of candidates)
                    # But check if it's actually better
                    if new_gain > best_new_gain:
                        best_new_gain = new_gain
                        best_replace = j
                
                if best_replace is not None:
                    # Perform swap
                    for idx in ball_i:
                        covered[idx] -= 1
                    chosen[i] = best_replace
                    for idx in balls_indices[best_replace]:
                        covered[idx] += 1
                    improved = True
                    break

        # 3. Try 2-for-1 swaps (Remove 2, Add 1)
        if not improved:
            n_chosen = len(chosen)
            for i in range(n_chosen):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, n_chosen):
                    if time.time() > deadline - 10:
                        break
                    
                    # Calculate lost words if we remove i and j
                    ball_i = balls_indices[chosen[i]]
                    ball_j = balls_indices[chosen[j]]
                    
                    # Words that are covered by i or j
                    # We need to check which ones become uncovered
                    lost = []
                    # Union of balls
                    # Efficiently: iterate over ball_i and ball_j
                    # A word w is lost if:
                    # 1. w is in ball_i or ball_j
                    # 2. covered[w] - (1 if w in ball_i else 0) - (1 if w in ball_j else 0) == 0
                    
                    # To avoid double counting in union, we can just check all w in ball_i U ball_j
                    # Since balls are small (15), we can just loop
                    # But we need to know if w is in ball_i and ball_j
                    # Let's just iterate over ball_i
                    for idx in ball_i:
                        in_i = 1
                        in_j = 1 if idx in ball_j else 0 # This is slow. 
                        # Better: create a set of ball_j
                        # Or just check covered count
                        # If covered[idx] == 1 and idx in ball_i: 
                        #   If idx in ball_j: removing i and j removes 2 covers -> 0. Lost.
                        #   If idx not in ball_j: removing i removes 1 cover -> 0. Lost.
                        # If covered[idx] == 2:
                        #   If idx in ball_i and idx in ball_j: removing both -> 0. Lost.
                        #   If idx in ball_i and not in ball_j: removing i -> 1. Safe.
                        #   If idx not in ball_i (so in ball_j): removing j -> 1. Safe.
                        
                        # So:
                        # If idx in ball_i:
                        #   if covered[idx] == 1: Lost (regardless of j, because i is one of the covers)
                        #   if covered[idx] == 2 and idx in ball_j: Lost (both covers removed)
                        #   else: Safe
                        # If idx not in ball_i (but in ball_j):
                        #   if covered[idx] == 1: Lost
                        #   if covered[idx] == 2 and idx in ball_i: Impossible (not in ball_i)
                        #   So if covered[idx] == 1: Lost.
                        
                        # Let's simplify:
                        # w is lost if (w in ball_i and covered[w] - 1 == 0) OR 
                        #             (w in ball_j and covered[w] - (1 if w in ball_i else 0) - 1 == 0)
                        
                        # Let's just use a set for ball_j for fast lookup
                        set_j = set(ball_j.tolist())
                        
                        # Check ball_i
                        for idx in ball_i:
                            if idx in set_j:
                                # In both
                                if covered[idx] <= 2:
                                    # If covered is 1, it's covered by i and j? No, covered is total.
                                    # If covered[idx] == 1, it's covered by exactly one of them? No, it's in both balls, so it's covered by both?
                                    # No, "in ball" means distance <= 1. It doesn't mean it's covered by that codeword in the count sense?
                                    # Yes, covered[idx] counts how many chosen codewords are in ball(idx).
                                    # So if idx is in ball_i and ball_j, then i and j both contribute to covered[idx].
                                    # So if we remove i and j, we subtract 2.
                                    # If covered[idx] == 1, it's impossible that it's in both balls (would imply count >= 2).
                                    # So if idx in ball_i and ball_j, covered[idx] >= 2.
                                    # If covered[idx] == 2, then removing i and j makes it 0. Lost.
                                    # If covered[idx] > 2, Safe.
                                    if covered[idx] == 2:
                                        lost.append(idx)
                            else:
                                # Only in ball_i
                                # Removing i subtracts 1.
                                if covered[idx] == 1:
                                    lost.append(idx)
                        
                        # Check ball_j (excluding those already in ball_i)
                        for idx in ball_j:
                            if idx in set_j and idx in set(ball_i.tolist()):
                                continue # Already processed
                            # Only in ball_j
                            if covered[idx] == 1:
                                lost.append(idx)
                            
                    if not lost:
                        # Both are redundant simultaneously?
                        # Remove both
                        for idx in ball_i:
                            covered[idx] -= 1
                        for idx in ball_j:
                            if covered[idx] > 0: # Avoid double decrement if overlap? 
                                # Wait, if idx is in both, we decremented twice?
                                # My logic above: 
                                # If in both, covered[idx] >= 2.
                                # We need to decrement by 2.
                                # My loop:
                                # for idx in ball_i: covered[idx] -= 1
                                # for idx in ball_j: covered[idx] -= 1
                                # This correctly decrements by 2 for overlaps.
                                pass
                        # But we need to check if we double decremented non-overlaps?
                        # No, ball_i and ball_j are separate lists.
                        # If idx is in both, it appears in both loops. Correct.
                        # If idx is only in ball_i, it appears in first loop. Correct.
                        
                        chosen.pop(i)
                        chosen.pop(j-1) # Adjust index
                        improved = True
                        break
                        
                    # If lost is not empty, try to find a replacement
                    # Candidates must cover all lost words
                    if len(lost) <= 4: # Limit size for efficiency
                        candidates = set(balls_indices[lost[0]].tolist())
                        for l in lost[1:]:
                            candidates &= set(balls_indices[l].tolist())
                        
                        best_replace = None
                        best_new_gain = 0
                        for k in candidates:
                            if k in chosen:
                                continue
                            ball_k = balls_indices[k]
                            new_gain = np.sum(covered[ball_k] == 0)
                            if new_gain > best_new_gain:
                                best_new_gain = new_gain
                                best_replace = k
                        
                        if best_replace is not None:
                            # Perform swap
                            for idx in ball_i:
                                covered[idx] -= 1
                            for idx in ball_j:
                                covered[idx] -= 1
                            chosen.pop(i)
                            chosen.pop(j-1)
                            chosen.append(best_replace)
                            for idx in balls_indices[best_replace]:
                                covered[idx] += 1
                            improved = True
                            break
                            
                if improved:
                    break

    # Convert to lists
    return [list(all_words[idx]) for idx in chosen]
# EVOLVE-BLOCK-END
