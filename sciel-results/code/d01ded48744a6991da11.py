import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    # Parameters from task
    # q=3, n=6, radius=1
    N = q ** n
    deadline = time.time() + 110.0  # 10s margin
    
    # Precompute balls for all words in {0..q-1}^n
    # Represent words as indices 0..N-1
    # Convert index to base-q digits
    def idx_to_digits(idx):
        return [int(idx // (q ** i) % q) for i in range(n)]
    
    def digits_to_idx(digits):
        idx = 0
        for i, d in enumerate(digits):
            idx += d * (q ** i)
        return idx

    # Precompute ball membership as a list of lists of indices
    # balls[c] contains all indices within distance 1 from c
    balls = []
    for idx in range(N):
        digits = idx_to_digits(idx)
        ball = [idx]
        for pos in range(n):
            for delta in range(1, q):
                new_digits = digits[:]
                new_digits[pos] = (digits[pos] + delta) % q
                ball.append(digits_to_idx(new_digits))
        balls.append(ball)
        
    # For faster "is x in ball(c)" checks, we can use a boolean matrix or sets
    # Since N=729, a 729x729 boolean matrix is ~0.5MB, which is fine.
    # Or just use sets for the balls if needed, but we mostly need to update coverage.
    # We will use the list of balls for updates.

    # Initial code: Tetracode direct sum
    # Tetracode words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            initial_codewords.append(digits_to_idx(word))
            
    codewords = initial_codewords[:]
    cov = np.zeros(N, dtype=np.int32)
    
    for c in codewords:
        cov[balls[c]] += 1
        
    best_codewords = codewords[:]
    best_size = len(codewords)
    
    # Simulated Annealing Parameters
    # We want to minimize len(codewords).
    # We will perform moves: Delete, Swap.
    # Temperature T starts high and cools down.
    
    T = 1.0
    cooling = 0.995
    min_T = 1e-5
    
    while time.time() < deadline and T > min_T:
        # 1. Try to Delete a random codeword
        if len(codewords) > 0:
            i = random.randint(0, len(codewords) - 1)
            c = codewords[i]
            # Check if c is redundant
            # If min coverage in ball(c) is > 1, we can remove c
            if np.min(cov[balls[c]]) > 1:
                # Remove c
                cov[balls[c]] -= 1
                codewords.pop(i)
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = codewords[:]
                # Do not cool down on success? Or cool down slowly.
                # Usually, we continue with the same T or decrease slightly.
                # Let's decrease T slightly even on success to encourage exploration later.
                T *= cooling
                continue
                
        # 2. Try a Swap
        if len(codewords) == 0:
            break
            
        i = random.randint(0, len(codewords) - 1)
        c_old = codewords[i]
        c_new = random.randint(0, N - 1)
        
        if c_new == c_old:
            T *= cooling
            continue
            
        # Tentative removal of c_old
        cov[balls[c_old]] -= 1
        
        # Find uncovered words
        missing = np.where(cov < 1)[0]
        
        if len(missing) == 0:
            # c_old was redundant, but we didn't delete it in step 1?
            # This can happen if we just did a swap that made it redundant, or random chance.
            # Actually, if min(cov[balls[c_old]]) > 1, we would have deleted it in step 1.
            # So if missing is empty, it means c_old was NOT the only cover for any word in its ball,
            # but it might be the only cover for words OUTSIDE its ball? No, removing c_old only affects ball(c_old).
            # So if missing is empty, c_old is redundant.
            # Why didn't we delete it? Maybe we didn't pick it in step 1.
            # We can just replace it with c_new. Size stays same.
            # But wait, if it's redundant, why keep it? We should delete it.
            # Let's delete it instead of swapping.
            cov[balls[c_old]] += 1 # Revert for now, we will delete in next iteration or just delete now.
            # Let's just delete it to reduce size.
            cov[balls[c_old]] -= 1
            codewords.pop(i)
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = codewords[:]
            T *= cooling
            continue
        else:
            # c_old is essential. Check if c_new covers all missing words.
            # missing is an array of indices.
            # Check if all missing are in ball(c_new)
            # We can check this by iterating missing or using a set.
            # For speed, let's use a set for ball(c_new) if not already available.
            # Or just check: is set(missing).issubset(balls[c_new])?
            # Creating a set every time is expensive.
            # Let's precompute ball sets.
            
            # Actually, let's just check if the number of missing words is small.
            # If not, it's unlikely c_new covers all.
            valid_swap = True
            for m in missing:
                if m not in balls[c_new]: # This is O(len(missing) * len(ball)) if balls is list.
                    # Wait, balls[c_new] is a list. 'in' is O(N).
                    # This is too slow if missing is large.
                    # Let's use a boolean array for membership?
                    # Or precomputed sets.
                    pass
                    
            # Let's optimize the check.
            # Precompute ball_sets outside the loop?
            # We didn't do that in the seed. Let's do it now.
            
            # Re-initialize ball_sets if not done.
            # To avoid re-doing work, we should have precomputed this.
            # Since we are in the loop, let's assume we can do a quick check.
            # If len(missing) > 10, it's very unlikely a single random word covers all.
            # We can skip the check if len(missing) is large.
            
            if len(missing) > 10:
                valid_swap = False
            else:
                # Check if all missing are in ball(c_new)
                # Convert balls[c_new] to a set for this check? No, that's expensive.
                # Let's just iterate.
                b_new = balls[c_new]
                # Optimization: if len(b_new) < len(missing), impossible.
                if len(b_new) < len(missing):
                    valid_swap = False
                else:
                    for m in missing:
                        if m not in b_new:
                            valid_swap = False
                            break
                            
            if valid_swap:
                # Accept swap
                cov[balls[c_new]] += 1
                codewords[i] = c_new
                # Size is same, so no improvement in best_size, but state changed.
                # We might enable a deletion later.
                T *= cooling
            else:
                # Reject swap, revert
                cov[balls[c_old]] += 1
                T *= cooling
        # End Swap

        # End while loop iteration
        # Note: The T update is done in branches. Need to ensure it happens always.
        # Let's restructure to ensure T updates.

    # Return the best code found
    result = [idx_to_digits(c) for c in best_codewords]
    return result
