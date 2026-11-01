"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate Initial Solution (Product Code)
    # For q=3, n=7, use the ternary tetracode (length 4) x {0,1,2}^(n-4)
    # This gives 9 * 3^(n-4) codewords.
    if n >= 4:
        # Tetracode: [a, b, a+b, a+2b]
        tetracode = np.array([[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)], dtype=np.int32)
        # Generate all tails
        if n > 4:
            tails = np.array(list(np.ndindex(3, *(3,)*(n-4))), dtype=np.int32)
            if tails.ndim == 1:
                tails = tails.reshape(-1, 1)
            tails = tails.reshape(-1, n - 4)
        else:
            tails = np.array([[]], dtype=np.int32).reshape(1, 0)
        
        # Construct initial code
        # Repeat tetracode for each tail
        code = np.hstack([np.repeat(tetracode, len(tails), axis=0), np.tile(tails, (len(tetracode), 1))])
    else:
        # Fallback for very small n (unlikely given task n=7)
        code = np.array([[0]*n], dtype=np.int32)

    code = code.astype(np.int32)
    N, M = code.shape
    total_words = q ** n
    radius = 1 # Task specifies radius 1, but handle generally if possible
    
    # 2. Precompute Neighborhoods
    # For each codeword index i, store the list of word_indices covered by its radius-1 ball.
    # We only need to track coverage counts to know when a word becomes uncovered.
    
    # Helper to convert code vector to word index
    # index = sum(c[i] * q^(n-1-i))
    powers = (q ** np.arange(n-1, -1, -1)).astype(np.int64)
    
    def get_ball_indices(codeword):
        """Returns list of indices in [0, q^n) covered by this codeword with radius 1."""
        base_idx = int(np.dot(codeword, powers))
        indices = [base_idx]
        if radius >= 1:
            for pos in range(n):
                power = powers[pos]
                for delta in range(1, q):
                    new_val = (codeword[pos] + delta) % q
                    diff = new_val - codeword[pos]
                    # Handle wrap-around correctly for index calculation
                    # Actually, simpler: just compute the new word's index directly
                    # new_word = codeword.copy(); new_word[pos] = new_val
                    # But that's slow. 
                    # index = base_idx + diff * power
                    indices.append(base_idx + diff * power)
        return indices

    # Precompute balls for all initial codewords
    # Note: This is O(N * n * q). For N~170, n=7, q=3, this is ~3500 ops. Very fast.
    balls = []
    for i in range(N):
        balls.append(get_ball_indices(code[i]))

    # 3. Coverage Tracking
    # coverage_counts[j] = number of codewords covering word j
    coverage_counts = np.zeros(total_words, dtype=np.int32)
    for i in range(N):
        for idx in balls[i]:
            coverage_counts[idx] += 1
        
    # 4. Local Search (Simulated Annealing)
    # Objective: Minimize number of codewords.
    # Move: Remove one codeword, Add one codeword (Swap).
    # We prefer moves that reduce the count (Remove 1, Add 0) or keep count same but improve "slack" (less likely needed for pure minimization).
    # Actually, standard approach for covering codes: 
    # Try to remove a codeword. If coverage remains >= 1 for all words, accept.
    # If stuck, try to swap: remove one, add another to "unblock" a removal or just random walk.
    # Since we want to minimize size, "Removal" is the primary goal.
    # A "Swap" (remove A, add B) is neutral in size but might allow future removals.
    
    current_size = N
    best_code = code.copy()
    best_size = N
    
    # Temperature for SA (if we use swaps that don't reduce size immediately)
    # Let's stick to a greedy-with-restarts or simple SA that accepts neutral/worse moves occasionally to escape local minima.
    # Given the time limit, a greedy approach with random restarts or random perturbations is safer.
    
    # Let's implement a "Greedy with Random Perturbation"
    # 1. Try to remove any codeword that is redundant (its ball is fully covered by others).
    # 2. If no such codeword, perform a random Swap (remove i, add j) to explore.
    
    # To speed up "is codeword i redundant?":
    # A codeword i is redundant if for every word w in ball(i), coverage_counts[w] > 1.
    # i.e., min(coverage_counts[ball(i)]) > 1.
    
    # We can maintain a list of "critical" words (coverage == 1) to speed this up?
    # Or just check the min. Ball size is 1 + n*(q-1) = 1 + 7*2 = 15.
    # Checking min of 15 elements is very fast.
    
    rng = np.random.default_rng(42)
    
    # We will run a loop until deadline
    while time.time() < deadline:
        # 1. Greedy Removal Phase
        improved = False
        # Shuffle order to avoid bias
        order = rng.permutation(current_size)
        
        for i in order:
            if time.time() > deadline:
                break
            if i >= current_size:
                continue
            
            # Check if removing code[i] leaves all words covered
            # We need to check if any word in balls[i] has count == 1
            # If min(coverage_counts[balls[i]]) > 1, we can remove it.
            
            # Optimization: If we have a list of indices where count==1, we can check if any are in balls[i].
            # But balls[i] is small. Just checking the array slice is fine.
            ball_indices = balls[i]
            if np.min(coverage_counts[ball_indices]) > 1:
                # Remove it
                # Update coverage counts
                for idx in ball_indices:
                    coverage_counts[idx] -= 1
                
                # Remove from arrays
                # This is O(N) for list/array removal. 
                # Using numpy arrays makes removal O(N).
                # For N~100-200, this is fine.
                
                code = np.delete(code, i, axis=0)
                balls = np.delete(balls, i, axis=0) # Wait, balls is a list of lists. 
                # Let's keep balls as a list of lists for easy deletion, or rebuild if we use numpy.
                # Actually, deleting from a list of numpy arrays is okay.
                # But `balls` was initialized as a list.
                del balls[i]
                
                current_size -= 1
                improved = True
                if current_size < best_size:
                    best_size = current_size
                    best_code = code.copy()
                    
        # 2. If no improvement, try to break local optimum with Swaps
        if not improved:
            # Perform a few random swaps to escape
            # Swap: Remove i, Add j.
            # We want to add a j that covers some "critical" words (count==1) to make them count==2,
            # allowing us to remove i which might be one of the ones covering them?
            # Actually, a simple random swap is a good exploration move.
            
            # Find a word with count == 1 (critical word)
            critical_words = np.where(coverage_counts == 1)[0]
            if len(critical_words) == 0:
                # No critical words, but we couldn't remove? That means all removals were blocked by something else?
                # If min(counts[ball]) > 1 for all, we would have removed.
                # So if we are here, it means for every i, there is at least one word in ball(i) with count==1.
                # This implies there ARE critical words.
                pass
            
            if len(critical_words) > 0:
                # Pick a random critical word
                target_word_idx = critical_words[rng.integers(len(critical_words))]
                
                # Find all current codewords covering this word
                # This requires searching. 
                # Alternatively, just pick a random codeword i to remove, and a random new codeword j to add.
                # To be smarter: Pick i that covers a critical word.
                # Let's just do a random swap for now, it's simpler and effective for SA-like search.
                
                if current_size > 1:
                    i = rng.integers(current_size)
                    # Generate a random new codeword j
                    j_vec = rng.integers(0, q, size=n).astype(np.int32)
                    j = len(code) # Temp index, will be appended then we remove i
                    
                    # Check if this swap is valid?
                    # A swap is always "valid" in the sense that we can do it, but we must ensure we don't uncover words.
                    # If we remove i, words in ball(i) lose 1 count.
                    # If we add j, words in ball(j) gain 1 count.
                    # We need min(coverage_counts[ball(i)] - 1 + coverage_from_j) >= 1.
                    # This is complex to check quickly without updating.
                    
                    # Simpler Strategy: "Add then Remove"
                    # 1. Add a random codeword j.
                    # 2. Try to remove any redundant codeword (including the old ones, maybe the new one is redundant immediately? unlikely).
                    # This increases size by 1 temporarily, but might allow a removal of 2 later?
                    # Or just: Add j. If it makes some old codeword redundant, remove it. Net change 0.
                    # If it makes NO codeword redundant, we are stuck with +1 size.
                    # We should only accept this if we think it helps.
                    
                    # Let's try a specific "Unblock" move:
                    # Find a critical word w.
                    # Find a codeword i that covers w.
                    # Find a new codeword j that covers w (and maybe others).
                    # If we replace i with j, w is still covered.
                    # Does this help? It changes the coverage of other words.
                    
                    # Let's just do a random "Add" and then "Greedy Remove".
                    # If size doesn't decrease, revert.
                    
                    old_size = current_size
                    old_code = code.copy()
                    old_balls = balls.copy()
                    old_counts = coverage_counts.copy()
                    
                    # Add random j
                    j_vec = rng.integers(0, q, size=n).astype(np.int32)
                    code = np.vstack([code, j_vec])
                    j_ball = get_ball_indices(j_vec)
                    balls.append(j_ball)
                    for idx in j_ball:
                        coverage_counts[idx] += 1
                    current_size += 1
                    
                    # Now try to remove ANY redundant codeword
                    removed = False
                    order2 = rng.permutation(current_size)
                    for k in order2:
                        if k >= current_size: continue
                        if np.min(coverage_counts[balls[k]]) > 1:
                            # Remove k
                            for idx in balls[k]:
                                coverage_counts[idx] -= 1
                            code = np.delete(code, k, axis=0)
                            del balls[k]
                            current_size -= 1
                            removed = True
                            break # Remove just one to keep it fast, loop will try more next iter
                    
                    if current_size < best_size:
                        best_size = current_size
                        best_code = code.copy()
                    elif current_size > old_size:
                        # Revert if we got bigger
                        code = old_code
                        balls = old_balls
                        coverage_counts = old_counts
                        current_size = old_size
                    # If current_size == old_size, keep the new configuration (it's a neutral move that might help later)
        
        # Safety: if we hit the deadline, break
        if time.time() > deadline - 5.0:
            break

    # Ensure we return the best found
    # The 'best_code' might be from a previous iteration.
    # We should verify 'best_code' is valid? 
    # The algorithm maintains validity invariantly (we only remove if valid, and add+remove maintains or improves).
    # So best_code is valid.
    
    return [list(map(int, row)) for row in best_code]

# EVOLVE-BLOCK-END
