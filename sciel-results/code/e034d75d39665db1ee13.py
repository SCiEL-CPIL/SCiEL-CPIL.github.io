import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: aim to finish by 118s to be well within the 120s kill switch
    deadline = start_time + 118.0
    
    total_words = q ** n
    
    # Helper functions for conversion
    def int_to_tuple(x, n, q):
        t = [0] * n
        for i in range(n):
            t[i] = x % q
            x //= q
        return t

    def tuple_to_int(t, q):
        x = 0
        for i in range(n-1, -1, -1):
            x = x * q + t[i]
        return x

    # Precompute balls (neighbors within radius)
    # For radius=1, ball size is 1 + n*(q-1)
    # We store balls as lists of integers for fast coverage updates
    balls = [None] * total_words
    for i in range(total_words):
        t_i = int_to_tuple(i, n, q)
        ball = [i]
        if radius >= 1:
            for pos in range(n):
                orig_val = t_i[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    t_new = t_i[:]
                    t_new[pos] = new_val
                    ball.append(tuple_to_int(t_new, q))
        balls[i] = ball
        
    # Precompute M (Coverage Matrix)
    # M[i, j] = 1 if j is in balls[i]
    M = np.zeros((total_words, total_words), dtype=np.uint8)
    for i in range(total_words):
        M[i, balls[i]] = 1

    def greedy_cover():
        """
        Fast greedy set cover to generate an initial valid solution.
        """
        covered = np.zeros(total_words, dtype=bool)
        codewords = []
        
        while True:
            if time.time() > deadline - 5:
                break
            if not np.any(~covered):
                break
            
            # scores = M.dot(~covered) gives number of uncovered words covered by each candidate
            scores = M.dot(~covered)
            best_idx = np.argmax(scores)
            
            if scores[best_idx] == 0:
                break
                
            codewords.append(best_idx)
            covered[balls[best_idx]] = True
            
        return codewords

    def is_valid(codewords):
        """
        Check if the set of codewords covers the entire space.
        """
        if not codewords:
            return total_words == 0
        covered = np.zeros(total_words, dtype=bool)
        for c in codewords:
            covered[balls[c]] = True
        return np.all(covered)

    def local_search_remove(codewords):
        """
        Iteratively remove redundant codewords.
        """
        if not codewords:
            return codewords
        
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            cover_count = np.zeros(total_words, dtype=np.int32)
            for c in codewords:
                cover_count[balls[c]] += 1
            
            for i in range(len(codewords)):
                c = codewords[i]
                # Check if removing c leaves all words in balls[c] covered by others
                # i.e., cover_count[w] > 1 for all w in balls[c]
                # Equivalent to: min(cover_count[w] for w in balls[c]) > 1
                # Or: sum(cover_count[w] == 1 for w in balls[c]) == 0
                if np.min(cover_count[balls[c]]) > 1:
                    codewords.pop(i)
                    changed = True
                    break # Restart loop to update cover_count
            
        return codewords

    def simulated_annealing(codewords, target_size=None):
        """
        Simulated Annealing to minimize the number of codewords.
        State: list of codewords.
        Move: Swap a random codeword with a random other codeword.
        Cost: Number of codewords (penalized if invalid).
        """
        if not codewords:
            return codewords
        
        # Start with a copy
        current = codewords[:]
        
        # Initial temperature
        # The "energy" landscape is discrete and rugged.
        # We define energy as:
        # If valid: len(current)
        # If invalid: len(current) + 1000 * (number of uncovered words)
        # This strong penalty ensures we prioritize validity, but allows "jumps" 
        # if the gain in size is huge? No, usually we want to stay valid.
        # Actually, standard SA for set cover often allows invalid states.
        # Let's use: Cost = len(current) + 100 * uncovered_count.
        
        def get_cost(state):
            if not state:
                return 100000
            covered = np.zeros(total_words, dtype=bool)
            for c in state:
                covered[balls[c]] = True
            uncovered = np.sum(~covered)
            return len(state) + 100 * uncovered

        current_cost = get_cost(current)
        
        # We want to reach a target size, or just minimize.
        # Let's run for a fixed number of iterations or time.
        
        # Initial temperature should be high enough to accept bad moves.
        # Cost difference for a swap is usually small (0, +/- 1, or penalty changes).
        # Penalty changes can be large (100 * delta_uncovered).
        # Let's set T_initial = 50.
        
        T_initial = 50.0
        T_final = 0.1
        # Number of iterations
        # Each iteration is a swap + cost update.
        # Cost update involves O(N * ball_size) if we recompute from scratch.
        # N=2187, ball_size=15. ~33k ops.
        # 1000 iterations ~ 33M ops. Python might be slow.
        # We can optimize cost update?
        # If we swap c_out with c_in:
        # 1. Remove c_out: update covered.
        # 2. Add c_in: update covered.
        # 3. Compute uncovered count.
        
        # Let's try to optimize the cost calculation using the M matrix or just numpy sums.
        # Actually, maintaining a `cover_count` array is better.
        
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in current:
            cover_count[balls[c]] += 1
        
        def get_cost_from_counts(state_size, counts):
            uncovered = np.sum(counts == 0)
            return state_size + 100 * uncovered

        current_cost = get_cost_from_counts(len(current), cover_count)
        
        # SA Loop
        # We will run until deadline - 5s
        # We need to be careful with the number of iterations.
        # Let's do a batch of swaps.
        
        # To make SA effective, we need to explore.
        # Swap: pick random i, j in current. current[i] = current[j].
        # This keeps size constant! 
        # Wait, if we keep size constant, we can only find valid covers of the SAME size.
        # To reduce size, we need to allow "Remove" moves that might lead to invalid states, 
        # or "Replace" moves where we remove one and add none (size -1) if it doesn't break validity too much?
        
        # Alternative: Variable size SA.
        # Moves:
        # 1. Remove a random codeword. (Size -1)
        # 2. Add a random codeword. (Size +1)
        # 3. Swap two codewords. (Size 0)
        
        # If we allow Remove, we might become invalid. The penalty (100 * uncovered) will be high.
        # If we become valid again with smaller size, we accept.
        
        # Let's implement this.
        
        # Optimization: 
        # Instead of full re-computation, we can track changes.
        # But for simplicity and robustness within time limit, let's use a "batch" approach 
        # or just ensure the inner loop is fast enough.
        
        # Let's estimate speed.
        # One move:
        # - Update cover_count (O(ball_size) = 15 ops)
        # - Compute cost (O(N) = 2187 ops for np.sum)
        # Total ~2200 ops.
        # 10,000 moves ~ 22M ops.
        # In Python, 22M simple ops might take ~1-2 seconds? 
        # Numpy sum is fast, but Python loop overhead for 10k iterations is small.
        # Let's try to run as many iterations as possible.
        
        # To speed up cost calculation, we can maintain `uncovered_count` incrementally?
        # When we remove c:
        #   for w in balls[c]:
        #       if cover_count[w] == 1: uncovered_count += 1
        #       cover_count[w] -= 1
        # When we add c:
        #   for w in balls[c]:
        #       if cover_count[w] == 0: uncovered_count -= 1
        #       cover_count[w] += 1
        # This makes cost update O(ball_size) = 15 ops!
        # This is MUCH faster.
        
        current_uncovered = np.sum(cover_count == 0)
        current_size = len(current)
        current_cost = current_size + 100 * current_uncovered
        
        best_state = current[:]
        best_cost = current_cost
        
        # Cooling schedule
        # Let's do a geometric cooling.
        # T = T_initial * (T_final/T_initial)^iter
        # Or just decrease T slowly.
        
        # We'll run for a fixed time slice, say 2 seconds, then check time.
        # Actually, let's just run until deadline.
        
        # To avoid getting stuck in a local minimum of the same size, 
        # we need to occasionally accept "worse" moves.
        
        # Let's use a random walk with restarts if stuck?
        # Or just standard SA.
        
        # Initial T
        T = T_initial
        
        # Number of iterations per second?
        # Let's just loop until time is up.
        
        # To ensure we don't just oscillate, we can add a "restart" mechanism 
        # if no improvement for X iterations.
        
        no_improve_count = 0
        # If no improvement for 1000 iterations, restart T? Or randomize state?
        # Randomizing state is expensive. Just increase T slightly?
        
        while time.time() < deadline - 2:
            # Pick a random move type
            # 1: Remove (prob 0.3)
            # 2: Add (prob 0.3)
            # 3: Swap (prob 0.4)
            
            move_type = random.random()
            
            new_state = current[:]
            new_size = current_size
            new_uncovered = current_uncovered
            
            if move_type < 0.3:
                # Remove
                if current_size == 0:
                    continue
                idx = random.randint(0, current_size - 1)
                c_out = current[idx]
                new_state.pop(idx)
                new_size -= 1
                
                # Update counts
                for w in balls[c_out]:
                    if cover_count[w] == 1:
                        new_uncovered += 1
                    cover_count[w] -= 1
                # Note: We are modifying cover_count in place. 
                # If we reject the move, we must revert.
                
            elif move_type < 0.6:
                # Add
                c_in = random.randint(0, total_words - 1)
                # Check if already in state? 
                # If already in state, adding is a no-op for coverage but increases size? 
                # No, codewords must be distinct. 
                # If c_in is in current, we can't add it.
                # We need to check membership. 
                # Using a set for membership is O(1).
                # But maintaining a set in sync with list is O(1) per op.
                
                # Let's maintain a set `current_set`.
                # For now, let's just check `if c_in in current`. 
                # `in` for list is O(N). N~200. 200 ops. Acceptable.
                if c_in in current:
                    continue
                
                new_state.append(c_in)
                new_size += 1
                
                for w in balls[c_in]:
                    if cover_count[w] == 0:
                        new_uncovered -= 1
                    cover_count[w] += 1
                    
            else:
                # Swap
                if current_size < 2:
                    continue
                idx1 = random.randint(0, current_size - 1)
                idx2 = random.randint(0, current_size - 1)
                if idx1 == idx2:
                    continue
                
                c_out = current[idx1]
                c_in = random.randint(0, total_words - 1)
                if c_in in current:
                    continue
                
                new_state[idx1] = c_in
                # Size stays same
                
                # Update counts: Remove c_out, Add c_in
                for w in balls[c_out]:
                    if cover_count[w] == 1:
                        new_uncovered += 1
                    cover_count[w] -= 1
                    
                for w in balls[c_in]:
                    if cover_count[w] == 0:
                        new_uncovered -= 1
                    cover_count[w] += 1

            new_cost = new_size + 100 * new_uncovered
            delta = new_cost - current_cost
            
            # Acceptance probability
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                current = new_state
                current_size = new_size
                current_uncovered = new_uncovered
                current_cost = new_cost
                
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_state = current[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Reject: Revert cover_count
                if move_type < 0.3:
                    # Revert Remove -> Add back c_out
                    c_out = current[random.randint(0, current_size - 1)] # This is wrong, we need the specific c_out
                    # We stored c_out in the branch above, but scope is tricky.
                    # Let's restructure to store c_out/c_in outside.
                    pass
                
                # To simplify reverting, let's just recompute cover_count from current if rejected?
                # No, that's slow.
                # Let's explicitly handle the revert in the branches.
                
                # I will refactor the loop to make reverting clean.
                # Break and restart logic below.
                break # This break is a placeholder for the messy revert logic. Let's rewrite the loop properly.

    # The above SA loop had a revert issue. Let's write a cleaner version.
    
    def simulated_annealing_clean(codewords):
        if not codewords:
            return codewords
        
        current = codewords[:]
        current_set = set(current)
        
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in current:
            cover_count[balls[c]] += 1
            
        current_uncovered = np.sum(cover_count == 0)
        current_size = len(current)
        current_cost = current_size + 100 * current_uncovered
        
        best_state = current[:]
        best_cost = current_cost
        
        T = 50.0
        # Cooling: T *= 0.999 per iteration? Or time-based?
        # Let's do time-based cooling or fixed steps.
        # Let's run for a fixed number of steps, e.g., 5000, then check time.
        
        steps_per_batch = 2000
        
        while time.time() < deadline - 2:
            for _ in range(steps_per_batch):
                if time.time() > deadline - 2:
                    break
                    
                move_type = random.random()
                
                c_out = None
                c_in = None
                idx_out = -1
                
                if move_type < 0.3:
                    # Remove
                    if current_size == 0:
                        continue
                    idx_out = random.randint(0, current_size - 1)
                    c_out = current[idx_out]
                    
                    # Apply
                    new_size = current_size - 1
                    new_uncovered = current_uncovered
                    for w in balls[c_out]:
                        if cover_count[w] == 1:
                            new_uncovered += 1
                        cover_count[w] -= 1
                        
                    new_cost = new_size + 100 * new_uncovered
                    delta = new_cost - current_cost
                    
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        # Accept
                        current.pop(idx_out)
                        current_set.discard(c_out)
                        current_size = new_size
                        current_uncovered = new_uncovered
                        current_cost = new_cost
                        
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_state = current[:]
                    else:
                        # Revert
                        for w in balls[c_out]:
                            if cover_count[w] == 0:
                                pass # new_uncovered was incremented, so it was 1, now 0? 
                                # Wait. If cover_count[w] was 1, new_uncovered += 1.
                                # Revert: cover_count[w] += 1. 
                                # If it becomes 1, new_uncovered -= 1?
                                # Actually, just re-add.
                                pass
                            cover_count[w] += 1
                            # We don't need to update current_uncovered for revert if we just restore counts.
                            # The "new_uncovered" variable was local.
                            # We just restore cover_count.
                            
                elif move_type < 0.6:
                    # Add
                    c_in = random.randint(0, total_words - 1)
                    if c_in in current_set:
                        continue
                    
                    new_size = current_size + 1
                    new_uncovered = current_uncovered
                    for w in balls[c_in]:
                        if cover_count[w] == 0:
                            new_uncovered -= 1
                        cover_count[w] += 1
                        
                    new_cost = new_size + 100 * new_uncovered
                    delta = new_cost - current_cost
                    
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        # Accept
                        current.append(c_in)
                        current_set.add(c_in)
                        current_size = new_size
                        current_uncovered = new_uncovered
                        current_cost = new_cost
                        
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_state = current[:]
                    else:
                        # Revert
                        for w in balls[c_in]:
                            cover_count[w] -= 1
                            
                else:
                    # Swap
                    if current_size < 1:
                        continue
                    idx_out = random.randint(0, current_size - 1)
                    c_out = current[idx_out]
                    c_in = random.randint(0, total_words - 1)
                    if c_in in current_set:
                        continue
                    
                    new_size = current_size
                    new_uncovered = current_uncovered
                    
                    # Remove c_out
                    for w in balls[c_out]:
                        if cover_count[w] == 1:
                            new_uncovered += 1
                        cover_count[w] -= 1
                        
                    # Add c_in
                    for w in balls[c_in]:
                        if cover_count[w] == 0:
                            new_uncovered -= 1
                        cover_count[w] += 1
                        
                    new_cost = new_size + 100 * new_uncovered
                    delta = new_cost - current_cost
                    
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        # Accept
                        current[idx_out] = c_in
                        current_set.discard(c_out)
                        current_set.add(c_in)
                        current_uncovered = new_uncovered
                        current_cost = new_cost
                        
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_state = current[:]
                    else:
                        # Revert
                        # Re-add c_out
                        for w in balls[c_out]:
                            cover_count[w] += 1
                        # Remove c_in
                        for w in balls[c_in]:
                            cover_count[w] -= 1

                # Cooling
                T *= 0.999
                if T < 0.1:
                    T = 0.1
                    
            # Occasionally restart T to escape local minima?
            # Or just keep going.
            
        return best_state

    # Main Strategy:
    # 1. Generate a valid cover using Greedy.
    # 2. Apply Local Search Remove to minimize.
    # 3. Apply Simulated Annealing to escape local optima.
    
    initial_cover = greedy_cover()
    initial_cover = local_search_remove(initial_cover)
    
    best_codewords = initial_cover[:]
    best_size = len(best_codewords)
    
    # Run SA
    # SA might take a while. We run it for the remaining time.
    # To allow multiple SA runs, we can split the time.
    # Let's run SA for 50% of remaining time, then check.
    # Or just run it until deadline.
    
    # Let's run SA.
    sa_result = simulated_annealing_clean(best_codewords)
    
    # SA returns best_state found.
    # It might be invalid if it got stuck? 
    # No, `best_state` is only updated if `new_cost < best_cost`.
    # `best_cost` starts with a valid state.
    # If a new state is invalid, its cost is high (penalty).
    # So it won't be accepted as best unless the valid state was worse? 
    # Valid state cost = Size. Invalid state cost = Size + 100*Uncovered.
    # Since Uncovered >= 1, Invalid Cost >= Size + 100.
    # So an invalid state can never be better than a valid state of the same or smaller size?
    # If Size increases by 1, but Uncovered decreases by 100? Impossible.
    # So best_state should always be valid if the initial state was valid.
    
    if len(sa_result) < best_size:
        best_codewords = sa_result
        best_size = len(best_codewords)
        
    # Final check: ensure validity
    if not is_valid(best_codewords):
        # Fallback to greedy if SA broke it (shouldn't happen)
        best_codewords = greedy_cover()
        best_codewords = local_search_remove(best_codewords)
        
    # Convert to output format
    result = []
    for c in best_codewords:
        result.append(int_to_tuple(c, n, q))
        
    return result
