import time
import numpy as np

def construct(q, n, radius):
    # Deadline guard: 150s limit, aim for 110s safety margin
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Specific parameters for the task: q=3, n=7, radius=1
    total_words = q ** n
    
    # Helper to convert int to base-q tuple
    def int_to_tuple(x, n, q):
        t = [0] * n
        for i in range(n):
            t[i] = x % q
            x //= q
        return t

    # Precompute the "ball" of each word: which words are within radius?
    # For q=3, n=7, radius=1, each word covers 1 + n*(q-1) = 1 + 7*2 = 15 words.
    
    balls = [None] * total_words
    for i in range(total_words):
        t_i = int_to_tuple(i, n, q)
        ball = [i]
        for pos in range(n):
            orig_val = t_i[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                t_new = t_i[:]
                t_new[pos] = new_val
                x_new = 0
                for k in range(n-1, -1, -1):
                    x_new = x_new * q + t_new[k]
                ball.append(x_new)
        balls[i] = ball

    # Phase 0: Greedy Initialization
    # We want a valid starting point. Greedy Set Cover is a good deterministic start.
    covered = np.zeros(total_words, dtype=bool)
    initial_codewords = []
    
    # To speed up init, we can precompute the "coverage potential" or just loop.
    # 2187 words * 15 neighbors. Greedy steps ~ 150-200.
    # 200 * 2187 * 15 = ~6.5M ops. This is fast in Python if optimized, but loops are slow.
    # Let's use a slightly smarter greedy or just accept the time cost for init.
    
    # Optimization for Greedy Init:
    # Instead of checking all 2187 every time, we can maintain a list of "uncovered" words.
    # But a word is covered if any codeword covers it.
    # Let's stick to the simple loop but break early if possible? No.
    # Actually, for 2187 items, a pure Python loop for greedy init might take 1-2 seconds. That's acceptable.
    
    while time.time() < deadline - 90: # Leave plenty of time for ILS
        best_idx = -1
        best_count = -1
        # Iterate through all words to find the one covering the most uncovered points
        for i in range(total_words):
            cnt = 0
            for j in balls[i]:
                if not covered[j]:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        
        if best_count == 0:
            break
        
        initial_codewords.append(best_idx)
        for j in balls[best_idx]:
            covered[j] = True
            
    # Ensure validity (just in case greedy didn't finish due to time, though unlikely)
    final_covered = np.zeros(total_words, dtype=bool)
    for c in initial_codewords:
        final_covered[balls[c]] = True
    if not np.all(final_covered):
        uncovered_indices = np.where(~final_covered)[0]
        for u in uncovered_indices:
            if u not in initial_codewords:
                initial_codewords.append(u)
                final_covered[balls[u]] = True

    # Convert to binary vector
    x = np.zeros(total_words, dtype=np.uint8)
    for c in initial_codewords:
        x[c] = 1
        
    # Calculate initial cover_count
    cover_count = np.zeros(total_words, dtype=np.uint8)
    for c in initial_codewords:
        cover_count[balls[c]] += 1
        
    current_size = len(initial_codewords)
    best_size = current_size
    best_x = x.copy()
    
    # Iterated Local Search (ILS)
    # Strategy:
    # 1. Local Optimize (Greedy Remove): Try to remove any codeword that is redundant.
    #    A codeword c is redundant if all j in balls[c] have cover_count[j] > 1.
    #    If we find one, remove it, update counts, and repeat.
    # 2. Perturb: If stuck (no redundant codewords), we need to escape the local optimum.
    #    Perturbation: Pick a random codeword c, remove it, and then repair the solution
    #    by adding random codewords until the solution is valid again.
    #    This "Random-Insert" repair ensures we move to a different valid region.
    
    # To make ILS efficient, we need a fast way to find redundant codewords.
    # We can maintain a list of "potentially redundant" codewords? 
    # Or just scan. Scanning 2187 items is fast.
    
    # We will run this loop until deadline.
    # To avoid getting stuck in the same local optimum, the perturbation must be strong enough.
    
    # We also track the "best" solution found so far.
    
    # Helper: Greedy Remove Pass
    def greedy_remove(x, cover_count, size):
        # Returns new x, cover_count, size
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            # Scan all codewords
            for i in range(total_words):
                if x[i] == 1:
                    # Check if i is redundant
                    redundant = True
                    for j in balls[i]:
                        if cover_count[j] == 1:
                            redundant = False
                            break
                    if redundant:
                        # Remove i
                        x[i] = 0
                        for j in balls[i]:
                            cover_count[j] -= 1
                        size -= 1
                        changed = True
                        # Restart scan? Or continue? 
                        # Continuing is faster, but might miss other redundancies created immediately?
                        # Removing i decreases counts, so it might make OTHER codewords redundant.
                        # So restarting is safer for finding the local optimum, but slower.
                        # Let's break and restart to ensure we find the "deepest" local optimum quickly.
                        break
        return x, cover_count, size

    # Helper: Repair (Random Insert)
    # Given an invalid x (with some uncovered words), add random codewords until valid.
    def repair(x, cover_count, size):
        # Find uncovered words
        # We need to know which words are uncovered.
        # We can just loop until no uncovered words remain.
        # To be efficient, we can pick a random uncovered word and add a codeword covering it.
        
        # First, identify uncovered words.
        # Since we just removed one codeword, only words in its ball might be uncovered.
        # But in general perturbation, we might remove multiple? No, we remove one.
        # So we can just check the ball of the removed codeword.
        # Wait, the repair function should be general.
        
        # Let's just do a generic repair:
        # While there are uncovered words:
        # 1. Find an uncovered word u.
        # 2. Pick a random codeword c from balls[u].
        # 3. Add c to x.
        
        # To find uncovered words efficiently:
        # We can maintain a list of uncovered words? 
        # Or just scan cover_count for 0s. Scanning 2187 is fast.
        
        while time.time() < deadline - 5:
            # Find an uncovered word
            uncovered_idx = -1
            for j in range(total_words):
                if cover_count[j] == 0:
                    uncovered_idx = j
                    break
            
            if uncovered_idx == -1:
                # Valid
                break
            
            # Pick a random codeword from balls[uncovered_idx]
            # balls[uncovered_idx] contains all words that cover uncovered_idx
            candidates = balls[uncovered_idx]
            # Pick one randomly
            c = candidates[np.random.randint(len(candidates))]
            
            # Add c
            if x[c] == 0:
                x[c] = 1
                for j in balls[c]:
                    cover_count[j] += 1
                size += 1
            # If x[c] was already 1, it means we picked a codeword that was already there?
            # But if cover_count[uncovered_idx] was 0, then no codeword in balls[uncovered_idx] was in x.
            # So x[c] must be 0.
            
        return x, cover_count, size

    # Main ILS Loop
    # We start with the greedy solution.
    # We repeatedly:
    # 1. Greedy Remove to local optimum.
    # 2. If size < best_size, update best.
    # 3. Perturb: Remove a random codeword, then Repair.
    # 4. Repeat.
    
    # To avoid infinite loops on the same perturbation, we can keep a history or just rely on randomness.
    
    # Optimization: 
    # The "Greedy Remove" scan is O(N * BallSize). 
    # The "Repair" is O(NumUncovered * 1).
    
    # Let's run this.
    
    # We need to be careful with the "Restart scan" in greedy_remove. 
    # If we break on the first removal, we might do many passes.
    # For N=2187, BallSize=15, one pass is ~30k ops. 
    # If we remove 10 words, 10 passes = 300k ops. Very fast.
    
    # Perturbation strategy detail:
    # Instead of just removing one random codeword, we can remove a random codeword 
    # and then repair. This is a "kick".
    
    # Let's add a "cooling" or "diversification" mechanism? 
    # Maybe increase the number of removals in perturbation as we get stuck?
    # For now, stick to single removal + repair.
    
    # To ensure we explore, we can also try "Swap" perturbations? 
    # But Remove+Repair is simpler and effective for set cover.
    
    # Let's run the loop.
    
    # To prevent getting stuck in a cycle, we can track the last few sizes.
    # But random repair should escape.
    
    # One issue: If the current solution is already optimal (or near), 
    # removing a codeword and repairing might result in a larger solution.
    # That's fine, we keep the best.
    
    # Let's also add a "Greedy Add" step? No, Repair does that.
    
    # Let's refine the Greedy Remove to be faster.
    # We can maintain a list of "active" codewords.
    # But scanning 2187 is so fast it's not worth the complexity.
    
    # Let's implement the loop.
    
    # To avoid modifying the best_x directly, we work on a copy.
    
    # We will use a "temperature" like mechanism? No, deterministic ILS.
    # But we can vary the perturbation strength.
    # Start with 1 removal. If stuck for a while, try 2 removals?
    # Let's keep it simple: 1 removal.
    
    # To speed up the "Find uncovered" in repair, we can keep a list of uncovered words.
    # But since we only remove one codeword at a time, the set of newly uncovered words
    # is a subset of the ball of the removed codeword.
    # So we can just check the ball of the removed codeword for zeros.
    
    # Let's optimize repair to only check the affected area.
    
    def repair_fast(x, cover_count, size, affected_words):
        # affected_words: list of words that might be uncovered (ball of removed codeword)
        # We need to cover any word in affected_words that has cover_count 0.
        
        # Create a set of words to cover
        to_cover = set()
        for w in affected_words:
            if cover_count[w] == 0:
                to_cover.add(w)
                
        while to_cover and time.time() < deadline - 5:
            # Pick a random word from to_cover
            # Convert to list for random choice
            u = list(to_cover)[np.random.randint(len(to_cover))]
            
            # Pick a random codeword from balls[u]
            candidates = balls[u]
            c = candidates[np.random.randint(len(candidates))]
            
            # Add c
            if x[c] == 0:
                x[c] = 1
                for j in balls[c]:
                    cover_count[j] += 1
                size += 1
                
            # Update to_cover
            # Any word in balls[c] that was in to_cover is now covered
            # We need to re-evaluate to_cover.
            # Simplest: just rebuild to_cover from affected_words?
            # Or remove covered ones.
            
            # Let's just rebuild to_cover from affected_words to be safe.
            # affected_words is small (15).
            new_to_cover = set()
            for w in affected_words:
                if cover_count[w] == 0:
                    new_to_cover.add(w)
            to_cover = new_to_cover
            
        return x, cover_count, size

    # Main Loop
    # We'll run this until time runs out.
    
    # To avoid immediate repetition, we can store the last removed codeword.
    
    last_removed = -1
    
    while time.time() < deadline - 5:
        # 1. Local Optimize (Greedy Remove)
        x, cover_count, size = greedy_remove(x, cover_count, size)
        
        # 2. Update Best
        if size < best_size:
            best_size = size
            best_x = x.copy()
            
        # 3. Perturb
        # Pick a random codeword to remove
        # We should pick from the current codewords.
        # To avoid picking the same one repeatedly, we can pick randomly from x.
        
        # Get list of current codewords
        # This is O(N). 
        # We can maintain a list of active codewords to make this O(1) or O(K).
        # Let's maintain active_codewords list.
        
        # Actually, creating a list of indices where x[i]==1 is O(N).
        # N=2187. This is fast.
        
        active = np.where(x == 1)[0]
        
        if len(active) == 0:
            break # Should not happen for valid cover
            
        # Pick random active codeword
        # Avoid last_removed if possible
        if len(active) > 1:
            # Filter out last_removed
            candidates_remove = [i for i in active if i != last_removed]
            if not candidates_remove:
                candidates_remove = active
            c_remove = candidates_remove[np.random.randint(len(candidates_remove))]
        else:
            c_remove = active[0]
            
        last_removed = c_remove
        
        # Remove it
        x[c_remove] = 0
        affected = balls[c_remove]
        for j in affected:
            cover_count[j] -= 1
        size -= 1
        
        # 4. Repair
        x, cover_count, size = repair_fast(x, cover_count, size, affected)
        
        # The solution is now valid again.
        # Loop back to Greedy Remove.

    # Final Check and Return
    # Ensure best_x is valid
    check_cover = np.zeros(total_words, dtype=np.uint8)
    for i in range(total_words):
        if best_x[i] == 1:
            check_cover[balls[i]] += 1
            
    if np.any(check_cover == 0):
        # Repair best_x if it somehow became invalid (shouldn't happen if we tracked best_x correctly)
        uncovered_indices = np.where(check_cover == 0)[0]
        for u in uncovered_indices:
            if best_x[u] == 0:
                best_x[u] = 1
                check_cover[balls[u]] += 1
                
    codewords = [i for i in range(total_words) if best_x[i] == 1]
    result = []
    for c in codewords:
        result.append(int_to_tuple(c, n, q))
    return result

# EVOLVE-BLOCK-END
