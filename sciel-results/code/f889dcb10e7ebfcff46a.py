import itertools
import random
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters if ever called (though task is specific)
        # Simple greedy for general case
        total_words = q ** n
        # Precompute balls
        balls = []
        for w in range(total_words):
            word = [0] * n
            val = w
            for i in range(n):
                word[i] = val % q
                val //= q
            ball = [w]
            for i in range(n):
                for d in range(1, q):
                    new_word = word.copy()
                    new_word[i] = (word[i] + d) % q
                    new_val = 0
                    for j in range(n):
                        new_val = new_val * q + new_word[j]
                    ball.append(new_val)
            balls.append(ball)
            
        count = [0] * total_words
        S = []
        
        # Greedy
        while True:
            if time.time() > deadline: break
            best_gain = -1
            best_c = -1
            for c in range(total_words):
                gain = 0
                for w in balls[c]:
                    if count[w] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_gain == 0:
                break
            S.append(best_c)
            for w in balls[best_c]:
                count[w] += 1
                
        # Reduce
        changed = True
        while changed:
            changed = False
            for i in range(len(S) - 1, -1, -1):
                c = S[i]
                redundant = True
                for w in balls[c]:
                    if count[w] <= 1:
                        redundant = False
                        break
                if redundant:
                    S.pop(i)
                    for w in balls[c]:
                        count[w] -= 1
                    changed = True
                    
        return [[w % q for w in [int((c // (q**i)) % q) for i in range(n)]] for c in S]

    # Specific optimized path for q=3, n=6, radius=1
    total_words = 729
    
    # Precompute balls
    balls = []
    for w in range(total_words):
        word = [0] * 6
        val = w
        for i in range(6):
            word[i] = val % 3
            val //= 3
        ball = [w]
        for i in range(6):
            for d in [1, 2]:
                new_word = word.copy()
                new_word[i] = (word[i] + d) % 3
                new_val = 0
                for j in range(6):
                    new_val = new_val * 3 + new_word[j]
                ball.append(new_val)
        balls.append(ball)
        
    count = [0] * total_words
    S = []
    
    # 1. Greedy Construction
    while True:
        if time.time() > deadline:
            break
        best_gain = -1
        best_c = -1
        for c in range(total_words):
            gain = 0
            for w in balls[c]:
                if count[w] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_gain == 0:
            break
        S.append(best_c)
        for w in balls[best_c]:
            count[w] += 1
            
    # 2. Reduction Phase
    def reduce_set():
        changed = True
        while changed:
            changed = False
            for i in range(len(S) - 1, -1, -1):
                c = S[i]
                redundant = True
                for w in balls[c]:
                    if count[w] <= 1:
                        redundant = False
                        break
                if redundant:
                    S.pop(i)
                    for w in balls[c]:
                        count[w] -= 1
                    changed = True
                    # Early exit from inner loop if we want, but full scan is safer for stability
                    # Actually, breaking out of for i loop is good to restart scan
                    break 
            # If we broke, the while changed loop will restart
            
    reduce_set()
    
    # 3. Local Search: Swap to enable Reduction
    # Helper to check if a swap is valid
    def is_valid_swap(c_out, c_in):
        # Check if c_in covers all words uniquely covered by c_out
        for w in balls[c_out]:
            if count[w] == 1:
                # c_out is the only one covering w
                if w not in balls[c_in]:
                    return False
        return True

    # Helper to apply swap
    def apply_swap(c_out, c_in):
        S.remove(c_out)
        S.append(c_in)
        for w in balls[c_out]:
            count[w] -= 1
        for w in balls[c_in]:
            count[w] += 1

    # Helper to find redundant codeword
    def find_redundant():
        for c in S:
            redundant = True
            for w in balls[c]:
                if count[w] <= 1:
                    redundant = False
                    break
            if redundant:
                return c
        return -1

    # Main Local Search Loop
    neutral_moves = 0
    while time.time() < deadline:
        improvement_made = False
        
        # Try to find a swap that leads to a reduction
        # Iterate through all c_out
        candidates = []
        for c_out in S:
            # Find valid c_in
            valid_ins = []
            for c_in in range(total_words):
                if c_in in S:
                    continue
                if is_valid_swap(c_out, c_in):
                    valid_ins.append(c_in)
            
            if not valid_ins:
                continue
                
            # For each valid swap, check if it creates a redundancy
            for c_in in valid_ins:
                # Simulate swap
                # We don't want to actually modify S/count until we are sure
                # We can check redundancy in the hypothetical set S' = (S \ {c_out}) U {c_in}
                
                # A codeword c is redundant in S' if all w in balls[c] have count' > 1
                # count'[w] = count[w] - 1 (if w in balls[c_out]) + 1 (if w in balls[c_in])
                
                # Check if c_in is redundant?
                # c_in covers uniques, so count'[w] for w in uniques is 1. So c_in is NOT redundant.
                
                # Check if any c in S \ {c_out} becomes redundant
                # c becomes redundant if for all w in balls[c], count'[w] > 1
                # count'[w] > 1 implies:
                # If w in balls[c_out] and w in balls[c_in]: count[w] > 1
                # If w in balls[c_out] and w NOT in balls[c_in]: count[w] - 1 > 1 => count[w] > 2
                # If w NOT in balls[c_out] and w in balls[c_in]: count[w] + 1 > 1 => count[w] >= 0 (always true)
                # If w NOT in balls[c_out] and w NOT in balls[c_in]: count[w] > 1
                
                # This check is expensive for all c.
                # Optimization: Only check codewords that "interact" with the swap.
                # Codewords c such that balls[c] intersects balls[c_out] or balls[c_in].
                
                # However, a simpler heuristic: Just perform the swap, check for redundancy, and if found, remove.
                # If not found, revert?
                # Or, just perform the swap if it's "promising".
                
                # Let's just perform the swap, check reduction, and if no reduction, revert.
                # This is a "try and see" approach.
                
                # To avoid overhead, we can limit the number of swaps tried per c_out.
                # But 729 is small.
                
                # Apply swap
                apply_swap(c_out, c_in)
                
                # Check for redundancy
                red_c = find_redundant()
                if red_c != -1:
                    # Remove redundant
                    S.remove(red_c)
                    for w in balls[red_c]:
                        count[w] -= 1
                    improvement_made = True
                    break # Break out of c_in loop
                else:
                    # Revert swap
                    apply_swap(c_in, c_out)
                    
            if improvement_made:
                break # Break out of c_out loop
                
        if improvement_made:
            neutral_moves = 0
            # Run full reduction again to catch cascading redundancies
            reduce_set()
        else:
            # No improving swap found. Try a neutral swap to escape local minimum.
            # Pick a random c_out and a random valid c_in.
            c_out = random.choice(S)
            valid_ins = []
            for c_in in range(total_words):
                if c_in in S:
                    continue
                if is_valid_swap(c_out, c_in):
                    valid_ins.append(c_in)
            
            if valid_ins:
                c_in = random.choice(valid_ins)
                apply_swap(c_out, c_in)
                # Try to reduce, but don't revert if it fails
                # This changes the state
                reduce_set()
                neutral_moves += 1
                # If we've done too many neutral moves, we might be cycling.
                # But time limit handles this.
            else:
                # No valid swaps at all? Stuck hard.
                # Break? Or just wait?
                # If no valid swaps, we are in a local minimum of the swap graph.
                # We can try to remove a codeword and re-add greedily?
                # For now, just break out of local search to save time for other things (if any)
                # Or just stop.
                break

    # Convert indices to lists
    result = []
    for c in S:
        word = [0] * 6
        val = c
        for i in range(6):
            word[i] = val % 3
            val //= 3
        result.append(word)
        
    return result
