import time
import numpy as np

def construct(q, n, radius):
    # Deadline guard: 150s limit, aim for 115s safety margin
    start_time = time.time()
    deadline = start_time + 115.0
    
    total_words = q ** n
    
    # Helper to convert int to base-q tuple
    def int_to_tuple(x, n, q):
        t = [0] * n
        for i in range(n):
            t[i] = x % q
            x //= q
        return t

    # Precompute the "ball" of each word: which words are within radius?
    # For q=3, n=7, radius=1, each word covers 1 + n*(q-1) = 15 words.
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
    covered = np.zeros(total_words, dtype=bool)
    initial_codewords = []
    
    while True:
        if time.time() > deadline - 90:
            break
        best_idx = -1
        best_count = -1
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
        covered[balls[best_idx]] = True
        
    # Ensure validity if greedy timed out
    final_covered = np.zeros(total_words, dtype=bool)
    for c in initial_codewords:
        final_covered[balls[c]] = True
    if not np.all(final_covered):
        uncovered_indices = np.where(~final_covered)[0]
        for u in uncovered_indices:
            if u not in initial_codewords:
                initial_codewords.append(u)
                final_covered[balls[u]] = True

    # State representation: list of codeword indices
    current_codewords = list(initial_codewords)
    best_codewords = list(initial_codewords)
    best_size = len(best_codewords)
    
    # Helper to check if a set of codewords is valid
    def is_valid(codewords):
        cover = np.zeros(total_words, dtype=np.uint8)
        for c in codewords:
            cover[balls[c]] += 1
        return np.all(cover > 0)
    
    # Helper to compute coverage counts for a set
    def get_coverage(codewords):
        cover = np.zeros(total_words, dtype=np.uint8)
        for c in codewords:
            cover[balls[c]] += 1
        return cover
    
    # Iterated Local Search
    # Strategy: 
    # 1. Start with current solution.
    # 2. Perform Local Search (Greedy Remove): Try to remove codewords if redundant.
    #    Actually, a better LS for covering codes is "Swap" or "Remove+Add".
    #    Let's use a specific Local Search: 
    #    - Try to remove any codeword if it doesn't break validity.
    #    - If stuck, try swapping one codeword for another that reduces size? 
    #    No, swap keeps size same. We want to reduce size.
    #    So LS should be: Try to remove. If no removal possible, we are in a local optimum.
    #    Then Perturb: Add k random codewords, then run Greedy Remove again.
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 5:
        # Local Search Phase: Greedy Remove
        # We want to find a minimal valid set. 
        # A simple greedy removal: iterate through codewords, if removing one keeps validity, remove it.
        # This is O(N^2 * ball_size) per pass. 
        # To speed up, we can use the coverage array.
        
        current_set = set(current_codewords)
        cover = get_coverage(current_codewords)
        
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            # Iterate in reverse to allow popping
            for i in range(len(current_codewords) - 1, -1, -1):
                c = current_codewords[i]
                # Check if c is redundant: all j in balls[c] have cover[j] > 1
                redundant = True
                for j in balls[c]:
                    if cover[j] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove c
                    current_codewords.pop(i)
                    for j in balls[c]:
                        cover[j] -= 1
                    changed = True
                    # Update best if improved
                    if len(current_codewords) < best_size:
                        best_size = len(current_codewords)
                        best_codewords = list(current_codewords)
            
        # If we are stuck (no changes in last pass), we are in a local optimum.
        # Check time
        if time.time() > deadline - 10:
            break
            
        # Perturbation Phase
        # Strategy: Add a few random codewords, then run Local Search again.
        # How many? Maybe 2-5.
        # Or, a smarter perturbation: "Swap" a codeword with a random one, then LS.
        # But Swap doesn't change size. The LS will then try to remove.
        # Let's try: Pick a random codeword to remove (even if it breaks validity? No, must stay valid).
        # If we can't remove any, we are stuck.
        # So Perturb: 
        # 1. Pick a random codeword c_in_current.
        # 2. Pick a random codeword c_out_not_in_current.
        # 3. Swap them.
        # 4. Run LS (Remove).
        # This changes the "shape" of the cover, potentially allowing more removals.
        
        if len(current_codewords) > 0:
            idx_remove = rng.integers(len(current_codewords))
            c_remove = current_codewords[idx_remove]
            
            # Pick a random candidate to add
            # Ideally one that covers many currently singly-covered words? 
            # For simplicity, just random.
            c_add = rng.integers(total_words)
            
            # If c_add is already in set, skip or pick another?
            # Just add it. Duplicates are handled by set logic in LS? 
            # My current_codewords is a list. 
            # Let's ensure c_add is not in current_set to make it a true swap.
            while c_add in current_set:
                c_add = rng.integers(total_words)
            
            # Perform Swap
            current_codewords[idx_remove] = c_add
            current_set.remove(c_remove)
            current_set.add(c_add)
            
            # Update coverage array for LS
            # Decrement for c_remove, Increment for c_add
            for j in balls[c_remove]:
                cover[j] -= 1
            for j in balls[c_add]:
                cover[j] += 1
            
            # The solution might now be invalid if c_remove was essential and c_add doesn't cover its unique neighbors.
            # But LS (Greedy Remove) assumes a valid start? 
            # If invalid, Greedy Remove will just remove things and make it worse? 
            # Actually, if invalid, `cover[j]==1` check might pass for some, but we have `cover[j]==0`.
            # We need to handle invalid states in LS or ensure validity.
            # To be safe, let's check validity. If invalid, revert? 
            # Or, modify LS to handle invalid: 
            # If invalid, we MUST add words. 
            # This makes the loop complex. 
            # Alternative: Only swap if it preserves validity? 
            # Check: After swap, is cover > 0 everywhere?
            # We can check quickly: 
            # The only words that might become uncovered are those in balls[c_remove] that were ONLY covered by c_remove.
            # i.e., cover_old[j] == 1 for j in balls[c_remove].
            # If any such j is NOT in balls[c_add], then j becomes uncovered.
            
            valid_swap = True
            for j in balls[c_remove]:
                # Note: cover currently reflects the SWAP (dec c_remove, inc c_add)
                # So cover[j] is the NEW count.
                if cover[j] == 0:
                    valid_swap = False
                    break
            
            if not valid_swap:
                # Revert Swap
                current_codewords[idx_remove] = c_remove
                current_set.remove(c_add)
                current_set.add(c_remove)
                for j in balls[c_add]:
                    cover[j] -= 1
                for j in balls[c_remove]:
                    cover[j] += 1
                # Try a different perturbation? Or just restart LS?
                # Let's just continue to next iteration (which will try LS on current state, which is valid)
                # But we want to perturb! 
                # Maybe try adding a random word (increasing size) then LS?
                # "Greedy-Remove + Random-Insert"
                
                # Insert a random word
                c_ins = rng.integers(total_words)
                if c_ins not in current_set:
                    current_codewords.append(c_ins)
                    current_set.add(c_ins)
                    for j in balls[c_ins]:
                        cover[j] += 1
                    # Now run LS to remove the extra weight and potentially others
                else:
                    # If it was already there, just try a different swap later
                    pass
            else:
                # Valid swap, proceed to LS (which happens at top of loop)
                pass
        else:
            # Empty set? Unlikely.
            break

    # Final Verification and Cleanup
    # Ensure best_codewords is valid
    if not is_valid(best_codewords):
        # Repair best_codewords
        cover = get_coverage(best_codewords)
        uncovered = np.where(cover == 0)[0]
        for u in uncovered:
            if u not in best_codewords:
                best_codewords.append(u)
        
        # Try to prune again?
        # Quick prune
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            cover = get_coverage(best_codewords)
            for i in range(len(best_codewords) - 1, -1, -1):
                c = best_codewords[i]
                redundant = True
                for j in balls[c]:
                    if cover[j] == 1:
                        redundant = False
                        break
                if redundant:
                    best_codewords.pop(i)
                    changed = True

    # Convert to result format
    result = []
    for c in best_codewords:
        result.append(int_to_tuple(c, n, q))
    return result

# EVOLVE-BLOCK-END
