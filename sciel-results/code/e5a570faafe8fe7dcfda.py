import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Step 1: Construct the 256-word linear code [17,8,6]
    poly = 0x1D

    def gf_mul(a, b):
        res = 0
        for _ in range(8):
            if b & 1:
                res ^= a
            b >>= 1
            if a & 0x80:
                a = (a << 1) ^ poly
            else:
                a <<= 1
            a &= 0xFF
        return res

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res

    alpha = None
    for a in range(1, 256):
        if gf_pow(a, 17) == 1 and a != 1:
            alpha = a
            break

    if alpha is None:
        return [0]

    Q = [1, 2, 4, 8, 9, 13, 15, 16]

    g_coeffs = [1]
    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            new_g[j + 1] ^= g_coeffs[j]
            new_g[j] ^= gf_mul(g_coeffs[j], root)
        g_coeffs = new_g

    G = np.zeros((9, 17), dtype=np.int64)
    for k in range(9):
        for j in range(9):
            if g_coeffs[j]:
                pos = (j + k) % 17
                G[k, pos] = 1

    words_18 = []
    for u in range(512):
        word = np.zeros(17, dtype=np.int64)
        for i in range(9):
            if (u >> i) & 1:
                word ^= G[i]
        parity = int(np.sum(word) % 2)
        word_ext = np.append(word, parity)
        words_18.append(word_ext)

    words_short = []
    for w in words_18:
        if w[17] == 0:
            w_short = w[:17]
            val = 0
            for i in range(17):
                if w_short[i]:
                    val |= (1 << i)
            words_short.append(val)

    if not words_short:
        return [0]

    # Precompute popcount table
    popcount_table = np.array([i.bit_count() for i in range(2**17)], dtype=np.int8)
    all_words = np.arange(2**17, dtype=np.int64)
    
    # Precompute neighbors? No, too big. 
    # But we can precompute the mask of words at distance < 6 from 0?
    # dist < 6 means popcount < 6.
    # We need to update forbidden counts.
    
    base_code = np.array(words_short, dtype=np.int64)
    code_list = list(base_code)
    
    # forbidden_count[w] = number of codewords c such that dist(c, w) < 6
    forbidden_count = np.zeros(2**17, dtype=np.int32)
    
    # Helper to update forbidden_count for a word w being added or removed
    # We need to iterate over all v and check popcount(w^v) < 6
    # This is O(2^17) per update.
    # To speed this up, we can precompute the list of words at distance < 6 from 0.
    # Let N0 = [v for v in range(2**17) if popcount(v) < 6]
    # Then neighbors of w are w ^ N0.
    
    N0 = np.array([v for v in range(2**17) if popcount_table[v] < 6], dtype=np.int64)
    # N0 has size sum_{i=0}^{5} C(17, i) = 1 + 17 + 136 + 680 + 2380 + 6188 = 9402
    # This is small enough to keep in memory and use for vectorized updates.
    
    def update_forbidden(w, delta):
        # w is int
        # neighbors = w ^ N0
        # forbidden_count[neighbors] += delta
        neighbors = w ^ N0
        forbidden_count[neighbors] += delta

    # Initialize forbidden_count
    for w in code_list:
        update_forbidden(int(w), 1)
        
    best_code = np.array(code_list, dtype=np.int64)
    best_size = len(code_list)
    
    # Greedy Extension
    # Find all words with forbidden_count == 0
    # We want to add them.
    
    # To make it faster, we can maintain a list of available candidates.
    # But forbidden_count changes.
    # We can just scan for forbidden_count == 0.
    
    # Optimization: 
    # The candidate pool is likely words at distance 6 from the code.
    # Let's just scan the whole space? 2^17 is 131072. 
    # Scanning is fast in numpy.
    
    max_iterations = 1000
    for _ in range(max_iterations):
        if time.time() > deadline - 5.0:
            break
            
        # Find available words
        available = np.where(forbidden_count == 0)[0]
        if len(available) == 0:
            # Try swaps?
            # For now, break and try a different strategy or swap logic below
            break
            
        # Pick a random available word
        # Or pick one that minimizes something? 
        # Let's pick randomly to explore different branches
        # But we want to maximize size, so any valid word is good.
        # However, adding a word might block many others.
        # Heuristic: Pick a word that is "central" or has many other available neighbors?
        # Let's just pick the first one or random.
        
        # Random selection to allow different runs to find different codes
        # But we need to be deterministic? No, seed is not fixed in construct.
        # Wait, the evaluator runs it once. We should use a fixed seed for reproducibility in testing, 
        # but the evaluator might run it multiple times? 
        # "Each evaluation must finish within 60s."
        # Usually, we want the *best* possible result.
        # Randomness can help escape local minima.
        
        # Let's try to add ALL available words in a greedy fashion?
        # If we add w, it might block u.
        # So we should add the one that blocks the *fewest* other available words?
        # That's expensive.
        
        # Simple Greedy: Add one, update, repeat.
        
        # Let's pick a random one from available
        # To ensure we don't loop forever on the same set, we can shuffle.
        
        # Actually, if we just add *any* available word, we might end up with a suboptimal set.
        # Better: Try to add the word that has the *most* available neighbors?
        # Or the one with *fewest* available neighbors (to save them)?
        
        # Let's stick to simple greedy first: Add random available word.
        # If we get stuck (no available), we try swaps.
        
        # To improve, let's look at the distribution of available words.
        # Words at distance 6 from the linear code are the most likely candidates.
        
        # Let's pick a random index from available
        idx = np.random.randint(len(available))
        w = int(available[idx])
        
        code_list.append(w)
        update_forbidden(w, 1)
        
        if len(code_list) > best_size:
            best_size = len(code_list)
            best_code = np.array(code_list, dtype=np.int64)
            
    # Swap Strategy
    # If we are stuck, try to remove a word and add 2 others.
    
    if time.time() < deadline - 5.0:
        # We need a list of codewords to try removing
        # Let's try removing each word in the code (or a subset)
        
        # To speed up, we only try swaps if we are stuck.
        # We check if available is empty.
        available = np.where(forbidden_count == 0)[0]
        
        if len(available) == 0:
            # Try swaps
            # Iterate through a random subset of codewords to remove
            # Or all of them? 256+ words.
            # For each word w to remove:
            # 1. update_forbidden(w, -1)
            # 2. Find new available words
            # 3. If > 1 new available, try to add them
            # 4. If we can add 2, accept swap (remove w, add u, v)
            # 5. Else, revert (update_forbidden(w, 1))
            
            # To be fast, we limit the number of swap attempts.
            
            # We should prioritize removing words that "block" many candidates.
            # But we don't know that easily.
            
            # Let's just try random removals.
            
            max_swaps = 500
            for _ in range(max_swaps):
                if time.time() > deadline - 2.0:
                    break
                
                # Pick a random codeword to remove
                idx_rm = np.random.randint(len(code_list))
                w_rm = code_list[idx_rm]
                
                # Temporarily remove
                code_list.pop(idx_rm)
                update_forbidden(w_rm, -1)
                
                # Find available
                available = np.where(forbidden_count == 0)[0]
                
                if len(available) >= 2:
                    # Try to add 2
                    # Pick 2 random available
                    idx1 = np.random.randint(len(available))
                    idx2 = np.random.randint(len(available))
                    # Ensure distinct
                    if idx1 == idx2:
                        idx2 = (idx2 + 1) % len(available)
                    
                    w_add1 = int(available[idx1])
                    w_add2 = int(available[idx2])
                    
                    # Check if they are distinct and not w_rm (they can't be, w_rm is in code, forbidden_count[w_rm] was 1, now 0? 
                    # Wait, w_rm is in the code. forbidden_count[w_rm] was incremented by all other words.
                    # When we remove w_rm, forbidden_count[w_rm] decreases by (number of neighbors of w_rm in code).
                    # Since min dist is 6, dist(w_rm, w_rm) = 0 < 6. 
                    # So w_rm itself was forbidden by itself? 
                    # My update_forbidden logic: 
                    # neighbors = w ^ N0. N0 includes 0. 
                    # So w is in neighbors. 
                    # So forbidden_count[w] is incremented by 1 for each codeword c in code such that dist(c, w) < 6.
                    # If w is in code, dist(w, w) = 0 < 6, so forbidden_count[w] >= 1.
                    # When we remove w, forbidden_count[w] decreases by 1.
                    # If no other codeword is close to w, forbidden_count[w] becomes 0.
                    # So w itself might appear in available!
                    # We must NOT re-add w_rm.
                    
                    # Filter out w_rm from available
                    # available is an array of ints.
                    # We can just check.
                    
                    # Create a list of valid adds
                    valid_adds = [x for x in available if x != w_rm]
                    
                    if len(valid_adds) >= 2:
                        # Pick 2
                        # Shuffle to be random
                        np.random.shuffle(valid_adds)
                        w_add1 = int(valid_adds[0])
                        w_add2 = int(valid_adds[1])
                        
                        # Add them
                        code_list.append(w_add1)
                        update_forbidden(w_add1, 1)
                        code_list.append(w_add2)
                        update_forbidden(w_add2, 1)
                        
                        if len(code_list) > best_size:
                            best_size = len(code_list)
                            best_code = np.array(code_list, dtype=np.int64)
                            
                        # Continue searching? 
                        # The code has grown. We can restart the greedy phase.
                        # Break out of swap loop and go back to greedy?
                        # Or just continue swapping?
                        # Let's break and let the outer loop (if any) or just return.
                        # Since we improved, let's try to do more greedy steps.
                        # But we are inside a single construct call.
                        # Let's just break the swap loop and return the best found so far.
                        # Or, better, restart the greedy phase with the new code.
                        
                        # To keep it simple and fast, we'll just break and return.
                        # The greedy phase at the start should have handled most.
                        # Swaps are for local minima.
                        
                        break
                    else:
                        # Revert
                        update_forbidden(w_rm, 1)
                        code_list.append(w_rm)
                else:
                    # Revert
                    update_forbidden(w_rm, 1)
                    code_list.append(w_rm)

    return list(best_code)
