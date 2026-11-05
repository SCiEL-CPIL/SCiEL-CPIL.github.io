import time
import sys

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        # Fallback
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            import itertools
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            import itertools
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_words = 3 ** n
    words = [list(t) for t in __import__('itertools').product(range(3), repeat=n)]
    
    # Map word tuple to index
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute ball masks
    # ball_masks[i] is a bitmask of words within distance 1 of words[i]
    ball_masks = [0] * total_words
    # Also precompute coverers: coverers[w] = list of codeword indices c such that w is in ball_masks[c]
    coverers = [[] for _ in range(total_words)]
    
    for i, w in enumerate(words):
        mask = 0
        # Distance 0
        mask |= (1 << i)
        # Distance 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val != orig_val:
                    new_w = w.copy()
                    new_w[pos] = new_val
                    idx = word_to_idx[tuple(new_w)]
                    mask |= (1 << idx)
        ball_masks[i] = mask
        # Add i to coverers for all words in its ball
        # This is O(15 * 2187) which is fine
        # But we can just iterate through the bits? No, iterating 2187 bits is slow.
        # Better: When building mask, we know the indices.
        
    # Re-do coverers efficiently
    coverers = [[] for _ in range(total_words)]
    for i, w in enumerate(words):
        # i covers itself and its neighbors
        coverers[i].append(i)
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val != orig_val:
                    new_w = w.copy()
                    new_w[pos] = new_val
                    idx = word_to_idx[tuple(new_w)]
                    coverers[idx].append(i)
                    
    all_mask = (1 << total_words) - 1
    
    # Initial Greedy Solution to get an upper bound
    uncovered = all_mask
    greedy_code = []
    # Count how many times each word is covered? No, standard greedy:
    # Pick codeword that covers most uncovered words.
    # To speed up, we can just run a simple greedy.
    for _ in range(200): # Safety bound
        if uncovered == 0:
            break
        best_c = -1
        max_gain = 0
        for c in range(total_words):
            gain = bin(uncovered & ball_masks[c]).count('1')
            if gain > max_gain:
                max_gain = gain
                best_c = c
        if best_c == -1:
            break
        greedy_code.append(best_c)
        uncovered &= ~ball_masks[best_c]
        
    best_size = len(greedy_code)
    best_code = greedy_code[:]
    
    # Exact Search (Branch and Bound)
    # Try to find a solution of size best_size - 1, best_size - 2, ...
    # Time limit: 100 seconds (leave 20s for overhead)
    start_time = time.time()
    time_limit = 100.0
    
    # We will search for depth limit = best_size - 1
    # If found, update best_size and best_code, then try best_size - 2, etc.
    
    # To make the search faster, we need to order the coverers.
    # We'll use a recursive function.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def search(uncovered_mask, depth, current_code, max_depth):
        nonlocal best_size, best_code, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return False
            
        if uncovered_mask == 0:
            if depth < best_size:
                best_size = depth
                best_code = current_code[:]
            return True
            
        if depth >= best_size:
            return False
            
        # Choose the uncovered word with the fewest coverers
        # We need to iterate through uncovered bits to find the one with min len(coverers[w])
        # This is expensive if done naively.
        # Optimization: Precompute coverers length? It's static.
        # We can just pick the first uncovered word? Algorithm X says min degree.
        
        # Find min degree uncovered word
        min_deg = float('inf')
        target_w = -1
        temp_mask = uncovered_mask
        while temp_mask:
            bit = temp_mask & -temp_mask
            w_idx = bit.bit_length() - 1
            deg = len(coverers[w_idx])
            if deg < min_deg:
                min_deg = deg
                target_w = w_idx
            temp_mask ^= bit
            
        if target_w == -1:
            return False # Should not happen if uncovered_mask != 0
            
        # Try all codewords that cover target_w
        for c in coverers[target_w]:
            new_uncovered = uncovered_mask & ~ball_masks[c]
            current_code.append(c)
            if search(new_uncovered, depth + 1, current_code, max_depth):
                current_code.pop()
                return True
            current_code.pop()
            
        return False
        
    # Try to improve
    for target_size in range(best_size - 1, 185, -1): # Down to 186 (known lower bound is 156, best known 186)
        # Reset time? No, global time limit.
        # We just try to find ANY solution of size <= target_size.
        # Actually, the search above finds the *first* solution of size < best_size.
        # We should structure it to find the minimum.
        
        # Let's just run the search with max_depth = target_size
        # But the `search` function above returns True if it finds *any* solution of size < best_size.
        # It updates best_size.
        
        # To be safe, let's just run the search once with a high limit and let it prune.
        # But we want to guarantee we finish.
        
        pass # The logic above is a bit mixed. Let's refine.
        
    # Refined Search Strategy:
    # 1. Get greedy size G.
    # 2. Set best_size = G.
    # 3. Run DFS with pruning `depth >= best_size`.
    # 4. If a solution of size K < G is found, update best_size = K.
    # 5. Continue searching? The DFS might get stuck in a bad branch.
    # 6. Given time constraints, we might only find one improvement.
    
    # Let's implement the DFS properly.
    
    def dfs(uncovered_mask, depth, current_code):
        nonlocal best_size, best_code
        
        if time.time() - start_time > time_limit:
            return
            
        if uncovered_mask == 0:
            if depth < best_size:
                best_size = depth
                best_code = current_code[:]
            return
            
        if depth >= best_size:
            return
            
        # Select variable (uncovered word)
        min_deg = float('inf')
        target_w = -1
        temp_mask = uncovered_mask
        while temp_mask:
            bit = temp_mask & -temp_mask
            w_idx = bit.bit_length() - 1
            deg = len(coverers[w_idx])
            if deg < min_deg:
                min_deg = deg
                target_w = w_idx
            temp_mask ^= bit
            
        if target_w == -1:
            return
            
        # Heuristic: Order coverers by how many *other* uncovered words they cover?
        # Simple: Just iterate.
        for c in coverers[target_w]:
            new_uncovered = uncovered_mask & ~ball_masks[c]
            current_code.append(c)
            dfs(new_uncovered, depth + 1, current_code)
            current_code.pop()
            
            if time.time() - start_time > time_limit:
                return

    dfs(all_mask, 0, [])
    
    return [words[i] for i in best_code]
