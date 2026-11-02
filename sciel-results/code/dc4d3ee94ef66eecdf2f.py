import time
import random
import sys

def construct(q, n, radius):
    # Parameters for the specific task instance
    # The task specifies q=3, n=8, radius=1.
    # We assume the evaluator passes these, but we can hardcode optimizations for this specific case
    # while keeping the function signature general.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback to seed logic for other parameters to ensure validity
        # though the task is specific.
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = 3**n
    # Precompute balls
    # A word is a tuple of 8 ints.
    # We map word -> index (0 to 6560)
    # index = w0 + 3*w1 + 9*w2 + ...
    
    def word_to_idx(w):
        idx = 0
        pow3 = 1
        for i in range(n):
            idx += w[i] * pow3
            pow3 *= 3
        return idx

    def idx_to_word(idx):
        w = [0]*n
        for i in range(n):
            w[i] = idx % 3
            idx //= 3
        return w

    # Precompute balls for all indices
    # balls[i] is a list of indices
    balls = [None] * total_words
    for i in range(total_words):
        w = idx_to_word(i)
        ball = [i]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val != orig:
                    w[pos] = new_val
                    ball.append(word_to_idx(w))
            w[pos] = orig
        balls[i] = ball

    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_code = None
    best_size = total_words # Upper bound
    
    # Run multiple restarts
    while time.time() < deadline:
        # 1. Randomized Greedy Construction
        # We want to cover all words.
        # uncovered: list of indices not yet covered
        # We can maintain a count of how many times each word is covered.
        # But for greedy, we usually pick a word, then a cover.
        
        # Let's try a different greedy:
        # Start with empty set.
        # While there are uncovered words:
        #   Pick a random uncovered word u.
        #   Pick a random codeword c from balls[u] that is not yet in the set (or just any, we can dedupe later).
        #   Add c.
        
        # This is O(N * K) roughly.
        
        cover_count = [0] * total_words
        code_set = set()
        
        # To speed up "pick random uncovered", we can maintain a list of uncovered indices.
        # But removing from a list is O(N).
        # Given N=6561, O(N) is small.
        
        uncovered_list = list(range(total_words))
        random.shuffle(uncovered_list)
        
        while uncovered_list:
            if time.time() > deadline - 5.0:
                break # Stop construction if running out of time
            
            # Pick a random uncovered word
            # Since we are modifying the list, we pick from the end to avoid O(N) removal?
            # Or just pop a random index.
            idx_u = random.randint(0, len(uncovered_list) - 1)
            u = uncovered_list[idx_u]
            uncovered_list[idx_u] = uncovered_list[-1]
            uncovered_list.pop()
            
            # Pick a random codeword from balls[u]
            c = random.choice(balls[u])
            
            if c in code_set:
                # If we already have this codeword, it might cover u?
                # Wait, if u was uncovered, and c is in code_set, then u should have been covered.
                # This implies a bug in logic or u was already covered but still in list?
                # Actually, if c is in code_set, it was added earlier.
                # When it was added, it covered u. So u should have been removed from uncovered_list.
                # Unless we didn't remove it properly.
                # Let's ensure we remove u from uncovered_list when it becomes covered.
                # This approach is tricky to maintain strictly.
                
                # Alternative: Just add c. If it's a duplicate, it does nothing.
                # But we need to update cover_count and uncovered_list.
                pass
            
            # Add c
            if c not in code_set:
                code_set.add(c)
                for w in balls[c]:
                    if cover_count[w] == 0:
                        # w was uncovered, now covered
                        # We need to remove w from uncovered_list.
                        # This is O(N).
                        # To avoid O(N) search, we can mark it and clean up later?
                        # Or just use a set for uncovered?
                        pass
                    cover_count[w] += 1
        
        # This greedy is hard to implement efficiently with dynamic list removal in Python without overhead.
        # Let's use a simpler, robust heuristic:
        
        # **Strategy: Random Subset + Local Deletion**
        # 1. Generate a random set of codewords of size S (e.g., 600).
        # 2. Check if it covers. If not, add random codewords until it does.
        # 3. Perform local deletion.
        
        pass # Skipping the complex greedy above, using the simpler restart strategy below.

        # --- Restart Strategy ---
        
        # 1. Random Initialization
        # Start with a set of size approx 500-600.
        # We can just pick random codewords until covered.
        
        code_list = []
        code_set = set()
        cover_count = [0] * total_words
        uncovered_count = total_words
        
        # Add random codewords until covered
        # To avoid infinite loops, we limit iterations, but 6561 is small.
        # We can pick randomly.
        
        while uncovered_count > 0:
            if time.time() > deadline - 10.0:
                break
            
            c = random.randint(0, total_words - 1)
            if c in code_set:
                continue
                
            code_set.add(c)
            code_list.append(c)
            
            # Update coverage
            for w in balls[c]:
                if cover_count[w] == 0:
                    uncovered_count -= 1
                cover_count[w] += 1
        
        if uncovered_count > 0:
            continue # Failed to cover in time, restart or accept? Accept if we have a valid cover? No, must be valid.
            
        # 2. Local Deletion
        # We have a valid cover. Try to remove codewords.
        # We iterate through the list.
        # To make it faster, we can shuffle the list and try to remove.
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            # Shuffle to avoid bias
            random.shuffle(code_list)
            new_code_list = []
            for c in code_list:
                # Check if we can remove c
                removable = True
                for w in balls[c]:
                    if cover_count[w] == 1:
                        removable = False
                        break
                
                if removable:
                    # Remove c
                    for w in balls[c]:
                        cover_count[w] -= 1
                    code_set.remove(c)
                    changed = True
                else:
                    new_code_list.append(c)
            
            code_list = new_code_list
        
        # Update best
        if len(code_list) < best_size:
            best_size = len(code_list)
            best_code = code_list.copy()
            # print(f"New best: {best_size}") # Avoid print overhead

    # Convert indices back to words
    if best_code is None:
        # Fallback to seed
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [idx_to_word(c) for c in best_code]
