import time

def construct(q, n, radius):
    start_time = time.time()
    # Use 110s deadline to be safe against 120s limit
    deadline = start_time + 110.0
    
    total_words = q ** n
    
    # Precompute index <-> word mappings
    # This is faster than doing it on the fly in the inner loop
    # But we can also just use integers as words if we are careful.
    # Let's stick to indices for the sets, and convert at the end.
    
    # Generate all words as tuples of integers
    all_words = []
    for i in range(total_words):
        w = [0] * n
        temp = i
        for j in range(n - 1, -1, -1):
            w[j] = temp % q
            temp //= q
        all_words.append(tuple(w))
        
    # Precompute balls
    # balls[i] is a set of indices covered by codeword i
    balls = [set() for _ in range(total_words)]
    
    # Helper to calculate neighbors for a given word index
    # We can reuse the word tuple from all_words
    
    for c_idx in range(total_words):
        c_word = all_words[c_idx]
        balls[c_idx].add(c_idx)
        
        if radius >= 1:
            for pos in range(n):
                orig_val = c_word[pos]
                for new_val in range(q):
                    if new_val != orig_val:
                        # Calculate index of neighbor
                        # Since base is q, changing digit at pos changes index by (new_val - orig_val) * q^(n-1-pos)
                        # But we don't have powers precomputed.
                        # Let's just build the tuple and convert, or precompute powers.
                        pass # We'll do a slightly more efficient index calc below if needed, 
                             # but for n=7, building tuple and converting is fast enough in precomp.
            
            # More efficient index calculation for neighbors
            # Precompute powers of q
            powers = [1] * n
            for i in range(1, n):
                powers[i] = powers[i-1] * q
            
            for pos in range(n):
                orig_val = c_word[pos]
                for new_val in range(q):
                    if new_val != orig_val:
                        delta = (new_val - orig_val) * powers[n - 1 - pos]
                        n_idx = c_idx + delta
                        balls[c_idx].add(n_idx)
                        
                        # Note: This only works for radius 1.
                        # If radius > 1, we need to generate all combinations.
                        # Given the task is specifically K_3(7,1), R=1 is the main case.
                        # If R>1, the above logic is insufficient.
                        # However, the prompt says "The evaluator passes this task's alphabet size q, length n and covering radius."
                        # And the task is "Football pool problem for 7 matches: ternary covering code K_3(7,1)".
                        # It is highly likely radius=1.
                        # To be safe, I will implement a general ball generator if radius > 1,
                        # but optimize for radius 1.

    # Re-implement ball generation to be general but fast for R=1
    balls = [set() for _ in range(total_words)]
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # For R=1, we can just use the delta method.
    # For R>1, we need itertools.product of positions to change.
    if radius == 1:
        for c_idx in range(total_words):
            balls[c_idx].add(c_idx)
            for pos in range(n):
                orig_val = all_words[c_idx][pos]
                for new_val in range(q):
                    if new_val != orig_val:
                        delta = (new_val - orig_val) * powers[n - 1 - pos]
                        balls[c_idx].add(c_idx + delta)
    else:
        # General case: generate all words at distance <= radius
        # This is slower but handles any radius
        # For small n, q, R, it's fine.
        # We can use a BFS or direct generation.
        # Direct generation:
        # Choose k positions to change, where 1 <= k <= radius
        # For each combination of positions, choose new values.
        import itertools
        
        # Precompute combinations of positions
        pos_combos = []
        for k in range(1, radius + 1):
            for combo in itertools.combinations(range(n), k):
                pos_combos.append(combo)
                
        for c_idx in range(total_words):
            balls[c_idx].add(c_idx)
            c_word = all_words[c_idx]
            for combo in pos_combos:
                # Generate all variations for these positions
                # This is q^k possibilities minus 1 (the original)
                # But we need to ensure we don't generate duplicates if k is large?
                # No, combinations are unique sets of positions.
                # For each position in combo, we pick a new value.
                # We need to iterate over all q^k assignments.
                # We can use product.
                vals = list(range(q))
                for new_vals in itertools.product(vals, repeat=len(combo)):
                    # Check if any val is different
                    # Actually, we just construct the index.
                    n_idx = c_idx
                    for i, pos in enumerate(combo):
                        orig = c_word[pos]
                        new = new_vals[i]
                        if new != orig:
                            n_idx += (new - orig) * powers[n - 1 - pos]
                    balls[c_idx].add(n_idx)

    # Greedy Set Cover
    uncovered = set(range(total_words))
    selected = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_c = -1
        max_covered = 0
        
        for c_idx in range(total_words):
            # Optimization: skip if c_idx is already selected? 
            # No, we can re-select? No, set cover usually doesn't need to re-select.
            # But greedy picks the best *new* coverage.
            # A selected codeword covers 0 new words (since its ball is removed from uncovered).
            # So we can just check all.
            
            covered_new = len(balls[c_idx] & uncovered)
            if covered_new > max_covered:
                max_covered = covered_new
                best_c = c_idx
                
        if best_c == -1 or max_covered == 0:
            break
            
        selected.append(best_c)
        uncovered.difference_update(balls[best_c])
        
    return [list(all_words[idx]) for idx in selected]
