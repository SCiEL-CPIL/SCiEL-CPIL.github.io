import time
import itertools
import math

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other cases, though task is specific
        # Simple greedy for general case
        words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(words)}
        
        def dist(w1, w2):
            return sum(1 for a, b in zip(w1, w2) if a != b)
        
        balls = []
        for w in words:
            ball = [w]
            for pos in range(n):
                for val in range(q):
                    if val != w[pos]:
                        nw = list(w)
                        nw[pos] = val
                        ball.append(tuple(nw))
            balls.append(set(ball))
            
        covered = [False] * len(words)
        selected = []
        
        while any(not c for c in covered):
            best_c = -1
            max_gain = 0
            for i, ball in enumerate(balls):
                if i in selected: continue
                gain = sum(1 for w in ball if not covered[word_to_idx[w]])
                if gain > max_gain:
                    max_gain = gain
                    best_c = i
            if best_c == -1: break
            selected.append(best_c)
            for w in balls[best_c]:
                covered[word_to_idx[w]] = True
                
        return [list(w) for w in (words[i] for i in selected)]

    start_time = time.time()
    deadline = 145.0 # Safety margin

    # Precompute balls as bitmasks (Python int)
    # 729 words, index 0..728
    # Ball of w includes w and all words at dist 1
    all_words_mask = (1 << 729) - 1
    
    def num_to_vec(num, length=6):
        vec = [0] * length
        for i in range(length):
            vec[i] = num % 3
            num //= 3
        return vec

    def vec_to_num(vec):
        num = 0
        for i in range(len(vec)):
            num += vec[i] * (3 ** i)
        return num

    # Precompute ball masks
    ball_masks = [0] * 729
    for i in range(729):
        v = num_to_vec(i)
        mask = 1 << i
        for pos in range(6):
            orig = v[pos]
            for new_val in range(3):
                if new_val != orig:
                    v[pos] = new_val
                    j = vec_to_num(v)
                    mask |= (1 << j)
            v[pos] = orig
        ball_masks[i] = mask

    # Precompute for each word w, the list of codewords that cover it
    # coverers[w] = list of c such that w in ball(c)
    coverers = [[] for _ in range(729)]
    for c in range(729):
        m = ball_masks[c]
        # Iterate set bits
        mm = m
        while mm:
            lsb = mm & -mm
            w = lsb.bit_length() - 1
            coverers[w].append(c)
            mm ^= lsb

    # Greedy Initial Solution
    def greedy_cover():
        covered_mask = 0
        selected = []
        while covered_mask != all_words_mask:
            best_c = -1
            max_gain = 0
            # Optimization: only check candidates not yet selected? 
            # Actually, we can just check all, but skip if already in selected
            # For speed, we can maintain a list of remaining candidates
            for c in range(729):
                if c in selected:
                    continue
                gain = (ball_masks[c] & ~covered_mask).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_c = c
            if best_c == -1:
                break
            selected.append(best_c)
            covered_mask |= ball_masks[best_c]
        return selected, covered_mask

    initial_selected, initial_covered = greedy_cover()
    best_size = len(initial_selected)
    best_solution = initial_selected[:]

    # Backtracking Search
    # State: covered_mask, current_selected (list), depth
    # We want to find a solution with size < best_size
    
    # To speed up backtracking, we can use a "forced" choice strategy.
    # If a word w is uncovered, we MUST pick a codeword from coverers[w].
    
    # Optimization: Precompute for each word w, the candidates sorted by "usefulness" (e.g., size of ball, or number of other uncovered words they cover)
    # But "usefulness" changes as we cover words.
    
    # Let's use a recursive search that picks the uncovered word with the fewest options (MRV).
    # To find MRV quickly, we need to count options for each uncovered word.
    # This is expensive. 
    # Alternative: Just pick the first uncovered word? No, MRV is crucial.
    
    # Given 729 words, iterating through all uncovered words to find MRV is O(N^2) per node.
    # With bitsets, finding the first uncovered word is fast.
    # Let's try a simpler heuristic: Pick the uncovered word w that is covered by the *fewest* available codewords.
    
    # To make this fast, we can approximate.
    # Or, we can just use the first uncovered word, but that might be weak.
    
    # Let's try a hybrid: 
    # 1. Greedy gives a good start (size ~80-90).
    # 2. Local Search (1-move, 2-move) on the bitset representation.
    #    1-move: Remove c if (covered_mask & ~ball_masks[c]) | (union of others) == covered_mask.
    #    Actually, simpler: c is removable if for every w in ball(c), there exists c' != c in selected such that w in ball(c').
    #    This is equivalent to: ball(c) is a subset of the union of balls of (selected \ {c}).
    
    def local_search(selected):
        selected_set = set(selected)
        covered_mask = 0
        for c in selected_set:
            covered_mask |= ball_masks[c]
        
        improved = True
        while improved:
            improved = False
            # 1-move: try to remove
            for c in list(selected_set):
                # Check if c is essential
                # Union of all others
                others_union = 0
                for c2 in selected_set:
                    if c2 != c:
                        others_union |= ball_masks[c2]
                
                if (others_union & all_words_mask) == all_words_mask:
                    # c is redundant
                    selected_set.remove(c)
                    covered_mask = others_union
                    improved = True
                    break # Restart loop to check for new redundancies
            
            # 2-move: try to swap
            # For each pair (c1, c2), check if we can remove them and add one c3
            # such that c3 covers the words that were ONLY covered by c1 or c2.
            # unique1 = ball(c1) & ~Union(others except c1, c2)
            # unique2 = ball(c2) & ~Union(others except c1, c2)
            # required = unique1 | unique2
            # Find c3 such that required is subset of ball(c3).
            
            # This is O(N^3) if done naively.
            # We can limit this.
            
            if time.time() > deadline - 10:
                break
            
            # Efficient 2-move check
            # We only need to check pairs if we are stuck.
            # Let's just do 1-move for now to ensure correctness and speed.
            # If 1-move is exhausted, we are in a local minimum.
            
        return list(selected_set)

    # Run local search on initial
    best_solution = local_search(initial_selected)
    best_size = len(best_solution)

    # If we have time, try more greedy restarts with different randomizations?
    # Or use the backtracking to prove optimality or find better?
    
    # Given the time limit, a robust local search is safer than a potentially exploding backtracker.
    # Let's enhance the local search with 2-moves but optimized.
    
    def enhanced_local_search(selected):
        selected_set = set(selected)
        covered_mask = 0
        for c in selected_set:
            covered_mask |= ball_masks[c]
        
        # Precompute union of all balls for quick removal check?
        # No, we need union of *others*.
        
        improved = True
        while improved:
            improved = False
            
            # 1-move
            for c in list(selected_set):
                others_union = 0
                for c2 in selected_set:
                    if c2 != c:
                        others_union |= ball_masks[c2]
                if (others_union & all_words_mask) == all_words_mask:
                    selected_set.remove(c)
                    covered_mask = others_union
                    improved = True
                    break
            
            if time.time() > deadline - 5:
                break
                
            # 2-move (Swap)
            # Only try if size is still large
            if len(selected_set) > 73:
                # Pick a random pair to swap? Or systematic?
                # Systematic is better for finding the minimum.
                # But O(N^3) is too slow for N=80. 80^3 = 512,000. 
                # Inside, we need to find c3. 
                # We can precompute for each word w, the candidates.
                
                # Let's try a limited 2-move:
                # Pick c1, c2.
                # Calculate required mask.
                # Iterate through all 729 codewords to see if any covers required mask.
                # This is 729 * (N^2) operations. 729 * 6400 ~ 4M. Might be okay in Python if bit operations are fast?
                # 4M iterations of Python loops is ~0.1-0.2s.
                
                if time.time() > deadline - 20:
                    break

                c_list = list(selected_set)
                for i in range(len(c_list)):
                    if time.time() > deadline - 10:
                        break
                    c1 = c_list[i]
                    for j in range(i + 1, len(c_list)):
                        c2 = c_list[j]
                        
                        # Union of others excluding c1, c2
                        others_union = 0
                        for c3 in selected_set:
                            if c3 != c1 and c3 != c2:
                                others_union |= ball_masks[c3]
                        
                        # Words that MUST be covered by the new word
                        # They are in ball(c1) or ball(c2) but NOT in others_union
                        required = (ball_masks[c1] | ball_masks[c2]) & ~others_union
                        
                        if required == 0:
                            # c1 and c2 are redundant together?
                            # This implies 1-move would have caught it if they were individually redundant?
                            # No, if they cover each other's unique parts, they might both be essential individually but redundant as a pair?
                            # Actually if required is 0, it means others_union covers everything c1 and c2 cover.
                            # So we can remove BOTH.
                            selected_set.remove(c1)
                            selected_set.remove(c2)
                            covered_mask = others_union
                            improved = True
                            break
                        
                        # Find c_new that covers required
                        # c_new must be in selected_set? No, can be any codeword.
                        # But we want to keep size same or reduce.
                        # If we remove 2 and add 1, size reduces by 1.
                        
                        found_new = False
                        # Check all 729 codewords
                        # Optimization: Only check if required is not too large?
                        # Ball size is 19. Required is subset of 38 words.
                        
                        for c_new in range(729):
                            if c_new in selected_set:
                                continue # We are removing c1, c2, so c_new shouldn't be in the remaining set? 
                                        # Actually, if c_new is in selected_set, we are just swapping.
                                        # If c_new is NOT in selected_set, we are adding a new one.
                                        # Both are valid.
                            
                            # Check if ball(c_new) covers required
                            if (ball_masks[c_new] & required) == required:
                                # Perform swap
                                selected_set.remove(c1)
                                selected_set.remove(c2)
                                selected_set.add(c_new)
                                # Recalculate covered mask
                                covered_mask = 0
                                for c in selected_set:
                                    covered_mask |= ball_masks[c]
                                improved = True
                                found_new = True
                                break
                        
                        if found_new:
                            break
                    if improved:
                        break
        return list(selected_set)

    # Run enhanced local search
    best_solution = enhanced_local_search(initial_selected)
    best_size = len(best_solution)

    # If we still have time, try more initializations
    if time.time() < deadline - 30:
        # Try random greedy? 
        # Or just return best found
        pass

    # Verify and return
    # Check if best_solution covers all
    final_covered = 0
    for c in best_solution:
        final_covered |= ball_masks[c]
        
    if (final_covered & all_words_mask) == all_words_mask:
        return [num_to_vec(c) for c in best_solution]
    else:
        # Fallback
        return [num_to_vec(c) for c in initial_selected]

# EVOLVE-BLOCK-END
