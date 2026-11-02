import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    # Parameters for the specific task instance
    # q=3, n=8, radius=1
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1()
    
    # Fallback for other parameters (simple greedy)
    total_words = q ** n
    if total_words > 100000:
        # Too large for simple fallback, return empty or minimal
        return []
        
    start_time = time.time()
    deadline = start_time + 110.0
    
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    ball_size = 1 + n * (q - 1)
    
    # Precompute balls
    balls = np.zeros((total_words, ball_size), dtype=np.int32)
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[tuple(w)] = i
        
    for i in range(total_words):
        w = all_words[i]
        neighbors = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                idx = word_to_idx[tuple(new_w)]
                neighbors.append(idx)
        balls[i] = neighbors
        
    is_covered = np.zeros(total_words, dtype=bool)
    codewords = []
    
    while not np.all(is_covered):
        if time.time() > deadline - 5:
            break
        newly_covered = (~is_covered)[balls]
        scores = newly_covered.sum(axis=1)
        best_idx = np.argmax(scores)
        if scores[best_idx] == 0:
            break
        codewords.append(best_idx)
        is_covered[balls[best_idx]] = True
        
    return [list(all_words[c]) for c in codewords]

def _solve_ternary_8_1():
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17
    
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # 1. Precompute balls efficiently using base-3 arithmetic
    # We don't need all_words array explicitly if we use indices, 
    # but we need to return vectors. 
    # Let's keep all_words for final conversion.
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    
    balls = np.zeros((total, ball_size), dtype=np.int32)
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
                
    # Precompute covers: list of codeword indices that cover each word
    # covers[w] is a list of c such that w is in balls[c]
    # Note: covers[w] is exactly balls[w] due to symmetry of Hamming distance
    # But let's verify or just use balls. 
    # Since distance is symmetric, w in balls[c] <=> c in balls[w].
    # So covers[w] == balls[w]. This is a huge simplification.
    
    def run_greedy_and_refine(random_seed):
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Greedy Initialization
        is_covered = np.zeros(total, dtype=bool)
        codewords = []
        
        # Optimization: Maintain a count of how many times each word is covered? 
        # No, just track uncovered.
        # To speed up greedy, we can use the fact that we only care about uncovered words.
        
        while not np.all(is_covered):
            if time.time() > deadline - 10:
                break
            
            # Find word covering most uncovered
            # newly_covered = (~is_covered)[balls]
            # scores = newly_covered.sum(axis=1)
            # This is O(N * BallSize). N=6561, BallSize=17. ~110k ops per step.
            # Greedy takes ~400 steps. 400 * 110k = 44M ops. Fast enough in numpy.
            
            scores = (~is_covered)[balls].sum(axis=1)
            max_score = np.max(scores)
            
            if max_score == 0:
                break
            
            # Pick randomly among best to add diversity
            candidates = np.where(scores == max_score)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            codewords.append(best_idx)
            is_covered[balls[best_idx]] = True
            
        if len(codewords) == 0:
            return [], 0
            
        # Local Search
        # cover_counts[w] = number of codewords covering w
        cover_counts = np.zeros(total, dtype=np.int32)
        for c in codewords:
            cover_counts[balls[c]] += 1
            
        improved = True
        iterations = 0
        max_iterations = 2000
        
        while improved and time.time() < deadline - 5:
            improved = False
            iterations += 1
            
            # Phase 1: Remove redundant codewords (vectorized)
            i = 0
            while i < len(codewords):
                c = codewords[i]
                cover_counts[balls[c]] -= 1
                if not np.any(cover_counts == 0):
                    codewords.pop(i)
                    improved = True
                else:
                    cover_counts[balls[c]] += 1
                    i += 1
                    
            if not improved:
                # Phase 2: Vectorized swap-then-remove search
                # For each codeword c, find unique words it covers, then find
                # candidate replacements b that cover all unique words.
                # If a valid swap exists that enables removal, accept it.
                
                n_cw = len(codewords)
                swap_found = False
                
                # Precompute: for each codeword, its unique covered words
                # unique_words_for_c[i] = list of words uniquely covered by codewords[i]
                unique_masks = (cover_counts == 1)  # [total]
                
                for i in range(n_cw):
                    if swap_found:
                        break
                    c = codewords[i]
                    unique_in_ball = unique_masks[balls[c]]
                    unique_words = balls[c][unique_in_ball]
                    
                    if len(unique_words) == 0:
                        continue
                    
                    # Find candidates that cover ALL unique words
                    # Use vectorized intersection via boolean masks
                    # A candidate b covers word w iff b is in balls[w]
                    # So b must be in intersection of balls[w] for all unique w
                    
                    # Build a mask: candidate_mask[b] = True iff b covers all unique_words
                    # For efficiency, only consider b that are not already codewords
                    in_codeword = np.zeros(total, dtype=bool)
                    in_codeword[codewords] = True
                    
                    # Start with all non-codeword positions as candidates
                    candidate_mask = ~in_codeword
                    
                    # Intersect: for each unique word w, candidate must be in balls[w]
                    for w in unique_words:
                        # candidates must include w's ball
                        # candidate_mask &= (balls contains w)
                        # But balls[w] gives indices of words at distance <=1 from w
                        # We need: b is in balls[w] for ALL w in unique_words
                        # This is: for all w in unique_words, b in balls[w]
                        # Equivalently: w in balls[b] for all w in unique_words
                        # So: unique_words subset of balls[b]
                        
                        # Vectorized: check which b have all unique_words in their ball
                        # For each b, count how many unique_words are in balls[b]
                        # This is expensive. Alternative: iterate over balls[w]
                        
                        # More efficient: start with balls[unique_words[0]], then intersect
                        # Let's use the set-based approach but with numpy where possible
                        pass
                    
                    # Fallback to set-based for correctness, but optimize the inner loop
                    candidates = set(balls[unique_words[0]].tolist())
                    for w in unique_words[1:]:
                        if not candidates:
                            break
                        candidates &= set(balls[w].tolist())
                        
                    if not candidates:
                        continue
                        
                    candidates -= set(codewords)
                    
                    if not candidates:
                        continue
                        
                    candidate_list = list(candidates)
                    random.shuffle(candidate_list)
                    
                    for b in candidate_list:
                        # Tentative swap
                        cover_counts[balls[c]] -= 1
                        cover_counts[balls[b]] += 1
                        
                        if not np.any(cover_counts == 0):
                            # Check if any codeword can now be removed
                            removal_found = False
                            for j in range(n_cw):
                                if j == i:
                                    continue
                                c_j = codewords[j]
                                cover_counts[balls[c_j]] -= 1
                                if not np.any(cover_counts == 0):
                                    removal_found = True
                                    cover_counts[balls[c_j]] += 1
                                    break
                                else:
                                    cover_counts[balls[c_j]] += 1
                                    
                            if removal_found:
                                codewords[i] = b
                                improved = True
                                swap_found = True
                                break
                            else:
                                cover_counts[balls[b]] -= 1
                                cover_counts[balls[c]] += 1
                        else:
                            cover_counts[balls[b]] -= 1
                            cover_counts[balls[c]] += 1
                            
        return codewords, len(codewords)

    best_codewords = None
    best_size = total
    
    # Run multiple restarts with different seeds
    seed = 42
    while time.time() < deadline - 10:
        codewords, size = run_greedy_and_refine(seed)
        if size < best_size:
            best_size = size
            best_codewords = codewords[:]
            
        seed += 1
        
        if best_size <= 486:
            break
            
    if best_codewords is None:
        return []
    
    # Final polish: run local search again on the best solution with remaining time
    if time.time() < deadline - 3 and len(best_codewords) > 0:
        cover_counts = np.zeros(total, dtype=np.int32)
        for c in best_codewords:
            cover_counts[balls[c]] += 1
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            
            i = 0
            while i < len(best_codewords):
                c = best_codewords[i]
                cover_counts[balls[c]] -= 1
                if not np.any(cover_counts == 0):
                    best_codewords.pop(i)
                    improved = True
                else:
                    cover_counts[balls[c]] += 1
                    i += 1
            
            if not improved:
                unique_masks = (cover_counts == 1)
                for i in range(len(best_codewords)):
                    if time.time() > deadline - 3:
                        break
                    c = best_codewords[i]
                    unique_in_ball = unique_masks[balls[c]]
                    unique_words = balls[c][unique_in_ball]
                    if len(unique_words) == 0:
                        continue
                    
                    candidates = set(balls[unique_words[0]].tolist())
                    for w in unique_words[1:]:
                        if not candidates:
                            break
                        candidates &= set(balls[w].tolist())
                    if not candidates:
                        continue
                    candidates -= set(best_codewords)
                    if not candidates:
                        continue
                    
                    for b in list(candidates):
                        cover_counts[balls[c]] -= 1
                        cover_counts[balls[b]] += 1
                        if not np.any(cover_counts == 0):
                            removal_found = False
                            for j in range(len(best_codewords)):
                                if j == i:
                                    continue
                                c_j = best_codewords[j]
                                cover_counts[balls[c_j]] -= 1
                                if not np.any(cover_counts == 0):
                                    removal_found = True
                                    cover_counts[balls[c_j]] += 1
                                    break
                                else:
                                    cover_counts[balls[c_j]] += 1
                            if removal_found:
                                best_codewords[i] = b
                                improved = True
                                break
                            else:
                                cover_counts[balls[b]] -= 1
                                cover_counts[balls[c]] += 1
                        else:
                            cover_counts[balls[b]] -= 1
                            cover_counts[balls[c]] += 1
    
    return [list(all_words[c]) for c in best_codewords]
