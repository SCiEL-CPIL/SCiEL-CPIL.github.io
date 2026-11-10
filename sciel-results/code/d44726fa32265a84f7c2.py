import time
import numpy as np
from scipy.optimize import linprog

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    total_words = q ** n
    
    def int_to_tuple(x, n, q):
        t = [0] * n
        for i in range(n):
            t[i] = x % q
            x //= q
        return t
    
    # Precompute balls using vectorized operations
    # For each word, compute all words at distance <= radius
    # Use numpy for efficiency
    all_words = np.array([list(int_to_tuple(i, n, q)) for i in range(total_words)])
    
    # Precompute balls as boolean masks for each word
    # ball_mask[i] is a boolean array of length total_words
    # True if word j is within radius of word i
    ball_masks = np.zeros((total_words, total_words), dtype=np.uint8)
    
    # Compute distances in chunks to save memory
    # Actually, let's compute balls more efficiently
    # For radius=1, ball of word i = {i} ∪ {words at distance 1}
    
    # Let's use a different approach: precompute which positions differ
    # For each pair, we can check if they're within distance 1
    # But that's O(n^2) which is too slow
    
    # Better: for each word, generate all words at distance <= radius
    balls = []
    for i in range(total_words):
        t_i = all_words[i]
        ball_set = set()
        ball_set.add(i)
        
        if radius >= 1:
            for pos in range(n):
                orig_val = t_i[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    # Create new word
                    t_new = t_i.copy()
                    t_new[pos] = new_val
                    # Convert back to index
                    x_new = 0
                    for k in range(n - 1, -1, -1):
                        x_new = x_new * q + t_new[k]
                    ball_set.add(x_new)
        
        balls.append(sorted(ball_set))
    
    # Build ball_masks for fast coverage counting
    # Use uint8 to save memory
    for i in range(total_words):
        for j in balls[i]:
            ball_masks[i, j] = 1
    
    # Phase 1: LP Relaxation to get initial weights
    # Minimize sum(x_i) subject to A @ x >= 1, 0 <= x_i <= 1
    # where A[i,j] = 1 if word j covers word i
    
    # This is a huge LP (2187 variables, 2187 constraints)
    # scipy linprog might be slow, but let's try with time guard
    
    lp_done = False
    lp_weights = None
    
    if time.time() < deadline - 30:
        try:
            # A matrix: A[i,j] = 1 if word j covers word i
            # A is the transpose of ball_masks
            A_mat = ball_masks.T.astype(float)  # shape (total_words, total_words)
            b_vec = np.ones(total_words)
            
            # Objective: minimize sum(x)
            c_vec = np.ones(total_words)
            
            bounds = [(0, 1)] * total_words
            
            # Use method='highs' which is faster
            result = linprog(c_vec, A_ub=-A_mat, b_ub=-b_vec, bounds=bounds, method='highs')
            
            if result.success:
                lp_weights = result.x
                lp_done = True
        except Exception as e:
            pass
    
    # Phase 2: Weighted Greedy Initialization
    covered = np.zeros(total_words, dtype=np.uint8)
    initial_codewords = []
    
    # Use LP weights to guide greedy selection
    # Words with higher LP values are "more important" to cover
    if lp_done and lp_weights is not None:
        # Sort words by LP weight (descending)
        word_order = np.argsort(-lp_weights)
    else:
        word_order = np.arange(total_words)
    
    # Standard greedy: pick word that covers most uncovered
    # But we can bias by LP weights
    remaining_time = deadline - time.time()
    if remaining_time < 5:
        # Emergency: just use standard greedy
        for i in range(total_words):
            if not np.all(covered > 0):
                break
            # Count uncovered in ball[i]
            cnt = np.sum(covered[balls[i]] == 0)
            if cnt == 0:
                continue
            # For now, just do standard greedy
            pass
    
    # Do standard greedy (fast)
    uncovered_count = total_words
    while uncovered_count > 0:
        if time.time() > deadline - 60:
            break
        
        best_idx = -1
        best_cnt = -1
        
        # Sample a subset if too slow
        if total_words > 500:
            sample_size = min(total_words, 200)
            candidates = np.random.choice(total_words, size=sample_size, replace=False)
        else:
            candidates = np.arange(total_words)
        
        for i in candidates:
            cnt = np.sum(covered[balls[i]] == 0)
            # Add small bias from LP weights if available
            if lp_done and lp_weights is not None:
                cnt += 0.01 * lp_weights[i]
            if cnt > best_cnt:
                best_cnt = cnt
                best_idx = i
        
        if best_cnt <= 0:
            break
        
        initial_codewords.append(best_idx)
        covered[balls[best_idx]] += 1
        uncovered_count = np.sum(covered == 0)
    
    # Ensure validity
    final_covered = np.zeros(total_words, dtype=np.uint8)
    for c in initial_codewords:
        final_covered[balls[c]] += 1
    uncovered_indices = np.where(final_covered == 0)[0]
    for u in uncovered_indices:
        if u not in initial_codewords:
            initial_codewords.append(u)
            final_covered[balls[u]] += 1
    
    current_codewords = list(initial_codewords)
    best_codewords = list(initial_codewords)
    best_size = len(best_codewords)
    
    # Phase 3: Local Search with Smart Perturbations
    rng = np.random.default_rng(42)
    
    def get_coverage(codewords):
        cover = np.zeros(total_words, dtype=np.uint8)
        for c in codewords:
            cover[balls[c]] += 1
        return cover
    
    while time.time() < deadline - 10:
        cover = get_coverage(current_codewords)
        
        # Local Search: Greedy Remove
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            # Try to remove codewords in random order
            indices = np.arange(len(current_codewords))
            np.random.shuffle(indices)
            
            for i in indices:
                if i >= len(current_codewords):
                    continue
                c = current_codewords[i]
                # Check if redundant
                redundant = True
                for j in balls[c]:
                    if cover[j] == 1:
                        redundant = False
                        break
                if redundant:
                    current_codewords.pop(i)
                    cover[balls[c]] -= 1
                    changed = True
                    if len(current_codewords) < best_size:
                        best_size = len(current_codewords)
                        best_codewords = list(current_codewords)
        
        if time.time() > deadline - 15:
            break
        
        # Perturbation: Targeted swap
        # Find a codeword that is "weakly supported" (covers few unique words)
        if len(current_codewords) > 0:
            # Compute for each codeword how many words it uniquely covers
            unique_cover_count = np.zeros(len(current_codewords))
            for idx, c in enumerate(current_codewords):
                unique_cover_count[idx] = np.sum(cover[balls[c]] == 1)
            
            # Pick a codeword with low unique coverage
            if np.any(unique_cover_count == 0):
                weak_indices = np.where(unique_cover_count == 0)[0]
                idx_remove = weak_indices[np.random.randint(len(weak_indices))]
            else:
                idx_remove = np.argmin(unique_cover_count)
            
            c_remove = current_codewords[idx_remove]
            
            # Find a good replacement: one that covers many of the uniquely covered words
            # Words that would become uncovered if we remove c_remove
            would_become_uncovered = [j for j in balls[c_remove] if cover[j] == 1]
            
            if would_become_uncovered:
                # Score potential replacements
                best_add = -1
                best_score = -1
                
                # Sample candidates
                sample_size = min(total_words, 100)
                candidates = np.random.choice(total_words, size=sample_size, replace=False)
                
                for c_add in candidates:
                    if c_add in current_codewords:
                        continue
                    # Score: how many of would_become_uncovered does c_add cover?
                    score = sum(1 for j in would_become_uncovered if ball_masks[c_add, j] == 1)
                    # Bonus for covering other uncovered words
                    score += 0.1 * np.sum(cover[balls[c_add]] == 0)
                    if score > best_score:
                        best_score = score
                        best_add = c_add
                
                if best_add >= 0:
                    # Perform swap
                    current_codewords[idx_remove] = best_add
                    cover[balls[c_remove]] -= 1
                    cover[balls[best_add]] += 1
            else:
                # No unique words, just try to add a word that covers many uncovered
                # But there are no uncovered, so just do a random swap
                c_add = np.random.randint(total_words)
                if c_add not in current_codewords:
                    current_codewords[idx_remove] = c_add
                    cover[balls[c_remove]] -= 1
                    cover[balls[c_add]] += 1
    
    # Final verification
    final_cover = np.zeros(total_words, dtype=np.uint8)
    for c in best_codewords:
        final_cover[balls[c]] += 1
    uncovered = np.where(final_cover == 0)[0]
    for u in uncovered:
        if u not in best_codewords:
            best_codewords.append(u)
    
    # Final prune
    changed = True
    while changed and time.time() < deadline - 2:
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
    
    # Convert to result
    result = []
    for c in best_codewords:
        result.append(int_to_tuple(c, n, q))
    return result
