"""
Optimized covering code search for K_3(6,1).
Uses Tetracode initialization, fast local search with coverage counts, 
and iterated local search with 1-1 swaps.
"""
import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    total_words = q ** n
    
    # Precompute index conversions
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute balls
    # balls[i] is a list of indices of words covered by codeword i
    balls = [None] * total_words
    for c in range(total_words):
        s = [c]
        w = idx_to_word(c)
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for nv in range(q):
                    if nv != orig:
                        w[pos] = nv
                        s.append(word_to_idx(w))
                        w[pos] = orig
        balls[c] = s

    # Helper to compute coverage from a boolean array x
    def compute_coverage(x):
        cov = np.zeros(total_words, dtype=np.int32)
        # Vectorized addition is hard with lists of variable length, 
        # but for n=6, q=3, balls are small (13 elements).
        # We can use a loop which is fast enough in Python if optimized.
        for i in range(total_words):
            if x[i]:
                cov[balls[i]] += 1
        return cov

    def is_valid(x):
        # Check if all words are covered
        # Instead of full coverage, we can just check min >= 1
        # But we need coverage for local search.
        pass

    def local_search(x, max_time=None):
        """
        Perform local search to minimize the number of codewords.
        1. Try to remove any redundant codeword.
        2. Try 1-1 swaps: remove c1, add c2 if c2 covers the deficit.
        """
        if max_time is None:
            max_time = deadline
        
        x = x.copy()
        cov = compute_coverage(x)
        selected = set(np.nonzero(x)[0].tolist())
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            
            # Phase 1: Remove redundant codewords
            # A codeword c is redundant if for all w in balls[c], cov[w] > 1
            to_remove = []
            for c in list(selected):
                # Check if any word in c's ball has count 1
                # If cov[w] == 1 for any w in balls[c], c is NOT redundant
                # We can break early if we find such a w
                redundant = True
                for w in balls[c]:
                    if cov[w] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(c)
            
            if to_remove:
                improved = True
                for c in to_remove:
                    x[c] = 0
                    selected.remove(c)
                    for w in balls[c]:
                        cov[w] -= 1
                
            # Phase 2: 1-1 Swaps
            # Try to find a pair (c_out, c_in) such that removing c_out 
            # and adding c_in results in a valid cover and same size (or we just 
            # try to swap to enable further removals? No, size is same).
            # Actually, a swap keeps size constant. The goal is to reach a state 
            # where Phase 1 can remove more.
            # So we try swaps that might "loosen" the constraints.
            
            # To keep it simple and fast, we only do swaps if we are stuck in Phase 1.
            if not to_remove and time.time() < max_time:
                # Try a few random swaps
                # Pick a random c_out
                # Find words that would become uncovered (cov[w]==1)
                # Find c_in that covers all of them and is not selected
                # This is expensive. 
                # Alternative: Just do a random perturbation if stuck for a while?
                # For now, let's stick to removals which is the main driver.
                # If we want swaps, we can do:
                # for c_out in selected (sampled):
                #    deficit = [w for w in balls[c_out] if cov[w]==1]
                #    if deficit:
                #       for c_in in unselected (sampled):
                #           if all(w in balls[c_in] for w in deficit):
                #               swap!
                
                # Let's implement a limited swap search
                unselected = [i for i in range(total_words) if not x[i]]
                if not unselected: continue
                
                # Sample a few c_out
                for _ in range(10):
                    c_out = random.choice(list(selected))
                    deficit = []
                    for w in balls[c_out]:
                        if cov[w] == 1:
                            deficit.append(w)
                    
                    if not deficit:
                        # This should have been caught in Phase 1
                        continue
                    
                    # Try to find c_in that covers all deficit
                    # Sample a few c_in
                    for c_in in random.sample(unselected, min(50, len(unselected))):
                        # Check if c_in covers all deficit
                        # balls[c_in] is a list. Converting to set is expensive.
                        # We can just check membership.
                        covers_all = True
                        # Optimization: Check the most restrictive deficit word first?
                        # Or just check all.
                        for w in deficit:
                            if w not in balls[c_in]:
                                covers_all = False
                                break
                        if covers_all:
                            # Perform swap
                            x[c_out] = 0
                            x[c_in] = 1
                            selected.remove(c_out)
                            selected.add(c_in)
                            
                            # Update coverage
                            for w in balls[c_out]:
                                cov[w] -= 1
                            for w in balls[c_in]:
                                cov[w] += 1
                            
                            improved = True
                            break
                    if improved:
                        break

        return x

    def greedy_cover(x, seed=0):
        """Complete the cover using greedy heuristic."""
        rng = random.Random(seed)
        cov = compute_coverage(x)
        uncovered = set(np.nonzero(cov == 0)[0].tolist())
        
        while len(uncovered) > 0 and time.time() < deadline:
            best_cand = -1
            best_cnt = 0
            # To speed up, we can limit candidates or use randomization
            # But for correctness, we need to find the best.
            # With 729 words, checking all is fast.
            for cand in range(total_words):
                if x[cand]:
                    continue
                cnt = 0
                for w in balls[cand]:
                    if w in uncovered:
                        cnt += 1
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_cand = cand
            
            if best_cand == -1 or best_cnt == 0:
                break # No improvement possible
            
            x[best_cand] = 1
            for w in balls[best_cand]:
                cov[w] += 1
                if w in uncovered:
                    uncovered.discard(w)
        
        return x

    # --- Initialization ---
    # Tetracode based initialization
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    def get_tetracode_init():
        x = np.zeros(total_words, dtype=np.int32)
        for ta in range(3):
            for tb in range(3):
                for h in tetracode:
                    idx = word_to_idx(h + [ta, tb])
                    x[idx] = 1
        return x

    best_x = None
    best_count = float('inf')

    def consider(x):
        nonlocal best_x, best_count
        # Ensure validity
        cov = compute_coverage(x)
        if not np.all(cov >= 1):
            # Try to fix with greedy
            x = greedy_cover(x, seed=1)
            cov = compute_coverage(x)
            if not np.all(cov >= 1):
                return
        
        # Local search
        x = local_search(x)
        
        # Check validity again (local search should preserve it, but just in case)
        cov = compute_coverage(x)
        if not np.all(cov >= 1):
            return
            
        c = int(np.sum(x))
        if c < best_count:
            best_count = c
            best_x = x.copy()

    # 1. Start with Tetracode
    x_init = get_tetracode_init()
    consider(x_init)

    # 2. Iterated Local Search
    # We will perturb the best solution and try to improve it.
    iterations = 0
    while time.time() < deadline - 10.0:
        iterations += 1
        # Perturbation: Start from best_x, remove K random codewords, 
        # complete with greedy, then local search.
        
        # Choose K based on iteration or random
        # Smaller K for fine tuning, larger for exploration
        K = random.randint(2, 10)
        
        x = best_x.copy()
        selected_indices = np.nonzero(x)[0]
        
        if len(selected_indices) <= K:
            # Can't remove more than we have
            continue
            
        # Remove K random ones
        to_remove = random.sample(selected_indices.tolist(), K)
        for idx in to_remove:
            x[idx] = 0
            
        # Complete
        x = greedy_cover(x, seed=iterations)
        
        # Local Search
        consider(x)
        
        # Occasionally, try a fresh start with different randomization in greedy
        if iterations % 5 == 0 and time.time() < deadline - 20.0:
             x_fresh = np.zeros(total_words, dtype=np.int32)
             x_fresh = greedy_cover(x_fresh, seed=iterations*100)
             consider(x_fresh)

    if best_x is None:
        # Fallback
        best_x = get_tetracode_init()
        best_x = local_search(best_x)

    return [idx_to_word(int(i)) for i in np.nonzero(best_x)[0]]
