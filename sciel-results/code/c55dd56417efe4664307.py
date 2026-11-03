import time
import random
import numpy as np
import sys

_DEADLINE = 110.0
_START = [time.time()]

def _elapsed():
    return time.time() - _START[0]

def _time_left():
    return _DEADLINE - _elapsed()

def construct(q, n, radius):
    _START[0] = time.time()
    
    # Fallback for non-standard parameters
    if (q, n, radius) != (3, 8, 1):
        import itertools
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    rng = random.Random(12345)
    N = 3 ** 8  # 6561
    BALL_SIZE = 17

    # Precompute digits and idx conversions
    def idx_to_digits(i):
        d = [0] * 8
        for p in range(8):
            d[p] = i % 3
            i //= 3
        return d

    def digits_to_idx(d):
        v = 0
        mul = 1
        for p in range(8):
            v += d[p] * mul
            mul *= 3
        return v

    # Precompute balls as a 2D array: ball_matrix[i] contains the 17 indices in ball of i
    ball_matrix = np.zeros((N, BALL_SIZE), dtype=np.int32)
    for i in range(N):
        d = idx_to_digits(i)
        ball_matrix[i, 0] = i
        idx = 1
        for p in range(8):
            orig = d[p]
            for nv in range(3):
                if nv != orig:
                    d[p] = nv
                    ball_matrix[i, idx] = digits_to_idx(d)
                    idx += 1
                    d[p] = orig

    # Precompute inverse mapping: word_to_coders[w] = list of codewords c such that w in ball(c)
    # Since ball is symmetric, this is just the transpose of the incidence structure.
    # We can build this efficiently.
    word_to_coders = [[] for _ in range(N)]
    for c in range(N):
        for w in ball_matrix[c]:
            word_to_coders[w].append(c)

    # 1. Criticality Greedy Initialization
    # We want to cover all N words.
    # uncovered_mask is a boolean array of size N
    uncovered = np.ones(N, dtype=bool)
    chosen = []
    chosen_set = set()
    
    # Precompute potential coders for each word for criticality check
    # crit[w] = len(word_to_coders[w]) - (number of chosen coders that cover w)
    # Initially crit[w] = len(word_to_coders[w]) = 17 for all w.
    # As we add codewords, we decrement crit for words in their ball.
    crit = np.array([len(word_to_coders[w]) for w in range(N)], dtype=np.int32)
    
    while uncovered.any():
        if _time_left() < 30.0:
            break
            
        # Find the uncovered word with minimum criticality
        # We only care about uncovered words.
        # To speed up, we can just pick a random uncovered word if we don't want to scan all N.
        # But N=6561 is small. Scanning is fast.
        # Actually, finding min crit among uncovered words:
        # We can maintain a priority queue, but updates are complex.
        # Simple scan:
        min_crit = N + 1
        target_w = -1
        
        # Optimization: Only check a subset? No, N is small.
        # But doing this in a Python loop is slow.
        # Vectorized approach:
        # We want argmin of crit where uncovered is True.
        # Set crit of covered words to a large number.
        temp_crit = crit.copy()
        temp_crit[~uncovered] = N + 1
        target_w = int(np.argmin(temp_crit))
        min_crit = temp_crit[target_w]
        
        if min_crit == 0:
            # Should not happen if logic is correct, but safety break
            break
            
        # Find best codeword c that covers target_w and maximizes new coverage
        # Candidates are word_to_coders[target_w]
        best_c = -1
        best_gain = -1
        
        for c in word_to_coders[target_w]:
            if c in chosen_set:
                continue
            # Gain = number of uncovered words in ball(c)
            # Vectorized: sum(uncovered[ball_matrix[c]])
            gain = np.sum(uncovered[ball_matrix[c]])
            if gain > best_gain:
                best_gain = gain
                best_c = c
                
        if best_c == -1:
            # Fallback: pick any
            for c in word_to_coders[target_w]:
                if c not in chosen_set:
                    best_c = c
                    break
        if best_c == -1:
            break
            
        chosen.append(best_c)
        chosen_set.add(best_c)
        
        # Update uncovered and crit
        b = ball_matrix[best_c]
        uncovered[b] = False
        crit[b] -= 1

    # Ensure full coverage (safety)
    if not uncovered.all() == False: # If any remain
         if not (~uncovered).all():
             for i in range(N):
                 if uncovered[i]:
                     chosen.append(i)
                     chosen_set.add(i)
                     uncovered[ball_matrix[i]] = False
                     crit[ball_matrix[i]] -= 1

    # 2. Local Search: Removals and Swaps
    # covcount: how many times each word is covered
    covcount = np.zeros(N, dtype=np.int32)
    for c in chosen:
        covcount[ball_matrix[c]] += 1

    def try_removals(chosen_set, covcount, time_budget=10.0):
        t_start = time.time()
        improved = True
        while improved and time.time() - t_start < time_budget and _time_left() > 15.0:
            improved = False
            # Shuffle to avoid order bias
            order = list(chosen_set)
            rng.shuffle(order)
            for c in order:
                if c not in chosen_set:
                    continue
                # Check if any word in ball(c) is uniquely covered by c
                # min_cov = np.min(covcount[ball_matrix[c]])
                # If min_cov >= 2, we can remove c
                if np.min(covcount[ball_matrix[c]]) >= 2:
                    covcount[ball_matrix[c]] -= 1
                    chosen_set.discard(c)
                    improved = True
        return chosen_set, covcount

    chosen_set, covcount = try_removals(chosen_set, covcount)

    def local_search_swaps(chosen_set, covcount, time_budget=60.0):
        t_start = time.time()
        best_set = set(chosen_set)
        best_size = len(chosen_set)
        
        # Iterative improvement
        while time.time() - t_start < time_budget and _time_left() > 10.0:
            improved = False
            
            # 1. Try removals again
            chosen_set, covcount = try_removals(chosen_set, covcount, time_budget=2.0)
            if len(chosen_set) < best_size:
                best_size = len(chosen_set)
                best_set = set(chosen_set)
                improved = True
            
            # 2. Try swaps
            # Calculate waste for each codeword: sum(max(0, covcount[w]-1) for w in ball(c))
            # Vectorized waste calculation for all codewords is expensive if done naively.
            # But we can compute total waste: sum(max(0, covcount-1))
            # We want to reduce total waste.
            
            # Identify codewords with high waste to try removing first
            # Waste of c = sum(covcount[ball(c)] - 1) for covcount[ball(c)] > 0
            # Actually, waste is just the number of "excess" coverings.
            
            # Instead of calculating waste for all, we can iterate through codewords
            # and try to swap the one that seems most redundant.
            
            order = list(chosen_set)
            rng.shuffle(order)
            
            for c in order:
                if _time_left() < 10.0:
                    break
                if c not in chosen_set:
                    continue
                    
                b_c = ball_matrix[c]
                # Words that would become uncovered if we remove c
                critical_words = b_c[covcount[b_c] == 1]
                
                if len(critical_words) == 0:
                    # Pure removal
                    covcount[b_c] -= 1
                    chosen_set.discard(c)
                    improved = True
                    if len(chosen_set) < best_size:
                        best_size = len(chosen_set)
                        best_set = set(chosen_set)
                    continue
                
                # We need to find a replacement c' that covers all critical_words
                # Candidates: intersection of word_to_coders[w] for w in critical_words
                # And not in chosen_set, and not c
                
                # Start with candidates from the first critical word
                if len(critical_words) == 0:
                    continue
                    
                candidates = set(word_to_coders[int(critical_words[0])])
                
                for w in critical_words[1:]:
                    candidates &= set(word_to_coders[int(w)])
                    if not candidates:
                        break
                
                candidates.discard(c)
                candidates -= chosen_set
                
                if not candidates:
                    continue
                
                # Try a few candidates
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                # Calculate current total waste
                # waste = sum(covcount - 1) where covcount > 1
                # To avoid O(N) sum every time, we can track it or just compute it.
                # N=6561 is small, summing is fast.
                current_waste = np.sum(np.maximum(covcount - 1, 0))
                
                for c_prime in cand_list[:5]: # Try up to 5 candidates
                    b_cp = ball_matrix[c_prime]
                    
                    # Simulate swap: remove c, add c_prime
                    # delta_waste = waste_after - waste_before
                    
                    # Words affected: union(b_c, b_cp)
                    # We can compute new covcount locally
                    # new_cov = covcount.copy()
                    # new_cov[b_c] -= 1
                    # new_cov[b_cp] += 1
                    # new_waste = np.sum(np.maximum(new_cov - 1, 0))
                    
                    # Optimized:
                    # Only words in b_c or b_cp change.
                    # Let's just do the copy for simplicity, it's fast enough.
                    new_cov = covcount.copy()
                    new_cov[b_c] -= 1
                    new_cov[b_cp] += 1
                    
                    # Check validity: no negative counts (should be true if c' covers criticals)
                    # new_cov[b_c] must be >= 0. Since criticals were 1, they become 0.
                    # Others were >= 2, become >= 1. So valid.
                    
                    new_waste = np.sum(np.maximum(new_cov - 1, 0))
                    
                    if new_waste < current_waste:
                        # Accept swap
                        covcount = new_cov
                        chosen_set.discard(c)
                        chosen_set.add(c_prime)
                        improved = True
                        if len(chosen_set) < best_size:
                            best_size = len(chosen_set)
                            best_set = set(chosen_set)
                        break # Restart loop with new state
                        # Note: We break the inner loop, but the outer 'for c in order' continues.
                        # It's better to restart the whole 'for c in order' to pick the next high-waste one.
                        # But 'order' is fixed. We can just continue.
                        
            if not improved:
                # No improvement in this pass
                break
                
        return best_set, best_size

    chosen_set, best_size = local_search_swaps(chosen_set, covcount)
    
    # Rebuild covcount from best_set
    covcount = np.zeros(N, dtype=np.int32)
    for c in chosen_set:
        covcount[ball_matrix[c]] += 1
        
    # Final removals
    chosen_set, covcount = try_removals(chosen_set, covcount, time_budget=5.0)
    
    # Final verification
    final_covered = np.zeros(N, dtype=np.int8)
    for c in chosen_set:
        final_covered[ball_matrix[c]] = 1
    if not final_covered.all():
        for i in range(N):
            if not final_covered[i]:
                chosen_set.add(i)

    chosen = list(chosen_set)
    return [idx_to_digits(i) for i in chosen]
