import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 150s kill

    total_words = q ** n

    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors for all words (radius-1 ball)
    neighbors = np.empty(total_words, dtype=object)
    for idx in range(total_words):
        w = idx_to_word(idx)
        nbrs = [idx]
        for i in range(n):
            orig = w[i]
            for new_val in range(q):
                if new_val == orig:
                    continue
                w[i] = new_val
                nbrs.append(word_to_idx(w))
            w[i] = orig
        neighbors[idx] = np.array(nbrs, dtype=np.int64)

    ball_size = int(len(neighbors[0]))
    
    # Precompute incidence matrix
    incidence = np.zeros((total_words, total_words), dtype=np.int8)
    for idx in range(total_words):
        incidence[idx, neighbors[idx]] = 1
    
    neighbor_lists = [neighbors[i] for i in range(total_words)]

    def build_coverage_count(code_list):
        """Return an int array where count[j] = number of codewords whose ball contains j."""
        cnt = np.zeros(total_words, dtype=np.int32)
        for ci in code_list:
            cnt[neighbor_lists[ci]] += 1
        return cnt

    def full_coverage_check(code_list):
        cov = np.zeros(total_words, dtype=bool)
        for ci in code_list:
            cov[neighbors[ci]] = True
        return cov

    # --- Start with tetracode direct product (81 words) as guaranteed valid seed ---
    # Tetracode: 9 words of length 4, perfect code radius 1 over Z_3
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Build initial code from tetracode × all tails
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            cw = h + list(tail)
            initial_code.append(word_to_idx(cw))
    
    code_indices = list(initial_code)
    
    # --- Local search: aggressive reduction ---
    
    def try_remove_redundant(code_list):
        """Remove all redundant codewords. Returns new list."""
        cnt = build_coverage_count(code_list)
        changed = True
        while changed:
            changed = False
            for i in range(len(code_list)):
                ci = code_list[i]
                if np.all(cnt[neighbor_lists[ci]] >= 2):
                    code_list.pop(i)
                    cnt = build_coverage_count(code_list)
                    changed = True
                    break
        return code_list

    def try_single_swap(code_list):
        """Try to swap one codeword for another that allows further removals.
        Returns (new_list, improved)."""
        cnt = build_coverage_count(code_list)
        in_code = np.zeros(total_words, dtype=bool)
        in_code[code_list] = True
        
        for i in range(len(code_list)):
            ci = code_list[i]
            nbrs_ci = neighbor_lists[ci]
            # Words that become uncovered if we remove ci
            lost = nbrs_ci[cnt[nbrs_ci] == 1]
            if len(lost) == 0:
                # Redundant
                code_list.pop(i)
                return code_list, True
            
            # Find candidates that cover all lost words
            covers_all = (incidence[:, lost] == 1).all(axis=1)
            covers_all[in_code] = False
            
            if not np.any(covers_all):
                continue
            
            cands = np.where(covers_all)[0]
            
            # Score: maximize coverage of singly-covered words (enables future removals)
            single_mask = (cnt == 1).astype(np.int32)
            scores = incidence[cands].dot(single_mask)
            
            # Also consider: minimize total overlap (words with cnt >= 2)
            overlap_mask = (cnt >= 2).astype(np.int32)
            overlap_scores = incidence[cands].dot(overlap_mask)
            
            # Combined score: maximize single coverage, minimize overlap
            combined = scores - 0.5 * overlap_scores
            best_c = int(cands[np.argmax(combined)])
            
            code_list[i] = best_c
            return code_list, True
        
        return code_list, False

    def try_double_swap(code_list):
        """Try to swap two codewords simultaneously for better coverage.
        Returns (new_list, improved)."""
        cnt = build_coverage_count(code_list)
        in_code = np.zeros(total_words, dtype=bool)
        in_code[code_list] = True
        
        m = len(code_list)
        if m < 2:
            return code_list, False
        
        # Try pairs of codewords to remove
        for i in range(m):
            if time.time() - start_time > DEADLINE - 10:
                return code_list, False
            ci = code_list[i]
            nbrs_ci = neighbor_lists[ci]
            
            for j in range(i + 1, m):
                cj = code_list[j]
                nbrs_cj = neighbor_lists[cj]
                
                # Combined lost set
                cnt_without = cnt.copy()
                cnt_without[nbrs_ci] -= 1
                cnt_without[nbrs_cj] -= 1
                
                lost = np.where(cnt_without == 0)[0]
                if len(lost) == 0:
                    # Both redundant
                    code_list.pop(j)
                    code_list.pop(i)
                    return code_list, True
                
                # Find two candidates that together cover all lost
                # This is expensive; use a heuristic
                # First candidate must cover some lost words
                covers_any = (incidence[:, lost] == 1).any(axis=1)
                covers_any[in_code] = False
                covers_any[ci] = False
                covers_any[cj] = False
                
                if not np.any(covers_any):
                    continue
                
                # Try the best single candidate for lost words
                covers_single = (incidence[:, lost] == 1).all(axis=1)
                covers_single[in_code] = False
                covers_single[ci] = False
                covers_single[cj] = False
                
                if np.any(covers_single):
                    cands = np.where(covers_single)[0]
                    single_mask = (cnt == 1).astype(np.int32)
                    scores = incidence[cands].dot(single_mask)
                    best_c = int(cands[np.argmax(scores)])
                    code_list[i] = best_c
                    code_list.pop(j)
                    return code_list, True
                
                # If no single candidate covers all, try two
                # Pick candidate that covers most lost words
                lost_count = len(lost)
                if lost_count <= 6:  # Only try for small lost sets
                    cands_partial = np.where(covers_any)[0]
                    if len(cands_partial) > 0:
                        # Score by number of lost words covered
                        lost_inc = incidence[:, lost]
                        cov_counts = lost_inc[cands_partial].sum(axis=1)
                        best_partial = int(cands_partial[np.argmax(cov_counts)])
                        
                        # Now find second candidate to cover remaining
                        remaining_lost = lost[lost_inc[best_partial] == 0]
                        if len(remaining_lost) == 0:
                            code_list[i] = best_partial
                            code_list.pop(j)
                            return code_list, True
                        
                        if len(remaining_lost) <= 3:
                            covers_rem = (incidence[:, remaining_lost] == 1).all(axis=1)
                            covers_rem[in_code] = False
                            covers_rem[best_partial] = False
                            covers_rem[ci] = False
                            covers_rem[cj] = False
                            if np.any(covers_rem):
                                rem_cands = np.where(covers_rem)[0]
                                single_mask2 = (cnt == 1).astype(np.int32)
                                scores2 = incidence[rem_cands].dot(single_mask2)
                                best_c2 = int(rem_cands[np.argmax(scores2)])
                                code_list[i] = best_partial
                                code_list[j] = best_c2
                                return code_list, True
        
        return code_list, False

    # Iterative local search
    max_iterations = 50
    for iteration in range(max_iterations):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        improved = True
        while improved:
            improved = False
            
            # Remove redundants
            before = len(code_indices)
            code_indices = try_remove_redundant(code_indices)
            if len(code_indices) < before:
                improved = True
                continue
            
            # Try single swap
            code_indices, imp = try_single_swap(code_indices)
            if imp:
                improved = True
                continue
            
            # Try double swap
            code_indices, imp = try_double_swap(code_indices)
            if imp:
                improved = True
                continue
        
        if time.time() - start_time > DEADLINE - 15:
            break

    # --- Final verification and emergency fill ---
    final_cov = full_coverage_check(code_indices)
    if not np.all(final_cov):
        covered = final_cov
        while not np.all(covered):
            if time.time() - start_time > DEADLINE - 2:
                break
            covered_int = covered.astype(np.int8)
            new_cov = ball_size - incidence.dot(covered_int)
            best_idx = int(np.argmax(new_cov))
            best_new = int(new_cov[best_idx])
            if best_new <= 0:
                break
            code_indices.append(best_idx)
            covered[neighbor_lists[best_idx]] = True

    # Deduplicate
    seen = set()
    unique_code = []
    for idx in code_indices:
        if idx not in seen:
            seen.add(idx)
            unique_code.append(idx)

    return [idx_to_word(idx) for idx in unique_code]
