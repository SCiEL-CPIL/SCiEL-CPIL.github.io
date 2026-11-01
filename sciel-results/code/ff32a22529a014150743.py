import itertools
import time
import random
import sys

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        code = []
        covered = set()
        all_words = list(itertools.product(range(q), repeat=n))
        for w in all_words:
            if w not in covered:
                code.append(list(w))
                # Mark neighbors as covered
                w_list = list(w)
                covered.add(w)
                for pos in range(n):
                    orig = w_list[pos]
                    for nv in range(q):
                        if nv != orig:
                            w_list[pos] = nv
                            covered.add(tuple(w_list))
                    w_list[pos] = orig
        return code

    start_time = time.time()
    deadline = start_time + 115.0  # 5 second margin before 120s limit

    # Precompute all 729 words and their neighbors
    all_words = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    neighbors = [[] for _ in range(729)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)  # self
        w_list = list(w)
        for pos in range(6):
            orig_val = w_list[pos]
            for new_val in range(3):
                if new_val != orig_val:
                    w_list[pos] = new_val
                    neighbors[i].append(word_to_idx[tuple(w_list)])
            w_list[pos] = orig_val

    # Initial 81-word code from tetracode direct sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [word_to_idx[tuple(h + tail)] for h in tetracode for tail in itertools.product(range(3), repeat=2)]

    def compute_coverage(indices):
        cov = [0] * 729
        for idx in indices:
            for n_idx in neighbors[idx]:
                cov[n_idx] += 1
        return cov

    def greedy_remove(code_indices, coverage):
        """Try to remove redundant codewords. Returns (new_code, new_coverage)."""
        current = code_indices[:]
        cov = coverage[:]
        changed = True
        while changed:
            changed = False
            check_order = current[:]
            random.shuffle(check_order)
            for idx in check_order:
                if idx not in current:
                    continue
                # Check if all neighbors have coverage >= 2
                can_remove = True
                for n_idx in neighbors[idx]:
                    if cov[n_idx] < 2:
                        can_remove = False
                        break
                if can_remove:
                    current.remove(idx)
                    for n_idx in neighbors[idx]:
                        cov[n_idx] -= 1
                    changed = True
        return current, cov

    def try_swap(code_indices, coverage, idx_to_replace):
        """Try replacing idx_to_replace with a different codeword.
        Returns (new_code, new_coverage, improved) if swap enables further removal."""
        original_code = code_indices[:]
        original_cov = coverage[:]
        
        # Find a candidate to replace with
        # Try all possible words not already in the code
        in_code = set(code_indices)
        candidates = [i for i in range(729) if i not in in_code]
        random.shuffle(candidates)
        
        best_result = None
        best_size = len(code_indices)
        
        for cand in candidates[:50]:  # Limit swaps tried
            new_code = [c for c in code_indices if c != idx_to_replace]
            new_code.append(cand)
            new_cov = original_cov[:]
            # Remove coverage from old codeword
            for n_idx in neighbors[idx_to_replace]:
                new_cov[n_idx] -= 1
            # Add coverage from new codeword
            for n_idx in neighbors[cand]:
                new_cov[n_idx] += 1
            
            # Check validity
            valid = all(new_cov[i] >= 1 for i in range(729))
            if not valid:
                continue
            
            # Try greedy removal from this new state
            new_code2, new_cov2 = greedy_remove(new_code, new_cov)
            
            if len(new_code2) < best_size:
                best_size = len(new_code2)
                best_result = (new_code2, new_cov2)
        
        if best_result:
            return best_result[0], best_result[1], True
        return original_code, original_cov, False

    best_code = initial_code[:]
    best_size = len(best_code)
    best_cov = compute_coverage(best_code)
    
    # Multiple restarts with different strategies
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        # Start from initial code
        current_code = initial_code[:]
        current_cov = compute_coverage(current_code)
        
        # Strategy 1: Pure greedy removal with random order
        if attempt < 10:
            current_code, current_cov = greedy_remove(current_code, current_cov)
        
        # Strategy 2: Greedy removal + swap attempts
        elif attempt < 20:
            current_code, current_cov = greedy_remove(current_code, current_cov)
            # Try swaps on a few random codewords
            if current_code:
                swap_candidates = current_code[:]
                random.shuffle(swap_candidates)
                for idx in swap_candidates[:5]:
                    if time.time() > deadline:
                        break
                    new_code, new_cov, improved = try_swap(current_code, current_cov, idx)
                    if improved and len(new_code) < len(current_code):
                        current_code = new_code
                        current_cov = new_cov
                        # Continue greedy removal after swap
                        current_code, current_cov = greedy_remove(current_code, current_cov)
        
        # Strategy 3: Perturb initial code slightly, then greedy remove
        else:
            # Randomly replace 1-3 codewords with random words, then greedy remove
            perturbed = current_code[:]
            num_replacements = random.randint(1, 3)
            in_code = set(perturbed)
            for _ in range(num_replacements):
                if not perturbed:
                    break
                idx_to_replace = random.choice(perturbed)
                # Find a random replacement
                while True:
                    cand = random.randint(0, 728)
                    if cand not in in_code:
                        break
                perturbed.remove(idx_to_replace)
                perturbed.append(cand)
                in_code.remove(idx_to_replace)
                in_code.add(cand)
            
            current_cov = compute_coverage(perturbed)
            current_code, current_cov = greedy_remove(current_code, current_cov)
        
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = current_code[:]
            best_cov = current_cov[:]
    
    # Final polish: try more aggressive swap search on the best solution
    if time.time() < deadline - 10:
        current_code = best_code[:]
        current_cov = best_cov[:]
        
        for swap_round in range(5):
            if time.time() > deadline:
                break
            if not current_code:
                break
            progress = False
            swap_candidates = current_code[:]
            random.shuffle(swap_candidates)
            for idx in swap_candidates:
                if time.time() > deadline:
                    break
                new_code, new_cov, improved = try_swap(current_code, current_cov, idx)
                if improved and len(new_code) < len(current_code):
                    current_code = new_code
                    current_cov = new_cov
                    progress = True
                    # Greedy remove after successful swap
                    current_code, current_cov = greedy_remove(current_code, current_cov)
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = current_code[:]
                        best_cov = current_cov[:]
            if not progress:
                break
    
    # Convert to result format
    result = [list(all_words[idx]) for idx in best_code]
    
    # Verify validity (safety check)
    if len(result) > 0:
        final_cov = [0] * 729
        for c in result:
            idx = word_to_idx[tuple(c)]
            for n_idx in neighbors[idx]:
                final_cov[n_idx] += 1
        if any(c == 0 for c in final_cov):
            # Fallback to initial valid code
            result = [list(all_words[idx]) for idx in initial_code]
    
    return result
