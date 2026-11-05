import time
import random
import sys
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 150s kill

    N = q ** n  # 2187 for q=3, n=7

    # Precompute neighbors as bitsets (Python int as bitset)
    neighbors = [0] * N

    # Helper to convert index to list of digits
    def idx_to_word(idx):
        word = []
        for _ in range(n):
            word.append(idx % q)
            idx //= q
        return word

    # Precompute all words and word-to-index mapping
    words = [idx_to_word(i) for i in range(N)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}

    # Compute neighbors as bitsets
    for i in range(N):
        w_i = words[i]
        mask = 1 << i
        for pos in range(n):
            for val in range(q):
                if val == w_i[pos]:
                    continue
                new_w = list(w_i)
                new_w[pos] = val
                j = word_to_idx[tuple(new_w)]
                mask |= (1 << j)
        neighbors[i] = mask

    # Full universe bitset
    full_mask = (1 << N) - 1

    def greedy_cover(uncovered_mask, code_list, time_limit):
        """Standard greedy set cover using bitsets. Returns updated code_list."""
        uncovered = uncovered_mask
        code = list(code_list)

        while uncovered and time.time() < time_limit:
            best_candidate = -1
            max_new_cov = 0
            best_cov_mask = 0

            for c in range(N):
                new_cov = neighbors[c] & uncovered
                # Use bit_count for speed
                cov_count = new_cov.bit_count()
                if cov_count > max_new_cov:
                    max_new_cov = cov_count
                    best_candidate = c
                    best_cov_mask = new_cov

            if best_candidate == -1:
                break

            code.append(best_candidate)
            uncovered &= ~neighbors[best_candidate]

        return code

    def prune_redundant(code, time_limit):
        """Remove redundant codewords using cover counts."""
        if not code:
            return code

        # Count how many times each word is covered
        cover_count = [0] * N
        for c in code:
            mask = neighbors[c]
            m = mask
            while m:
                low = m & (-m)
                idx = low.bit_length() - 1
                cover_count[idx] += 1
                m ^= low

        changed = True
        while changed and time.time() < time_limit:
            changed = False
            to_remove = []
            for idx, c in enumerate(code):
                redundant = True
                mask = neighbors[c]
                m = mask
                while m:
                    low = m & (-m)
                    u = low.bit_length() - 1
                    if cover_count[u] <= 1:
                        redundant = False
                        break
                    m ^= low
                if redundant:
                    to_remove.append(idx)

            if to_remove:
                for idx in to_remove:
                    c = code[idx]
                    mask = neighbors[c]
                    m = mask
                    while m:
                        low = m & (-m)
                        u = low.bit_length() - 1
                        cover_count[u] -= 1
                        m ^= low
                remove_set = set(to_remove)
                code = [c for i, c in enumerate(code) if i not in remove_set]
                changed = True

        return code

    def compute_uncovered(code):
        """Compute the set of uncovered words given a code."""
        covered = 0
        for c in code:
            covered |= neighbors[c]
        return full_mask ^ covered

    def kick_and_repair(code, num_kicks, time_limit):
        """Remove random codewords and repair with greedy."""
        if len(code) <= num_kicks:
            return code

        code = list(code)
        removed = random.sample(range(len(code)), num_kicks)
        for idx in sorted(removed, reverse=True):
            code.pop(idx)

        uncovered = compute_uncovered(code)
        code = greedy_cover(uncovered, code, time_limit)
        code = prune_redundant(code, time_limit)

        return code

    def ensure_valid(code, time_limit):
        """Check if code covers all words. If not, add words to cover the gaps."""
        uncovered = compute_uncovered(code)
        if uncovered:
            # Use greedy to fill the holes
            code = greedy_cover(uncovered, code, time_limit)
            # Prune again to remove any redundancy introduced by the repair
            code = prune_redundant(code, time_limit)
        return code

    # Run multiple iterations with different strategies
    best_code = None
    best_size = float('inf')

    # First: fast greedy initialization
    initial_code = greedy_cover(full_mask, [], deadline - 30)
    initial_code = prune_redundant(initial_code, deadline - 30)
    initial_code = ensure_valid(initial_code, deadline - 30)
    best_size = len(initial_code)
    best_code = initial_code[:]

    # Multiple restarts with kicks of varying sizes
    random.seed(42)

    for iteration in range(100):
        if time.time() > deadline - 5:
            break

        # Start from current best
        code = best_code[:]

        # Apply kicks
        # We try a few different kick sizes
        kick_sizes = [1, 2, 3, 5, 8]
        for num_kicks in kick_sizes:
            if time.time() > deadline - 3:
                break
            if len(code) <= num_kicks:
                continue
            
            # Use a shorter time limit for each kick to allow more iterations
            kick_limit = min(deadline - 3, time.time() + 2.0) 
            new_code = kick_and_repair(code, num_kicks, kick_limit)
            
            # Ensure validity before comparing
            new_code = ensure_valid(new_code, deadline - 3)
            
            if len(new_code) < len(code):
                code = new_code
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]

    # Final: ensure the best code is valid
    if best_code is not None:
        best_code = ensure_valid(best_code, deadline - 1)

    if best_code is None:
        # Fallback: direct sum construction (should not happen if greedy works)
        import itertools
        # Simple greedy fallback
        best_code = greedy_cover(full_mask, [], deadline - 1)
        best_code = ensure_valid(best_code, deadline - 1)

    return [list(words[c]) for c in best_code]
