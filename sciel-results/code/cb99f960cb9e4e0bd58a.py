"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave margin for 150s limit

    # Fallback for non-specific parameters
    if q != 3 or n != 7 or radius != 1:
        if n <= 4 and q == 3 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbor lists as numpy arrays of int32 for fast operations
    neighbors_list = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(nb, dtype=np.int32))

    # Build cover matrix efficiently using the neighbor lists
    cover_matrix = np.zeros((N, N), dtype=np.bool_)
    for c in range(N):
        cover_matrix[c, neighbors_list[c]] = True

    def greedy_cover():
        """Construct a valid covering code using greedy set cover."""
        uncovered_mask = np.ones(N, dtype=bool)
        code = []
        
        while uncovered_mask.any():
            # Vectorized: compute gain for all candidates at once
            gains = cover_matrix[:, uncovered_mask].sum(axis=1)
            best_c = int(np.argmax(gains))
            best_count = int(gains[best_c])
            
            if best_count == 0:
                break
            
            code.append(best_c)
            uncovered_mask[neighbors_list[best_c]] = False
            
        return code

    def compute_coverage(code):
        """Compute coverage array from a list of codewords."""
        coverage = np.zeros(N, dtype=np.int32)
        for c in code:
            coverage[neighbors_list[c]] += 1
        return coverage

    def is_redundant(c, coverage):
        """Check if codeword c is redundant."""
        return np.min(coverage[neighbors_list[c]]) > 1

    def reduce_code(code, coverage):
        """Remove all redundant codewords from the code."""
        changed = True
        while changed:
            changed = False
            for c in list(code):
                if is_redundant(c, coverage):
                    code.remove(c)
                    coverage[neighbors_list[c]] -= 1
                    changed = True
        return code, coverage

    def find_best_replacement(uncovered_indices, current_code):
        """Find the single best codeword to cover the uncovered indices."""
        if len(uncovered_indices) == 0:
            return None, 0
        
        # Create a mask for uncovered indices
        # To be efficient, we can sum gains directly
        # But uncovered_indices might be sparse.
        # cover_matrix[:, uncovered_indices].sum(axis=1) is O(N * len(uncovered))
        # N=2187, len(uncovered) <= 2187. Max 4.8M ops. Fast.
        
        in_code = np.zeros(N, dtype=bool)
        for c in current_code:
            in_code[c] = True
            
        gains = cover_matrix[:, uncovered_indices].sum(axis=1)
        gains[in_code] = 0
        
        best_c = int(np.argmax(gains))
        best_count = int(gains[best_c])
        
        return best_c, best_count

    # Main Algorithm
    # 1. Initial Greedy
    code = greedy_cover()
    coverage = compute_coverage(code)
    
    # 2. Initial Reduction
    code, coverage = reduce_code(code, coverage)
    
    best_code = list(code)
    best_size = len(code)
    
    # 3. Simulated Annealing / Hill Climbing
    T = 1.0
    cooling = 0.995
    
    while time.time() < deadline:
        # Update best
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
            
        if len(code) < 2:
            break
            
        # Pick a random codeword to swap
        c_remove = random.choice(code)
        
        # Temporarily remove it
        code.remove(c_remove)
        coverage[neighbors_list[c_remove]] -= 1
        
        # Find uncovered words
        uncovered = np.where(coverage == 0)[0]
        
        if len(uncovered) == 0:
            # Free removal!
            pass # Code is already updated
        else:
            # Find best replacement
            c_new, count = find_best_replacement(uncovered, code)
            
            if c_new is not None and count > 0:
                # Add the new codeword
                code.append(c_new)
                coverage[neighbors_list[c_new]] += 1
            else:
                # No good replacement found, revert
                code.append(c_remove)
                coverage[neighbors_list[c_remove]] += 1
                T *= cooling
                continue
        
        # After swap (or removal), try to reduce
        old_size = len(code)
        code, coverage = reduce_code(code, coverage)
        new_size = len(code)
        
        delta = new_size - old_size
        
        # Acceptance criterion
        if delta < 0:
            # Improvement
            pass
        elif delta == 0:
            # Same size, accept with probability 0.5 to allow random walk
            if random.random() > 0.5:
                # Revert if we don't accept? 
                # Actually, if we accept, we keep the new code.
                # If we reject, we need to revert.
                # Reverting is complex because reduce_code changed it.
                # Simpler: Just always accept delta=0 to keep moving?
                # Or store previous state?
                # Let's just accept delta=0 to explore the space.
                pass
        else:
            # Size increased, reject (revert)
            # We need to restore the previous state.
            # This is tricky because we modified code and coverage in place.
            # To avoid complex rollback, we can work on copies.
            # But copies are expensive.
            # Alternative: Only try moves that are guaranteed to not increase size?
            # Swap 1 for 1 keeps size same. Reduce might decrease it.
            # So delta should be <= 0 always?
            # Yes, because we started with size S, removed 1 (S-1), added 1 (S), then reduced (<= S).
            # So delta should never be > 0.
            # If delta > 0, something went wrong.
            pass
        
        T *= cooling

    # Final verification and return
    # Ensure best_code is valid
    final_coverage = compute_coverage(best_code)
    if np.any(final_coverage == 0):
        # Fallback to full space if somehow invalid (should not happen)
        best_code = list(range(N))
        
    result = [idx_to_tuple(cw) for cw in best_code]
    return result
# EVOLVE-BLOCK-END
