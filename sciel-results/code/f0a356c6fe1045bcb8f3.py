"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Optimized for q=3, n=6, radius=1 using bitmask operations for fast coverage checks.
    """
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: simple greedy
        all_words = list(itertools.product(range(q), repeat=n))
        neighbors = {}
        for w in all_words:
            nb = set()
            for v in all_words:
                dist = sum(1 for i in range(n) if w[i] != v[i])
                if dist <= radius:
                    nb.add(v)
            neighbors[w] = nb
        
        uncovered = set(all_words)
        code = []
        while uncovered:
            best_word = None
            max_covered = -1
            for w in all_words:
                if w in code:
                    continue
                covered_count = len(neighbors[w] & uncovered)
                if covered_count > max_covered:
                    max_covered = covered_count
                    best_word = w
            if best_word is None:
                break
            code.append(best_word)
            uncovered -= neighbors[best_word]
        return [list(c) for c in code]

    # Specific optimization for q=3, n=6, radius=1
    total_words = 3 ** n
    all_words = list(itertools.product(range(3), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbor masks
    # mask[i] is an integer where bit j is 1 if all_words[j] is within distance 1 of all_words[i]
    neighbor_masks = [0] * total_words
    for i, w in enumerate(all_words):
        mask = 0
        # Add self
        mask |= (1 << i)
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = new_val
                new_w = tuple(new_w_list)
                j = word_to_idx[new_w]
                mask |= (1 << j)
        neighbor_masks[i] = mask
    
    total_mask = (1 << total_words) - 1
    
    # Start with tetracode-based construction (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = tuple(h + list(tail))
            initial_indices.append(word_to_idx[w])
    
    code_indices = set(initial_indices)
    
    # Helper to check if a set of indices covers everything
    def is_covering(idx_set):
        m = 0
        for idx in idx_set:
            m |= neighbor_masks[idx]
        return m == total_mask
    
    # 1. Greedy Removal
    # We want to remove as many as possible
    changed = True
    while changed:
        changed = False
        for idx in list(code_indices):
            if is_covering(code_indices - {idx}):
                code_indices.remove(idx)
                changed = True
                break # Restart to find other removable words
    
    # 2. Local Search: 1-for-1 Swaps
    # Try to swap a codeword with another to potentially allow further removals
    # Or just to find a different local minimum
    # We iterate a few times to ensure we don't get stuck in a bad local min
    time_limit = 110 # Leave some buffer for overhead
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        code_list = list(code_indices)
        for i, idx_remove in enumerate(code_list):
            # Try swapping with any word not in code
            # Optimization: Only try swaps that might be beneficial?
            # For now, try all 729 words
            for idx_add in range(total_words):
                if idx_add in code_indices:
                    continue
                new_set = (code_indices - {idx_remove}) | {idx_add}
                if is_covering(new_set):
                    code_indices = new_set
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # If no swap improved things, try a random perturbation?
            # Or break if we've been stuck for a while?
            # Let's try a few random restarts if we get stuck at >72
            if len(code_indices) > 72:
                # Perturb: remove a random word, add a random word, if still covering
                # Actually, if is_covering fails, we can't do that.
                # So we can only move to other covering codes.
                # If we are stuck, it means no 1-for-1 swap maintains coverage.
                # This is a local minimum for 1-for-1 moves.
                # We can try 2-for-2 moves, but that's O(N^4) which is too slow.
                # Let's just break and hope this is the global min or close.
                break
            
            # If we reached 72, we are done (lower bound is 71, so 72 is great)
            if len(code_indices) <= 72:
                break

    result = [list(all_words[idx]) for idx in code_indices]
    return result

# EVOLVE-BLOCK-END
