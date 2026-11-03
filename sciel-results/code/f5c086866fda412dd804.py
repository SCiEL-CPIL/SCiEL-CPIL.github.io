"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Classic recursive construction
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap
            max_overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    max_overlap = o
                    break
            new += block[max_overlap:]
        word = new
    
    # Try to shorten the word
    if n >= 3 and time.time() < deadline:
        word = try_shorten_fast(word, n, deadline)
    
    return "".join(map(str, word))

def try_shorten_fast(word, n, deadline):
    """
    Optimized local search to remove redundant symbols.
    Uses the fact that deleting a symbol only affects O(n) windows.
    """
    word = list(word)
    
    # Helper to get all permutations and their indices
    def get_perm_indices(w):
        perm_indices = defaultdict(list)
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i + n])
            if len(set(window)) == n:
                perm_indices[window].append(i)
        return perm_indices

    # Initial state
    perm_indices = get_perm_indices(word)
    total_perms = len(perm_indices)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # We iterate from right to left to avoid index shifting issues during deletion
        # But we need to re-evaluate indices after each deletion
        # To be efficient, we can try all deletions in one pass, but if we delete one,
        # indices change.
        # Given the small number of expected deletions (8 for n=7), we can do a full scan.
        
        # To speed up, we can check candidates.
        # A candidate index i is valid if removing it doesn't lose any permutation.
        
        # We check from right to left.
        for i in range(len(word) - 1, -1, -1):
            if time.time() >= deadline:
                return word
            
            # Identify windows affected by deletion at i
            # Windows starting at k such that [k, k+n) intersects [i, i+1)
            # i.e., k <= i < k+n  =>  i-n+1 <= k <= i
            start_k = max(0, i - n + 1)
            end_k = min(i, len(word) - n)
            
            if start_k > end_k:
                # No windows affected? This shouldn't happen if i is in range.
                # If i is at the very end or start, it might affect fewer.
                # Actually, if i is the last char, it affects window len-n.
                # If i is the first char, it affects window 0.
                pass

            # Collect permutations that ONLY appear in the affected range
            at_risk = set()
            for k in range(start_k, end_k + 1):
                p = tuple(word[k:k + n])
                if len(set(p)) == n:
                    # Check if this permutation appears elsewhere
                    # perm_indices[p] contains all original indices
                    # We need to know if it appears at an index NOT in [start_k, end_k]
                    # Note: perm_indices is based on the CURRENT word
                    indices = perm_indices[p]
                    # If all indices are in [start_k, end_k], it is at risk
                    has_safe_index = False
                    for idx in indices:
                        if idx < start_k or idx > end_k:
                            has_safe_index = True
                            break
                    if not has_safe_index:
                        at_risk.add(p)
            
            # If no at-risk permutations, deletion is safe
            if not at_risk:
                # Perform deletion
                word = word[:i] + word[i+1:]
                # Update perm_indices?
                # Instead of updating incrementally, recompute if word is not too large.
                # For n=7, len~6000, recomputing is O(6000*7) ~ 42000 ops.
                # This is fast.
                perm_indices = get_perm_indices(word)
                improved = True
                # Restart the loop to check the new state from the right again
                # Or continue? Since we deleted from the right, the left part is unchanged.
                # But indices shifted.
                # It's safer to restart the while loop to ensure we don't miss interactions.
                # However, restarting is expensive if we do it 8 times.
                # 8 * 6000 * 7 is trivial.
                break
            
    return word
# EVOLVE-BLOCK-END
