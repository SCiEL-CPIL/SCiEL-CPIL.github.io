"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

_START = time.time()
_DEADLINE = 85.0  # safety margin under the 90s kill limit


def _elapsed():
    return time.time() - _START


def construct():
    """
    Constructs 101 lists of denominators (1-10) such that the sum of their
    modular inverses (mod 101) equals the list index.
    Minimizes the maximum length of any list.
    
    Approach:
    1. Precompute all 2^10 - 1 = 1023 non-empty subsets of {1,...,10}
    2. For each subset, compute the sum of modular inverses mod 101
    3. Find the minimum k such that every residue 0..100 is covered by at least one subset of size <= k
    4. For the optimal k, assign to each residue a subset of size exactly k (or less) that covers it
    5. Use a greedy set-cover-like approach to minimize the maximum row length
    """
    n = 10
    mod = 101
    
    # Precompute modular inverses
    inv = [0] * (n + 1)
    for i in range(1, n + 1):
        inv[i] = pow(i, -1, mod)
    
    # Precompute all non-empty subsets: (mask, sum_mod, size)
    all_subsets = []
    subset_sum = [0] * (1 << n)
    subset_size = [0] * (1 << n)
    
    for mask in range(1, 1 << n):
        lsb = mask & (-mask)
        bit_idx = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        subset_sum[mask] = (subset_sum[prev_mask] + inv[bit_idx + 1]) % mod
        subset_size[mask] = subset_size[prev_mask] + 1
        all_subsets.append((mask, subset_sum[mask], subset_size[mask]))
    
    # Group subsets by size
    subsets_by_size = {}
    for mask, s, size in all_subsets:
        if size not in subsets_by_size:
            subsets_by_size[size] = []
        subsets_by_size[size].append((mask, s))
    
    # For each size k, build a mapping from residue to list of masks that produce it
    residue_to_masks = {}
    for k in sorted(subsets_by_size.keys()):
        residue_to_masks[k] = {}
        for mask, s in subsets_by_size[k]:
            if s not in residue_to_masks[k]:
                residue_to_masks[k][s] = []
            residue_to_masks[k][s].append(mask)
    
    # Find minimum k such that all 101 residues are covered
    min_k = None
    for k in sorted(subsets_by_size.keys()):
        covered = set(residue_to_masks[k].keys())
        # Also include smaller sizes
        for kk in range(1, k):
            covered.update(residue_to_masks[kk].keys())
        if len(covered) >= mod:
            min_k = k
            break
    
    if min_k is None:
        # Fallback: use all subsets
        min_k = max(subsets_by_size.keys())
    
    # Now assign to each residue a mask of size <= min_k
    # To minimize the maximum row length, we want to use size exactly min_k when possible
    # but use smaller sizes if that helps
    # Actually, the objective is to minimize the MAX row length, so we just need each row to have
    # size <= min_k. We can use any subset of size <= min_k.
    
    # For each residue, pick a mask of size <= min_k
    # Prefer smaller sizes to keep things simple, but we must ensure we have a valid assignment
    # Actually, we just need one valid subset per residue with size <= min_k
    
    best_for_residue = [None] * mod
    
    # Try sizes from min_k down to 1
    for k in range(min_k, 0, -1):
        for r in range(mod):
            if best_for_residue[r] is None and r in residue_to_masks[k]:
                # Pick the first mask (smallest mask for determinism)
                best_for_residue[r] = (k, min(residue_to_masks[k][r]))
    
    # Check if all residues are covered
    all_covered = all(best_for_residue[r] is not None for r in range(mod))
    
    if not all_covered:
        # Fallback: use any valid subset
        for mask, s, size in all_subsets:
            if best_for_residue[s] is None:
                best_for_residue[s] = (size, mask)
    
    # Construct the result rows
    rows = []
    max_len = 0
    for r in range(mod):
        if best_for_residue[r] is None:
            rows.append([1])  # fallback
            max_len = max(max_len, 1)
        else:
            size, mask = best_for_residue[r]
            terms = []
            for i in range(1, n + 1):
                if (mask >> (i - 1)) & 1:
                    terms.append(i)
            rows.append(terms)
            max_len = max(max_len, len(terms))
    
    return rows
# EVOLVE-BLOCK-END
