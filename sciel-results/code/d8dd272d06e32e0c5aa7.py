import time
import random
import numpy as np
import itertools
from collections import deque

def construct(q, n, radius):
    """
    Construct a ternary covering code K_3(6,1).
    Returns a list of codewords (lists of integers).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Fallback for non-standard parameters
    if q != 3 or n != 6 or radius != 1:
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_words = 3 ** n  # 729
    sphere_size = 1 + n * (q - 1)  # 13
    
    # --- 1. Precompute Data Structures ---
    
    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[n - 1 - i] = temp % q
            temp //= q
        return tuple(w)
    
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    # Precompute spheres as bitmasks for fast set operations
    # spheres_mask[i] is a Python int bitmask where bit j is set if word j is in sphere i
    spheres_mask = [0] * total_words
    spheres_list = [[] for _ in range(total_words)]  # Also keep list form for iteration
    
    for idx in range(total_words):
        w = list(idx_to_word(idx))
        sphere_set = {idx}
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                sphere_set.add(word_to_idx(w))
                w[pos] = orig_val
        spheres_list[idx] = list(sphere_set)
        # Build bitmask
        mask = 0
        for w_idx in sphere_set:
            mask |= (1 << w_idx)
        spheres_mask[idx] = mask
    
    # Precompute coverers: for each word w, which codewords cover it
    coverers = [[] for _ in range(total_words)]
    for c in range(total_words):
        for w in spheres_list[c]:
            coverers[w].append(c)
    
    # Convert coverers to bitmasks for fast "does c cover w" check
    # coverers_mask[w] has bit c set if word w is in sphere c
    coverers_mask = [0] * total_words
    for w in range(total_words):
        mask = 0
        for c in coverers[w]:
            mask |= (1 << c)
        coverers_mask[w] = mask
    
    # --- 2. Initial Code Construction ---
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_codewords.append(h + list(tail))
    
    initial_indices = [word_to_idx(cw) for cw in initial_codewords]
    
    # --- 3. Helper Functions for Search ---
    
    def compute_counts(code_indices):
        counts = np.zeros(total_words, dtype=np.int32)
        for idx in code_indices:
            counts[spheres_list[idx]] += 1
        return counts

    def reduce_code(code_indices, counts=None):
        """
        Greedily remove redundant codewords using CSP-style pruning.
        Returns (new_code_indices, new_counts)
        """
        if counts is None:
            counts = compute_counts(code_indices)
        
        code_set = set(code_indices)
        changed = True
        while changed:
            changed = False
            # Process in random order for diversity
            for idx in list(code_set):
                # Check if all neighbors are covered by >1 codeword
                if np.min(counts[spheres_list[idx]]) > 1:
                    code_set.remove(idx)
                    counts[spheres_list[idx]] -= 1
                    changed = True
        
        return list(code_set), counts

    def is_valid_swap(c_remove, c_add, code_set, counts):
        """
        Check if replacing c_remove with c_add keeps the code valid using bitmask operations.
        """
        if c_remove not in code_set or c_add in code_set:
            return False
        
        # Find critical words: those in Sphere(c_remove) with count == 1
        # Use bitmask to find words with count 1 in sphere_remove
        sphere_remove_mask = spheres_mask[c_remove]
        sphere_add_mask = spheres_mask[c_add]
        
        # For each word in sphere_remove, check if count == 1 and if it's covered by c_add
        for w in spheres_list[c_remove]:
            if counts[w] == 1:
                # Check if w is in sphere_add using bitmask
                if not (sphere_add_mask >> w) & 1:
                    return False
        return True

    def find_valid_swaps(c_remove, current_set, counts, max_candidates=None):
        """
        Find all valid c_add values for a given c_remove using CSP-style filtering.
        Uses arc consistency to prune candidates.
        """
        critical_words = [w for w in spheres_list[c_remove] if counts[w] == 1]
        if not critical_words:
            return []
        
        # Start with candidates from the first critical word
        candidates = set(coverers[critical_words[0]])
        candidates -= current_set
        
        # Apply arc consistency: for each remaining critical word, intersect with its coverers
        for w in critical_words[1:]:
            if not candidates:
                break
            # Keep only candidates that cover w
            filtered = set()
            for c in candidates:
                if (coverers_mask[w] >> c) & 1:
                    filtered.add(c)
            candidates = filtered
        
        if max_candidates and len(candidates) > max_candidates:
            # Randomly sample if too many
            candidates = set(random.sample(list(candidates), max_candidates))
        
        return list(candidates)

    # --- 4. Main Search Loop ---
    
    rng = random.Random(42)
    
    # Start with initial code
    current_code, current_counts = reduce_code(initial_indices)
    best_code = current_code[:]
    best_counts = current_counts.copy()
    best_len = len(best_code)
    
    all_indices = list(range(total_words))
    
    no_improve_count = 0
    max_no_improve = 30
    
    while time.time() < deadline:
        # 1. Reduce
        current_code, current_counts = reduce_code(current_code, current_counts)
        
        if len(current_code) < best_len:
            best_len = len(current_code)
            best_code = current_code[:]
            best_counts = current_counts.copy()
            no_improve_count = 0
            if best_len <= 71:
                break
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Perturb with CSP-guided random restart
            if current_code:
                current_set = set(current_code)
                # Try multiple random perturbations
                found_valid = False
                for _ in range(50):
                    c_remove = rng.choice(current_code)
                    # Find valid swaps using CSP filtering
                    valid_adds = find_valid_swaps(c_remove, current_set, current_counts, max_candidates=20)
                    if valid_adds:
                        c_add = rng.choice(valid_adds)
                        current_set.remove(c_remove)
                        current_set.add(c_add)
                        current_counts[spheres_list[c_remove]] -= 1
                        current_counts[spheres_list[c_add]] += 1
                        current_code = list(current_set)
                        found_valid = True
                        no_improve_count = 0
                        break
                
                if not found_valid:
                    # Reset to best code
                    current_code = best_code[:]
                    current_counts = best_counts.copy()
                    no_improve_count = 0
            continue

        # 2. Try to find a beneficial swap using CSP-guided search
        improved = False
        current_set = set(current_code)
        
        candidates = current_code[:]
        rng.shuffle(candidates)
        
        for c_remove in candidates:
            if c_remove not in current_set:
                continue
            
            # Use CSP to find valid c_add candidates
            valid_adds = find_valid_swaps(c_remove, current_set, current_counts, max_candidates=100)
            
            if not valid_adds:
                continue
            
            # Try each valid swap
            for c_add in valid_adds:
                if c_add in current_set:
                    continue
                
                # Perform swap
                current_set.remove(c_remove)
                current_set.add(c_add)
                current_counts[spheres_list[c_remove]] -= 1
                current_counts[spheres_list[c_add]] += 1
                
                # Try to reduce
                new_code, new_counts = reduce_code(list(current_set), current_counts)
                
                if len(new_code) < len(current_code):
                    current_code = new_code
                    current_counts = new_counts
                    current_set = set(current_code)
                    improved = True
                    break
                else:
                    # Revert
                    current_set.remove(c_add)
                    current_set.add(c_remove)
                    current_counts[spheres_list[c_add]] -= 1
                    current_counts[spheres_list[c_remove]] += 1
            
            if improved:
                break
            if time.time() > deadline - 5.0:
                break

    # Final verification
    final_counts = np.zeros(total_words, dtype=np.int32)
    for idx in best_code:
        final_counts[spheres_list[idx]] += 1
    if np.any(final_counts == 0):
        return [list(idx_to_word(idx)) for idx in initial_indices]
    
    return [list(idx_to_word(idx)) for idx in best_code]
