import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    DL = t0 + 112

    # For q=3, n=8, radius=1:
    # total_words = 3^8 = 6561
    # cover_size = 1 + 8*2 = 17
    total_words = q ** n
    num_coords = n
    alphabet = q
    cover_size = 1 + num_coords * (alphabet - 1)

    def idx_to_digits(idx):
        digits = [0] * num_coords
        for i in range(num_coords - 1, -1, -1):
            digits[i] = idx % alphabet
            idx //= alphabet
        return digits

    def digits_to_idx(digits):
        idx = 0
        for d in digits:
            idx = idx * alphabet + d
        return idx

    # Precompute coverage and reverse coverage arrays
    # coverage[cw_idx, j] = word_idx covered by codeword cw_idx at position j
    # covers_word[w_idx, j] = codeword_idx that covers word w_idx at position j
    coverage = np.zeros((total_words, cover_size), dtype=np.int32)
    covers_word = np.zeros((total_words, cover_size), dtype=np.int32)

    for idx in range(total_words):
        digits = idx_to_digits(idx)
        coverage[idx, 0] = idx
        covers_word[idx, 0] = idx
        j = 1
        for pos in range(num_coords):
            for val in range(alphabet):
                if val == digits[pos]:
                    continue
                new_digits = digits[:]
                new_digits[pos] = val
                new_idx = digits_to_idx(new_digits)
                coverage[idx, j] = new_idx
                # The word new_idx is covered by idx (as a neighbor)
                # Also idx is covered by new_idx (as a neighbor)
                covers_word[idx, j] = idx
                j += 1
        # For covers_word, we need to fill in the reverse: for each idx,
        # the words that cover idx are idx itself and all its neighbors
        for pos in range(num_coords):
            for val in range(alphabet):
                if val == digits[pos]:
                    continue
                new_digits = digits[:]
                new_digits[pos] = val
                new_idx = digits_to_idx(new_digits)
                # new_idx covers idx
                # Find position j in covers_word[new_idx] where we can put idx
                # Actually, covers_word[new_idx, j] should include idx
                # But we already filled covers_word[new_idx, 0] = new_idx
                # and covers_word[new_idx, j] for j>=1 are the neighbors of new_idx
                # idx is one of those neighbors of new_idx
                # We need to find the right j. This is tricky.
                # Let's just rebuild covers_word properly.
    
    # Rebuild covers_word properly
    covers_word = np.zeros((total_words, cover_size), dtype=np.int32)
    for idx in range(total_words):
        digits = idx_to_digits(idx)
        covers_word[idx, 0] = idx  # idx covers itself
        j = 1
        for pos in range(num_coords):
            for val in range(alphabet):
                if val == digits[pos]:
                    continue
                new_digits = digits[:]
                new_digits[pos] = val
                new_idx = digits_to_idx(new_digits)
                # new_idx is a neighbor of idx, so new_idx covers idx
                covers_word[idx, j] = new_idx
                j += 1

    # Precompute: for each codeword, which words does it cover?
    # That's coverage[cw, :]

    # --- Greedy Set Cover ---
    uncovered = np.ones(total_words, dtype=np.uint8)
    count = np.zeros(total_words, dtype=np.int32)

    # Initialize: count[cw] = number of words covered by cw (all words initially uncovered)
    # Since every cw covers cover_size words, count[cw] = cover_size for all cw
    count[:] = cover_size

    codewords = []
    num_covered = 0

    while num_covered < total_words:
        if time.time() > DL:
            break
        best_cw = int(np.argmax(count))
        if count[best_cw] <= 0:
            break
        codewords.append(best_cw)
        for j in range(cover_size):
            w_idx = coverage[best_cw, j]
            if uncovered[w_idx]:
                uncovered[w_idx] = 0
                num_covered += 1
                for k in range(cover_size):
                    cw_idx = covers_word[w_idx, k]
                    count[cw_idx] -= 1

    # If greedy didn't finish (shouldn't happen), add remaining
    if num_covered < total_words:
        # Add all remaining words as codewords (fallback)
        for w_idx in range(total_words):
            if uncovered[w_idx]:
                codewords.append(w_idx)
                for j in range(cover_size):
                    wi = coverage[w_idx, j]
                    uncovered[wi] = 0
        num_covered = total_words

    # --- Local Search: Minimize ---
    # Build cover_count: for each word, how many codewords in current set cover it
    cw_set = set(codewords)
    cover_count = np.zeros(total_words, dtype=np.int32)
    for cw in cw_set:
        for j in range(cover_size):
            w_idx = coverage[cw, j]
            cover_count[w_idx] += 1

    # Iterative removal: remove a codeword if all words it covers are covered by >=2
    improved = True
    removal_round = 0
    while improved and time.time() < DL - 5:
        improved = False
        removal_round += 1
        # Find critical words (covered by exactly 1 codeword)
        critical_mask = (cover_count == 1)
        
        # A codeword is removable if it doesn't cover any critical word
        # Build a "blocked" mask: codewords that cover at least one critical word
        blocked = np.zeros(total_words, dtype=np.uint8)
        for w_idx in np.where(critical_mask)[0]:
            for k in range(cover_size):
                cw_idx = covers_word[w_idx, k]
                blocked[cw_idx] = 1

        # Try each codeword in the set
        for cw in list(cw_set):
            if blocked[cw]:
                continue
            # Check if all words covered by cw have cover_count > 1
            removable = True
            for j in range(cover_size):
                w_idx = coverage[cw, j]
                if cover_count[w_idx] <= 1:
                    removable = False
                    break
            if removable:
                cw_set.remove(cw)
                for j in range(cover_size):
                    w_idx = coverage[cw, j]
                    cover_count[w_idx] -= 1
                improved = True
                break  # Restart

    # --- Perturbation-based local search (ILS) ---
    # Strategy: try to swap codewords to reduce size
    # Or: remove k codewords randomly, then greedily add back to cover holes
    best_size = len(cw_set)
    
    if time.time() < DL - 20:
        # Multiple perturbation attempts
        num_perturbs = 0
        while time.time() < DL - 10:
            num_perturbs += 1
            
            # Random perturbation: remove 2-5 random codewords
            num_remove = random.randint(2, min(5, len(cw_set) - 1))
            if len(cw_set) <= num_remove + 1:
                break
                
            removed = random.sample(list(cw_set), num_remove)
            for cw in removed:
                cw_set.remove(cw)
                for j in range(cover_size):
                    w_idx = coverage[cw, j]
                    cover_count[w_idx] -= 1
            
            # Now fix holes: while there are uncovered words, add best greedy codeword
            # Repeatedly add the codeword that covers the most uncovered words
            max_add = 50  # Safety limit on additions per perturbation
            for _ in range(max_add):
                if time.time() > DL - 5:
                    break
                # Find uncovered words
                uncovered_mask = (cover_count == 0)
                if not np.any(uncovered_mask):
                    break
                
                # For each candidate codeword not in set, compute how many uncovered words it covers
                # To be fast, only consider codewords that cover at least one uncovered word
                # Use the covers_word structure
                gain = np.zeros(total_words, dtype=np.int32)
                for w_idx in np.where(uncovered_mask)[0]:
                    for k in range(cover_size):
                        cw_idx = covers_word[w_idx, k]
                        if cw_idx not in cw_set:
                            gain[cw_idx] += 1
                
                # Find best candidate
                if gain.max() <= 0:
                    break
                best_add = int(np.argmax(gain))
                if best_add in cw_set:
                    break
                cw_set.add(best_add)
                for j in range(cover_size):
                    w_idx = coverage[best_add, j]
                    cover_count[w_idx] += 1
            
            # Check if valid
            if np.any(cover_count == 0):
                # Invalid, restore
                for cw in removed:
                    cw_set.add(cw)
                    for j in range(cover_size):
                        w_idx = coverage[cw, j]
                        cover_count[w_idx] += 1
                # Also remove any extra added codewords
                # We need to track what was added
                # This is getting complex. Let's just abort perturbation if invalid
                continue
            
            # If valid and better, accept
            if len(cw_set) < best_size:
                best_size = len(cw_set)
            
            # Try removal again after perturbation
            for _ in range(50):
                if time.time() > DL - 5:
                    break
                improved2 = False
                critical_mask2 = (cover_count == 1)
                blocked2 = np.zeros(total_words, dtype=np.uint8)
                for w_idx in np.where(critical_mask2)[0]:
                    for k in range(cover_size):
                        cw_idx = covers_word[w_idx, k]
                        blocked2[cw_idx] = 1
                for cw in list(cw_set):
                    if blocked2[cw]:
                        continue
                    removable2 = True
                    for j in range(cover_size):
                        w_idx = coverage[cw, j]
                        if cover_count[w_idx] <= 1:
                            removable2 = False
                            break
                    if removable2:
                        cw_set.remove(cw)
                        for j in range(cover_size):
                            w_idx = coverage[cw, j]
                            cover_count[w_idx] -= 1
                        improved2 = True
                        if len(cw_set) < best_size:
                            best_size = len(cw_set)
                        break
                if not improved2:
                    break

    # Final check: ensure validity
    if np.any(cover_count == 0):
        # Fallback: add all missing
        for w_idx in np.where(cover_count == 0)[0]:
            # Add a codeword that covers w_idx
            for k in range(cover_size):
                cw_idx = covers_word[w_idx, k]
                if cw_idx not in cw_set:
                    cw_set.add(cw_idx)
                    for j in range(cover_size):
                        wi = coverage[cw_idx, j]
                        cover_count[wi] += 1
                    break

    # Convert to result
    result = [idx_to_digits(cw) for cw in cw_set]
    
    # Ensure distinctness
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)
    
    return unique_result
