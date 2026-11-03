import time
import random

def construct(q, n, radius):
    start_time = time.time()
    limit = 140.0  # Safety margin for 150s limit

    total_words = q ** n

    # Precompute powers of q for index conversion
    powers = [q ** i for i in range(n)]

    def get_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % q
            idx //= q
        return digits

    # Precompute neighbor masks
    neighbor_masks = [0] * total_words

    for idx in range(total_words):
        mask = 1 << idx
        for i in range(n):
            original = (idx // powers[i]) % q
            base = idx - original * powers[i]
            for new_val in range(q):
                if new_val != original:
                    neighbor_idx = base + new_val * powers[i]
                    mask |= (1 << neighbor_idx)
        neighbor_masks[idx] = mask

    def greedy_cover(use_random):
        uncovered = (1 << total_words) - 1
        solution = []
        while uncovered:
            if time.time() - start_time > limit:
                break
            best_word = -1
            max_covered = 0
            # Collect candidates with max coverage
            if use_random:
                # Random tie-breaking
                # To be efficient, we can just pick the first one with max coverage,
                # but shuffle the order of iteration?
                # Or collect all max and pick random.
                # Collecting all is O(total_words).
                # Let's just shuffle the range.
                order = list(range(total_words))
                random.shuffle(order)
            else:
                order = range(total_words)
                
            for w in order:
                count = (neighbor_masks[w] & uncovered).bit_count()
                if count > max_covered:
                    max_covered = count
                    best_word = w
                    if use_random:
                        # Don't break, keep looking for others with same count?
                        # Or just break?
                        # If we break, it's not truly random tie-breaking.
                        # Let's collect all.
                        pass
            
            # This approach is getting complicated.
            # Simpler: Just use the standard greedy.
            # The variation comes from the local search or different initializations.
            # Actually, standard greedy is very strong.
            # Let's stick to the single greedy run but ensure local search is optimal.
            pass
        
        return solution

    # Let's just use the single greedy run as before, it's likely optimal enough.
    # The main gain was from the faster local search.
    
    # Revert to single greedy
    uncovered = (1 << total_words) - 1
    solution_indices = []

    while uncovered:
        if time.time() - start_time > limit:
            break

        best_word = -1
        max_covered = 0

        for w in range(total_words):
            count = (neighbor_masks[w] & uncovered).bit_count()
            if count > max_covered:
                max_covered = count
                best_word = w

        if best_word == -1:
            break

        solution_indices.append(best_word)
        uncovered &= ~neighbor_masks[best_word]

    # Local Search: Use persistent coverage counts for efficient removal
    new_solution = list(solution_indices)
    
    # Build initial cov
    cov = [0] * total_words
    for w in new_solution:
        mask = neighbor_masks[w]
        m = mask
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            cov[v] += 1
            m ^= lb

    def is_redundant(w, cov):
        mask = neighbor_masks[w]
        m = mask
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            if cov[v] == 1:
                return False
            m ^= lb
        return True

    # Phase 1: Greedy removal using persistent coverage counts
    changed = True
    while changed and (time.time() - start_time < limit):
        changed = False
        for i in range(len(new_solution) - 1, -1, -1):
            if time.time() - start_time > limit:
                break
            w = new_solution[i]
            if is_redundant(w, cov):
                # Remove w
                mask = neighbor_masks[w]
                m = mask
                while m:
                    lb = m & (-m)
                    v = lb.bit_length() - 1
                    cov[v] -= 1
                    m ^= lb
                new_solution.pop(i)
                changed = True
                break  # Restart loop as indices changed

    # Convert indices back to lists of digits
    return [get_digits(idx) for idx in new_solution]
