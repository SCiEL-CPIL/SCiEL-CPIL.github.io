import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q != 3 or n != 6 or radius != 1:
        return _greedy_set_cover(q, n, radius, deadline)

    space_size = q ** n  # 729

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

    # Precompute neighbors
    neighbors = []
    for i in range(space_size):
        w = idx_to_word(i)
        nbs = [i]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val != orig:
                    w[pos] = new_val
                    nbs.append(word_to_idx(w))
            w[pos] = orig
        neighbors.append(nbs)
    
    # Convert to numpy arrays for speed
    neighbors_np = [np.array(nb, dtype=np.int32) for nb in neighbors]
    
    # Initial construction: Tetracode sum
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    tails = list(itertools.product(range(q), repeat=n - 4))
    initial_codewords = [h + list(tail) for h in tetracode for tail in tails]
    initial_indices = [word_to_idx(w) for w in initial_codewords]
    
    def build_counts(indices):
        counts = np.zeros(space_size, dtype=np.int32)
        for ci in indices:
            counts[neighbors_np[ci]] += 1
        return counts

    def is_valid(counts):
        return np.all(counts >= 1)

    def local_search(indices, counts, deadline, max_iters=100000):
        """
        Local search with 1-deletion, 2-deletion, and 1-swap (to enable deletion).
        """
        current = list(indices)
        cnt = counts.copy()
        current_set = set(current)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            progress = False
            
            # 1. Try 1-deletion
            for i in range(len(current) - 1, -1, -1):
                ci = current[i]
                can_remove = True
                for nb in neighbors_np[ci]:
                    if cnt[nb] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for nb in neighbors_np[ci]:
                        cnt[nb] -= 1
                    current.pop(i)
                    current_set.discard(ci)
                    progress = True
                    break
            
            if progress:
                continue

            # 2. Try 2-deletion
            found_two = False
            # Sample pairs to avoid O(N^2) if N is large, but N~73 is small
            # We can iterate all pairs if N < 100
            if len(current) < 100:
                for i in range(len(current)):
                    if time.time() > deadline:
                        return current, cnt
                    ci = current[i]
                    for j in range(i + 1, len(current)):
                        cj = current[j]
                        valid = True
                        # Check if removing ci and cj leaves any word uncovered
                        # A word is uncovered if it was only covered by ci or cj
                        # and not by any other.
                        # Efficient check:
                        # Union of neighbors of ci and cj.
                        # For each nb in union, check if cnt[nb] > (1 if nb in N(ci) else 0) + (1 if nb in N(cj) else 0)
                        # Actually, simpler:
                        # Temporarily decrement, check min, increment back? No, too slow.
                        
                        # Check essential neighbors of ci
                        # If nb in N(ci) and cnt[nb]==1, then nb MUST be covered by cj (if nb in N(cj)) or some other.
                        # If nb in N(ci) and cnt[nb]==1 and nb not in N(cj), then we cannot remove ci.
                        # So:
                        # For all nb in N(ci): if cnt[nb]==1 and nb not in N(cj) -> invalid.
                        # For all nb in N(cj): if cnt[nb]==1 and nb not in N(ci) -> invalid.
                        
                        # We can precompute neighbor sets for O(1) check?
                        # Let's use sets for N(ci) and N(cj) for this check.
                        # Since N is small (19 words), sets are fine.
                        
                        set_ci = set(neighbors[ci])
                        set_cj = set(neighbors[cj])
                        
                        # Check ci's essential neighbors
                        for nb in neighbors_np[ci]:
                            if cnt[nb] == 1 and nb not in set_cj:
                                valid = False
                                break
                        if not valid:
                            continue
                            
                        # Check cj's essential neighbors
                        for nb in neighbors_np[cj]:
                            if cnt[nb] == 1 and nb not in set_ci:
                                valid = False
                                break
                                
                        if valid:
                            # Perform removal
                            for nb in neighbors_np[ci]:
                                cnt[nb] -= 1
                            for nb in neighbors_np[cj]:
                                cnt[nb] -= 1
                            current.pop(j)
                            current.pop(i)
                            current_set.discard(ci)
                            current_set.discard(cj)
                            found_two = True
                            break
                    if found_two:
                        break
            
            if found_two:
                continue

            # 3. Try 1-swap (to enable future deletions)
            # This is harder. We swap ci with cand.
            # This is only useful if it allows a deletion later.
            # For now, let's stick to deletions and rely on ILS perturbations for swaps.
            
            if not progress and not found_two:
                break

        return current, cnt

    def perturb(indices, k=2):
        """Randomly swap k codewords with random others."""
        current = list(indices)
        rng = random.Random()
        for _ in range(k):
            i = rng.randint(0, len(current) - 1)
            # Pick a random candidate not in current
            while True:
                cand = rng.randint(0, space_size - 1)
                if cand not in set(current):
                    break
            current[i] = cand
        return current

    # Main Loop: Iterated Local Search
    best_size = float('inf')
    best_code = None
    
    # Initial Greedy
    cw_indices = list(initial_indices)
    counts = build_counts(cw_indices)
    code, counts = local_search(cw_indices, counts, deadline - 20)
    
    if len(code) < best_size:
        best_size = len(code)
        best_code = code.copy()
        
    # ILS
    while time.time() < deadline - 5:
        # Perturb best code
        if best_code is None:
            break
        perturbed = perturb(best_code, k=2)
        
        # Local Search
        p_counts = build_counts(perturbed)
        new_code, new_counts = local_search(perturbed, p_counts, deadline - 5)
        
        if len(new_code) < best_size:
            best_size = len(new_code)
            best_code = new_code.copy()
            
        # If we found 72, we can stop early or continue to find 71
        if best_size <= 72:
             # Try a bit more to see if 71 is possible, but 72 is huge win
             pass

    # Final Polish
    if best_code is not None and time.time() < deadline - 2:
        final_counts = build_counts(best_code)
        final_code, _ = local_search(best_code, final_counts, deadline)
        if len(final_code) < best_size:
            best_code = final_code
            best_size = len(final_code)

    if best_code is None:
        return [idx_to_word(ci) for ci in initial_indices]
        
    return [idx_to_word(ci) for ci in best_code]

def _greedy_set_cover(q, n, radius, deadline):
    """Greedy set cover for general parameters."""
    all_words = list(itertools.product(range(q), repeat=n))
    total = q ** n

    def hamming_dist(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)

    coverage = {}
    for i, w in enumerate(all_words):
        covered = set()
        for j, v in enumerate(all_words):
            if hamming_dist(w, v) <= radius:
                covered.add(j)
        coverage[i] = covered

    uncovered = set(range(total))
    code = []

    while uncovered and time.time() < deadline - 5:
        best_i = -1
        best_count = 0
        for i in range(total):
            count = len(uncovered & coverage[i])
            if count > best_count:
                best_count = count
                best_i = i
            if time.time() > deadline - 5:
                break
        if best_i == -1:
            break
        code.append(list(all_words[best_i]))
        uncovered -= coverage[best_i]

    return code
