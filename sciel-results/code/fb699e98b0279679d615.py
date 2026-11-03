import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    # Evaluator kills at 150s; leave safety margin
    deadline = start_time + 142.0

    total_space = q ** n

    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Precompute neighbor lists (sparse representation)
    # Each word covers itself + n*(q-1) neighbors
    neighbors = []
    for i in range(total_space):
        nbrs = [i]
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                nbrs.append(i + diff * powers[pos])
        neighbors.append(nbrs)

    best_size = total_space
    best_selected = list(range(total_space))

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    def polish(selected, time_limit):
        """Remove redundant codewords iteratively."""
        selected = selected[:]
        if not selected:
            return selected

        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            for nb in neighbors[w]:
                counts[nb] += 1

        changed = True
        while changed and time.time() < time_limit:
            changed = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                # Check if any neighbor is covered only by w
                # If min(counts[neighbors[w]]) > 1, w is redundant
                if np.min(counts[neighbors[w]]) > 1:
                    # Remove w
                    for nb in neighbors[w]:
                        counts[nb] -= 1
                    selected.pop(idx)
                    changed = True

        return selected

    def local_search(selected, time_limit):
        """Try to reduce the set by swapping out one word for another."""
        selected = selected[:]
        if not selected:
            return selected

        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            for nb in neighbors[w]:
                counts[nb] += 1

        # Precompute for speed: which words cover which points?
        # Actually, we just need to check if a candidate covers the unique points of w.
        
        for _ in range(50):
            if time.time() > time_limit:
                break
            improved = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                
                # Find unique points covered by w
                unique_points = []
                for nb in neighbors[w]:
                    if counts[nb] == 1:
                        unique_points.append(nb)
                
                if not unique_points:
                    # w is redundant
                    for nb in neighbors[w]:
                        counts[nb] -= 1
                    selected.pop(idx)
                    improved = True
                    continue

                # Try to find a replacement w2 that covers all unique_points
                # and adds some new coverage (gain > 0 is ideal, but covering unique is mandatory)
                # We want to maximize new coverage gained by w2 not in w.
                
                best_gain = -1
                best_word = -1
                
                # Optimization: Only check words that cover at least one unique point?
                # Or just scan all? 2187 is small.
                for w2 in range(total_space):
                    if w2 == w:
                        continue
                    
                    # Check if w2 covers all unique points
                    covers_all = True
                    for up in unique_points:
                        # Check if up is in neighbors[w2]
                        # Since neighbors[w2] is a list, 'in' is O(K).
                        # To speed up, we could have a set for neighbors, but list is okay for 27 items.
                        if up not in neighbors[w2]:
                            covers_all = False
                            break
                    
                    if not covers_all:
                        continue

                    # Calculate gain: number of points in neighbors[w2] not in neighbors[w]
                    # and not already covered by others? 
                    # Actually, gain relative to current state:
                    # New points covered = points in w2 that are NOT covered by current selected \ {w}
                    # Current coverage counts include w. So counts[nb] > 1 means covered by others.
                    # If we remove w, points with counts==1 become uncovered.
                    # Points with counts>1 remain covered.
                    # So w2 must cover the unique points (counts==1).
                    # The "gain" is how many points with counts>1 does w2 cover? No, that's not gain.
                    # The gain is how many points does w2 cover that w does NOT cover?
                    # If w2 covers a point p that w also covers, and p is covered by others, it's fine.
                    # If w2 covers a point p that w does NOT cover, and p is currently covered by others, 
                    # adding w2 increases count, no harm.
                    # If w2 covers a point p that w does NOT cover, and p is NOT covered by others (counts[p]==0? No, space is covered), 
                    # wait, the space is fully covered by selected.
                    
                    # Let's look at the structure:
                    # We remove w. Points with counts==1 (unique to w) become uncovered.
                    # We add w2. w2 must cover those unique points.
                    # Does w2 cover any other points? Yes.
                    # Does this help reduce size? No, size stays same (swap).
                    # Does it help future removals? Yes, if w2 overlaps more with other codewords, 
                    # it makes other codewords more likely to be redundant.
                    # So we want w2 to have high overlap with existing codewords (excluding w).
                    # Overlap with existing = number of points in neighbors[w2] where counts > 1 (after removing w? No, before).
                    # If we remove w, counts for unique points drop to 0.
                    # For non-unique points, counts stay same or drop by 1 if w was the only other? No, counts>1 means at least 2.
                    # So removing w keeps counts >= 1 for non-unique points.
                    # We want w2 to cover points that are "hard" to cover?
                    
                    # Simple heuristic: Maximize overlap of w2 with other selected words.
                    # Overlap = count of nb in neighbors[w2] such that counts[nb] > 1 (before removal of w).
                    # If w2 overlaps heavily, it becomes redundant sooner.
                    
                    overlap = 0
                    for nb in neighbors[w2]:
                        if counts[nb] > 1:
                            overlap += 1
                    
                    if overlap > best_gain:
                        best_gain = overlap
                        best_word = w2

                if best_word >= 0:
                    # Perform swap
                    # Remove w
                    for nb in neighbors[w]:
                        counts[nb] -= 1
                    # Add w2
                    for nb in neighbors[best_word]:
                        counts[nb] += 1
                    selected[idx] = best_word
                    improved = True

            if not improved:
                break

        return selected

    def run_greedy(randomized=False, alpha=1.0):
        """Greedy set cover."""
        covered = np.zeros(total_space, dtype=bool)
        selected = []
        remaining_count = total_space

        while remaining_count > 0 and time.time() < deadline - 5:
            best_gain = -1
            best_word = -1
            candidates = []

            for w in range(total_space):
                # Calculate gain: how many uncovered points does w cover?
                # gain = sum(1 for nb in neighbors[w] if not covered[nb])
                # Using numpy for speed
                gain = np.sum(~covered[neighbors[w]])
                
                if randomized:
                    gain = gain + random.gauss(0, 1.0) * alpha
                
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
                    candidates = [w]
                elif gain == best_gain and not randomized:
                    candidates.append(w)

            if not candidates:
                break

            if randomized and len(candidates) > 1:
                best_word = random.choice(candidates)
            else:
                best_word = candidates[0]

            selected.append(best_word)
            covered[neighbors[best_word]] = True
            remaining_count -= best_gain # Approximate, but we check > 0

        return selected

    # Strategy 1: Standard greedy + polish
    if time.time() < deadline - 10:
        sol = run_greedy(randomized=False)
        polished = polish(sol[:], deadline - 10)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    # Strategy 2: Multiple randomized greedy restarts with polish
    num_restarts = 100 # Reduced count because greedy is faster now, but polish is also faster
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break

        sol = run_greedy(randomized=True, alpha=random.uniform(0.1, 0.5))
        if len(sol) < best_size + 5:  # Only polish if close
            polished = polish(sol[:], deadline - 15)
            if len(polished) < best_size:
                best_size = len(polished)
                best_selected = polished[:]

    # Strategy 3: Local search on best solution
    if time.time() < deadline - 5:
        ls = local_search(best_selected[:], deadline - 5)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Strategy 4: Polish + local search + polish again
    if time.time() < deadline - 3:
        polished = polish(best_selected[:], deadline - 3)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]
        if time.time() < deadline - 2:
            ls2 = local_search(best_selected[:], deadline - 2)
            if len(ls2) < best_size:
                best_size = len(ls2)
                best_selected = ls2[:]

    # Final polish
    if time.time() < deadline - 1:
        polished = polish(best_selected[:], deadline - 1)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
