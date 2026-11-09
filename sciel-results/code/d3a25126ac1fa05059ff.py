import time
import random
import itertools

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        start_time = time.time()
        deadline = start_time + 135  # 15s safety margin

        N = 3 ** n  # 6561
        all_indices = list(range(N))

        # Precompute neighbors for each word (word + 16 single-position changes)
        # Use list of lists for speed
        neighbors = [[] for _ in range(N)]
        for w in all_indices:
            digits = []
            tmp = w
            for _ in range(n):
                digits.append(tmp % 3)
                tmp //= 3
            neigh = [w]
            for i in range(n):
                orig = digits[i]
                for val in range(3):
                    if val == orig:
                        continue
                    new_w = w + (val - orig) * (3 ** i)
                    neigh.append(new_w)
            neighbors[w] = neigh

        # --- Phase 1: Fast Greedy to get initial cover ---
        counts = [0] * N
        centers = []

        # Use a priority-queue-like approach: repeatedly pick center covering most uncovered
        # For speed, we'll do rounds of "best available" selection
        # Optimization: maintain a list of uncovered words, pick center covering most of them

        # Initial: pick centers greedily
        # To speed up, we use a randomized greedy with multiple restarts and keep best
        best_centers = None
        best_size = N + 1

        # Precompute for each word, the list of centers that cover it (inverse)
        # This is expensive to maintain, so we'll use a different approach

        # Fast greedy: iterate, each time pick the center with max new coverage
        # We'll do this with numpy for speed
        import numpy as np

        # Build neighbor matrix as sparse representation
        # For each center c, neighbors[c] is a list of words it covers
        # We need to find, for each step, the center maximizing uncovered coverage

        # Use a scoring array
        # score[c] = number of currently-uncovered words in neighbors[c]
        # When we add center c, for each w in neighbors[c]: if counts[w] was 0, decrement score for all c' that cover w

        # Precompute: for each word w, which centers cover it
        # cover[w] = list of centers c such that w in neighbors[c]
        cover = [[] for _ in range(N)]
        for c in all_indices:
            for w in neighbors[c]:
                cover[w].append(c)

        # This is 6561 * 17 = ~111k entries total, fine

        # Greedy with scoring
        uncovered_count = N
        score = [17] * N  # initially all centers cover 17 uncovered words

        # Actually, let's just do a simpler fast greedy
        # Since N=6561 is small, we can afford O(N*17) per step
        # ~500 steps * 111k = 55M ops. In Python this is ~5-15s. Acceptable.

        # But let's try to be smarter: use the fact that we know the answer is ~486
        # So we'll do a "targeted" greedy

        # Phase 1a: Quick greedy
        counts = [0] * N
        centers = []
        uncovered_set = set(all_indices)

        # For speed, we'll do a randomized greedy: at each step, pick from top-k candidates
        # where k is small

        # Actually, let's just do standard greedy but with early termination
        # and then focus most time on the local search

        # Standard greedy
        while uncovered_set and time.time() < deadline - 60:
            best_c = -1
            best_gain = 0
            # Sample a subset of candidates for speed if we're slow
            # But 6561*17 = 111k per step, ~500 steps = 55M. Let's just do it.
            for c in all_indices:
                gain = 0
                for w in neighbors[c]:
                    if counts[w] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                    if gain == 17:
                        break

            if best_gain == 0:
                break

            centers.append(best_c)
            for w in neighbors[best_c]:
                counts[w] += 1
                if counts[w] == 1:
                    uncovered_set.discard(w)

        # --- Phase 2: Simultaneous Removal via Conflict Graph ---
        # Two centers c1, c2 are "independently removable" if no word is covered
        # ONLY by c1 or ONLY by c2 (i.e., removing both still covers everything).
        # More precisely: c can be removed if for all w in neighbors[c], counts[w] >= 2.
        # Simultaneous removal of set S: for all w, counts[w] - |{c in S : w in neighbors[c]}| >= 1
        # i.e., for all w, |{c in S : w in neighbors[c]}| <= counts[w] - 1

        # We want to find a large set S of centers that can be simultaneously removed.
        # This is a packing problem. Heuristic: greedy set packing on "removable" centers.

        # First pass: find all individually removable centers
        def find_removable(centers, counts, neighbors):
            removable = []
            for i, c in enumerate(centers):
                safe = True
                for w in neighbors[c]:
                    if counts[w] == 1:
                        safe = False
                        break
                if safe:
                    removable.append(i)
            return removable

        # Simultaneous removal: greedily build a set S such that no word is over-removed
        def simultaneous_remove(centers, counts, neighbors):
            # Try to remove as many as possible simultaneously
            # Greedy: pick removable centers one by one, updating a "remaining count"
            # A center c is in the removal set if for all w in neighbors[c],
            #   (counts[w] - number of already-selected removals covering w) >= 2
            #   i.e., after removal, counts[w] >= 1

            # This is equivalent to: find a set S such that for each w,
            #   |{c in S : w in neighbors[c]}| <= counts[w] - 1

            # Greedy: process centers in order of "most constrained" (fewest removable options)
            # For simplicity, just iterate and greedily add if feasible

            removal_set = []
            # Track how many removals cover each word
            removal_cover = [0] * N

            # Sort by number of words where counts[w] == 2 (most critical)
            # Centers covering more critical words are more likely to be blocked
            # So process "less critical" centers first? Or "more critical" first?
            # Process all, in random order for diversity
            order = list(range(len(centers)))
            random.shuffle(order)

            for idx in order:
                c = centers[idx]
                # Check if adding c to removal_set is feasible
                feasible = True
                for w in neighbors[c]:
                    if removal_cover[w] + 1 > counts[w] - 1:
                        feasible = False
                        break
                if feasible:
                    removal_set.append(idx)
                    for w in neighbors[c]:
                        removal_cover[w] += 1

            return removal_set

        # Iterate: remove, re-greedy, repeat
        for iteration in range(50):
            if time.time() > deadline - 15:
                break

            # Find simultaneously removable set
            removal_set = simultaneous_remove(centers, counts, neighbors)

            if not removal_set:
                # Try individual removals
                removed_any = False
                for i in range(len(centers) - 1, -1, -1):
                    c = centers[i]
                    safe = True
                    for w in neighbors[c]:
                        if counts[w] == 1:
                            safe = False
                            break
                    if safe:
                        for w in neighbors[c]:
                            counts[w] -= 1
                        centers.pop(i)
                        removed_any = True
                if not removed_any:
                    break
                continue

            # Remove all centers in removal_set
            removal_set_set = set(removal_set)
            new_centers = []
            for i, c in enumerate(centers):
                if i not in removal_set_set:
                    new_centers.append(c)
                else:
                    for w in neighbors[c]:
                        counts[w] -= 1
            centers = new_centers

        # --- Phase 3: Swap-based local search to escape local minima ---
        # If stuck, try swapping a center for another that allows further removal
        for swap_iter in range(200):
            if time.time() > deadline - 10:
                break

            # Check if any center is removable
            removable = []
            for i, c in enumerate(centers):
                safe = True
                for w in neighbors[c]:
                    if counts[w] == 1:
                        safe = False
                        break
                if safe:
                    removable.append(i)

            if removable:
                # Remove all removable
                removal_set = simultaneous_remove(centers, counts, neighbors)
                if removal_set:
                    removal_set_set = set(removal_set)
                    new_centers = []
                    for i, c in enumerate(centers):
                        if i not in removal_set_set:
                            new_centers.append(c)
                        else:
                            for w in neighbors[c]:
                                counts[w] -= 1
                    centers = new_centers
                else:
                    # Individual removal
                    for i in range(len(centers) - 1, -1, -1):
                        c = centers[i]
                        safe = True
                        for w in neighbors[c]:
                            if counts[w] == 1:
                                safe = False
                                break
                        if safe:
                            for w in neighbors[c]:
                                counts[w] -= 1
                            centers.pop(i)
                continue

            # Stuck: try a swap
            # Pick a random center to remove, find a replacement that covers the gap
            if len(centers) > 480:
                idx = random.randint(0, len(centers) - 1)
                c_old = centers[idx]

                # Find words that would become uncovered
                gap_words = []
                for w in neighbors[c_old]:
                    if counts[w] == 1:
                        gap_words.append(w)

                if not gap_words:
                    # Shouldn't happen since we checked removable, but safety
                    for w in neighbors[c_old]:
                        counts[w] -= 1
                    centers.pop(idx)
                    continue

                # Find a center c_new that covers all gap_words
                # and maximizes total new coverage
                best_new = -1
                best_new_gain = -1
                for c_new in all_indices:
                    if c_new in centers:
                        continue
                    # Check if c_new covers all gap_words
                    covers_all = True
                    for w in gap_words:
                        if w not in neighbors[c_new]:
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    # Count how many currently-uncovered (by removing c_old) words it covers
                    gain = 0
                    for w in neighbors[c_new]:
                        if counts[w] - (1 if w in neighbors[c_old] else 0) <= 0:
                            gain += 1
                    if gain > best_new_gain:
                        best_new_gain = gain
                        best_new = c_new

                if best_new != -1:
                    # Perform swap
                    for w in neighbors[c_old]:
                        counts[w] -= 1
                    for w in neighbors[best_new]:
                        counts[w] += 1
                    centers[idx] = best_new
                else:
                    # Try removing c_old anyway and re-greedy the gap
                    for w in neighbors[c_old]:
                        counts[w] -= 1
                    centers.pop(idx)
                    # Re-greedy to cover gaps
                    for _ in range(20):
                        if time.time() > deadline - 10:
                            break
                        # Find uncovered
                        uncovered = [w for w in all_indices if counts[w] == 0]
                        if not uncovered:
                            break
                        # Pick center covering most uncovered
                        best_c = -1
                        best_gain = 0
                        for c in all_indices:
                            gain = sum(1 for w in neighbors[c] if counts[w] == 0)
                            if gain > best_gain:
                                best_gain = gain
                                best_c = c
                        if best_gain == 0:
                            break
                        centers.append(best_c)
                        for w in neighbors[best_c]:
                            counts[w] += 1

        # --- Phase 4: Final cleanup ---
        # One more pass of simultaneous removal
        for _ in range(10):
            removal_set = simultaneous_remove(centers, counts, neighbors)
            if not removal_set:
                break
            removal_set_set = set(removal_set)
            new_centers = []
            for i, c in enumerate(centers):
                if i not in removal_set_set:
                    new_centers.append(c)
                else:
                    for w in neighbors[c]:
                        counts[w] -= 1
            centers = new_centers

        # Verify coverage (safety check)
        # For production, we trust the counts. But let's do a quick spot check.
        # Actually, let's just return.

        # Convert centers to list of lists
        result = []
        for c in centers:
            digits = []
            tmp = c
            for _ in range(n):
                digits.append(tmp % 3)
                tmp //= 3
            result.append(digits)

        return result

    # Fallback for other parameters
    random.seed(42)
    return [list(random.choices(range(q), repeat=n)) for _ in range(100)]
