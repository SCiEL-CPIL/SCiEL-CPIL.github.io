"""Covering code K_q(n,1) via greedy set-cover + local search (2-opt / swap)."""
import time
import random
import itertools
import numpy as np

_DEADLINE = 115.0  # seconds; keep safety margin under the 120 s kill


def _time_left(deadline):
    return deadline - time.time()


def construct(q, n, radius):
    deadline = time.time() + _DEADLINE
    rng = random.Random(12345)

    if radius != 1:
        # Fallback: full space (only valid for radius >= n-1 trivially, but keep valid)
        return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Precompute all words as integers in base-q
    total = q ** n
    # For speed, represent words as ints; helper to get digit at position
    # We'll build a neighbor table: for each word, the set of words within distance 1.
    # But total can be 2187 (q=3,n=7), manageable.

    # Build list of all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # For each word, compute the ball of radius 1 (the word + all words at distance 1)
    # ball[word_idx] = frozenset of covered word indices
    balls = []
    for w in all_words:
        covered = [word_to_idx[w]]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                covered.append(word_to_idx[tuple(nw)])
        balls.append(frozenset(covered))

    # Greedy set cover: repeatedly pick the word whose ball covers the most uncovered words.
    uncovered = set(range(total))
    chosen = []
    # Precompute ball sizes for tie-breaking
    # Use a heap-free greedy: each step scan all candidates (total is small: 2187)
    # But we can do better: maintain for each candidate its current coverage count.
    # Since total is small, simple greedy is fine.

    # Optimize: represent uncovered as a set; for each candidate compute |ball ∩ uncovered|
    # To speed up, we can precompute ball as a set and just intersect.

    while uncovered and _time_left(deadline) > 5.0:
        best_idx = -1
        best_count = -1
        for idx in range(total):
            if idx in chosen_set if 'chosen_set' in dir() else True:
                pass
            # Compute coverage of idx among uncovered
            cov = len(balls[idx] & uncovered)
            if cov > best_count:
                best_count = cov
                best_idx = idx
                if cov == len(uncovered):
                    break
        if best_idx < 0:
            break
        chosen.append(all_words[best_idx])
        uncovered -= balls[best_idx]

    # Local search: try to remove redundant codewords
    def _is_redundant(chosen_list, balls_list, target_idx):
        # Check if removing target_idx still covers everything
        # covered_by_others = union of balls of all chosen except target
        # Must equal full space
        # Efficient: for each word, check if it's covered by some other chosen
        # Use: a word is NOT covered by others iff it's only covered by target among chosen.
        # Build: for each word, count how many chosen cover it.
        counts = np.zeros(total, dtype=np.int32)
        for i, w in enumerate(chosen_list):
            idx = word_to_idx[w]
            if i == target_idx:
                continue
            for c in balls_list[idx]:
                counts[c] += 1
        # Any word with count 0 is uncovered -> redundant check fails
        return np.all(counts > 0)

    # Simpler redundancy: a codeword w is redundant if every word in ball(w) is also in ball(w') for some other chosen w'
    # Equivalent: union of other balls covers the full space.
    def _remove_redundant(chosen_list):
        changed = True
        while changed and _time_left(deadline) > 3.0:
            changed = False
            n_chosen = len(chosen_list)
            for i in range(n_chosen):
                w = chosen_list[i]
                idx = word_to_idx[w]
                # Check: is ball(idx) subset of union of other balls?
                # i.e., is every word in ball(idx) covered by some other chosen?
                ball_set = balls[idx]
                # Compute union of other balls
                other_cover = set()
                for j in range(n_chosen):
                    if j == i:
                        continue
                    other_cover |= balls[word_to_idx[chosen_list[j]]]
                if ball_set <= other_cover:
                    chosen_list.pop(i)
                    changed = True
                    break
        return chosen_list

    chosen_list = list(chosen)
    chosen_list = _remove_redundant(chosen_list)

    # Local search: 2-opt swap — try replacing one codeword with another to reduce count
    # or try adding one and removing two.
    # Strategy: iteratively try to find a codeword whose removal is "almost" redundant
    # (covers few unique words), then see if we can add a different codeword that covers
    # those unique words plus more, allowing removal of another codeword.

    # Simpler: try all single replacements (remove i, add j) and keep if still covers.
    # Also try: remove i, add j, then remove k (net -1).

    def _covers_all(chosen_list):
        cover = set()
        for w in chosen_list:
            cover |= balls[word_to_idx[w]]
        return cover == set(range(total)) or len(cover) == total

    # Use numpy for speed in coverage checks
    def _cover_mask(chosen_list):
        mask = np.zeros(total, dtype=bool)
        for w in chosen_list:
            idx = word_to_idx[w]
            for c in balls[idx]:
                mask[c] = True
        return mask

    def _is_valid(chosen_list):
        mask = _cover_mask(chosen_list)
        return mask.all()

    # Phase 2: try to reduce by swapping
    # For each pair (i, j) where i is in chosen, j not in chosen:
    #   try chosen - i + j, check validity. If valid, accept (same size, but may enable further reductions)
    # More useful: try chosen - i + j - k (net -1)

    # Given time constraints, do a simple iterative improvement:
    # Repeatedly try to remove one codeword and add one different codeword that allows
    # removing a third.

    # Actually, the most effective simple local search for set cover is:
    # 1. Remove redundant (done)
    # 2. For each codeword i, compute its "unique coverage" (words only it covers)
    #    If unique coverage is small, try to find a codeword j (not chosen) whose ball
    #    covers all of i's unique coverage. Then remove i and add j. This keeps size same
    #    but may allow further removals.
    # 3. Repeat removal of redundant.

    # Let's implement a combined local search:
    improved = True
    while improved and _time_left(deadline) > 3.0:
        improved = False
        chosen_list = _remove_redundant(chosen_list)
        
        # Compute unique coverage for each chosen codeword
        n_ch = len(chosen_list)
        if n_ch == 0:
            break
        
        # For each chosen i, find words in ball(i) not covered by any other chosen
        for i in range(n_ch):
            if _time_left(deadline) < 3.0:
                break
            w_i = chosen_list[i]
            idx_i = word_to_idx[w_i]
            ball_i = balls[idx_i]
            
            # Find unique coverage of i
            unique = set()
            for c in ball_i:
                # Check if c is covered by any other chosen
                covered_by_other = False
                for j in range(n_ch):
                    if j == i:
                        continue
                    if c in balls[word_to_idx[chosen_list[j]]]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    unique.add(c)
            
            if not unique:
                continue
            
            # Try to find a codeword j (not in chosen) whose ball covers all of unique
            # and ideally covers more uncovered-like words
            # For efficiency, only try candidates that cover at least one unique word
            candidates = []
            for j_idx in range(total):
                if all_words[j_idx] in chosen_list:
                    continue
                ball_j = balls[j_idx]
                if ball_j >= unique:  # covers all unique words
                    candidates.append(j_idx)
            
            if candidates:
                # Pick the candidate that maximizes additional coverage
                best_j = None
                best_score = -1
                for j_idx in candidates:
                    # Score: how many words in ball_j are not in any other chosen's ball
                    # (i.e., how much "new" coverage j adds beyond what i already provides)
                    ball_j = balls[j_idx]
                    # New coverage = ball_j - union of other chosen balls
                    # Approximate: just use |ball_j| as tiebreaker, or count words only covered by j
                    # For simplicity, pick the one with largest ball intersection with current uncovered... 
                    # Actually all are covered, so just pick first or random
                    best_j = j_idx
                    break  # take first valid candidate for speed
                
                if best_j is not None:
                    # Replace i with best_j
                    chosen_list[i] = all_words[best_j]
                    improved = True
                    break  # restart the loop

    # Final redundancy removal
    chosen_list = _remove_redundant(chosen_list)

    # Verify validity
    if not _is_valid(chosen_list):
        # Fallback to greedy-only result
        chosen_list = list(chosen)
        chosen_list = _remove_redundant(chosen_list)

    return [list(w) for w in chosen_list]
