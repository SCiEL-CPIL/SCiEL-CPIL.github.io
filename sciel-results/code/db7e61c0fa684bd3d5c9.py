"""Covering code K_3(8,1): minimize codewords so every word is within Hamming distance 1."""
import time
import random
import numpy as np

_DEADLINE = 110.0
_START = [time.time()]

def _elapsed():
    return time.time() - _START[0]

def _time_left():
    return _DEADLINE - _elapsed()

def construct(q, n, radius):
    _START[0] = time.time()
    if (q, n, radius) != (3, 8, 1):
        import itertools
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    rng = random.Random(12345)
    N = 3 ** 8  # 6561

    # Precompute digits and idx conversions
    def idx_to_digits(i):
        d = [0] * 8
        for p in range(8):
            d[p] = i % 3
            i //= 3
        return d

    def digits_to_idx(d):
        v = 0
        mul = 1
        for p in range(8):
            v += d[p] * mul
            mul *= 3
        return v

    # Precompute balls as a 2D array: ball_matrix[i] contains the 17 indices in ball of i
    # Ball size = 1 + 8*2 = 17
    BALL_SIZE = 17
    ball_matrix = np.zeros((N, BALL_SIZE), dtype=np.int32)
    for i in range(N):
        d = idx_to_digits(i)
        ball_matrix[i, 0] = i
        idx = 1
        for p in range(8):
            orig = d[p]
            for nv in range(3):
                if nv != orig:
                    d[p] = nv
                    ball_matrix[i, idx] = digits_to_idx(d)
                    idx += 1
                    d[p] = orig

    # Precompute inverse coverage: for each word j, which codewords cover it
    # inv_matrix[j] = list of codeword indices i such that j is in ball_matrix[i]
    # We'll build this on the fly during search, but precompute the structure
    # Actually, let's precompute for each word j, the 17 words at distance 1 from j
    # That's already ball_matrix[j]! The ball of j contains j and its 16 neighbors.
    # A codeword i covers word j iff j is in ball_matrix[i] iff i is in ball_matrix[j]
    # (by symmetry of Hamming distance)
    # So inv_matrix = ball_matrix (same structure, symmetric)

    # Tetracode seed
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_set = []
    for h in tetracode:
        for t in tetracode:
            w = h + t
            seed_set.append(digits_to_idx(w))
    seed_set = list(set(seed_set))

    # Greedy fill with vectorized gain computation
    covered = np.zeros(N, dtype=np.int8)
    chosen = list(seed_set)
    for i in seed_set:
        covered[ball_matrix[i]] = 1

    def greedy_fill(chosen, covered, time_budget=20.0):
        t_start = time.time()
        while time.time() - t_start < time_budget and _time_left() > 20.0:
            # Vectorized: for each candidate i, gain = sum of (1 - covered[j]) for j in ball_matrix[i]
            # = BALL_SIZE - sum(covered[ball_matrix[i]])
            gains = BALL_SIZE - covered[ball_matrix].sum(axis=1)
            best_i = int(np.argmax(gains))
            if gains[best_i] <= 0:
                break
            chosen.append(best_i)
            covered[ball_matrix[best_i]] = 1
        return chosen, covered

    chosen, covered = greedy_fill(chosen, covered)
    
    # Additional greedy passes with different orderings to escape local optima
    for _ in range(3):
        if _time_left() < 20.0:
            break
        # Try adding codewords that maximize coverage of currently singly-covered words
        t_start = time.time()
        while time.time() - t_start < 5.0 and _time_left() > 15.0:
            # Find words with covcount == 1 (critical)
            covcount_temp = np.zeros(N, dtype=np.int32)
            for i in chosen:
                covcount_temp[ball_matrix[i]] += 1
            critical = np.where(covcount_temp == 1)[0]
            
            if len(critical) == 0:
                break
            
            # For each candidate not in chosen, compute how many critical words it covers
            chosen_set_temp = set(chosen)
            gains = np.zeros(N, dtype=np.int32)
            for c in critical:
                for nb in ball_matrix[c]:
                    if nb not in chosen_set_temp:
                        gains[nb] += 1
            
            best_i = int(np.argmax(gains))
            if gains[best_i] <= 0:
                break
            
            chosen.append(best_i)
            covered[ball_matrix[best_i]] = 1
        
        # Now try removals to undo any bad additions
        covcount_temp = np.zeros(N, dtype=np.int32)
        for i in chosen:
            covcount_temp[ball_matrix[i]] += 1
        
        for i in list(chosen):
            b = ball_matrix[i]
            if np.min(covcount_temp[b]) >= 2:
                covcount_temp[b] -= 1
                chosen.remove(i)
                covered[ball_matrix[i]] = 0
                # Recompute covered
                for j in ball_matrix[i]:
                    if covcount_temp[j] > 0:
                        covered[j] = 1

    if not covered.all():
        for i in range(N):
            if not covered[i]:
                chosen.append(i)
                covered[ball_matrix[i]] = 1

    # Build coverage count
    covcount = np.zeros(N, dtype=np.int32)
    for i in chosen:
        covcount[ball_matrix[i]] += 1

    # Local search: removals
    chosen_set = set(chosen)
    def try_removals(chosen_set, covcount, time_budget=15.0):
        t_start = time.time()
        improved = True
        while improved and time.time() - t_start < time_budget and _time_left() > 15.0:
            improved = False
            order = list(chosen_set)
            rng.shuffle(order)
            for i in order:
                if i not in chosen_set:
                    continue
                if np.min(covcount[ball_matrix[i]]) >= 2:
                    covcount[ball_matrix[i]] -= 1
                    chosen_set.discard(i)
                    improved = True
        return chosen_set, covcount

    chosen_set, covcount = try_removals(chosen_set, covcount)

    # Simulated annealing local search with swaps and multi-swaps
    def improved_local_search(chosen_set, covcount, time_budget=55.0):
        t_start = time.time()
        best_set = set(chosen_set)
        best_size = len(chosen_set)
        cur_set = set(chosen_set)
        cur_size = len(cur_set)
        
        # Temperature schedule
        T = 5.0
        T_min = 0.1
        cool_rate = 0.999
        
        while time.time() - t_start < time_budget and _time_left() > 10.0:
            # Try removals first (always accept)
            made_progress = False
            removal_order = list(cur_set)
            rng.shuffle(removal_order)
            for i in removal_order:
                if i not in cur_set:
                    continue
                b = ball_matrix[i]
                if np.min(covcount[b]) >= 2:
                    covcount[b] -= 1
                    cur_set.discard(i)
                    cur_size -= 1
                    made_progress = True
                    if cur_size < best_size:
                        best_size = cur_size
                        best_set = set(cur_set)
            
            if not made_progress:
                # Try swaps: remove i, add j (1-for-1)
                i = rng.choice(list(cur_set))
                b = ball_matrix[i]
                would_uncover = b[covcount[b] == 1]
                
                if len(would_uncover) == 0:
                    covcount[b] -= 1
                    cur_set.discard(i)
                    cur_size -= 1
                    if cur_size < best_size:
                        best_size = cur_size
                        best_set = set(cur_set)
                    continue
                
                target_words = would_uncover.tolist()
                candidate_set = None
                for tw in target_words:
                    covering = set(ball_matrix[tw].tolist()) - cur_set
                    if candidate_set is None:
                        candidate_set = covering
                    else:
                        candidate_set &= covering
                    if not candidate_set:
                        break
                
                if candidate_set:
                    candidates = list(candidate_set)
                    best_j = None
                    best_score = -1
                    for j in candidates:
                        bj = ball_matrix[j]
                        score = int(np.sum(covcount[bj] >= 2))
                        if score > best_score:
                            best_score = score
                            best_j = j
                    
                    if best_j is not None:
                        bj = ball_matrix[best_j]
                        covcount[b] -= 1
                        covcount[bj] += 1
                        cur_set.discard(i)
                        cur_set.add(best_j)
                        if cur_size < best_size:
                            best_size = cur_size
                            best_set = set(cur_set)
                else:
                    # No exact swap found; try accepting a worsening move via SA
                    # Pick a random i and a random j not in cur_set
                    i = rng.choice(list(cur_set))
                    b = ball_matrix[i]
                    would_uncover = b[covcount[b] == 1]
                    
                    if len(would_uncover) > 0 and len(would_uncover) <= 3:
                        # Try to find j that covers most of would_uncover
                        target_words = would_uncover.tolist()
                        best_j = None
                        best_covered = 0
                        # Sample a few candidates
                        all_candidates = set()
                        for tw in target_words[:2]:
                            all_candidates.update(set(ball_matrix[tw].tolist()) - cur_set)
                        
                        if all_candidates:
                            sample = rng.sample(list(all_candidates), min(10, len(all_candidates)))
                            for j in sample:
                                bj = ball_matrix[j]
                                covered_count = sum(1 for tw in target_words if tw in bj)
                                if covered_count > best_covered:
                                    best_covered = covered_count
                                    best_j = j
                            
                            if best_j is not None and best_covered >= len(target_words) - 1:
                                bj = ball_matrix[best_j]
                                # Compute cost: words that become uncovered
                                new_uncovered = 0
                                for tw in target_words:
                                    if tw not in bj:
                                        new_uncovered += 1
                                # Accept if new_uncovered < len(target_words) or with probability
                                if new_uncovered < len(target_words):
                                    covcount[b] -= 1
                                    covcount[bj] += 1
                                    cur_set.discard(i)
                                    cur_set.add(best_j)
                                    if cur_size < best_size:
                                        best_size = cur_size
                                        best_set = set(cur_set)
                                elif T > T_min:
                                    delta = new_uncovered - len(target_words)
                                    if rng.random() < np.exp(-delta / T):
                                        covcount[b] -= 1
                                        covcount[bj] += 1
                                        cur_set.discard(i)
                                        cur_set.add(best_j)
                                        if cur_size < best_size:
                                            best_size = cur_size
                                            best_set = set(cur_set)
            
            # Cool down
            T = max(T * cool_rate, T_min)
            
            # Periodically reset temperature if stuck
            if T <= T_min and cur_size >= best_size:
                T = 2.0
        
        # Restore best
        cur_set = best_set
        return cur_set, covcount

    chosen_set, covcount = improved_local_search(chosen_set, covcount)

    # Final removals pass with more budget
    chosen_set, covcount = try_removals(chosen_set, covcount, time_budget=15.0)

    # Final verification
    final_covered = np.zeros(N, dtype=np.int8)
    for i in chosen_set:
        final_covered[ball_matrix[i]] = 1
    if not final_covered.all():
        for i in range(N):
            if not final_covered[i]:
                chosen_set.add(i)
    
    # One more removal pass after final verification
    final_covcount = np.zeros(N, dtype=np.int32)
    for i in chosen_set:
        final_covcount[ball_matrix[i]] += 1
    
    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        for i in list(chosen_set):
            if i not in chosen_set:
                continue
            b = ball_matrix[i]
            if np.min(final_covcount[b]) >= 2:
                final_covcount[b] -= 1
                chosen_set.discard(i)
                improved = True
    
    chosen = list(chosen_set)
    return [idx_to_digits(i) for i in chosen]
