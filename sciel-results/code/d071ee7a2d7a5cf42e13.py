"""Covering code K_3(8,1): minimize codewords so every word is within Hamming distance 1."""
import time
import random
import numpy as np

_DEADLINE = 110.0  # wall-clock safety margin under the 150s kill
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
    N = 3 ** n
    
    # Precompute digits and indices
    def idx_to_digits(i):
        d = [0] * n
        for p in range(n):
            d[p] = i % 3
            i //= 3
        return d

    def digits_to_idx(d):
        v = 0
        mul = 1
        for p in range(n):
            v += d[p] * mul
            mul *= 3
        return v

    # Precompute balls
    ball = np.empty(N, dtype=object)
    for i in range(N):
        d = idx_to_digits(i)
        s = [i]
        for p in range(n):
            orig = d[p]
            for nv in range(3):
                if nv != orig:
                    d[p] = nv
                    s.append(digits_to_idx(d))
            d[p] = orig
        ball[i] = np.array(s, dtype=np.int32)
        
    # Precompute coverers: for each word w, list of codewords c such that w in ball(c)
    # This is the transpose of the ball relation.
    # coverers[w] = list of i such that w in ball[i]
    coverers = [set() for _ in range(N)]
    for i in range(N):
        for w in ball[i]:
            coverers[w].add(i)

    # 1. Initial Greedy Construction
    # Tetracode seed
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_set = []
    for h in tetracode:
        for t in tetracode:
            w = h + t
            seed_set.append(digits_to_idx(w))
    seed_set = list(set(seed_set))

    covered = np.zeros(N, dtype=bool)
    covcount = np.zeros(N, dtype=np.int32)
    chosen = set(seed_set)
    for i in seed_set:
        covcount[ball[i]] += 1
        covered[ball[i]] = True

    # Greedy fill
    while not covered.all() and _time_left() > 20.0:
        # Find an uncovered word
        unc_idx = np.where(~covered)[0]
        if len(unc_idx) == 0:
            break
        w = unc_idx[rng.randint(0, len(unc_idx)-1)]
        candidates = [c for c in coverers[w] if c not in chosen]
        if not candidates:
            break
        # Pick the one that covers most new words
        best_c = -1
        best_gain = 0
        for c in candidates:
            b = ball[c]
            gain = int(np.sum(~covered[b]))
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_c != -1:
            chosen.add(best_c)
            covcount[ball[best_c]] += 1
            covered[ball[best_c]] = True
        else:
            c = candidates[0]
            chosen.add(c)
            covcount[ball[c]] += 1
            covered[ball[c]] = True

    # Ensure valid
    if not covered.all():
        for i in range(N):
            if not covered[i]:
                chosen.add(i)
                covcount[ball[i]] += 1
                covered[ball[i]] = True

    # 2. Local Search: Remove
    in_chosen = np.zeros(N, dtype=bool)
    for i in chosen:
        in_chosen[i] = True

    def try_remove_all():
        improved = True
        while improved and _time_left() > 2.0:
            improved = False
            order = list(chosen)
            rng.shuffle(order)
            for i in order:
                if not in_chosen[i]:
                    continue
                b = ball[i]
                if int(np.min(covcount[b])) >= 2:
                    covcount[b] -= 1
                    chosen.discard(i)
                    in_chosen[i] = False
                    improved = True
                    if _time_left() < 1.0:
                        return

    try_remove_all()

    # 3. Fast Local Search: Swaps using set-based coverer lookups
    # Key optimization: use coverers[w] as a set for O(1) membership tests
    # and precompute ball sets for fast intersection checks
    
    # Precompute ball as sets for faster operations
    ball_sets = [set(b.tolist()) for b in ball]
    
    def fast_swap_search():
        """Fast 1-swap local search."""
        while _time_left() > 2.0:
            try_remove_all()
            if _time_left() < 3.0:
                break
            if len(chosen) == 0:
                break
            
            i = rng.choice(list(chosen))
            b_i = ball[i]
            
            # Check if i is removable
            if int(np.min(covcount[b_i])) >= 2:
                covcount[b_i] -= 1
                chosen.discard(i)
                in_chosen[i] = False
                continue
            
            # Find words that would become uncovered if we remove i
            would_uncover = b_i[covcount[b_i] == 1]
            
            if len(would_uncover) == 0:
                covcount[b_i] -= 1
                chosen.discard(i)
                in_chosen[i] = False
                continue

            # Find candidates j that cover all would_uncover words and are not in chosen
            # Use intersection of coverer sets
            cands = None
            for w in would_uncover:
                cw = coverers[w]
                if cands is None:
                    cands = cw.copy()
                else:
                    cands.intersection_update(cw)
                if not cands:
                    break
            
            if cands is None or not cands:
                continue
            
            # Remove already chosen candidates
            cands -= chosen
            
            if not cands:
                continue
            
            j = rng.choice(list(cands))
            b_j = ball[j]
            
            # Perform swap
            covcount[b_i] -= 1
            covcount[b_j] += 1
            chosen.discard(i)
            in_chosen[i] = False
            chosen.add(j)
            in_chosen[j] = True
    
    def two_opt_search():
        """2-opt: remove two codewords, add two new ones."""
        while _time_left() > 5.0:
            if len(chosen) < 2:
                break
            
            # Pick two random codewords to try removing
            order = list(chosen)
            rng.shuffle(order)
            i1 = order[0]
            i2 = order[1]
            
            b_i1 = ball[i1]
            b_i2 = ball[i2]
            
            # Words that would become uncovered if we remove both
            # A word is uncovered iff it's in ball[i1] or ball[i2] and covcount==1 for it
            # More precisely: after removing both, covcount[w] decreases by 1 if w in ball[i1], by 1 if w in ball[i2]
            # A word becomes uncovered if its new covcount would be 0
            
            # Compute which words are at risk
            # For words in both balls: covcount decreases by 2
            # For words in only one ball: covcount decreases by 1
            # A word is uncovered if new_covcount == 0
            
            union_balls = set(b_i1.tolist()) | set(b_i2.tolist())
            intersect_balls = set(b_i1.tolist()) & set(b_i2.tolist())
            
            would_uncover = []
            for w in union_balls:
                if w in intersect_balls:
                    if covcount[w] == 2:
                        would_uncover.append(w)
                else:
                    if covcount[w] == 1:
                        would_uncover.append(w)
            
            if not would_uncover:
                # Can remove both
                covcount[b_i1] -= 1
                covcount[b_i2] -= 1
                chosen.discard(i1)
                chosen.discard(i2)
                in_chosen[i1] = False
                in_chosen[i2] = False
                continue
            
            # Find two candidates j1, j2 (not in chosen) that cover all would_uncover
            # Use intersection approach: for each w in would_uncover, get coverers[w] - chosen
            # Then find pairs that together cover all would_uncover
            
            # First, get candidate sets for each uncovered word
            cand_sets = []
            for w in would_uncover:
                cs = coverers[w] - chosen
                if not cs:
                    return  # No solution, give up on this 2-opt
                cand_sets.append(cs)
            
            # Find a pair (j1, j2) such that for every w in would_uncover, j1 or j2 is in coverers[w]
            # This is a small set cover problem with k=2
            # Iterate over candidates from the first set
            j1_candidates = list(cand_sets[0])
            if len(j1_candidates) > 50:
                rng.shuffle(j1_candidates)
                j1_candidates = j1_candidates[:50]
            
            for j1 in j1_candidates:
                # What does j1 cover from would_uncover?
                b_j1 = set(ball[j1].tolist())
                remaining = [w for w in would_uncover if w not in b_j1]
                
                if not remaining:
                    # j1 alone covers all, so we can just do a 1-swap (already handled)
                    # But for 2-opt, we need to add exactly 2, so pick any j2
                    # Actually, if j1 covers all, we can add j1 and skip j2
                    # But that's a 1-swap, not 2-opt. Let's handle it as: add j1, don't add j2
                    # This reduces to 1-swap, which is fine
                    covcount[b_i1] -= 1
                    covcount[b_i2] -= 1
                    covcount[ball[j1]] += 1
                    chosen.discard(i1)
                    chosen.discard(i2)
                    in_chosen[i1] = False
                    in_chosen[i2] = False
                    chosen.add(j1)
                    in_chosen[j1] = True
                    return
                
                # Need j2 to cover all remaining
                # j2 must be in intersection of coverers[w] for w in remaining, minus chosen, minus {j1}
                j2_cands = None
                for w in remaining:
                    cw = (coverers[w] - chosen) - {j1}
                    if j2_cands is None:
                        j2_cands = cw.copy()
                    else:
                        j2_cands.intersection_update(cw)
                    if not j2_cands:
                        break
                
                if j2_cands is None or not j2_cands:
                    continue
                
                j2 = rng.choice(list(j2_cands))
                
                # Perform 2-opt swap
                covcount[b_i1] -= 1
                covcount[b_i2] -= 1
                covcount[ball[j1]] += 1
                covcount[ball[j2]] += 1
                chosen.discard(i1)
                chosen.discard(i2)
                in_chosen[i1] = False
                in_chosen[i2] = False
                chosen.add(j1)
                in_chosen[j1] = True
                chosen.add(j2)
                in_chosen[j2] = True
                return
    
    # Alternate between 1-swap and 2-opt
    use_two_opt = True
    while _time_left() > 2.0:
        if use_two_opt:
            two_opt_search()
        else:
            fast_swap_search()
        use_two_opt = not use_two_opt
    
    # Final cleanup
    try_remove_all()
    
    # Verify
    final_covered = np.zeros(N, dtype=bool)
    for i in chosen:
        final_covered[ball[i]] = True
    if not final_covered.all():
        for i in range(N):
            if not final_covered[i]:
                chosen.add(i)
                final_covered[ball[i]] = True

    return [idx_to_digits(i) for i in chosen]
