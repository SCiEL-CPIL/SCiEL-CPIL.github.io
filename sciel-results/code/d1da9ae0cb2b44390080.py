"""Combined covering-code constructor:
  - Subgradient-weighted greedy initialization (dual variables for uncovered words)
  - Repair-based SLS with critical-set tracking for O(1) removal checks
  - Symmetry-guided swap candidates (small Hamming distance bias)
  - Vectorized overlap-delta for fast acceptance tests

Entry point: construct(q, n, radius) -> list of codewords (each a list of n ints).
"""

import time
import itertools
import numpy as np
import random

_DEADLINE = 140.0


def _direct_sum(q, n, radius):
    """Always-valid baseline: tetracode direct sum for q=3, r=1, n>=4."""
    if n < 4:
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    base = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    return [h + list(tail) for h in base
            for tail in itertools.product(range(q), repeat=n - 4)]


def _idx(w, q):
    idx = 0
    for x in w:
        idx = idx * q + x
    return idx


def _ball_indices(c, q, n, radius):
    """Flat indices of all words within Hamming distance `radius` of center c."""
    if radius == 0:
        return [_idx(c, q)]
    ball = [_idx(c, q)]
    for pos in range(n):
        for v in range(q):
            if v == c[pos]:
                continue
            w = list(c)
            w[pos] = v
            ball.append(_idx(w, q))
    return ball


def _build_balls(q, n, radius):
    """Precompute balls as a list of numpy arrays of flat indices."""
    total = q ** n
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    balls = []
    for i in range(total):
        c = all_words[i]
        balls.append(np.array(_ball_indices(c.tolist(), q, n, radius), dtype=np.int32))
    return all_words, balls


def _covers_all(cw, q, n, radius, balls=None):
    """Exact coverage check."""
    total = q ** n
    if total > 5 * 10**6:
        return False
    if balls is None:
        return False
    covered = np.zeros(total, dtype=bool)
    for c in cw:
        idx = _idx(c, q)
        covered[balls[idx]] = True
    return bool(covered.all())


def _greedy_subgradient(q, n, radius, time_budget, seed=0):
    """Greedy set-cover with subgradient-weighted scoring.
    
    Instead of simply maximizing uncovered words covered, we weight each
    uncovered word by a dual variable that increases over time if it remains
    uncovered. This encourages the algorithm to focus on "hard" regions.
    """
    total = q ** n
    if total > 3 * 10**6:
        return None, False

    start = time.time()
    all_words, balls = _build_balls(q, n, radius)
    
    # Precompute ball sets for fast intersection
    ball_sets = [set(b) for b in balls]
    
    covered = np.zeros(total, dtype=bool)
    uncovered_count = total
    code = []
    
    # Dual variables: weight for each word (higher = more "important" to cover)
    dual = np.ones(total, dtype=np.float64)
    # Track how many times each word has been "seen" as uncovered
    uncovered_age = np.zeros(total, dtype=np.int32)
    
    # Precompute: for each candidate, which words it covers (already in balls)
    # We need to compute gain = sum of dual[w] for w in ball[c] where not covered[w]
    
    # For efficiency, precompute the ball membership as a matrix? Too large.
    # Instead, for each candidate, compute the weighted gain.
    
    # Optimization: maintain a "weighted_uncovered" array that we update incrementally.
    # gain[c] = sum(dual[w] for w in balls[c] if not covered[w])
    # When we cover new words, we need to update gains for all candidates that
    # contain those words. This is expensive.
    
    # Alternative: just compute gain on the fly but use numpy for the sum.
    # For 729 candidates, each with a ball of 13 words, this is 729*13 = 9477 ops per step.
    # With ~80 steps, that's ~758K ops. Very fast.
    
    # Let's just compute gain directly each time.
    
    while uncovered_count > 0:
        if time.time() - start > time_budget:
            return code, _covers_all(code, q, n, radius, balls)
        
        # Compute gain for each candidate
        best = -1
        best_gain = -1.0
        
        for i in range(total):
            if covered[i]:
                # Still might have gain from other words in its ball
                pass
            b = balls[i]
            # Weighted gain: sum of dual[w] for w in b where not covered[w]
            mask = ~covered[b]
            if not mask.any():
                gain = 0.0
            else:
                gain = float(np.sum(dual[b[mask]]))
            if gain > best_gain:
                best_gain = gain
                best = i
        
        if best < 0 or best_gain <= 0:
            # Shouldn't happen if uncovered_count > 0
            break
        
        # Add this codeword
        code.append(all_words[best].tolist())
        b = balls[best]
        new_uncovered = ~covered[b]
        newly_covered = int(np.sum(new_uncovered))
        uncovered_count -= newly_covered
        covered[b] = True
        
        # Update dual variables: increase weight for still-uncovered words
        # Subgradient step: for uncovered words, increase dual
        still_uncovered = ~covered
        uncovered_age[still_uncovered] += 1
        # dual[w] = 1 + alpha * age[w] (simple linear growth)
        # Or multiplicative: dual[w] *= (1 + beta) for uncovered
        # Let's use: dual[w] = 1.0 + 0.1 * uncovered_age[w]
        dual = 1.0 + 0.1 * uncovered_age.astype(np.float64)
    
    verified = _covers_all(code, q, n, radius, balls)
    return code, verified


def _sls_repair(code, q, n, radius, time_budget, seed=0):
    """SLS with critical-set tracking and symmetry-guided swaps."""
    total = q ** n
    if total > 3 * 10**6:
        return code, False

    rng = random.Random(seed)
    start = time.time()

    all_words, balls = _build_balls(q, n, radius)
    ball_sets = [set(b) for b in balls]  # For fast intersection/union
    
    # State
    state = [_idx(c, q) for c in code]
    
    # Coverage counts
    counts = np.zeros(total, dtype=np.int32)
    for wi in state:
        counts[balls[wi]] += 1
    
    # Critical set: words with count == 1
    critical = set(np.where(counts == 1)[0].tolist())
    
    # For each codeword index, track which critical words it covers
    # critical_covered_by[i] = set of critical words in ball[i]
    
    def compute_critical_covered(wi):
        b = balls[wi]
        return set(w for w in b if w in critical)
    
    # Precompute for current state
    crit_by_cw = {}
    for wi in state:
        crit_by_cw[wi] = compute_critical_covered(wi)
    
    cur_overlap = int(np.sum(counts * counts))
    best_state = list(state)
    best_overlap = cur_overlap
    best_size = len(state)
    
    no_improve = 0
    max_no_improve = 30000
    
    while time.time() - start < time_budget:
        if no_improve > max_no_improve:
            break
        
        # === Phase 1: Try removals (O(1) check via critical set) ===
        removed = False
        for i in range(len(state) - 1, -1, -1):
            if time.time() - start > time_budget:
                break
            wi = state[i]
            # Can remove if it covers no critical words
            if not crit_by_cw.get(wi):
                # Remove
                b = balls[wi]
                old_counts = counts[b].copy()
                counts[b] -= 1
                
                # Update critical set
                # Words that were count 1 and are now 0 -> uncovered (shouldn't happen if valid)
                # Words that were count 2 and are now 1 -> newly critical
                newly_critical = set(w for w in b if counts[w] == 1)
                no_longer_critical = set(w for w in b if old_counts[w] == 1)
                critical -= no_longer_critical
                critical |= newly_critical
                
                state.pop(i)
                del crit_by_cw[wi]
                
                # Update crit_by_cw for remaining codewords that share critical words
                # Only need to update if newly_critical or no_longer_critical is non-empty
                if newly_critical or no_longer_critical:
                    for j, wj in enumerate(state):
                        b_j = balls[wj]
                        # Recompute only if this codeword intersects the changed set
                        changed = newly_critical | no_longer_critical
                        if changed & ball_sets[wj]:
                            crit_by_cw[wj] = compute_critical_covered(wj)
                
                cur_overlap = int(np.sum(counts * counts))
                best_state = list(state)
                best_overlap = cur_overlap
                best_size = len(state)
                removed = True
                break  # Restart
        
        if removed:
            no_improve = 0
            continue
        
        # === Phase 2: Try swaps ===
        if len(state) > 0:
            i = rng.randrange(len(state))
            old_wi = state[i]
            b_old = balls[old_wi]
            set_old = ball_sets[old_wi]
            
            # Find critical words covered by old_wi (must be preserved)
            must_cover = set(w for w in b_old if counts[w] == 1)
            
            found_swap = False
            # Try symmetry-guided candidates: words at small Hamming distance
            # Plus random candidates
            attempts = 0
            max_attempts = 300
            while attempts < max_attempts:
                if time.time() - start > time_budget:
                    break
                attempts += 1
                
                if attempts < 100:
                    # Symmetry-guided: pick a random position, change value
                    # This gives candidates at Hamming distance 1
                    pos = rng.randrange(n)
                    new_val = rng.randrange(q)
                    if new_val == all_words[old_wi][pos]:
                        continue
                    w_list = all_words[old_wi].tolist()
                    w_list[pos] = new_val
                    new_wi = _idx(w_list, q)
                else:
                    new_wi = rng.randrange(total)
                
                if new_wi == old_wi:
                    continue
                
                b_new = balls[new_wi]
                set_new = ball_sets[new_wi]
                
                # Check: all critical words in old must be in new
                if must_cover and not must_cover.issubset(set_new):
                    continue
                
                # Compute overlap delta
                # delta = sum_{w in old\new} ((c-1)^2 - c^2) + sum_{w in new\old} ((c+1)^2 - c^2)
                only_old = set_old - set_new
                only_new = set_new - set_old
                
                delta = 0
                for w in only_old:
                    c = counts[w]
                    delta += (c - 1) ** 2 - c ** 2
                for w in only_new:
                    c = counts[w]
                    delta += (c + 1) ** 2 - c ** 2
                
                new_overlap = cur_overlap + delta
                
                if new_overlap < best_overlap or (new_overlap == best_overlap and rng.random() < 0.05):
                    # Accept
                    counts[b_old] -= 1
                    counts[b_new] += 1
                    state[i] = new_wi
                    cur_overlap = new_overlap
                    best_state = list(state)
                    best_overlap = new_overlap
                    
                    # Update critical set
                    changed = only_old | only_new
                    newly_critical = set(w for w in changed if counts[w] == 1)
                    no_longer_critical = set()
                    # Words in only_old that were count 1 before -> now 0 (uncovered, invalid!)
                    # Wait, we already checked must_cover. But there could be other count-1 words
                    # in only_old that are NOT in must_cover? No, must_cover IS all count-1 words in b_old.
                    # So all count-1 words in b_old are in must_cover, which is subset of set_new.
                    # So no word in only_old had count 1. Good.
                    # Words in only_new that were count 0 -> now 1 (newly critical)
                    # Words in only_new that were count 1 -> now 2 (no longer critical)
                    for w in only_new:
                        if counts[w] == 1:
                            newly_critical.add(w)
                        # If it was 1 before and is now 2, it's no longer critical
                    # We need to track what was critical before
                    # Simplify: just recompute critical for affected words
                    for w in changed:
                        if counts[w] == 1:
                            critical.add(w)
                        else:
                            critical.discard(w)
                    
                    # Update crit_by_cw
                    del crit_by_cw[old_wi]
                    crit_by_cw[new_wi] = compute_critical_covered(new_wi)
                    if changed:
                        for j, wj in enumerate(state):
                            if wj == new_wi:
                                continue
                            if changed & ball_sets[wj]:
                                crit_by_cw[wj] = compute_critical_covered(wj)
                    
                    found_swap = True
                    no_improve = 0
                    break
            
            if found_swap:
                continue
        
        no_improve += 1
    
    best_code = [all_words[wi].tolist() for wi in best_state]
    verified = _covers_all(best_code, q, n, radius, balls)
    return best_code, verified


def construct(q, n, radius):
    start = time.time()
    
    # Baseline
    if n >= 4 and radius >= 1 and q == 3:
        baseline = _direct_sum(q, n, radius)
    else:
        baseline = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    best = baseline
    best_len = len(baseline)
    
    total = q ** n
    if total > 3 * 10**6 or best_len <= 1:
        return best
    
    # Step 1: Subgradient-weighted greedy
    t_greedy = max(1.0, _DEADLINE - (time.time() - start) - 80.0)
    try:
        cand, verified = _greedy_subgradient(q, n, radius, t_greedy, seed=42)
    except Exception:
        cand, verified = None, False
    if cand is not None and verified and len(cand) < best_len:
        best = cand
        best_len = len(cand)
    
    # Step 2: SLS repair polish
    remaining = _DEADLINE - (time.time() - start) - 10.0
    if remaining > 3.0 and best_len > 1:
        try:
            polished, p_verified = _sls_repair(best, q, n, radius, remaining, seed=123)
        except Exception:
            polished, p_verified = None, False
        if polished is not None and p_verified and len(polished) < best_len:
            best = polished
            best_len = len(polished)
    
    # Step 3: If time remains, try SLS from a different starting point
    remaining2 = _DEADLINE - (time.time() - start) - 5.0
    if remaining2 > 5.0 and best_len > 1:
        # Try SLS from the baseline (direct sum) as well
        try:
            polished2, p2_verified = _sls_repair(baseline, q, n, radius, remaining2, seed=999)
        except Exception:
            polished2, p2_verified = None, False
        if polished2 is not None and p2_verified and len(polished2) < best_len:
            best = polished2
            best_len = len(polished2)
    
    # Safety: return baseline if running out of time
    if time.time() - start > _DEADLINE - 3.0:
        return baseline
    
    return best
