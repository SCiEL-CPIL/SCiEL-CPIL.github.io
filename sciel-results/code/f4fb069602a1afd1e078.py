# EVOLVE-BLOCK-START
import itertools
import time
import random

_DEADLINE = 150.0
_MARGIN = 12.0          # safety margin before the 150s kill
_SMALL_N = 24           # below this, prune every candidate word (exact)


def _hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def _balls(word, q, n):
    """All words within Hamming distance 1 of `word` (a list of tuples)."""
    out = [tuple(word)]
    for i in range(n):
        for v in range(q):
            if v != word[i]:
                w = list(word)
                w[i] = v
                out.append(tuple(w))
    return out

def _word_to_int(word, q, n):
    """Convert a word (list of ints) to an integer index."""
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx

def _int_to_word(idx, q, n):
    """Convert an integer index to a word (list of ints)."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word

def _local_search(codewords, q, n, radius, deadline, seed):
    """
    Local search to minimize the size of a covering code.
    Uses cover counts to quickly check validity of removals and swaps.
    """
    rng = random.Random(seed)
    # Convert to integers for faster hashing and indexing
    words = [tuple(w) for w in codewords]
    words_int = [_word_to_int(w, q, n) for w in words]
    
    # Precompute balls for all possible words in the space (q^n <= 729 is small)
    # ball_mask[i] is a set of point indices covered by word i
    all_points = q ** n
    ball_masks = [None] * all_points
    for i in range(all_points):
        w = _int_to_word(i, q, n)
        b = _balls(w, q, n)
        ball_masks[i] = set(_word_to_int(p, q, n) for p in b)
        
    # Initialize cover counts
    cover_counts = [0] * all_points
    active = set(words_int)
    for w in active:
        for p in ball_masks[w]:
            cover_counts[p] += 1
            
    def is_valid_removal(w):
        """Check if removing w keeps all points covered."""
        for p in ball_masks[w]:
            if cover_counts[p] <= 1:
                return False
        return True

    def apply_swap(remove_w, add_w):
        """
        Swap remove_w with add_w.
        Returns True if successful (and updates state), False otherwise.
        """
        # Check validity of new state
        # New count for p = old count - (p in ball(remove_w)) + (p in ball(add_w))
        # We need new_count[p] >= 1 for all p.
        
        # Optimization: Only check points in ball(remove_w) or ball(add_w)
        # Points outside both balls have count >= 1 (since old state was valid).
        # Points in ball(remove_w) \ ball(add_w): count decreases by 1. Must be >= 2 originally.
        # Points in ball(add_w) \ ball(remove_w): count increases by 1. Always >= 1.
        # Points in intersection: count unchanged.
        
        for p in ball_masks[remove_w]:
            if p in ball_masks[add_w]:
                continue # Unchanged
            if cover_counts[p] <= 1:
                return False # Would become 0
        
        # If valid, perform swap
        for p in ball_masks[remove_w]:
            cover_counts[p] -= 1
        for p in ball_masks[add_w]:
            cover_counts[p] += 1
            
        active.discard(remove_w)
        active.add(add_w)
        return True

    # Main Loop
    no_improvement_count = 0
    max_no_improvement = 5000 # Tunable
    
    while time.time() < deadline:
        # Phase 1: Try to remove redundant words
        removed_any = False
        # Iterate over a copy since we modify active
        for w in list(active):
            if is_valid_removal(w):
                # Remove w
                for p in ball_masks[w]:
                    cover_counts[p] -= 1
                active.discard(w)
                removed_any = True
                no_improvement_count = 0
                # Restart loop to check for new redundancies created by removal?
                # Actually, removal only decreases counts, so it doesn't create new redundancies
                # for OTHER words (it makes them MORE critical).
                # So we can just continue scanning.
                # But wait, if we remove w, other words become more critical.
                # So we shouldn't break, just continue.
        
        if removed_any:
            continue
        
        # Phase 2: If no removal possible, try swaps
        # Pick a random word from active to remove
        if not active:
            break
            
        candidates = list(active)
        remove_w = rng.choice(candidates)
        
        # Try a few random swaps
        swap_success = False
        # Try 100 random swaps per "stuck" state
        for _ in range(100):
            # Pick a random word from the whole space to add
            add_w = rng.randint(0, all_points - 1)
            if add_w in active:
                continue
            
            if apply_swap(remove_w, add_w):
                swap_success = True
                no_improvement_count = 0
                break
        
        if not swap_success:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Stuck, maybe try a different random seed or restart?
                # For now, just break to avoid wasting time
                break

    return [_int_to_word(w, q, n) for w in active]

def _prune(codewords, q, n, radius, deadline, small):
    """Remove redundant words. A word is redundant if its radius-1 ball is covered
    by the remaining words. We keep a valid covering at all times (only drop when
    the rest already covers that ball)."""
    words = [tuple(w) for w in codewords]
    balls = [_balls(w, q, n) for w in words]
    # Precompute the union of all balls to know the universe we must keep covered.
    universe = set()
    for b in balls:
        universe.update(b)

    # Order: drop words that are "least useful" first (fewest uniquely-covered points).
    def unique_count(i):
        mine = set(balls[i])
        others = set()
        for j in range(len(words)):
            if j != i:
                others.update(balls[j])
        return len(mine - others)

    order = sorted(range(len(words)), key=lambda i: unique_count(i))
    active = set(range(len(words)))
    covered_now = set()
    for i in range(len(words)):
        covered_now.update(balls[i])

    for i in order:
        if time.time() > deadline:
            break
        if i not in active:
            continue
        # Does removing word i keep the covering valid?
        # i.e. is balls[i] subset of (union of active \ {i})?
        # Recompute union of active without i (cheap enough for small active sets).
        if len(active) > small and (len(active) - 1) % 7 != 0:
            # For large sets, only attempt a cheap check on a sample to save time.
            sample = list(balls[i])
            others = set()
            for j in active:
                if j != i:
                    others.update(balls[j])
            if all(p in others for p in sample):
                active.discard(i)
        else:
            others = set()
            for j in active:
                if j != i:
                    others.update(balls[j])
            if all(p in others for p in balls[i]):
                active.discard(i)

    return [list(words[i]) for i in sorted(active)]

def _greedy_cover(q, n, radius, deadline, seed):
    """Greedy set-cover: repeatedly add the word whose radius-1 ball covers the most
    not-yet-covered points, with symmetry-guided tie-breaking."""
    rng = random.Random(seed)
    universe = set(itertools.product(range(q), repeat=n))
    uncovered = set(universe)
    if not uncovered:
        return []
    # Precompute balls lazily.
    balls_cache = {}
    def ball_of(w):
        t = tuple(w)
        if t not in balls_cache:
            balls_cache[t] = _balls(list(t), q, n)
        return balls_cache[t]

    chosen = []
    while uncovered and time.time() < deadline:
        best = None
        best_gain = -1
        best_key = None
        for w in itertools.product(range(q), repeat=n):
            b = ball_of(list(w))
            gain = len(uncovered.intersection(b))
            if gain > best_gain:
                best_gain = gain
                best = w
                best_key = _symmetry_key(list(w), q, n, rng)
            elif gain == best_gain and gain > 0:
                k = _symmetry_key(list(w), q, n, rng)
                if best_key is None or k < best_key:
                    best = w
                    best_key = k
        if best is None or best_gain <= 0:
            break
        chosen.append(list(best))
        uncovered.difference_update(ball_of(best))
    return chosen

def _symmetry_key(word, q, n, rng):
    """A canonical-ish symmetry signature used only for deterministic tie-breaking:
    coordinate permutation (sorted by symbol) + a random affine symbol relabel."""
    perm = list(range(n))
    rng.shuffle(perm)
    a = rng.randint(1, q - 1)
    b = rng.randint(0, q - 1)
    return tuple(sorted(((word[i] * a + b) % q, i) for i in perm))

def _direct_sum(short_code, q, n, k, radius):
    """Direct sum: every short codeword on the first k coords × every word on the rest."""
    tails = list(itertools.product(range(q), repeat=n - k))
    out = []
    for h in short_code:
        for t in tails:
            out.append(list(h) + list(t))
    return out

def _optimize_short_code(q, k, radius, deadline, seed):
    """Find a small covering code of length k, radius `radius` over q, via greedy + prune."""
    rng = random.Random(seed)
    universe = set(itertools.product(range(q), repeat=k))
    uncovered = set(universe)
    balls_cache = {}
    def ball_of(w):
        t = tuple(w)
        if t not in balls_cache:
            balls_cache[t] = _balls(list(t), q, k)
        return balls_cache[t]
    chosen = []
    while uncovered and time.time() < deadline:
        best = None
        best_gain = -1
        best_key = None
        for w in itertools.product(range(q), repeat=k):
            b = ball_of(list(w))
            gain = len(uncovered.intersection(b))
            if gain > best_gain:
                best_gain = gain
                best = w
                best_key = _symmetry_key(list(w), q, k, rng)
            elif gain == best_gain and gain > 0:
                kk = _symmetry_key(list(w), q, k, rng)
                if best_key is None or kk < best_key:
                    best = w
                    best_key = kk
        if best is None or best_gain <= 0:
            break
        chosen.append(list(best))
        uncovered.difference_update(ball_of(best))
    pruned = _prune(chosen, q, k, radius, deadline, small=_SMALL_N)
    return pruned

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + (_DEADLINE - _MARGIN)

    # --- Seed construction (always valid) as a baseline. ---
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n >= 4:
        baseline = _direct_sum(tetracode, q, n, 4, radius)
    else:
        baseline = list(itertools.product(range(q), repeat=n))
    best = baseline

    # --- Try: optimized short code of length k, direct-summed, then pruned. ---
    k_choices = []
    for k in range(2, n + 1):
        k_choices.append(k)

    remaining = deadline - time.time()
    if remaining > 5:
        k_choices.sort(key=lambda k: q ** (n - k))
        per_k = max(2.0, remaining / max(1, len(k_choices)))
        for k in k_choices:
            if time.time() > deadline:
                break
            sub_deadline = min(deadline, time.time() + per_k)
            try:
                short = _optimize_short_code(q, k, radius, sub_deadline, seed=12345 + k)
                if not short:
                    continue
                prod = _direct_sum(short, q, n, k, radius)
                if len(prod) >= len(best):
                    continue
                pruned = _prune(prod, q, n, radius, min(deadline, time.time() + per_k),
                                 small=_SMALL_N)
                if len(pruned) < len(best):
                    best = pruned
            except Exception:
                continue

    # --- Try: full greedy cover on the whole space, then prune (if room). ---
    remaining = deadline - time.time()
    if remaining > 8 and q ** n <= 40000:
        try:
            greedy = _greedy_cover(q, n, radius, min(deadline, time.time() + remaining - 3),
                                   seed=777)
            if greedy and len(greedy) < len(best):
                pruned = _prune(greedy, q, n, radius, deadline, small=_SMALL_N)
                if len(pruned) < len(best):
                    best = pruned
        except Exception:
            pass

    # --- NEW: Local Search on the best candidate found so far ---
    remaining = deadline - time.time()
    if remaining > 5:
        try:
            # Use the best code found so far as the starting point for local search
            # Local search can handle up to ~100 words very fast
            ls_code = _local_search(best, q, n, radius, deadline, seed=42)
            if len(ls_code) < len(best):
                best = ls_code
        except Exception:
            pass

    # Final safety: ensure we return before the hard limit.
    if time.time() > deadline:
        pass

    # De-duplicate while preserving order (defensive).
    seen = set()
    out = []
    for w in best:
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            out.append(list(w))
    return out
# EVOLVE-BLOCK-END
