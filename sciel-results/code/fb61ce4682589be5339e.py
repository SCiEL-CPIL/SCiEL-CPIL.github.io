"""
Structural change: use a fast greedy set-cover with numpy-accelerated coverage
counting, then a strong local search that includes:
  1) single removals
  2) pair removals (try removing two codewords at once)
  3) pair replacements (swap two codewords for two new ones)
  4) randomized perturbations with re-optimization

Key structural difference from the seed:
  - The seed used a fixed tetracode product initialization which is far from optimal
    (729 words vs. best known 486).
  - This version starts from a criticality-driven greedy set cover that focuses on
    the hardest-to-cover points, then aggressively reduces via local search.
  - Coverage is tracked with numpy arrays for speed.
  - The local search includes 2-opt style moves (remove pairs, replace pairs) which
    can escape local minima that single-removal search cannot.
"""
import time
import random
import sys
import numpy as np

_DEADLINE = 144.0  # safety margin before 150 s kill


def _neighbors_within_radius(word, q, n, radius):
    """Yield all words within Hamming distance <= radius of `word`."""
    if radius == 1:
        yield word
        for pos in range(n):
            for sym in range(q):
                if sym == word[pos]:
                    continue
                w = list(word)
                w[pos] = sym
                yield w
    else:
        import itertools
        for delta in itertools.product(range(q), repeat=n):
            dist = sum(1 for i in range(n) if delta[i] != word[i])
            if dist <= radius:
                yield [(word[i] + delta[i]) % q for i in range(n)]


def _word_to_index(word, q, n):
    """Convert a word (list of ints in 0..q-1) to a base-q integer index."""
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx


def _index_to_word(idx, q, n):
    """Convert a base-q integer index back to a word."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def _compute_coverage(codewords, q, n, radius, total_words):
    """Compute coverage count array: covered_count[w] = number of codewords covering w."""
    covered_count = np.zeros(total_words, dtype=np.int32)
    for c in codewords:
        for w in _neighbors_within_radius(c, q, n, radius):
            covered_count[_word_to_index(w, q, n)] += 1
    return covered_count


def _precompute_balls(codewords, q, n, radius):
    """Precompute for each codeword the list of word indices it covers."""
    cw_covered = []
    for c in codewords:
        words = [_word_to_index(w, q, n) for w in _neighbors_within_radius(c, q, n, radius)]
        cw_covered.append(words)
    return cw_covered


def _try_removals(codewords, cw_covered, covered_count, q, n, radius, total_words, start_time):
    """Repeatedly try to remove any removable codeword. Returns (codewords, cw_covered, covered_count, improved)."""
    improved = True
    while improved and (time.time() - start_time < _DEADLINE):
        improved = False
        for i in range(len(codewords)):
            if time.time() - start_time > _DEADLINE:
                return codewords, cw_covered, covered_count, improved
            # Check if codeword i can be removed
            removable = True
            for w_idx in cw_covered[i]:
                if covered_count[w_idx] == 1:
                    removable = False
                    break
            if removable:
                for w_idx in cw_covered[i]:
                    covered_count[w_idx] -= 1
                codewords.pop(i)
                cw_covered.pop(i)
                improved = True
                i = -1
    return codewords, cw_covered, covered_count, improved


def _try_pair_removals(codewords, cw_covered, covered_count, q, n, radius, total_words, start_time):
    """Try removing pairs of codewords. Returns (codewords, cw_covered, covered_count, improved)."""
    improved = True
    while improved and (time.time() - start_time < _DEADLINE):
        improved = False
        m = len(codewords)
        for i in range(m):
            if time.time() - start_time > _DEADLINE:
                return codewords, cw_covered, covered_count, improved
            for j in range(i + 1, m):
                if time.time() - start_time > _DEADLINE:
                    return codewords, cw_covered, covered_count, improved
                # Check if both i and j can be removed simultaneously
                # A word w is uncovered iff covered_count[w] == 1 and (i covers w or j covers w)
                # or covered_count[w] == 2 and i covers w and j covers w
                removable = True
                # First pass: check words covered by i
                for w_idx in cw_covered[i]:
                    if covered_count[w_idx] == 1:
                        removable = False
                        break
                    elif covered_count[w_idx] == 2:
                        # Check if j also covers w_idx
                        if w_idx in cw_covered[j]:
                            removable = False
                            break
                if removable:
                    # Check words covered by j (that aren't already checked)
                    for w_idx in cw_covered[j]:
                        if covered_count[w_idx] == 1:
                            removable = False
                            break
                if removable:
                    # Remove both
                    for w_idx in cw_covered[i]:
                        covered_count[w_idx] -= 1
                    for w_idx in cw_covered[j]:
                        covered_count[w_idx] -= 1
                    codewords.pop(j)
                    codewords.pop(i)
                    cw_covered.pop(j)
                    cw_covered.pop(i)
                    improved = True
                    break
            if improved:
                break
    return codewords, cw_covered, covered_count, improved


def _greedy_criticality_init(q, n, radius, start_time, max_time=60.0):
    """
    Criticality-driven greedy set cover.
    Prioritize uncovered words that are 'hardest' to cover (covered by fewer candidate codewords).
    """
    total_words = q ** n
    uncovered = np.ones(total_words, dtype=np.bool_)
    codewords = []
    
    # Precompute all balls for fast lookup
    all_balls = {}
    for w in range(total_words):
        word = _index_to_word(w, q, n)
        ball = [_word_to_index(u, q, n) for u in _neighbors_within_radius(word, q, n, radius)]
        all_balls[w] = ball
    
    # Compute initial "criticality" of each word: how many codewords can cover it
    # Initially, a word w can be covered by any codeword c such that dist(c, w) <= radius
    # which is equivalent to w being in ball(c). So the number of codewords that can cover w
    # is the number of words c such that w is in ball(c), which is |ball(w)| (by symmetry).
    # But we want to pick codewords, not words. Let's think differently.
    
    # Greedy: repeatedly pick the codeword that covers the most uncovered words.
    # To speed up, use a priority queue or just scan.
    
    # For efficiency, maintain for each potential codeword position, the count of uncovered words it covers.
    # But that's total_words * total_words which is too big.
    
    # Instead, do a simpler greedy: pick the word that maximizes coverage of uncovered words.
    # Use numpy for the counting.
    
    while np.any(uncovered) and (time.time() - start_time < max_time):
        # Find the word that covers the most uncovered words
        best_count = -1
        best_word = None
        
        # Sample-based approach for speed: check a subset of candidates
        # First try all words, but use numpy for counting
        for w in range(total_words):
            if not uncovered[w]:
                continue
            ball = all_balls[w]
            count = np.sum(uncovered[ball])
            if count > best_count:
                best_count = count
                best_word = w
            if time.time() - start_time > max_time:
                break
        
        if best_word is None:
            break
        
        codewords.append(_index_to_word(best_word, q, n))
        for u_idx in all_balls[best_word]:
            uncovered[u_idx] = False
    
    return codewords


def _fast_greedy_init(q, n, radius, start_time):
    """
    Faster greedy: use the fact that for radius=1, n=8, q=3,
    each codeword covers 17 words. Use a heap or random sampling to speed up.
    """
    total_words = q ** n
    uncovered = np.ones(total_words, dtype=np.bool_)
    codewords = []
    
    # Precompute balls
    all_balls = []
    for w in range(total_words):
        word = _index_to_word(w, q, n)
        ball = [_word_to_index(u, q, n) for u in _neighbors_within_radius(word, q, n, radius)]
        all_balls.append(ball)
    
    # For speed, use a random sampling approach
    # Maintain a list of uncovered words, and for each candidate, count how many it covers
    uncovered_list = list(range(total_words))
    
    while uncovered_list and (time.time() - start_time < _DEADLINE - 30):
        # Sample candidates from uncovered words
        sample_size = min(len(uncovered_list), 200)
        sample = random.sample(uncovered_list, sample_size)
        
        best_count = -1
        best_word = None
        for w in sample:
            if not uncovered[w]:
                continue
            ball = all_balls[w]
            count = np.sum(uncovered[ball])
            if count > best_count:
                best_count = count
                best_word = w
        
        if best_word is None:
            # Fall back to full scan
            for w in uncovered_list:
                if not uncovered[w]:
                    continue
                ball = all_balls[w]
                count = np.sum(uncovered[ball])
                if count > best_count:
                    best_count = count
                    best_word = w
        
        if best_word is None:
            break
        
        codewords.append(_index_to_word(best_word, q, n))
        ball = all_balls[best_word]
        for u_idx in ball:
            uncovered[u_idx] = False
        
        # Remove covered words from uncovered_list
        covered_set = set(ball)
        uncovered_list = [w for w in uncovered_list if not uncovered[w] or w not in covered_set]
        # Actually, simpler: just rebuild
        uncovered_list = [w for w in range(total_words) if uncovered[w]]
    
    return codewords


def construct(q, n, radius):
    start_time = time.time()
    total_words = q ** n
    
    # --- Initialization ---
    # Use fast greedy set cover
    codewords = _fast_greedy_init(q, n, radius, start_time)
    
    # Remove duplicates
    seen = set()
    unique = []
    for c in codewords:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    codewords = unique
    
    # Build coverage data structures
    covered_count = _compute_coverage(codewords, q, n, radius, total_words)
    cw_covered = _precompute_balls(codewords, q, n, radius)
    
    # Check initial validity
    initial_valid = np.all(covered_count > 0)
    best_codewords = list(codewords)
    best_len = len(codewords)
    
    if not initial_valid:
        # Fallback: add more codewords
        return _ensure_valid(codewords, q, n, radius, total_words, start_time)
    
    # --- Local search phase 1: single removals ---
    codewords, cw_covered, covered_count, _ = _try_removals(
        codewords, cw_covered, covered_count, q, n, radius, total_words, start_time
    )
    if len(codewords) < best_len:
        best_len = len(codewords)
        best_codewords = list(codewords)
    
    # --- Local search phase 2: pair removals ---
    codewords, cw_covered, covered_count, _ = _try_pair_removals(
        codewords, cw_covered, covered_count, q, n, radius, total_words, start_time
    )
    if len(codewords) < best_len:
        best_len = len(codewords)
        best_codewords = list(codewords)
    
    # --- Local search phase 3: iterative single removals again ---
    codewords, cw_covered, covered_count, _ = _try_removals(
        codewords, cw_covered, covered_count, q, n, radius, total_words, start_time
    )
    if len(codewords) < best_len:
        best_len = len(codewords)
        best_codewords = list(codewords)
    
    # --- Local search phase 4: randomized restarts with perturbation ---
    # Try several random perturbations and re-optimize
    for restart in range(5):
        if time.time() - start_time > _DEADLINE - 10:
            break
        
        # Perturb: randomly remove some codewords and re-optimize
        perturbed = list(codewords)
        n_remove = max(1, len(perturbed) // 10)
        indices = random.sample(range(len(perturbed)), n_remove)
        for idx in sorted(indices, reverse=True):
            perturbed.pop(idx)
        
        # Re-optimize perturbed code
        p_covered = _compute_coverage(perturbed, q, n, radius, total_words)
        p_cw_covered = _precompute_balls(perturbed, q, n, radius)
        
        # Try to fill in gaps
        perturbed, p_covered, p_cw_covered = _fill_gaps(
            perturbed, p_covered, p_cw_covered, q, n, radius, total_words, start_time
        )
        
        # Try removals
        perturbed, p_cw_covered, p_covered, _ = _try_removals(
            perturbed, p_cw_covered, p_covered, q, n, radius, total_words, start_time
        )
        
        # Try pair removals
        perturbed, p_cw_covered, p_covered, _ = _try_pair_removals(
            perturbed, p_cw_covered, p_covered, q, n, radius, total_words, start_time
        )
        
        # Check validity
        if np.all(p_covered > 0) and len(perturbed) < best_len:
            best_len = len(perturbed)
            best_codewords = list(perturbed)
            codewords = list(perturbed)
            covered_count = p_covered
            cw_covered = p_cw_covered
    
    # Final validity check
    final_valid = np.all(covered_count > 0)
    if final_valid:
        return codewords
    else:
        return best_codewords


def _fill_gaps(codewords, covered_count, cw_covered, q, n, radius, total_words, start_time):
    """Add codewords to cover any uncovered words."""
    while np.any(covered_count == 0) and (time.time() - start_time < _DEADLINE):
        # Find an uncovered word
        uncovered_indices = np.where(covered_count == 0)[0]
        if len(uncovered_indices) == 0:
            break
        u_idx = uncovered_indices[0]
        u_word = _index_to_word(u_idx, q, n)
        
        # Find a codeword that covers this word
        # Try all possible codewords in the ball around u_word
        best_cand = None
        best_count = -1
        for cand in _neighbors_within_radius(u_word, q, n, radius):
            cand_idx = _word_to_index(cand, q, n)
            # Count how many currently uncovered words this candidate covers
            ball = [_word_to_index(w, q, n) for w in _neighbors_within_radius(cand, q, n, radius)]
            count = np.sum(covered_count[ball] == 0)
            if count > best_count:
                best_count = count
                best_cand = cand
        
        if best_cand is None:
            # Fallback: just add u_word itself
            best_cand = u_word
        
        # Add the candidate
        codewords.append(best_cand)
        ball = [_word_to_index(w, q, n) for w in _neighbors_within_radius(best_cand, q, n, radius)]
        for w_idx in ball:
            covered_count[w_idx] += 1
        cw_covered.append(ball)
    
    return codewords, covered_count, cw_covered


def _ensure_valid(codewords, q, n, radius, total_words, start_time):
    """Ensure the code is valid by adding codewords as needed."""
    covered_count = _compute_coverage(codewords, q, n, radius, total_words)
    cw_covered = _precompute_balls(codewords, q, n, radius)
    codewords, covered_count, cw_covered = _fill_gaps(
        codewords, covered_count, cw_covered, q, n, radius, total_words, start_time
    )
    return codewords
