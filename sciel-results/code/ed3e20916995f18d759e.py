"""Football pool K_3(6,1): minimize number of ternary words covering {0,1,2}^6 within radius 1."""
import time
import random
import numpy as np

# Deadline guard: the evaluator kills at 150s; keep a safety margin.
_DEADLINE = 130.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE - _elapsed()


def _all_words(q, n):
    """Return all q^n words as a list of tuples (or numpy array)."""
    return np.array(np.meshgrid(*[np.arange(q)] * n, indexing='ij')).reshape(n, -1).T


def _hamming_dist_matrix(codewords, all_words):
    """Compute Hamming distance from each codeword to each word.
    codewords: (m, n) array
    all_words: (N, n) array
    Returns (m, N) array of distances.
    """
    # Use broadcasting: |c_i - w_j| summed over positions, but only count non-zero
    # Actually: Hamming distance = number of positions where they differ.
    # For small n, we can do: (codewords[:, None, :] != all_words[None, :, :]).sum(axis=2)
    # But for large N this is memory heavy. Let's do it in chunks.
    m = codewords.shape[0]
    N = all_words.shape[0]
    n = codewords.shape[1]
    # For n=6, m~73, N=729: 73*729*6 = ~318k comparisons, fine.
    diff = (codewords[:, None, :] != all_words[None, :, :])
    return diff.sum(axis=2)


def _is_covered(codewords, all_words, radius):
    """Check if every word is within radius of some codeword."""
    dists = _hamming_dist_matrix(codewords, all_words)
    return (dists.min(axis=0) <= radius).all()


def _greedy_cover(q, n, radius):
    """Greedy set cover: start empty, repeatedly add the word that covers the most uncovered words."""
    all_words = _all_words(q, n)
    N = all_words.shape[0]
    # Precompute for each candidate word (each of the N words), which words it covers
    # A word c covers word w if Hamming(c, w) <= radius.
    # For radius=1, n=6, q=3: each word covers 1 + 6*2 = 13 words.
    # We'll represent coverage as bitmasks or boolean arrays.
    # N=729, so we can use a boolean array of size N.
    
    # Precompute coverage sets: for each word i, the set of words it covers.
    # cov[i] = set of j such that Hamming(word_i, word_j) <= radius
    cov = np.zeros((N, N), dtype=bool)
    for i in range(N):
        dists = np.sum(all_words[i] != all_words, axis=1)
        cov[i] = dists <= radius
    
    uncovered = np.ones(N, dtype=bool)
    uncovered_count = N
    codewords = []
    
    while uncovered_count > 0:
        # For each candidate not yet in codewords, count how many uncovered words it covers
        # candidates: all N words (we can pick any word as a codeword)
        gains = cov[:, uncovered].sum(axis=1)  # (N,)
        # Pick the candidate with max gain
        best = np.argmax(gains)
        if gains[best] == 0:
            break  # shouldn't happen
        codewords.append(all_words[best].tolist())
        uncovered[cov[best]] = False
        uncovered_count -= gains[best]
        if _remaining() < 5:
            break
    
    return codewords


def _local_search(codewords, q, n, radius, time_limit=10):
    """Try to remove or swap codewords to reduce count while maintaining coverage.
    Returns a (possibly smaller) list of codewords."""
    all_words = _all_words(q, n)
    N = all_words.shape[0]
    end_time = time.time() + time_limit
    codewords = [list(c) for c in codewords]
    
    while time.time() < end_time:
        improved = False
        m = len(codewords)
        if m == 0:
            break
        
        # Try removal: for each codeword, check if removing it still covers everything.
        # Efficiently: compute for each word which codewords cover it.
        # cover_count[w] = number of codewords within radius of w.
        # If all cover_count[w] >= 1, and for the codeword being removed, 
        # every word it uniquely covers has another cover, then removal is valid.
        
        # Compute coverage counts
        cw = np.array(codewords)
        dists = _hamming_dist_matrix(cw, all_words)  # (m, N)
        within = dists <= radius  # (m, N) bool
        cover_count = within.sum(axis=0)  # (N,)
        
        # Try each removal
        for i in range(m):
            if time.time() > end_time:
                break
            # If we remove codeword i, words with cover_count==1 that were covered by i become uncovered.
            # i.e., words where within[i, w] is True and cover_count[w] == 1.
            unique_to_i = within[i] & (cover_count == 1)
            if not unique_to_i.any():
                # Safe to remove
                codewords.pop(i)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swap: replace codeword i with some other word j (not in codewords)
        # This is more expensive. For now, just do removals.
        # If no removal possible, try a simple swap heuristic:
        # Find the codeword with the most "redundant" coverage (covers many words that are also covered by others)
        # and try replacing it with a word that covers many uniquely-uncovered... but everything is covered.
        # Actually, swap: remove i, add j. Valid if after removal, all words still covered by remaining,
        # OR the words that become uncovered are covered by j.
        # Let's try: for each i, for a sample of candidate j's, check validity.
        
        if len(codewords) <= 73:
            # We're at or below known optimum, stop trying to reduce
            break
        
        # Attempt swaps: for each codeword i, try replacing with a few random words
        random.shuffle(range(m))
        swap_done = False
        for i in range(m):
            if time.time() > end_time:
                break
            # Temporarily remove i
            remaining = [c for k, c in enumerate(codewords) if k != i]
            cw_rem = np.array(remaining) if remaining else np.empty((0, n))
            if len(remaining) == 0:
                # Can't remove if it's the only one
                continue
            dists_rem = _hamming_dist_matrix(cw_rem, all_words)
            within_rem = dists_rem <= radius
            still_covered = within_rem.any(axis=0)
            uncovered_after_removal = ~still_covered
            
            if not uncovered_after_removal.any():
                # This shouldn't happen if we already tried removals, but just in case
                codewords.pop(i)
                swap_done = True
                break
            
            # Try to find a word j that covers all uncovered_after_removal
            # j must satisfy: for all w in uncovered_after_removal, Hamming(j, w) <= radius
            # Also j should ideally cover more to allow future removals.
            # For radius=1, j must be within 1 of each uncovered word.
            # The set of words within 1 of a word w is the "ball" of radius 1.
            # Intersection of balls for all uncovered words gives valid j's.
            
            # For small uncovered set, enumerate candidates.
            unc_idx = np.where(uncovered_after_removal)[0]
            if len(unc_idx) > 8:
                continue  # Too many to easily find a common cover
            
            # Find intersection of balls
            # Ball around w: all words at Hamming dist <= radius from w
            # For radius=1: w itself + all single-position changes
            candidate_set = None
            for w_idx in unc_idx:
                w = all_words[w_idx]
                ball = set()
                ball.add(tuple(w))
                for pos in range(n):
                    for val in range(q):
                        if val == w[pos]:
                            continue
                        new_w = w.copy()
                        new_w[pos] = val
                        ball.add(tuple(new_w))
                if candidate_set is None:
                    candidate_set = ball
                else:
                    candidate_set &= ball
            
            if candidate_set is None or len(candidate_set) == 0:
                continue
            
            # Pick a candidate j from candidate_set
            # Prefer one that covers the most total words (for future flexibility)
            best_j = None
            best_gain = -1
            for j_tuple in list(candidate_set)[:20]:  # Limit candidates
                j_arr = np.array(j_tuple)
                dists_j = np.sum(j_arr != all_words, axis=1)
                gain = (dists_j <= radius).sum()
                if gain > best_gain:
                    best_gain = gain
                    best_j = j_tuple
            
            if best_j is not None:
                codewords.pop(i)
                codewords.append(list(best_j))
                swap_done = True
                break
        
        if not swap_done:
            # No improvement possible with current heuristic
            break
    
    return codewords


def construct(q, n, radius):
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = 130.0
    
    # For the specific task: q=3, n=6, radius=1
    # Known optimum: 73, lower bound: 71
    
    if q == 3 and n == 6 and radius == 1:
        # Use a specialized approach
        # Start with a good initial code
        # The tetracode-based construction gives 81 words.
        # Let's start with greedy and then local search.
        
        # Greedy cover
        codewords = _greedy_cover(q, n, radius)
        
        if _remaining() < 10:
            return codewords
        
        # Local search to reduce
        codewords = _local_search(codewords, q, n, radius, time_limit=max(1, int(_remaining() - 5)))
        
        return codewords
    else:
        # General case: greedy
        return _greedy_cover(q, n, radius)
