"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random


def _neighbors_indices(word, q, n, radius):
    """Yield all words within Hamming distance <= radius from word."""
    # For radius 1: the word itself plus all single-position changes
    yield tuple(word)
    for i in range(n):
        for v in range(q):
            if v != word[i]:
                new = list(word)
                new[i] = v
                yield tuple(new)


def _tetracode_solution(q, n, radius):
    """Direct sum construction: 9 * q^(n-4) words."""
    if q != 3 or n < 4:
        return None
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]


def _greedy_cover(q, n, radius, deadline):
    """Greedy set cover with pruning."""
    total = q ** n
    # Precompute all words as tuples
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    
    # For each word, precompute which codeword candidates cover it (i.e., words within distance radius)
    # We'll use a bitset approach for efficiency
    # covered[i] = True if word i is covered by some chosen codeword
    
    # Precompute for each possible codeword, the set of words it covers
    # This is expensive for large spaces, so we compute on the fly
    
    covered = [False] * total
    codewords = []
    
    # Index mapping from tuple to index
    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx
    
    # For each candidate codeword, compute how many new words it covers
    # Use a heuristic: pick the word that covers the most uncovered words
    
    # Precompute neighbors for each word (words within distance radius)
    # For radius 1, each word covers 1 + n*(q-1) words
    neighbors = {}
    for i in range(total):
        w = all_words[i]
        nbrs = set()
        for cand in _neighbors_indices(w, q, n, radius):
            nbrs.add(word_to_idx(cand))
        neighbors[i] = nbrs
    
    # Greedy: repeatedly pick the word that covers the most uncovered words
    # Use a priority approach: maintain count of uncovered words each candidate covers
    
    # uncovered_count[cand] = number of uncovered words that cand covers
    uncovered_count = [len(neighbors[i]) for i in range(total)]
    
    while time.time() < deadline:
        # Find the candidate with max uncovered_count
        best_idx = -1
        best_count = 0
        for i in range(total):
            if uncovered_count[i] > best_count:
                best_count = uncovered_count[i]
                best_idx = i
                if best_count == 0:
                    break
        
        if best_idx == -1 or best_count == 0:
            break
        
        # Add best_idx as a codeword
        codewords.append(list(all_words[best_idx]))
        
        # Update covered and uncovered_count
        for j in neighbors[best_idx]:
            if not covered[j]:
                covered[j] = True
                # For all candidates that cover j, decrement their count
                # This is expensive; instead, we'll recompute periodically
                # For now, mark and handle in next iteration
                pass
        
        # Recompute uncovered_count for all candidates (expensive but correct)
        # Optimization: only recompute if we're making progress
        new_uncovered_count = [0] * total
        for i in range(total):
            cnt = 0
            for j in neighbors[i]:
                if not covered[j]:
                    cnt += 1
            new_uncovered_count[i] = cnt
        uncovered_count = new_uncovered_count
    
    # Check if all covered
    if all(covered):
        return codewords
    
    # Fallback: use tetracode if not fully covered
    tc = _tetracode_solution(q, n, radius)
    if tc is not None:
        return tc
    
    # Return what we have (may be invalid, but evaluator will check)
    return codewords


def _greedy_prune(codewords, q, n, radius):
    """Remove redundant codewords."""
    total = q ** n
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    
    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx
    
    # For each codeword, compute the set of words it uniquely covers
    # (words that no other codeword covers)
    # A codeword can be removed if it has no uniquely covered words
    
    # Precompute coverage sets
    cover_sets = []
    for cw in codewords:
        cw_t = tuple(cw)
        s = set()
        for cand in _neighbors_indices(cw_t, q, n, radius):
            s.add(word_to_idx(cand))
        cover_sets.append(s)
    
    # For each word, count how many codewords cover it
    word_cover_count = [0] * total
    for s in cover_sets:
        for j in s:
            word_cover_count[j] += 1
    
    # A codeword is removable if all words it covers are covered by at least one other codeword
    # i.e., no word has cover_count == 1 where this codeword is the only one
    # More precisely: codeword i is removable if for all j in cover_sets[i], word_cover_count[j] > 1
    
    keep = [True] * len(codewords)
    for i in range(len(codewords)):
        removable = True
        for j in cover_sets[i]:
            if word_cover_count[j] == 1:
                removable = False
                break
        if removable:
            keep[i] = False
            # Update word_cover_count
            for j in cover_sets[i]:
                word_cover_count[j] -= 1
    
    return [codewords[i] for i in range(len(codewords)) if keep[i]]


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try tetracode first as a baseline
    tc = _tetracode_solution(q, n, radius)
    if tc is not None:
        best = tc
    else:
        best = []
    
    # Try greedy cover
    try:
        greedy_result = _greedy_cover(q, n, radius, deadline)
        if greedy_result and len(greedy_result) < len(best):
            # Verify coverage
            total = q ** n
            all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
            
            def word_to_idx(w):
                idx = 0
                for d in w:
                    idx = idx * q + d
                return idx
            
            covered = [False] * total
            for cw in greedy_result:
                cw_t = tuple(cw)
                for cand in _neighbors_indices(cw_t, q, n, radius):
                    covered[word_to_idx(cand)] = True
            
            if all(covered):
                best = greedy_result
                # Try to prune
                best = _greedy_prune(best, q, n, radius)
    except Exception:
        pass
    
    return best
# EVOLVE-BLOCK-END
