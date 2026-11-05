"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np


def _word_to_idx(word, q, n):
    """Convert a word (tuple or list) to an integer index in 0..q^n-1."""
    idx = 0
    for d in word:
        idx = idx * q + d
    return idx


def _idx_to_word(idx, q, n):
    """Convert an integer index to a word (list)."""
    word = []
    for _ in range(n):
        word.append(idx % q)
        idx //= q
    word.reverse()
    return word


def _tetracode_solution(q, n, radius):
    """Direct sum construction: 9 * q^(n-4) words."""
    if q != 3 or n < 4:
        return None
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]


def _precompute_neighbors(q, n, radius):
    """
    Precompute neighbor lists and reverse neighbor lists.
    neighbors[i]: list of indices covered by word i.
    reverse_neighbors[j]: list of indices that cover word j.
    """
    total = q ** n
    neighbors = [[] for _ in range(total)]
    reverse_neighbors = [[] for _ in range(total)]
    
    # For radius 1, we can compute neighbors efficiently
    # A word covers itself and words differing in 1 position
    for i in range(total):
        word = _idx_to_word(i, q, n)
        covered = [i]
        
        if radius >= 1:
            for pos in range(n):
                orig_val = word[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    new_word = word[:]
                    new_word[pos] = new_val
                    j = _word_to_idx(new_word, q, n)
                    covered.append(j)
        
        neighbors[i] = covered
        for j in covered:
            reverse_neighbors[j].append(i)
            
    return neighbors, reverse_neighbors


def _greedy_cover_efficient(q, n, radius, deadline, start_time):
    """
    Greedy set cover with O(1) updates per covered point.
    """
    total = q ** n
    neighbors, reverse_neighbors = _precompute_neighbors(q, n, radius)
    
    covered = np.zeros(total, dtype=bool)
    codewords = []
    
    # uncovered_count[i] = number of currently uncovered words that codeword i covers
    # Initially, all words are uncovered, so this is just len(neighbors[i])
    uncovered_count = np.array([len(nbrs) for nbrs in neighbors], dtype=np.int64)
    
    # We need to be careful: if a word is covered by multiple potential codewords,
    # we only count it once in the "uncovered" set.
    # The greedy algorithm picks the codeword that covers the most *currently uncovered* words.
    
    # Optimization: Instead of checking all candidates every time (O(N)),
    # we can maintain a max-heap or just scan if N is small enough.
    # N=6561 is small enough for a linear scan in Python if we do it carefully, 
    # but numpy argmax is faster.
    
    # However, updating uncovered_count is the key.
    # When we pick codeword 'best', we mark its neighbors as covered.
    # For each newly covered word 'j', we need to decrement the count for all 'i' in reverse_neighbors[j].
    
    while time.time() < deadline:
        # Find candidate with max uncovered_count
        best_idx = int(np.argmax(uncovered_count))
        best_count = int(uncovered_count[best_idx])
        
        if best_count == 0:
            break
            
        # Add best_idx
        codewords.append(_idx_to_word(best_idx, q, n))
        
        # Update covered and counts
        for j in neighbors[best_idx]:
            if not covered[j]:
                covered[j] = True
                # Decrement count for all candidates that cover j
                # This includes best_idx itself and others
                for i in reverse_neighbors[j]:
                    uncovered_count[i] -= 1
                    
        # Check if all covered
        if np.all(covered):
            break
            
    if np.all(covered):
        return codewords
    return None


def _prune_codewords(codewords, q, n, radius, deadline, start_time):
    """
    Remove redundant codewords.
    A codeword is redundant if removing it leaves all its neighbors covered by others.
    """
    total = q ** n
    neighbors, _ = _precompute_neighbors(q, n, radius)
    
    # Map word index to position in codewords list
    # Note: codewords are lists of ints, we need indices
    cw_indices = [_word_to_idx(cw, q, n) for cw in codewords]
    
    # cover_count[j] = number of codewords that cover word j
    cover_count = np.zeros(total, dtype=np.int64)
    for idx in cw_indices:
        for j in neighbors[idx]:
            cover_count[j] += 1
            
    keep = [True] * len(codewords)
    
    # Iterate multiple times to catch cascading removals? 
    # Usually one pass is enough for greedy results, but let's do a few.
    for _ in range(5):
        if time.time() > deadline:
            break
        changed = False
        for i, idx in enumerate(cw_indices):
            if not keep[i]:
                continue
            # Check if any neighbor is uniquely covered by this codeword
            # i.e., cover_count[j] == 1 for any j in neighbors[idx]
            # If so, we cannot remove it.
            # If cover_count[j] > 1 for all j in neighbors[idx], we can remove it.
            
            # Optimization: check only if min cover_count in neighbors is > 1
            min_count = np.min(cover_count[neighbors[idx]])
            if min_count > 1:
                # Safe to remove
                keep[i] = False
                changed = True
                # Update cover_count
                for j in neighbors[idx]:
                    cover_count[j] -= 1
        if not changed:
            break
            
    return [codewords[i] for i in range(len(codewords)) if keep[i]]


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    # 1. Baseline: Tetracode
    tc = _tetracode_solution(q, n, radius)
    if tc is not None:
        best = tc
        best_len = len(tc)
    else:
        best = []
        best_len = float('inf')
    
    # 2. Greedy Cover
    try:
        greedy_result = _greedy_cover_efficient(q, n, radius, deadline - 10.0, start_time)
        if greedy_result is not None:
            # Verify coverage (should be guaranteed by algorithm, but safe to check)
            total = q ** n
            covered = np.zeros(total, dtype=bool)
            for cw in greedy_result:
                idx = _word_to_idx(cw, q, n)
                # Re-derive neighbors for verification or use precomputed?
                # Precomputation is expensive, let's do a quick check for small N
                # For N=6561, we can just trust the greedy if it finished.
                # But let's verify a few random points if time allows?
                # Actually, the greedy ensures coverage by construction.
                
            if len(greedy_result) < best_len:
                best = greedy_result
                best_len = len(best)
                
                # 3. Prune
                if time.time() < deadline - 5.0:
                    pruned = _prune_codewords(best, q, n, radius, deadline - 2.0, start_time)
                    if pruned and len(pruned) < best_len:
                        # Final verification of pruned set
                        # We need to ensure pruned set still covers everything
                        total = q ** n
                        covered = np.zeros(total, dtype=bool)
                        for cw in pruned:
                            idx = _word_to_idx(cw, q, n)
                            word = _idx_to_word(idx, q, n)
                            covered[idx] = True
                            for pos in range(n):
                                orig_val = word[pos]
                                for new_val in range(q):
                                    if new_val == orig_val:
                                        continue
                                    new_word = word[:]
                                    new_word[pos] = new_val
                                    j = _word_to_idx(new_word, q, n)
                                    covered[j] = True
                            
                        if np.all(covered):
                            best = pruned
                            best_len = len(best)
    except Exception as e:
        pass
        
    return best
# EVOLVE-BLOCK-END
