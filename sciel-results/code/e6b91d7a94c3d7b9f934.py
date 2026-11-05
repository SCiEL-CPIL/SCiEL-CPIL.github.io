"""Rewritten program for ternary covering code K_3(8,1)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product as _it_product

_start_time = time.time()
_DEADLINE = 120.0  # safety margin before 150s kill


def _tetracode_solution(q, n, radius):
    """Direct sum of ternary tetracode (9 words len 4, perfect radius 1) with full tails."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    words = []
    for h in tetracode:
        for tail in _it_product(range(q), repeat=n - 4):
            words.append(h + list(tail))
    return words


def _build_cov_matrix(q, n, radius):
    """Build the full coverage matrix: cov_matrix[i] is a bool array of words within radius of word i."""
    space_size = q ** n
    cov = np.zeros((space_size, space_size), dtype=bool)
    digits = np.zeros((space_size, n), dtype=np.int32)
    for i in range(space_size):
        x = i
        for pos in range(n - 1, -1, -1):
            digits[i, pos] = x % q
            x //= q
    cov[np.arange(space_size), np.arange(space_size)] = True
    if radius >= 1:
        for pos in range(n):
            weight = q ** (n - 1 - pos)
            for d in range(q):
                mask = digits[:, pos] != d
                if not np.any(mask):
                    continue
                src_indices = np.where(mask)[0]
                targets = src_indices + (d - digits[src_indices, pos]) * weight
                cov[src_indices, targets] = True
    return cov, digits


def _word_to_idx(q, word):
    idx = 0
    for d in word:
        idx = idx * q + d
    return idx


def _idx_to_word(q, n, idx):
    word = []
    for pos in range(n - 1, -1, -1):
        word.append(idx % q)
        idx //= q
    word.reverse()
    return word


def _fast_prune(q, n, radius, words, time_limit=120.0):
    """Efficient iterative pruning: remove codewords that are redundant."""
    space_size = q ** n
    cov, digits = _build_cov_matrix(q, n, radius)
    
    word_indices = np.array([_word_to_idx(q, w) for w in words], dtype=np.int64)
    k = len(word_indices)
    
    if k == 0:
        return words
    
    sel_cov = cov[word_indices]  # shape (k, space_size)
    coverage_count = sel_cov.sum(axis=0).astype(np.int32)
    
    active = np.ones(k, dtype=bool)
    
    # Multiple passes until no more removals possible
    improved = True
    while improved:
        if time.time() - _start_time > time_limit - 5:
            break
        improved = False
        
        # Find all codewords that can be removed in this pass
        removable = np.zeros(k, dtype=bool)
        for i in range(k):
            if not active[i]:
                continue
            only_by_me = sel_cov[i] & (coverage_count == 1)
            if not np.any(only_by_me):
                removable[i] = True
        
        if not np.any(removable):
            break
        
        # Remove all removable codewords at once
        for i in range(k):
            if removable[i]:
                coverage_count -= sel_cov[i]
                active[i] = False
                improved = True
        
        if time.time() - _start_time > time_limit - 5:
            break
    
    word_indices = word_indices[active]
    return [_idx_to_word(q, n, int(idx)) for idx in word_indices]


def _local_search_replace(q, n, radius, words, time_limit=120.0):
    """Try to replace each codeword with a different one that maintains coverage,
    then prune again. This is a 1-exchange local search."""
    space_size = q ** n
    cov, digits = _build_cov_matrix(q, n, radius)
    
    word_indices = np.array([_word_to_idx(q, w) for w in words], dtype=np.int64)
    k = len(word_indices)
    
    if k == 0:
        return words
    
    sel_cov = cov[word_indices]
    coverage_count = sel_cov.sum(axis=0).astype(np.int32)
    
    # Precompute neighbor lists for each codeword (within distance 1)
    # neighbors[i] = list of word indices at distance 1 from word_indices[i]
    neighbors = []
    for i in range(k):
        w = word_indices[i]
        nbrs = []
        for pos in range(n):
            weight = q ** (n - 1 - pos)
            cur_digit = (w // weight) % q
            for d in range(q):
                if d != cur_digit:
                    nbrs.append(w + (d - cur_digit) * weight)
        neighbors.append(nbrs)
    
    improved = True
    while improved:
        if time.time() - _start_time > time_limit - 5:
            break
        improved = False
        
        # Try removing each codeword and see if we can re-add a better one
        for i in range(k):
            if time.time() - _start_time > time_limit - 5:
                break
            
            # Check if codeword i can be removed
            only_covered = sel_cov[i] & (coverage_count == 1)
            if np.any(only_covered):
                continue
            
            # Try to find a replacement that covers the "orphaned" positions
            # First, what positions would become uncovered?
            coverage_without_i = coverage_count - sel_cov[i]
            orphaned = np.where(coverage_without_i == 0)[0]
            
            if len(orphaned) == 0:
                # Can simply remove
                coverage_count -= sel_cov[i]
                word_indices = np.delete(word_indices, i)
                sel_cov = np.delete(sel_cov, i, axis=0)
                neighbors.pop(i)
                k -= 1
                improved = True
                break
            
            # Try to find a single codeword that covers all orphaned positions
            # and doesn't break other coverage
            # Check neighbors of codeword i
            found_replacement = False
            for nbr in neighbors[i]:
                # Check if nbr covers all orphaned positions
                nbr_cov = cov[nbr]
                if not np.all(nbr_cov[orphaned]):
                    continue
                
                # Check if adding nbr would cause any position to go from 
                # coverage 1 to coverage 2 (waste) - actually we just need
                # that removing i and adding nbr maintains full coverage
                # Coverage change: -sel_cov[i] + cov[nbr]
                new_coverage = coverage_count - sel_cov[i] + cov[nbr]
                if np.all(new_coverage > 0):
                    # Do the swap
                    coverage_count = new_coverage
                    word_indices[i] = nbr
                    sel_cov[i] = cov[nbr]
                    # Recompute neighbors for i
                    w = nbr
                    neighbors[i] = []
                    for pos in range(n):
                        weight = q ** (n - 1 - pos)
                        cur_digit = (w // weight) % q
                        for d in range(q):
                            if d != cur_digit:
                                neighbors[i].append(w + (d - cur_digit) * weight)
                    improved = True
                    found_replacement = True
                    break
            
            if found_replacement:
                break
    
    return [_idx_to_word(q, n, int(idx)) for idx in word_indices]


def _greedy_cover_fast(q, n, radius, time_limit=120.0):
    """Fast greedy set cover using precomputed neighbor lists."""
    space_size = q ** n
    cov, digits = _build_cov_matrix(q, n, radius)
    
    uncovered_count = cov.sum(axis=1).astype(np.int32)
    uncovered = np.ones(space_size, dtype=bool)
    
    selected = []
    remaining = space_size
    
    while remaining > 0:
        if time.time() - _start_time > time_limit - 5:
            break
        
        best = int(np.argmax(uncovered_count))
        if uncovered_count[best] <= 0:
            break
        selected.append(best)
        
        newly_covered_mask = cov[best] & uncovered
        if not np.any(newly_covered_mask):
            break
        nc_indices = np.where(newly_covered_mask)[0]
        
        decrement = cov[:, nc_indices].sum(axis=1).astype(np.int32)
        uncovered_count -= decrement
        uncovered_count = np.maximum(uncovered_count, 0)
        
        uncovered[newly_covered_mask] = False
        remaining -= len(nc_indices)
        uncovered_count[best] = -1
    
    return [_idx_to_word(q, n, i) for i in selected]


def construct(q, n, radius):
    global _start_time
    _start_time = time.time()
    time_limit = _DEADLINE - 5  # 5s safety margin

    if q != 3 or n != 8 or radius != 1:
        return _tetracode_solution(q, n, radius)

    # Start with tetracode-based solution (729 words)
    seed_words = _tetracode_solution(q, n, radius)
    
    # Fast iterative pruning
    pruned = _fast_prune(q, n, radius, seed_words, time_limit)
    
    # Local search: try 1-exchanges to improve
    pruned = _local_search_replace(q, n, radius, pruned, time_limit)
    
    # Prune again after local search
    pruned = _fast_prune(q, n, radius, pruned, time_limit)
    
    # If time remains, try greedy from scratch and compare
    elapsed = time.time() - _start_time
    if elapsed < time_limit - 20 and len(pruned) > 490:
        greedy_fresh = _greedy_cover_fast(q, n, radius, time_limit=time_limit)
        if len(greedy_fresh) < len(pruned):
            pruned = greedy_fresh
            pruned = _fast_prune(q, n, radius, pruned, time_limit)
            pruned = _local_search_replace(q, n, radius, pruned, time_limit)
            pruned = _fast_prune(q, n, radius, pruned, time_limit)
    
    # Final polish: one more round of pruning
    if time.time() - _start_time < time_limit - 5:
        pruned = _fast_prune(q, n, radius, pruned, time_limit)
    
    return pruned
# EVOLVE-BLOCK-END
