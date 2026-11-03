import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    total_words = q ** n
    
    # Generate all words as a 2D array: shape (total_words, n)
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    
    # Build adjacency matrix: adj[i,j] = True if word j is within radius of word i
    # For radius=1: adj[i,j] = (i==j) or (hamming_dist(i,j) <= radius)
    if radius == 1:
        # Vectorized: for each position, check if words differ in at most 1 position
        # Two words are within distance 1 if they are equal or differ in exactly one position
        # adj[i,j] = sum over positions of (all_words[i,pos] != all_words[j,pos]) <= radius
        # This is expensive to compute fully, so use a smarter approach for radius=1
        # For radius=1: neighbor of word i is i itself plus all words that differ in exactly one position
        # Number of neighbors = 1 + n*(q-1)
        
        # Build neighbor list using vectorized operations
        # For each position p, and each value v != all_words[i,p], the neighbor is the word with v at position p
        # We can compute this efficiently
        
        # Alternative: compute the full distance matrix using broadcasting
        # dist[i,j] = number of positions where all_words[i] != all_words[j]
        # This requires n comparisons of shape (total_words, total_words)
        # For 729 words and n=6, this is 6 * 729 * 729 ≈ 3.2M operations - very fast
        
        diff = (all_words[:, np.newaxis, :] != all_words[np.newaxis, :, :])  # shape (total_words, total_words, n)
        dist = diff.sum(axis=2)  # shape (total_words, total_words)
        adj = dist <= radius  # shape (total_words, total_words), bool
    else:
        # General case (shouldn't be needed for this task but handle it)
        diff = (all_words[:, np.newaxis, :] != all_words[np.newaxis, :, :])
        dist = diff.sum(axis=2)
        adj = dist <= radius
    
    # adj[i] is a boolean mask of all words within radius of word i
    # adj[i, i] = True always
    
    # Greedy set cover using vectorized operations
    code_set = np.zeros(total_words, dtype=bool)
    uncovered = np.ones(total_words, dtype=bool)
    
    # Symmetry breaking: fix first codeword to all zeros (index 0)
    code_set[0] = True
    uncovered[adj[0]] = False
    
    # Greedy: repeatedly pick the word that covers the most uncovered words
    while uncovered.any():
        if time.time() > deadline - 25:
            break
        
        # For each candidate word i (not already in code), count how many uncovered words it covers
        # coverage[i] = sum of uncovered[j] for j where adj[i,j] is True
        # = (adj @ uncovered_vector)[i]
        uncovered_vec = uncovered.astype(np.int32)
        coverage = adj @ uncovered_vec  # shape (total_words,)
        
        # Exclude already-selected words
        coverage[code_set] = -1
        
        best_idx = np.argmax(coverage)
        if coverage[best_idx] <= 0:
            break
        
        code_set[best_idx] = True
        uncovered[adj[best_idx]] = False
    
    # Local search: remove redundant codewords
    # A codeword i is redundant if all words it covers are also covered by other codewords
    # i.e., adj[i] is a subset of (OR of adj[j] for j != i in code)
    
    def remove_redundant(code_set, adj, uncovered, deadline):
        """Remove redundant codewords from code_set. Returns modified code_set."""
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            code_indices = np.where(code_set)[0]
            for i in code_indices:
                if time.time() > deadline - 10:
                    break
                # Check if removing i still covers everything
                # Coverage without i: OR of adj[j] for all j in code_set, j != i
                others_mask = code_set.copy()
                others_mask[i] = False
                if not others_mask.any():
                    continue
                other_coverage = adj[others_mask].any(axis=0)
                if adj[i].all() and adj[i][i] and other_coverage[i]:
                    # All words covered by i are also covered by others
                    # Actually check: is adj[i] subset of other_coverage?
                    if np.all(other_coverage[adj[i]]):
                        code_set[i] = False
                        # Recompute uncovered
                        uncovered = ~adj[code_set].any(axis=0) if code_set.any() else np.ones(total_words, dtype=bool)
                        improved = True
                        break
        return code_set
    
    code_set = remove_redundant(code_set, adj, uncovered, deadline)
    
    # Local search: try to replace codewords to enable further removals
    # Strategy: for each codeword, try replacing it with a different word that covers the same "unique" words
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        code_indices = np.where(code_set)[0]
        for i in code_indices:
            if time.time() > deadline - 5:
                break
            
            # Find words uniquely covered by i (would be uncovered if i removed)
            others_mask = code_set.copy()
            others_mask[i] = False
            if not others_mask.any():
                continue
            other_coverage = adj[others_mask].any(axis=0)
            unique_to_i = adj[i] & ~other_coverage
            
            if not unique_to_i.any():
                # i is redundant
                code_set[i] = False
                improved = True
                break
            
            # Try to find a replacement word c' such that adj[c'] covers unique_to_i
            # and c' is not already in code_set
            # For efficiency, precompute which words cover each unique word
            # candidate_mask[c'] = all(unique_to_i[j] implies adj[c',j]) for all j in unique_to_i
            # = all(adj[c', unique_to_i_indices])
            
            unique_indices = np.where(unique_to_i)[0]
            # For each candidate c', check if it covers all unique_indices
            # adj[:, unique_indices] has shape (total_words, len(unique_indices))
            # candidate is valid if all entries are True
            covers_unique = adj[:, unique_indices].all(axis=1)  # shape (total_words,)
            
            # Exclude already-selected and self
            candidates = covers_unique & ~code_set
            candidates[i] = False
            
            if candidates.any():
                # Pick the candidate that covers the most total uncovered or is "best" in some way
                # For now, just pick the first one
                best_candidate = np.argmax(candidates)
                code_set[i] = False
                code_set[best_candidate] = True
                improved = True
                break
    
    # Final redundancy removal pass
    code_set = remove_redundant(code_set, adj, uncovered, deadline)
    
    # Convert to list of codewords
    code_indices = np.where(code_set)[0]
    code = [all_words[i].tolist() for i in code_indices]
    
    # Ensure distinctness (should already be distinct since we use a set)
    seen = set()
    unique_code = []
    for c in code:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            unique_code.append(c)
    
    # Final validation: check that all words are covered
    if code_set.any():
        final_covered = adj[code_set].any(axis=0)
        if not final_covered.all():
            # Fallback: greedy from scratch
            code_set = np.zeros(total_words, dtype=bool)
            uncovered = np.ones(total_words, dtype=bool)
            code_set[0] = True
            uncovered[adj[0]] = False
            
            while uncovered.any() and time.time() < deadline:
                uncovered_vec = uncovered.astype(np.int32)
                coverage = adj @ uncovered_vec
                coverage[code_set] = -1
                best_idx = np.argmax(coverage)
                if coverage[best_idx] <= 0:
                    break
                code_set[best_idx] = True
                uncovered[adj[best_idx]] = False
            
            code_indices = np.where(code_set)[0]
            code = [all_words[i].tolist() for i in code_indices]
            seen = set()
            unique_code = []
            for c in code:
                t = tuple(c)
                if t not in seen:
                    seen.add(t)
                    unique_code.append(c)
    
    return unique_code
