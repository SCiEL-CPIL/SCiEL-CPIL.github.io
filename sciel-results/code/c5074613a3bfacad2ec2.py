"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import numpy as np
from collections import defaultdict


def _hamming_dist(a, b):
    return sum(1 for i in range(len(a)) if a[i] != b[i])


def _neighbors(word, q, n, radius=1):
    """Generate all words within Hamming distance radius of word."""
    result = [list(word)]
    if radius >= 1:
        for i in range(n):
            for val in range(q):
                if val != word[i]:
                    new_word = list(word)
                    new_word[i] = val
                    result.append(new_word)
    return result


def _build_hamming_cosets(q, n):
    """
    Build the cosets of the ternary Hamming code H(q, n).
    The Hamming code has dimension n-1, so there are q cosets.
    Each coset contains q^(n-1) words.
    """
    if q != 3 or n != 6:
        return None, None
    
    # Ternary Hamming code H(3,6): parity check matrix is 2x6
    # Columns are all non-zero vectors in F_3^2 up to scalar
    # Standard form: columns are (1,0), (0,1), (1,1), (1,2), (2,1), (2,2)
    H = np.array([
        [1, 0, 1, 1, 2, 2],
        [0, 1, 1, 2, 1, 2]
    ], dtype=int)
    
    all_words = list(itertools.product(range(3), repeat=6))
    
    # Compute syndrome for each word
    coset_representatives = {}
    word_to_coset = {}
    
    for word in all_words:
        w = np.array(word, dtype=int)
        syndrome = tuple((H @ w) % 3)
        
        if syndrome not in coset_representatives:
            coset_representatives[syndrome] = list(word)
        
        word_to_coset[tuple(word)] = syndrome
    
    return coset_representatives, word_to_coset


def _get_coset_members(coset_rep, word_to_coset, all_words):
    """Get all words in the coset represented by coset_rep."""
    target_syndrome = word_to_coset[tuple(coset_rep)]
    members = []
    for w in all_words:
        if word_to_coset[w] == target_syndrome:
            members.append(list(w))
    return members


def _precompute_coverage(q, n, radius):
    """
    Precompute for each word, which cosets can cover it.
    A word is covered by a codeword if their Hamming distance <= radius.
    """
    all_words = list(itertools.product(range(q), repeat=n))
    
    # For each word, find all words within distance radius
    word_neighbors = {}
    for w in all_words:
        neighbors = _neighbors(list(w), q, n, radius)
        word_neighbors[w] = [tuple(nb) for nb in neighbors]
    
    return all_words, word_neighbors


def _greedy_initial_solution(q, n, radius, coset_representatives, word_to_coset, all_words, word_neighbors):
    """
    Greedy set cover: start with one representative from each coset,
    then iteratively add the codeword that covers the most uncovered words.
    """
    # Start with all coset representatives
    initial_codewords = [list(rep) for rep in coset_representatives.values()]
    
    covered = set()
    for cw in initial_codewords:
        cw_tuple = tuple(cw)
        for nb in word_neighbors[cw_tuple]:
            covered.add(nb)
    
    uncovered = set(all_words) - covered
    
    # Greedy: add codewords that cover most uncovered words
    codewords = initial_codewords[:]
    codeword_set = set(tuple(cw) for cw in codewords)
    
    while uncovered:
        best_cw = None
        best_new_covered = set()
        
        # For each candidate codeword, count how many uncovered words it covers
        for cw in all_words:
            if cw in codeword_set:
                continue
            
            new_covered = set()
            for nb in word_neighbors[cw]:
                if nb in uncovered:
                    new_covered.add(nb)
            
            if len(new_covered) > len(best_new_covered):
                best_new_covered = new_covered
                best_cw = cw
        
        if best_cw is None or len(best_new_covered) == 0:
            break
        
        codewords.append(list(best_cw))
        codeword_set.add(best_cw)
        uncovered -= best_new_covered
    
    return codewords


def _remove_redundant(codewords, q, n, radius, all_words, word_neighbors):
    """Remove redundant codewords from the list."""
    codeword_set = set(tuple(cw) for cw in codewords)
    covered = set()
    
    for cw in codewords:
        cw_tuple = tuple(cw)
        for nb in word_neighbors[cw_tuple]:
            covered.add(nb)
    
    improved = True
    while improved:
        improved = False
        for i in range(len(codewords)):
            cw = codewords[i]
            cw_tuple = tuple(cw)
            
            # Check if all neighbors of cw are covered by other codewords
            all_covered = True
            for nb in word_neighbors[cw_tuple]:
                if nb not in covered:
                    all_covered = False
                    break
                
                # Check if nb is covered by another codeword
                is_covered_by_other = False
                for j, other_cw in enumerate(codewords):
                    if j == i:
                        continue
                    other_tuple = tuple(other_cw)
                    if nb in word_neighbors[other_tuple]:
                        is_covered_by_other = True
                        break
                
                if not is_covered_by_other:
                    all_covered = False
                    break
            
            if all_covered:
                # Remove this codeword
                del codewords[i]
                codeword_set.discard(cw_tuple)
                
                # Recalculate covered
                covered = set()
                for cw in codewords:
                    cw_tuple = tuple(cw)
                    for nb in word_neighbors[cw_tuple]:
                        covered.add(nb)
                
                improved = True
                break
    
    return codewords


def _local_search_improve(codewords, q, n, radius, all_words, word_neighbors, max_iterations=100):
    """
    Local search: try to replace codewords with better ones or remove redundant ones.
    """
    best_codewords = codewords[:]
    best_size = len(codewords)
    
    random.seed(42)
    
    for iteration in range(max_iterations):
        current = best_codewords[:]
        current_set = set(tuple(cw) for cw in current)
        
        # Calculate covered words
        covered = set()
        for cw in current:
            cw_tuple = tuple(cw)
            for nb in word_neighbors[cw_tuple]:
                covered.add(nb)
        
        # Try to remove a random codeword
        if len(current) > 0:
            idx = random.randint(0, len(current) - 1)
            cw_to_remove = current[idx]
            cw_tuple = tuple(cw_to_remove)
            
            # Find words that would become uncovered
            would_uncover = set()
            for nb in word_neighbors[cw_tuple]:
                if nb not in covered:
                    would_uncover.add(nb)
                else:
                    # Check if covered by another codeword
                    is_covered_by_other = False
                    for j, other_cw in enumerate(current):
                        if j == idx:
                            continue
                        other_tuple = tuple(other_cw)
                        if nb in word_neighbors[other_tuple]:
                            is_covered_by_other = True
                            break
                    if not is_covered_by_other:
                        would_uncover.add(nb)
            
            if not would_uncover:
                # Can remove
                del current[idx]
                current_set.discard(cw_tuple)
                best_codewords = current[:]
                best_size = len(current)
                break
            
            else:
                # Try to find a replacement that covers would_uncover and more
                found_replacement = False
                # Try random candidates
                for _ in range(50):
                    candidate = random.choice(all_words)
                    if candidate in current_set:
                        continue
                    
                    # Check if candidate covers all would_uncover
                    covers_all = True
                    for w in would_uncover:
                        if w not in word_neighbors[candidate]:
                            covers_all = False
                            break
                    
                    if not covers_all:
                        continue
                    
                    # Check if this replacement reduces the total size or is neutral
                    # Try swapping
                    new_current = current[:]
                    new_current[idx] = list(candidate)
                    
                    # Calculate new covered set
                    new_covered = set()
                    for cw in new_current:
                        cw_tuple = tuple(cw)
                        for nb in word_neighbors[cw_tuple]:
                            new_covered.add(nb)
                    
                    # If this covers everything and same size, accept with some probability
                    if new_covered == covered or len(new_covered) >= len(covered):
                        # Check if we can now remove another codeword
                        # For simplicity, just accept if same size
                        best_codewords = new_current[:]
                        best_size = len(new_current)
                        found_replacement = True
                        break
        
        if not found_replacement and iteration % 10 == 0:
            # Try adding a codeword that covers the most uncovered words
            uncovered = set(all_words) - covered
            if uncovered:
                best_cw = None
                best_new_covered = set()
                
                for _ in range(100):
                    candidate = random.choice(all_words)
                    if tuple(candidate) in current_set:
                        continue
                    
                    new_covered = set()
                    for nb in word_neighbors[tuple(candidate)]:
                        if nb in uncovered:
                            new_covered.add(nb)
                    
                    if len(new_covered) > len(best_new_covered):
                        best_new_covered = new_covered
                        best_cw = candidate
                
                if best_cw is not None and len(best_new_covered) > 0:
                    # Add this codeword, then try to remove redundant ones
                    current.append(list(best_cw))
                    current_set.add(tuple(best_cw))
                    current = _remove_redundant(current, q, n, radius, all_words, word_neighbors)
                    
                    if len(current) < best_size:
                        best_codewords = current[:]
                        best_size = len(current)
    
    return best_codewords


def construct(q, n, radius):
    if n == 0:
        return [[]]
    
    # For small cases, use direct construction
    if n <= 4:
        if n == 4:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        elif n == 3:
            return [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        elif n == 2:
            return [[0, 0], [1, 1], [2, 2], [0, 1], [1, 0], [2, 2], [0, 2], [2, 0], [1, 2], [2, 1]]
        else:
            return [[i] for i in range(q)]
    
    # For the main case: q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        # Build Hamming cosets
        coset_representatives, word_to_coset = _build_hamming_cosets(q, n)
        all_words, word_neighbors = _precompute_coverage(q, n, radius)
        
        # Get initial solution using greedy
        codewords = _greedy_initial_solution(q, n, radius, coset_representatives, word_to_coset, all_words, word_neighbors)
        
        # Remove redundant codewords
        codewords = _remove_redundant(codewords, q, n, radius, all_words, word_neighbors)
        
        # Local search to improve
        codewords = _local_search_improve(codewords, q, n, radius, all_words, word_neighbors, max_iterations=200)
        
        return codewords
    
    # Fallback for other cases: use greedy set cover
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    all_words, word_neighbors = _precompute_coverage(q, n, radius)
    
    # Greedy set cover
    codewords = []
    covered = set()
    
    while len(covered) < len(all_words):
        best_cw = None
        best_new_covered = set()
        
        for cw in all_words:
            if tuple(cw) in set(tuple(c) for c in codewords):
                continue
            
            new_covered = set()
            for nb in word_neighbors[tuple(cw)]:
                if nb not in covered:
                    new_covered.add(nb)
            
            if len(new_covered) > len(best_new_covered):
                best_new_covered = new_covered
                best_cw = cw
        
        if best_cw is None or len(best_new_covered) == 0:
            break
        
        codewords.append(list(best_cw))
        covered |= best_new_covered
    
    # Remove redundant
    codewords = _remove_redundant(codewords, q, n, radius, all_words, word_neighbors)
    
    return codewords
# EVOLVE-BLOCK-END
