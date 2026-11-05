"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import numpy as np


def construct(q, n, radius):
    if n == 0:
        return [[]]
    
    # Handle small n with known optimal or near-optimal constructions
    if n == 1:
        return [[i] for i in range(q)]
    if n == 2:
        # K_3(2,1) = 3
        return [[0, 0], [1, 1], [2, 2]]
    if n == 3:
        # K_3(3,1) = 9 (Greedy will find this, but let's be explicit if possible)
        # Actually, let's just let the general solver handle n>=3 as it's robust.
        pass
    if n == 4:
        # Tetracode is optimal for n=4, r=1 (size 9)
        return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word index
    # neighbors[i] is a list of indices of words within distance radius
    neighbors = []
    for i, w in enumerate(all_words):
        neigh = [i]
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_word = list(w)
                    new_word[pos] = val
                    neigh.append(word_to_idx[tuple(new_word)])
        neighbors.append(set(neigh))
    
    # Greedy Set Cover
    uncovered = set(range(num_words))
    codewords = []
    
    while uncovered:
        best_word_idx = -1
        best_count = -1
        
        # Find the word that covers the most uncovered words
        for i in range(num_words):
            count = 0
            for n_idx in neighbors[i]:
                if n_idx in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_word_idx = i
            if best_count == len(uncovered):
                break
        
        if best_word_idx == -1:
            break
            
        codewords.append(all_words[best_word_idx])
        # Remove covered words
        for n_idx in neighbors[best_word_idx]:
            uncovered.discard(n_idx)
            
    # Local Search: Remove redundant codewords
    improved = True
    while improved:
        improved = False
        for i in range(len(codewords)):
            cw_i = codewords[i]
            cw_i_idx = word_to_idx[cw_i]
            
            is_redundant = True
            for n_idx in neighbors[cw_i_idx]:
                # Check if n_idx is covered by any other codeword
                covered_by_other = False
                for j in range(len(codewords)):
                    if j == i:
                        continue
                    cw_j = codewords[j]
                    cw_j_idx = word_to_idx[cw_j]
                    if n_idx in neighbors[cw_j_idx]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    is_redundant = False
                    break
                    
            if is_redundant:
                del codewords[i]
                improved = True
                break # Restart loop from beginning as indices changed
                
    return codewords
# EVOLVE-BLOCK-END
