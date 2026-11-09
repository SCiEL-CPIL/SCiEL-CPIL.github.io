"""
Ternary covering code K_3(8,1) construction.

Structural change: use a constraint-satisfaction / local-search approach with swap
operations. Start from a larger redundant product code, then iteratively:
1) Try to remove any codeword whose coverage is redundant (all its neighbors have
   coverage >= 2).
2) Otherwise, identify a "critical" neighbor (covered by exactly one active codeword),
   and attempt to swap that codeword with another candidate that also covers the
   critical word and improves (or maintains) coverage redundancy.

We use numpy for coverage counts and precomputed neighbor lists for speed.
"""

import time
import numpy as np
from itertools import product
import random


def construct(q, n, radius):
    start_time = time.time()
    deadline = 145.0  # safety margin before 120s limit (evaluator kills at 150s)

    if q != 3 or n != 8 or radius != 1:
        # Fallback for general parameters: brute-force product code
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n-4)]
        else:
            return [list(w) for w in product(range(q), repeat=n)]

    # Generate the initial 729-word product code
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_words = [h + list(tail) for h in tetracode for tail in product(range(3), repeat=4)]

    # Precompute all 3^8 = 6561 possible words and map to indices
    all_words_list = [tuple(w) for w in product(range(3), repeat=8)]
    word_to_idx = {w: i for i, w in enumerate(all_words_list)}
    idx_to_word = {i: w for i, w in enumerate(all_words_list)}
    num_words = len(all_words_list)

    # Precompute neighbor indices for each word in the space
    # For each word, its neighbors are itself + 8*2 = 16 words at distance 1
    def get_neighbor_indices(word_idx):
        """Return list of indices of words within Hamming distance 1 of word_idx."""
        word = idx_to_word[word_idx]
        result = [word_idx]
        for i in range(n):
            for v in range(q):
                if v != word[i]:
                    delta = (v - word[i]) * (3 ** (7 - i))
                    new_idx = word_idx + delta
                    result.append(new_idx)
        return result

    # Precompute neighbors for all 6561 words
    all_neighbors = [get_neighbor_indices(i) for i in range(num_words)]

    # Also precompute, for each word index, the list of codeword positions that cover it
    # We'll update this dynamically as we add/remove codewords
    # But for initialization, we can compute it

    # Initial codewords
    cw_indices = [word_to_idx[tuple(w)] for w in initial_words]
    num_cw = len(cw_indices)
    active = [True] * num_cw

    # Coverage count: how many active codewords cover each word
    cov_count = np.zeros(num_words, dtype=np.int32)
    for idx in cw_indices:
        for nb_idx in all_neighbors[idx]:
            cov_count[nb_idx] += 1

    # For each word, maintain a set of active codeword positions that cover it
    # This is expensive to maintain, but useful for swap operations
    # Instead, we'll compute coverage on the fly for candidates

    # Helper: check if a codeword can be removed
    def can_remove(cw_pos):
        if not active[cw_pos]:
            return False
        cw_idx = cw_indices[cw_pos]
        for nb_idx in all_neighbors[cw_idx]:
            if cov_count[nb_idx] <= 1:
                return False
        return True

    # Helper: compute the "value" of a codeword (number of neighbors with cov_count == 1)
    def critical_count(cw_pos):
        if not active[cw_pos]:
            return float('inf')
        cw_idx = cw_indices[cw_pos]
        count = 0
        for nb_idx in all_neighbors[cw_idx]:
            if cov_count[nb_idx] == 1:
                count += 1
        return count

    # Phase 1: Greedy removal pass
    # Repeatedly try to remove codewords with no critical neighbors
    for pass_num in range(10):
        if time.time() > deadline - 5:
            break
        removed_any = False
        for i in range(num_cw):
            if active[i] and can_remove(i):
                active[i] = False
                removed_any = True
                for nb_idx in all_neighbors[cw_indices[i]]:
                    cov_count[nb_idx] -= 1
        if not removed_any:
            break

    # Phase 2: Swap-based local search
    # For each active codeword with critical neighbors, try to swap it with another word
    # that covers the same critical neighbors and improves overall coverage
    rng = random.Random(42)
    
    # Build a list of all possible word indices for candidates
    all_word_indices = list(range(num_words))
    
    # Precompute for each word index, the set of neighbor word indices (already have all_neighbors)
    
    # For swap: we want to find a word w such that:
    # 1. w covers all the critical neighbors of the current codeword
    # 2. w is not already active
    # 3. Adding w and removing the current codeword doesn't create new critical neighbors
    #    (or reduces the total number of critical neighbors)
    
    # Let's track the total number of critical neighbors
    def total_critical():
        total = 0
        for i in range(num_cw):
            if active[i]:
                total += critical_count(i)
        return total
    
    # For efficiency, let's maintain a count of critical neighbors globally
    # A neighbor is critical if cov_count[neighbor] == 1
    # We can count this directly
    
    def count_critical_neighbors():
        # Count how many word indices have cov_count == 1
        return int(np.sum(cov_count == 1))
    
    # Phase 2: Iterative swaps
    # We'll do multiple passes, trying to improve the solution
    for outer_pass in range(20):
        if time.time() > deadline - 5:
            break
        
        # For each active codeword, check if it has critical neighbors
        # If so, try to find a replacement
        improved = False
        for i in range(num_cw):
            if time.time() > deadline - 5:
                break
            if not active[i]:
                continue
            
            cw_idx = cw_indices[i]
            cc = critical_count(i)
            if cc == 0:
                # This codeword has no critical neighbors, try to remove it
                if can_remove(i):
                    active[i] = False
                    for nb_idx in all_neighbors[cw_idx]:
                        cov_count[nb_idx] -= 1
                    improved = True
                continue
            
            # This codeword has critical neighbors
            # Find the critical neighbors
            critical_neighbors = [nb_idx for nb_idx in all_neighbors[cw_idx] if cov_count[nb_idx] == 1]
            if not critical_neighbors:
                continue
            
            # Try to find a replacement word that covers all critical neighbors
            # and is not already active
            # We'll sample some random candidates
            candidates = rng.sample(all_word_indices, min(50, len(all_word_indices)))
            
            best_candidate = None
            best_score = float('inf')
            
            for cand_idx in candidates:
                if any(cw_indices[j] == cand_idx and active[j] for j in range(num_cw)):
                    continue
                
                # Check if cand_idx covers all critical neighbors
                cand_neighbors = set(all_neighbors[cand_idx])
                if not all(nb in cand_neighbors for nb in critical_neighbors):
                    continue
                
                # Compute the score: how many of the current codeword's neighbors
                # would become uncovered if we remove it and add the candidate
                # Actually, we want to minimize the number of new critical neighbors
                # that would be created
                # Let's compute the change in critical count
                # For each neighbor of the current codeword, check if it would lose coverage
                # For each neighbor of the candidate, check if it would gain coverage
                
                # Simpler: just check if the candidate covers at least as many
                # of the current codeword's neighbors as the current codeword does
                # This is a heuristic
                score = len(cand_neighbors)
                # Actually, let's use a different heuristic: prefer candidates
                # that are far from the current codeword (more diverse)
                # Or prefer candidates that cover many neighbors with high coverage
                # Let's just use the number of neighbors with cov_count > 1
                # that the candidate covers
                score = 0
                for nb in all_neighbors[cand_idx]:
                    if cov_count[nb] > 1:
                        score += 1
                # We want to maximize this, so invert for min-heap
                score = -score
                
                if score < best_score:
                    best_score = score
                    best_candidate = cand_idx
            
            if best_candidate is not None:
                # Perform the swap
                # Remove the current codeword
                active[i] = False
                for nb_idx in all_neighbors[cw_idx]:
                    cov_count[nb_idx] -= 1
                
                # Add the new codeword
                # Find or create a slot for the new codeword
                # We'll just add it to the list
                cw_indices.append(best_candidate)
                active.append(True)
                for nb_idx in all_neighbors[best_candidate]:
                    cov_count[nb_idx] += 1
                improved = True
    
        if not improved:
            break

    # Phase 3: Final greedy removal pass
    for pass_num in range(10):
        if time.time() > deadline - 5:
            break
        removed_any = False
        for i in range(len(cw_indices)):
            if active[i] and can_remove(i):
                active[i] = False
                removed_any = True
                for nb_idx in all_neighbors[cw_indices[i]]:
                    cov_count[nb_idx] -= 1
        if not removed_any:
            break

    # Extract remaining codewords
    result = [list(idx_to_word[idx]) for i, idx in enumerate(cw_indices) if active[i]]

    return result
