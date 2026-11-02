"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

# Deadline guard
_DEADLINE = 115.0  # safety margin before 120s limit

def _time_left():
    return _DEADLINE - time.time()

def construct(q, n, radius):
    global _DEADLINE
    _DEADLINE = time.time() + 115.0  # 115s budget out of 120s
    
    # Specific optimization for the known hard case q=3, n=7, radius=1
    if q == 3 and n == 7 and radius == 1:
        return _solve_q3_n7_r1()
    
    # General fallback: simple greedy with numpy
    return _general_greedy(q, n, radius)

def _solve_q3_n7_r1():
    q, n, radius = 3, 7, 1
    total_words = q ** n  # 2187
    
    # Precompute all words as a 2D numpy array (total_words, n)
    # and also as a flat array of base-q integers for index mapping
    # For q=3, n=7, we can map word to index via base-3 conversion
    
    # 1. Generate all words
    all_words = np.array([list(map(int, np.base_repr(i, q).zfill(n))) for i in range(total_words)])
    
    # 2. Precompute the "ball" for each word as a boolean mask or list of indices
    # Ball size for r=1 is 1 + n*(q-1) = 1 + 7*2 = 15
    # We represent the ball of word i as a set of indices.
    # To speed up, we precompute the indices of neighbors for each word.
    
    # Mapping word -> index is just the integer i if we generate in order.
    # Let's verify: base_repr(0).zfill(7) -> 0000000 -> index 0.
    # base_repr(1).zfill(7) -> 0000001 -> index 1.
    # Yes, if we iterate i from 0 to total_words-1, the index is i.
    
    # Precompute neighbors for each word i
    # neighbors[i] is a list of indices
    neighbors = [set() for _ in range(total_words)]
    for i in range(total_words):
        w = all_words[i]
        neighbors[i].add(i)
        for pos in range(n):
            for v in range(q):
                if v != w[pos]:
                    new_w = w.copy()
                    new_w[pos] = v
                    # Convert new_w back to index
                    idx = 0
                    for val in new_w:
                        idx = idx * q + val
                    neighbors[i].add(idx)
    
    # 3. Greedy Set Cover
    uncovered = set(range(total_words))
    codewords = []
    
    # To make the greedy step fast, we need to find the center that covers the most uncovered points.
    # Naive: for each center in uncovered, count |uncovered & neighbors[center]|.
    # This is O(|uncovered| * ball_size). 
    # Initial |uncovered| = 2187. ball_size = 15.
    # 2187 * 15 = ~33k operations per step.
    # We need ~186 steps.
    # 186 * 33k = ~6 million operations. Python can do this in < 1 second.
    # Wait, the previous program was using Python sets and loops.
    # Let's try to optimize the selection.
    
    # Optimization: 
    # Instead of iterating all uncovered, we can maintain a count of how many uncovered points each center covers.
    # But when we remove a point, we have to decrement the count for all centers that cover it.
    # This is the "dual" greedy.
    
    # Let's stick to the direct greedy but optimized with lists/sets.
    # The previous program likely timed out or was slow due to poor implementation details 
    # (e.g. converting back and forth, or inefficient set intersections).
    
    # Let's use a slightly smarter heuristic: 
    # Start with a random seed or a good initial guess?
    # Actually, for K_3(7,1), a known good construction is the "tetracode" extension.
    # But greedy usually gets close.
    
    # Let's implement the dual greedy (also known as the "worst-case" or "most constrained" heuristic variant)
    # Or simply the standard greedy with efficient counting.
    
    # Efficient Standard Greedy:
    # uncovered_mask: boolean array of length total_words
    # uncovered_mask[i] = 1 if i is uncovered
    
    uncovered_mask = np.ones(total_words, dtype=bool)
    
    while np.any(uncovered_mask):
        if _time_left() < 2.0:
            break
            
        # Find the best center
        # We only need to consider centers that are themselves uncovered? 
        # No, a covered center might cover more *other* uncovered points.
        # However, in set cover, it is sufficient to consider any center.
        # But usually, picking from uncovered is a good heuristic.
        # Let's pick from ALL centers to be safe, but that's 2187 checks.
        
        # Let's try picking from uncovered first, as it's a common heuristic.
        # If that fails to improve, we might need to look at all.
        
        candidates = np.where(uncovered_mask)[0]
        
        best_center = -1
        max_covered = 0
        
        # Vectorized approach for counting covered points?
        # For each candidate c, covered = sum(uncovered_mask[neighbors[c]])
        # neighbors[c] is a list of indices.
        
        # Since candidates can be up to 2187, and ball size is 15:
        # We can do this in a loop.
        
        # Optimization: Use a "score" array that we update?
        # Let's just do the loop. It's fast enough in Python if we are careful.
        
        # To speed up, let's precompute neighbors as a numpy array of indices?
        # neighbors_array[c] is a list of ints.
        
        for c in candidates:
            # Get neighbors of c
            # neighbors[c] is a set. Converting to list or iterating is fast.
            # Sum of uncovered_mask at those indices.
            # np.sum(uncovered_mask[list(neighbors[c])])
            
            # Even faster: 
            # count = 0
            # for nb in neighbors[c]:
            #     if uncovered_mask[nb]: count += 1
            
            # Let's try the numpy sum version
            count = np.sum(uncovered_mask[list(neighbors[c])])
            if count > max_covered:
                max_covered = count
                best_center = c
                
        if best_center == -1:
            break
            
        codewords.append(best_center)
        
        # Update uncovered mask
        for nb in neighbors[best_center]:
            uncovered_mask[nb] = False
        
        # Check time
        if _time_left() < 5.0:
            break
    
    # Convert indices back to words
    result = [all_words[c].tolist() for c in codewords]
    
    # Local Search / Swap Optimization
    # Try to remove a codeword if its ball is covered by others.
    # Or swap two codewords to cover more?
    # Let's try a simple "reduction" phase.
    
    # Check if any codeword is redundant
    changed = True
    while changed and _time_left() > 5.0:
        changed = False
        new_codewords = []
        for i, c in enumerate(codewords):
            # Temporarily remove c
            remaining = [x for j, x in enumerate(codewords) if j != i]
            
            # Check if the union of balls of remaining covers the ball of c
            # Specifically, check if every point in ball(c) is covered by at least one ball in remaining.
            # If yes, c is redundant.
            
            # Optimization: Only check if the number of codewords is small or time is available.
            # For 186 codewords, checking each one against 185 others is expensive.
            # 186 * 185 * 15 (ball size) ~ 500k ops.
            
            is_redundant = True
            # We need to check if union of balls(remaining) covers ball(c)
            # Let's check each point in ball(c)
            for p in neighbors[c]:
                covered = False
                for r in remaining:
                    if p in neighbors[r]:
                        covered = True
                        break
                if not covered:
                    is_redundant = False
                    break
            
            if is_redundant:
                changed = True
                # Do not add c to new_codewords
            else:
                new_codewords.append(c)
        
        codewords = new_codewords
        
    result = [all_words[c].tolist() for c in codewords]
    
    return result

def _general_greedy(q, n, radius):
    # Fallback for other parameters
    total_words = q ** n
    all_words = np.array([list(map(int, np.base_repr(i, q).zfill(n))) for i in range(total_words)])
    
    # Precompute neighbors
    neighbors = [set() for _ in range(total_words)]
    for i in range(total_words):
        w = all_words[i]
        neighbors[i].add(i)
        # Generate all words within radius
        # For general radius, this is expensive.
        # Assuming radius is small (1 or 2).
        if radius == 1:
             for pos in range(n):
                for v in range(q):
                    if v != w[pos]:
                        new_w = w.copy()
                        new_w[pos] = v
                        idx = 0
                        for val in new_w:
                            idx = idx * q + val
                        neighbors[i].add(idx)
        else:
            # Brute force for small n
            for j in range(total_words):
                # Calculate Hamming distance
                dist = np.sum(all_words[i] != all_words[j])
                if dist <= radius:
                    neighbors[i].add(j)

    uncovered_mask = np.ones(total_words, dtype=bool)
    codewords = []
    
    while np.any(uncovered_mask) and _time_left() > 2.0:
        candidates = np.where(uncovered_mask)[0]
        best_center = -1
        max_covered = 0
        
        for c in candidates:
            count = np.sum(uncovered_mask[list(neighbors[c])])
            if count > max_covered:
                max_covered = count
                best_center = c
                
        if best_center == -1:
            break
            
        codewords.append(best_center)
        for nb in neighbors[best_center]:
            uncovered_mask[nb] = False
            
    result = [all_words[c].tolist() for c in codewords]
    return result

# EVOLVE-BLOCK-END
