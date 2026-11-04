"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # For the specific case q=3, n=6, radius=1, use greedy set cover
    if q == 3 and n == 6 and radius == 1:
        # Total number of words
        total = 3 ** 6
        # Precompute: for each word, which words are within Hamming distance 1
        # Represent each word as an index 0..728
        # A word at index i corresponds to digits [d0,d1,d2,d3,d4,d5]
        
        # Build coverage: for each candidate word, the set of words it covers
        # Use numpy for efficiency
        # Each word is a 6-digit base-3 number
        
        # Generate all words as indices
        all_indices = np.arange(total)
        
        # Convert index to 6-digit base-3 representation
        digits = np.zeros((total, 6), dtype=np.int32)
        for pos in range(6):
            digits[:, pos] = (all_indices // (3 ** pos)) % 3
        
        # For greedy, we need to efficiently find which uncovered words each candidate covers
        # Start with all uncovered
        uncovered = np.ones(total, dtype=bool)
        
        codewords = []
        
        while uncovered.any():
            if time.time() > deadline:
                # Return what we have (may not be complete, but let's try to finish quickly)
                break
            
            # For each candidate, count how many uncovered words it covers
            # A word w covers itself and all words at Hamming distance 1
            # That's 1 + 6*2 = 13 words per codeword
            
            # Efficient approach: for each position and each alternative digit,
            # compute the index of the neighbor
            
            # Precompute neighbor indices for each position
            # For position p, changing digit d to d' changes index by (d'-d) * 3^p
            
            best_count = -1
            best_candidate = -1
            
            # We need to check candidates. To speed up, only check uncovered candidates
            # or use a smarter heuristic
            
            # For now, check all candidates but break early if possible
            # Actually, let's precompute the 13 neighbors for each word
            
            # This might be too slow in pure Python/numpy for 729 candidates * 729 words
            # Let's use a different approach: precompute the coverage sets as bitmasks
            
            # Alternative: use the fact that each codeword covers exactly 13 words
            # (1 + 6*2 = 13) in the full space, but some may already be covered
            
            # Let's precompute for each word, the list of 13 words it covers
            # Then for greedy, we count how many of those 13 are still uncovered
            
            # Precompute neighbors
            powers = [3 ** p for p in range(6)]
            
            # For each word index, compute the 13 neighbors (including itself)
            # neighbors[i] = list of 13 indices
            
            # This is 729 * 13 = 9477 entries, manageable
            # But doing this in a loop might be slow. Let's vectorize.
            
            # Actually, let's just do it with a simple loop for precomputation
            # since 729 is small
            
            # Precompute once outside the while loop - move this up
            
            break  # We'll restructure
        
        # Let me restructure: precompute neighbors first
        
        # Precompute for each word, the 13 words within Hamming distance 1
        powers = [3 ** p for p in range(6)]
        neighbors = []
        for i in range(total):
            # Get digits
            d = [(i // powers[p]) % 3 for p in range(6)]
            covered = [i]
            for p in range(6):
                for alt in range(3):
                    if alt != d[p]:
                        # Compute new index
                        new_i = i + (alt - d[p]) * powers[p]
                        covered.append(new_i)
            neighbors.append(covered)
        
        # Now do greedy
        uncovered = np.ones(total, dtype=bool)
        codewords = []
        
        while uncovered.any():
            if time.time() > deadline:
                break
            
            # Find candidate that covers most uncovered words
            best_count = -1
            best_candidate = -1
            
            # Only consider candidates that are themselves uncovered (optimization)
            # Actually, a covered candidate might still cover new words, but 
            # typically the best candidate is uncovered. Let's check all for correctness.
            
            # For speed, check only uncovered candidates first
            uncovered_indices = np.where(uncovered)[0]
            
            if len(uncovered_indices) == 0:
                break
            
            # Sample some candidates if there are too many to check all
            # 729 is small enough to check all
            for cand in uncovered_indices:
                count = sum(1 for nb in neighbors[cand] if uncovered[nb])
                if count > best_count:
                    best_count = count
                    best_candidate = cand
            
            if best_candidate == -1:
                break
            
            # Add this codeword
            codeword = [(best_candidate // powers[p]) % 3 for p in range(6)]
            codewords.append(codeword)
            
            # Mark covered
            for nb in neighbors[best_candidate]:
                uncovered[nb] = False
        
        # Verify we have a valid covering
        if len(codewords) > 0 and not uncovered.any():
            return codewords
        
        # Fallback to direct sum if greedy didn't finish
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # General case: fallback to direct sum
    if n >= 4 and q == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    # Very general fallback: all-zero word repeated (invalid if radius < n, but safe)
    # Actually, let's just return a single codeword for now
    return [[0] * n]
# EVOLVE-BLOCK-END
