"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def get_direct_sum():
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [[0] * n]

    # For the specific case q=3, n=6, radius=1, use optimized greedy set cover
    if q == 3 and n == 6 and radius == 1:
        total = 3 ** 6
        powers = [3 ** p for p in range(n)]

        # Precompute for each word, the 13 words within Hamming distance 1
        neighbors = np.empty((total, 13), dtype=np.int32)
        for i in range(total):
            d = [(i // powers[p]) % 3 for p in range(n)]
            neighbors[i, 0] = i
            k = 1
            for p in range(n):
                for alt in range(3):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        neighbors[i, k] = new_i
                        k += 1

        # Now do greedy: prefer candidates that cover the most uncovered words
        uncovered = np.ones(total, dtype=np.int8)
        codewords = []

        while uncovered.sum() > 0:
            if time.time() > deadline:
                break

            # Find candidate that covers most uncovered words
            counts = np.sum(uncovered[neighbors], axis=1)
            best_candidate = np.argmax(counts)
            
            if counts[best_candidate] == 0:
                break

            codewords.append(best_candidate)
            uncovered[neighbors[best_candidate]] = 0

        # Verify we have a valid covering
        if len(codewords) > 0 and uncovered.sum() == 0:
            # Local Search: Swap and Delete
            # Maintain coverage counts and in_set flags
            coverage_counts = np.zeros(total, dtype=np.int32)
            in_set = np.zeros(total, dtype=bool)
            for idx_c in codewords:
                coverage_counts[neighbors[idx_c]] += 1
                in_set[idx_c] = True
            
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Iterate through codewords to find deletions or swaps
                for i in range(len(codewords)):
                    if time.time() > deadline:
                        return get_direct_sum() # Fallback to safe result
                    
                    idx_c = codewords[i]
                    # Temporarily remove
                    coverage_counts[neighbors[idx_c]] -= 1
                    in_set[idx_c] = False
                    
                    min_cov = np.min(coverage_counts)
                    
                    if min_cov > 0:
                        # Safe to delete
                        codewords.pop(i)
                        improved = True
                        break # Restart loop to re-index
                    else:
                        # Find holes
                        holes = (coverage_counts == 0)
                        num_holes = np.sum(holes)
                        
                        # Find best replacement
                        best_w = -1
                        best_count = 0
                        
                        # Check all candidates not in set
                        # Optimization: only check if potential improvement exists
                        # We need to cover all holes to swap (or at least try)
                        # If best_count < num_holes, we can't cover all holes with one swap?
                        # Actually, we can only swap if the new word covers ALL holes.
                        # If it doesn't cover all, the swap is invalid (leaves holes).
                        
                        # Iterate candidates
                        for w in range(total):
                            if in_set[w]:
                                continue
                            cnt = np.sum(holes[neighbors[w]])
                            if cnt > best_count:
                                best_count = cnt
                                best_w = w
                                if cnt == num_holes:
                                    break # Found a perfect cover for holes
                        
                        if best_w != -1 and best_count == num_holes:
                            # Perform Swap
                            codewords[i] = best_w
                            in_set[best_w] = True
                            coverage_counts[neighbors[best_w]] += 1
                            improved = True
                            break # Restart to check for deletions
                        else:
                            # Restore
                            coverage_counts[neighbors[idx_c]] += 1
                            in_set[idx_c] = True

            # Convert indices back to codeword lists
            final_codewords = []
            for idx_c in codewords:
                final_codewords.append([(idx_c // powers[p]) % 3 for p in range(n)])
            
            ds = get_direct_sum()
            if len(final_codewords) <= len(ds):
                return final_codewords
            else:
                return ds
        else:
            return get_direct_sum()

    # General case: fallback to direct sum
    return get_direct_sum()
# EVOLVE-BLOCK-END
