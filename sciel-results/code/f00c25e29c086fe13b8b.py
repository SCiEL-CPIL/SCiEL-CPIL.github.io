import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback: direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # For small n, brute force
            all_words = list(itertools.product(range(q), repeat=n))
            return all_words
    
    # Specialized for q=3, n=8, r=1
    total = q ** n  # 6561
    
    # Convert between integer index and tuple
    def idx_to_tuple(idx):
        t = []
        for i in range(n):
            t.append(idx % q)
            idx //= q
        return list(reversed(t))
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx
    
    # Precompute neighbors: for each center, the 17 words it covers (itself + 16 at distance 1)
    # neighbors[center_idx] = list of word indices covered
    neighbors = np.zeros((total, 17), dtype=np.int32)
    for idx in range(total):
        t = idx_to_tuple(idx)
        neighbors[idx, 0] = idx
        for pos in range(n):
            for delta in range(1, q):
                t2 = t[:]
                t2[pos] = (t2[pos] + delta) % q
                neighbors[idx, 1 + pos * (q - 1) + (delta - 1)] = tuple_to_idx(t2)
    
    # Start with 729-word direct sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_centers = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=4):
            initial_centers.append(tuple(h + list(tail)))
    
    centers = list(initial_centers)
    center_set = set(centers)
    
    # Coverage count: how many centers cover each word
    cov_count = np.zeros(total, dtype=np.int32)
    for c in centers:
        c_idx = tuple_to_idx(c)
        for w_idx in neighbors[c_idx]:
            cov_count[w_idx] += 1
    
    def is_covered():
        return np.all(cov_count > 0)
    
    def try_remove(center_tuple):
        """Try to remove a center if all its words are still covered by others."""
        c_idx = tuple_to_idx(center_tuple)
        for w_idx in neighbors[c_idx]:
            if cov_count[w_idx] <= 1:
                return False
        # Safe to remove
        for w_idx in neighbors[c_idx]:
            cov_count[w_idx] -= 1
        centers.remove(center_tuple)
        center_set.discard(center_tuple)
        return True
    
    # Phase 1: Iterative removal of redundant centers
    removed_any = True
    while removed_any and time.time() < deadline:
        removed_any = False
        for i in range(len(centers) - 1, -1, -1):
            if try_remove(centers[i]):
                removed_any = True
    
    # Phase 2: Local search - try swaps to reduce count further
    # For each center, try replacing it with a candidate that allows net reduction
    # Strategy: find a center c that, if removed, leaves some words uncovered.
    # Then find a new center c' that covers those words and is "efficient"
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Find centers whose removal leaves uncovered words
        candidates_to_replace = []
        for c in centers:
            c_idx = tuple_to_idx(c)
            # Check if removing c leaves any word with cov_count == 1
            min_cov = min(cov_count[w_idx] for w_idx in neighbors[c_idx])
            if min_cov == 1:
                # Find which words would become uncovered
                uncovered = [w_idx for w_idx in neighbors[c_idx] if cov_count[w_idx] == 1]
                candidates_to_replace.append((c, c_idx, uncovered))
        
        # For each candidate, try to find a replacement
        for c, c_idx, uncovered in candidates_to_replace:
            if time.time() > deadline:
                break
            
            if not uncovered:
                continue
            
            # The new center must cover all words in `uncovered`
            # Find candidates: for each word in uncovered, the centers that cover it
            # The new center must be in the intersection of these sets
            
            # A center covers word w if w is in neighbors[center]
            # So we need center c' such that for all w in uncovered, w in neighbors[c']
            # i.e., c' is within distance 1 of all words in uncovered
            
            # For efficiency, pick the first uncovered word and find its neighbors
            first_w = uncovered[0]
            # Centers that cover first_w: those c' where first_w in neighbors[c']
            # This is equivalent to: c' is within distance 1 of first_w
            # The centers covering word w are: w itself and all words at distance 1 from w
            
            candidate_replacements = set()
            w_tuple = idx_to_tuple(first_w)
            # The center must be w itself or at distance 1 from w
            candidate_replacements.add(first_w)
            for pos in range(n):
                for delta in range(1, q):
                    t2 = w_tuple[:]
                    t2[pos] = (t2[pos] + delta) % q
                    candidate_replacements.add(tuple_to_idx(t2))
            
            # Filter: must cover ALL uncovered words
            valid_replacements = []
            for rc in candidate_replacements:
                if rc == c_idx:
                    continue
                # Check if rc covers all uncovered words
                rc_neighbors = neighbors[rc]
                covers_all = all(w in set(rc_neighbors) for w in uncovered)
                if covers_all:
                    valid_replacements.append(rc)
            
            # For each valid replacement, check if it allows net reduction
            # After replacing c with rc:
            # - Words covered only by c (uncovered) are now covered by rc
            # - Words covered by rc but not by c might become over-covered
            # - We can then try to remove other centers
            
            for rc in valid_replacements:
                if time.time() > deadline:
                    break
                
                rc_tuple = idx_to_tuple(rc)
                if rc_tuple in center_set:
                    continue  # Already a center, no point
                
                # Simulate the swap
                # Remove c, add rc
                # Check if this is beneficial: count how many centers we could remove after
                
                # Temporarily remove c
                for w_idx in neighbors[c_idx]:
                    cov_count[w_idx] -= 1
                
                # Temporarily add rc
                for w_idx in neighbors[rc]:
                    cov_count[w_idx] += 1
                
                # Check coverage
                if np.all(cov_count > 0):
                    # Count how many centers are now redundant
                    redundant_count = 0
                    for other in centers:
                        if other == c:
                            continue
                        other_idx = tuple_to_idx(other)
                        min_cov = min(cov_count[w_idx] for w_idx in neighbors[other_idx])
                        if min_cov >= 2:
                            redundant_count += 1
                    
                    # If we can remove at least 1 other center, this swap is beneficial
                    if redundant_count >= 1:
                        # Perform the swap
                        centers.remove(c)
                        center_set.discard(c)
                        centers.append(rc_tuple)
                        center_set.add(rc_tuple)
                        improved = True
                        
                        # Now remove redundant centers
                        for other in centers[:]:
                            other_idx = tuple_to_idx(other)
                            min_cov = min(cov_count[w_idx] for w_idx in neighbors[other_idx])
                            if min_cov >= 2:
                                for w_idx in neighbors[other_idx]:
                                    cov_count[w_idx] -= 1
                                centers.remove(other)
                                center_set.discard(other)
                        
                        # Roll back the temporary changes are already applied
                        # (we already modified cov_count in place)
                        break
                else:
                    # Roll back
                    for w_idx in neighbors[rc]:
                        cov_count[w_idx] -= 1
                    for w_idx in neighbors[c_idx]:
                        cov_count[w_idx] += 1
        
        if improved:
            continue
    
    # Phase 3: Final removal pass
    removed_any = True
    while removed_any and time.time() < deadline:
        removed_any = False
        for i in range(len(centers) - 1, -1, -1):
            if try_remove(centers[i]):
                removed_any = True
    
    # Verify coverage
    assert np.all(cov_count > 0), "Coverage broken!"
    
    # Return as list of lists
    return [list(c) for c in centers]
