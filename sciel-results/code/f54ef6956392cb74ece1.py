import time
import random
import numpy as np

# Precomputation for q=3, n=6
Q = 3
N = 6
RADIUS = 1
TOTAL = 729

# Precompute neighbors (distance <= 1) and distance-2 neighbors
# neighbors[i]: list of j such that dist(i, j) <= 1
# dist2_neighbors[i]: list of j such that dist(i, j) == 2

def _precompute():
    neighbors = [[] for _ in range(TOTAL)]
    dist2_neighbors = [[] for _ in range(TOTAL)]
    
    # Base-3 powers
    powers = [1, 3, 9, 27, 81, 243]
    
    for i in range(TOTAL):
        # Get vector
        vec = [0] * N
        num = i
        for k in range(N):
            vec[k] = num % Q
            num //= Q
            
        neighbors[i].append(i)
        
        # Distance 1
        for pos in range(N):
            orig = vec[pos]
            for new in range(Q):
                if new != orig:
                    diff = new - orig
                    new_idx = i + diff * powers[pos]
                    neighbors[i].append(new_idx)
                    
        # Distance 2
        for pos1 in range(N):
            for pos2 in range(pos1 + 1, N):
                v1 = vec[pos1]
                v2 = vec[pos2]
                for n1 in range(Q):
                    if n1 == v1: continue
                    for n2 in range(Q):
                        if n2 == v2: continue
                        new_idx = i + (n1 - v1) * powers[pos1] + (n2 - v2) * powers[pos2]
                        dist2_neighbors[i].append(new_idx)
                        
    return neighbors, dist2_neighbors

# Global cache
_NEIGHBORS = None
_DIST2_NEIGHBORS = None

def _get_precomputed():
    global _NEIGHBORS, _DIST2_NEIGHBORS
    if _NEIGHBORS is None:
        _NEIGHBORS, _DIST2_NEIGHBORS = _precompute()
    return _NEIGHBORS, _DIST2_NEIGHBORS

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        neighbors = [[] for _ in range(q**n)]
        for i in range(q**n):
            vec = [(i // (q**k)) % q for k in range(n)]
            neighbors[i].append(i)
            for pos in range(n):
                for new in range(q):
                    if new != vec[pos]:
                        new_vec = vec[:]
                        new_vec[pos] = new
                        new_idx = sum(v * (q**k) for k, v in enumerate(new_vec))
                        neighbors[i].append(new_idx)
        
        code = []
        covered = set()
        all_words = set(range(q**n))
        while covered != all_words:
            best_c = None
            best_new = 0
            for c in all_words:
                if c in set(code): continue
                new_cov = len(set(neighbors[c]) - covered)
                if new_cov > best_new:
                    best_new = new_cov
                    best_c = c
            if best_c is None: break
            code.append(best_c)
            covered.update(neighbors[best_c])
        return [[(c // (q**k)) % q for k in range(n)] for c in code]

    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    neighbors, dist2_neighbors = _get_precomputed()
    
    # Convert dist2_neighbors to sets for fast membership testing
    dist2_sets = [set(lst) for lst in dist2_neighbors]
    
    # 1. Initial Construction: Greedy Set Cover
    # We want to cover all 729 words.
    # Greedy: pick the word that covers the most currently uncovered words.
    
    uncovered_mask = np.ones(TOTAL, dtype=bool)
    code = []
    code_set = set()
    
    # Precompute coverage contribution of each word?
    # Actually, just iterate.
    # To speed up, we can precompute the list of neighbors for each word.
    
    # Optimization: 
    # Maintain a count of how many times each word is covered.
    # Initially 0.
    # When we add a codeword, increment counts for its neighbors.
    # Uncovered words are those with count 0.
    # We want to pick c that maximizes (count==0 in neighbors[c]).
    
    # Since we start with 0 coverage, the first pick is arbitrary (or random).
    # Subsequent picks depend on the current coverage.
    
    # A standard greedy set cover is O(N * M) where M is number of sets (729) and N is universe size (729).
    # 729 * 729 * 19 (avg neighbors) ~ 10M ops. This is fast in Python if optimized, but let's be careful.
    
    # Better: 
    # For each word w, precompute its neighbors.
    # Loop:
    #   Find w that covers the most uncovered words.
    #   Add w to code.
    #   Mark its neighbors as covered.
    
    # Let's use a numpy array for coverage counts.
    coverage_counts = np.zeros(TOTAL, dtype=np.int32)
    
    # To find the best word efficiently:
    # We can just loop through all 729 words, calculate new coverage, and pick the max.
    # 729 iterations * (calculation of new coverage).
    # Calculation: sum(coverage_counts[neighbors[c]] == 0)
    
    while np.any(uncovered_mask):
        best_c = -1
        best_gain = -1
        
        for c in range(TOTAL):
            if c in code_set:
                continue
            # Calculate gain
            # neighbors[c] is a list of indices
            # We need to check how many of these are currently uncovered (coverage_counts == 0)
            # Since we update coverage_counts immediately, this is accurate.
            gain = 0
            for nb in neighbors[c]:
                if coverage_counts[nb] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_c = c
                
        if best_c == -1 or best_gain == 0:
            # Should not happen if space is finite and we haven't covered all
            break
            
        code.append(best_c)
        code_set.add(best_c)
        
        # Update coverage
        for nb in neighbors[best_c]:
            coverage_counts[nb] += 1
            
        if time.time() > deadline - 30:
             break # Leave time for SA

    # Initial size is likely around 80-90.
    # Now apply Simulated Annealing to reduce size and fix any potential issues (though greedy is always valid).
    # Objective: Minimize |code| + Penalty * (Number of distance-2 conflicts)
    # Wait, the problem DOES NOT require the code to be an independent set (distance > 1).
    # The problem ONLY requires that every word is within distance 1 of SOME codeword.
    # The "independent set" constraint was a heuristic in the previous program to try to find a small code.
    # But a valid covering code CAN have codewords at distance 2!
    # Example: 000000 and 001100 are both in the code. They are distance 2 apart.
    # Does this invalidate the code? No.
    # The only constraint is: For all w in {0,1,2}^6, exists c in Code such that dist(w, c) <= 1.
    
    # So the previous program's constraint "no two elements are at distance 2" was an UNNECESSARY restriction
    # that likely prevented finding the optimal 73!
    # By removing this restriction, we allow a much larger search space.
    
    # Let's verify: If we have a code of size 73, do we require it to be an independent set?
    # No. The lower bound 71 and upper bound 73 are for general covering codes.
    # The van Laarhoven et al. code of 73 is a general covering code.
    
    # So the strategy is:
    # 1. Start with a valid covering code (from Greedy).
    # 2. Use Simulated Annealing to REMOVE codewords if possible, or SWAP them to allow removal of others.
    # 3. The cost function is simply the SIZE of the code.
    # 4. Validity check: Every word must be covered.
    
    # Let's restart with a cleaner SA approach on the Greedy result.
    
    # State: code (list of indices)
    # Cost: len(code) + large_penalty * uncovered_count
    
    # To make SA efficient, we need to quickly calculate the change in uncovered_count when adding/removing/swapping.
    
    # Let's use the coverage_counts array.
    # Uncovered words are those with coverage_counts == 0.
    
    current_code = list(code)
    current_code_set = set(current_code)
    
    # Recalculate coverage_counts from scratch to be safe
    coverage_counts = np.zeros(TOTAL, dtype=np.int32)
    for c in current_code:
        for nb in neighbors[c]:
            coverage_counts[nb] += 1
            
    current_uncovered = np.sum(coverage_counts == 0)
    
    if current_uncovered == 0:
        current_cost = len(current_code)
    else:
        current_cost = len(current_code) + 1000 * current_uncovered
    
    best_code = list(current_code)
    best_cost = current_cost
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        if T < T_min:
            T = T_min # Or break and do final check? Let's keep going at T_min for exploration.
            
        # Choose a random move
        move_type = random.random()
        
        if len(current_code) > 0 and move_type < 0.4:
            # REMOVE
            idx = random.randint(0, len(current_code) - 1)
            c = current_code[idx]
            
            # Calculate delta
            # Removing c decreases coverage_counts for neighbors[c].
            # New uncovered count = old_uncovered + (number of neighbors of c that had coverage 1)
            
            # We need to count how many neighbors of c have coverage_counts == 1.
            # If coverage_counts[nb] == 1, removing c makes it 0 (uncovered).
            # If coverage_counts[nb] > 1, it stays covered.
            
            new_uncovered_delta = 0
            for nb in neighbors[c]:
                if coverage_counts[nb] == 1:
                    new_uncovered_delta += 1
            
            new_uncovered = current_uncovered + new_uncovered_delta
            new_cost = (len(current_code) - 1) + 1000 * new_uncovered
            
            if new_cost < current_cost or random.random() < np.exp(-(new_cost - current_cost) / T):
                # Accept
                # Apply removal
                for nb in neighbors[c]:
                    coverage_counts[nb] -= 1
                current_code.pop(idx)
                current_code_set.remove(c)
                current_uncovered = new_uncovered
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_code = list(current_code)
                    
        elif move_type < 0.8:
            # SWAP
            if len(current_code) == 0: continue
            idx = random.randint(0, len(current_code) - 1)
            c_old = current_code[idx]
            c_new = random.randint(0, TOTAL - 1)
            
            if c_new in current_code_set:
                continue
                
            # Calculate delta for removing c_old and adding c_new
            # This is complex. 
            # Let's just calculate the new uncovered count from scratch? No, too slow.
            # Let's calculate delta.
            
            # 1. Remove c_old
            # 2. Add c_new
            
            # We can simulate the counts.
            # Temporarily decrement neighbors of c_old.
            # Increment neighbors of c_new.
            # Count zeros.
            # Revert if rejected.
            
            # To avoid modifying the global array and reverting (which is error-prone),
            # let's compute the change in the number of zeros.
            
            # A word w is uncovered if count(w) == 0.
            # After swap:
            # count'(w) = count(w) - [w in N(c_old)] + [w in N(c_new)]
            
            # We only care about words where count(w) is small (0 or 1) or where the change happens.
            
            # Let's just do a local simulation on a copy of the relevant counts?
            # Or just calculate new_uncovered directly by checking the affected words.
            
            # Affected words are N(c_old) U N(c_new).
            # Size is at most 19 + 19 = 38.
            
            # Let's build a dict of counts for affected words.
            # Actually, we can just iterate through the union of neighbors.
            
            affected = set(neighbors[c_old]) | set(neighbors[c_new])
            
            new_uncovered = 0
            # We need to know the new count for each word in 'affected'
            # And the old count for words NOT in 'affected' (which remain covered if they were covered before? No, they remain same count).
            
            # Uncovered words outside 'affected' remain unchanged.
            # So, new_uncovered = (old_uncovered - (uncovered in affected)) + (uncovered in affected after swap)
            
            # Let's identify which words in 'affected' are currently uncovered.
            currently_uncovered_in_affected = 0
            for w in affected:
                if coverage_counts[w] == 0:
                    currently_uncovered_in_affected += 1
            
            # Calculate new counts for words in 'affected'
            new_uncovered_in_affected = 0
            for w in affected:
                new_count = coverage_counts[w]
                if w in neighbors[c_old]:
                    new_count -= 1
                if w in neighbors[c_new]:
                    new_count += 1
                if new_count == 0:
                    new_uncovered_in_affected += 1
            
            new_uncovered = current_uncovered - currently_uncovered_in_affected + new_uncovered_in_affected
            new_cost = len(current_code) + 1000 * new_uncovered # Size unchanged
            
            if new_cost < current_cost or random.random() < np.exp(-(new_cost - current_cost) / T):
                # Accept
                for nb in neighbors[c_old]:
                    coverage_counts[nb] -= 1
                for nb in neighbors[c_new]:
                    coverage_counts[nb] += 1
                current_code[idx] = c_new
                current_code_set.remove(c_old)
                current_code_set.add(c_new)
                current_uncovered = new_uncovered
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_code = list(current_code)
                    
        else:
            # ADD (if we have uncovered words or to explore larger solutions)
            # Only useful if we are stuck.
            c_new = random.randint(0, TOTAL - 1)
            if c_new in current_code_set:
                continue
                
            # Calculate delta
            # Adding c_new increases counts.
            # Uncovered words in N(c_new) that were 0 become >0.
            
            newly_covered = 0
            for w in neighbors[c_new]:
                if coverage_counts[w] == 0:
                    newly_covered += 1
            
            new_uncovered = current_uncovered - newly_covered
            new_cost = (len(current_code) + 1) + 1000 * new_uncovered
            
            if new_cost < current_cost or random.random() < np.exp(-(new_cost - current_cost) / T):
                # Accept
                for nb in neighbors[c_new]:
                    coverage_counts[nb] += 1
                current_code.append(c_new)
                current_code_set.add(c_new)
                current_uncovered = new_uncovered
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_code = list(current_code)
        
        # Cool down
        T *= cooling_rate
        
        # Check time
        if time.time() > deadline - 5:
            break
            
    # Final check
    # Ensure best_code is valid
    final_coverage = np.zeros(TOTAL, dtype=np.int32)
    for c in best_code:
        for nb in neighbors[c]:
            final_coverage[nb] += 1
            
    if np.all(final_coverage > 0):
        return [[(c // (3**k)) % 3 for k in range(6)] for c in best_code]
    else:
        # Fallback to initial greedy if best_code is invalid (should not happen if we track correctly)
        return [[(c // (3**k)) % 3 for k in range(6)] for c in current_code]

