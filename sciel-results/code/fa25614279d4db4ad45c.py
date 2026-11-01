import numpy as np
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Specific parameters for the task: q=3, n=6, radius=1
    # The problem statement guarantees these values, but we handle generally just in case.
    total_words = q ** n
    
    # Precompute neighbors for each word (for radius=1)
    # neighbor_matrix[i] is a boolean array of length total_words where True means word j is within distance 1 of word i
    # Actually, for memory and speed, let's store the list of neighbor indices.
    # With q=3, n=6, total_words=729.
    # Neighbors of a word: itself + 1*(n*(q-1)) = 1 + 1*6*2 = 13 neighbors.
    
    neighbors = [[] for _ in range(total_words)]
    for idx in range(total_words):
        # Convert idx to word
        word = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            word[i] = temp % q
            temp //= q
        
        neighbors[idx].append(idx)
        
        if radius >= 1:
            for pos in range(n):
                orig = word[pos]
                for new_val in range(q):
                    if new_val != orig:
                        word[pos] = new_val
                        nb_idx = 0
                        for i in range(n):
                            nb_idx = nb_idx * q + word[i]
                        neighbors[idx].append(nb_idx)
                word[pos] = orig

    # Convert to numpy arrays for faster boolean masking
    neighbors_np = [np.array(nb, dtype=np.int32) for nb in neighbors]
    
    # Precompute a "coverage" helper structure?
    # For fast checking if a set of codewords covers the whole space, we can maintain a count array.
    
    def is_valid_cover(codewords):
        """Check if the set of codewords covers all words."""
        covered = np.zeros(total_words, dtype=np.int8)
        for cw in codewords:
            covered[neighbors_np[cw]] = 1
        return covered.all()

    def greedy_cover(restart_seed=0):
        """Greedy set cover to find an initial valid solution."""
        rng = random.Random(restart_seed)
        
        uncovered = np.ones(total_words, dtype=bool)
        codewords = []
        
        # Precompute neighbor matrix for fast counting?
        # Let's use a simpler approach: 
        # counts[i] = number of uncovered words covered by candidate i
        # We can update this incrementally.
        
        # Initialize counts
        # counts[i] = sum(uncovered[neighbors_np[i]])
        # This is expensive to initialize fully for 729 candidates, but 729*13 is small.
        counts = np.zeros(total_words, dtype=np.int32)
        for i in range(total_words):
            counts[i] = np.sum(uncovered[neighbors_np[i]])
            
        while not uncovered.all() == False: # while not all covered
            if time.time() > deadline:
                return codewords if codewords else None
            
            # Find candidate with max count
            if counts.max() == 0:
                break # Should not happen if uncovered exists and radius covers self
            
            best_cw = int(counts.argmax())
            codewords.append(best_cw)
            
            # Update uncovered and counts
            new_uncovered = neighbors_np[best_cw]
            # Mark these as covered
            uncovered[new_uncovered] = False
            
            # Update counts for all candidates
            # For each candidate j, counts[j] decreases by the number of new_uncovered words that j covers
            # This is equivalent to: counts[j] -= sum(uncovered_before_update[neighbors_np[j]] & new_uncovered)
            # Since new_uncovered is small (13), we can iterate over the newly covered words
            # and decrement the count of any candidate that covers them.
            
            # Let's build a reverse neighbor list: for each word w, which candidates cover w?
            # This is just the neighbors of w.
            # So for each w in new_uncovered:
            #   for each j in neighbors[w]: counts[j] -= 1
            
            for w in new_uncovered:
                for j in neighbors[w]:
                    counts[j] -= 1
        
        return codewords

    def remove_redundant(codewords):
        """Remove redundant codewords iteratively."""
        if not codewords:
            return codewords
            
        # Coverage count: how many codewords cover each word
        coverage = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            coverage[neighbors_np[cw]] += 1
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(codewords)):
                if time.time() > deadline:
                    break
                cw = codewords[i]
                # If min coverage in ball(cw) is >= 2, then cw is redundant
                if np.min(coverage[neighbors_np[cw]]) >= 2:
                    coverage[neighbors_np[cw]] -= 1
                    codewords.pop(i)
                    changed = True
                    break # Restart loop as indices changed
        return codewords

    def local_search(codewords):
        """
        1-Swap local search.
        Try to replace a codeword with another to maintain coverage and potentially enable removals.
        Actually, a simpler and very effective heuristic for covering codes is:
        1. Try to remove a codeword. If it breaks coverage, find a replacement that restores it.
        2. Or, just try all swaps (i, j) where i is in set, j is not. If set - {i} + {j} is valid, accept.
        
        Given the small size (729 words, ~73 codewords), we can afford to be a bit thorough.
        """
        if time.time() > deadline:
            return codewords
            
        codewords = list(codewords)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle order to break symmetry?
            # random.shuffle(codewords) -- maybe not, keep deterministic for reproducibility if possible, 
            # but random restarts of the whole process handle that.
            
            for i in range(len(codewords)):
                if time.time() > deadline:
                    break
                cw_out = codewords[i]
                
                # Check if removing cw_out leaves any word uncovered
                # We need to know which words are ONLY covered by cw_out.
                # Let's compute coverage counts for the current set.
                # This is expensive inside the loop. 
                
                # Alternative: 
                # 1. Compute coverage counts for the whole set once.
                # 2. For each i, check if min(coverage[neighbors[cw_out]]) == 1.
                #    If min > 1, it's redundant (handled by remove_redundant).
                #    If min == 1, there is at least one word 'v' only covered by cw_out.
                #    To remove cw_out, we must add a new codeword cw_in that covers 'v'.
                #    We should pick a cw_in that covers 'v' AND covers many other words that are currently 
                #    only covered by cw_out (to minimize "damage" or actually maximize "help").
                
                # Let's precompute coverage for the current state
                coverage = np.zeros(total_words, dtype=np.int32)
                for cw in codewords:
                    coverage[neighbors_np[cw]] += 1
                
                # Now iterate through codewords to find improvement opportunities
                # We want to replace cw_out with cw_in such that the new set is valid.
                # Ideally, we want to find a cw_in that covers all words that cw_out uniquely covered.
                
                # Words uniquely covered by cw_out:
                unique_words = neighbors_np[cw_out][coverage[neighbors_np[cw_out]] == 1]
                
                if len(unique_words) == 0:
                    # Redundant, should have been caught by remove_redundant, but safe to remove
                    codewords.pop(i)
                    improved = True
                    break # Restart loop
                
                # Find candidates cw_in that cover ALL unique_words
                # A candidate cw_in covers unique_words if all unique_words are in neighbors[cw_in]
                # This is equivalent to: unique_words is a subset of neighbors[cw_in]
                
                # Efficient check:
                # For each candidate j not in codewords:
                #   if all(unique_words are in neighbors_np[j]):
                #       check if new set is valid (it should be, if j covers all unique words, 
                #       and other words were covered by others... wait.
                #       If j covers all unique words, does it guarantee the whole space is covered?
                #       Yes, because:
                #       - Words not unique to cw_out were covered by at least one other codeword in the original set.
                #       - We only removed cw_out. So those words are still covered.
                #       - The unique words are now covered by j.
                #       So the new set is valid.
                
                # So we just need to find a j not in codewords such that neighbors_np[j] contains unique_words.
                # To be a good swap, we might prefer j that covers many words with low coverage (to build redundancy for future removals).
                
                in_set = set(codewords)
                best_j = -1
                best_score = -1
                
                # Optimize: Only check j that covers at least one unique word?
                # Actually, j MUST cover ALL unique words.
                
                # Let's iterate through all possible j
                for j in range(total_words):
                    if j in in_set:
                        continue
                    
                    # Check if j covers all unique_words
                    # unique_words is an array of indices
                    # neighbors_np[j] is an array of indices
                    # We can use a boolean mask or set intersection
                    # Since unique_words is small (usually 1-3), we can just check membership
                    covers_all = True
                    for u in unique_words:
                        if u not in neighbors_np[j]: # O(13) check
                            covers_all = False
                            break
                            
                    if covers_all:
                        # Score this candidate
                        # Prefer candidates that cover many words with coverage == 1 (other than unique_words)?
                        # Or simply prefer candidates that add redundancy to words with low coverage.
                        # Let's score by: sum(1 for w in neighbors[j] if coverage[w] == 1)
                        # This helps create redundancy so we can remove more later.
                        score = np.sum(coverage[neighbors_np[j]] == 1)
                        
                        if score > best_score:
                            best_score = score
                            best_j = j
                            
                if best_j != -1:
                    # Perform swap
                    codewords[i] = best_j
                    improved = True
                    break # Restart loop to re-evaluate coverage
                    
        return codewords

    # Main Strategy
    best_result = None
    best_count = float('inf')
    
    # 1. Try multiple greedy restarts
    for seed in range(20):
        if time.time() > deadline:
            break
        initial = greedy_cover(restart_seed=seed)
        if initial:
            # Remove redundant
            reduced = remove_redundant(initial)
            # Local search
            optimized = local_search(reduced)
            
            if len(optimized) < best_count:
                best_count = len(optimized)
                best_result = optimized
    
    # 2. If we haven't found a good one, try a specific known construction?
    # The tetracode construction gives 81 words. 
    # Let's try to optimize that specifically if greedy didn't beat it.
    
    if best_count > 75: # If greedy is stuck > 75
        # Construct tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_codewords = []
        for tail_a in range(3):
            for tail_b in range(3):
                for h in tetracode:
                    word = h + [tail_a, tail_b]
                    idx = 0
                    for val in word:
                        idx = idx * q + val
                    initial_codewords.append(idx)
        
        reduced = remove_redundant(initial_codewords)
        optimized = local_search(reduced)
        
        if len(optimized) < best_count:
            best_count = len(optimized)
            best_result = optimized

    # 3. Further local search on the best result found so far, with random restarts of the local search order?
    # The local_search above is deterministic given the input. 
    # We can add randomization to the local search by shuffling the codewords before the swap loop.
    
    if best_result is not None:
        # Try to improve best_result further with randomized local search
        for rs in range(5):
            if time.time() > deadline:
                break
            # Copy and shuffle
            temp_result = best_result.copy()
            random.shuffle(temp_result)
            
            # Run local search again
            optimized = local_search(temp_result)
            
            if len(optimized) < best_count:
                best_count = len(optimized)
                best_result = optimized

    if best_result is None:
        # Fallback: return a valid cover, even if not optimal
        # Greedy should have found one. If not, construct one manually.
        # For q=3, n=6, radius=1, the all-zero word covers 13 words.
        # We just need to cover the rest.
        # This fallback is unlikely to be hit.
        return []

    # Convert indices back to words
    result_words = []
    for idx in best_result:
        word = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            word[i] = temp % q
            temp //= q
        result_words.append(word)
        
    return result_words

# EVOLVE-BLOCK-END
