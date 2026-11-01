import time
import random
import heapq
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Specific case for q=3, n=6, radius=1
    # General fallback for other parameters
    if q == 3 and n == 6 and radius == 1:
        # Total words
        total_words = q ** n
        
        # Generate all words as tuples
        all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute spheres: list of indices for each word within distance 1
        spheres = []
        for w in all_words:
            sphere = [word_to_idx[w]]
            for i in range(n):
                # Change coordinate i to all other q-1 values
                for val in range(q):
                    if val == w[i]:
                        continue
                    new_w = list(w)
                    new_w[i] = val
                    sphere.append(word_to_idx[tuple(new_w)])
            spheres.append(sphere)
        
        covered = [False] * total_words
        uncovered_count = total_words
        code = []
        picked = set()
        
        # For each word, track how many of its sphere neighbors are still uncovered
        # This is the "gain" if we select this word
        # Use a max-heap: (-gain, tiebreaker, word_idx)
        # Tiebreaker uses symmetry: we can use a canonical form based on coordinate permutations
        
        # Precompute a symmetry-based tiebreaker score for each word
        # The cubic group allows permuting coordinates and permuting symbols within each coordinate
        # For tie-breaking, we use the lexicographically smallest representative under coordinate permutation
        # This breaks ties consistently based on structural symmetry
        
        def symmetry_score(w):
            # Try all n! coordinate permutations, keep the lexicographically smallest
            # For efficiency, just use the word itself as tiebreaker (simpler)
            # Or use a hash of the sorted coordinates
            return w
        
        # Initialize heap with all words
        # gain[word_idx] = number of uncovered words in its sphere
        # Initially all are uncovered, so gain = len(spheres[word_idx]) = 13 for all
        # But we need to track which are actually uncovered
        
        # More efficient: maintain gain dynamically
        # gain[i] = count of uncovered words in spheres[i]
        gain = [0] * total_words
        for i in range(total_words):
            gain[i] = len(spheres[i])  # Initially all uncovered
        
        # Max-heap: (-gain, symmetry_tiebreaker, word_idx)
        heap = []
        for i in range(total_words):
            # Use negative gain for max-heap
            # Tiebreaker: use the word tuple itself for lexicographic ordering
            heap.append((-gain[i], all_words[i], i))
        
        heapq.heapify(heap)
        
        # Track which heap entries are stale (gain changed)
        # We'll use lazy deletion: when we pop, check if the gain matches current
        
        while uncovered_count > 0:
            if time.time() > deadline:
                break
            
            # Pop until we find a valid (non-stale) entry
            best_candidate = -1
            best_gain = 0
            
            while heap:
                neg_gain, word_tuple, cand_idx = heapq.heappop(heap)
                current_gain = gain[cand_idx]
                
                # If this entry is stale (gain has changed), skip
                if -neg_gain != current_gain:
                    continue
                
                # If this word is already picked, skip
                if cand_idx in picked:
                    continue
                
                best_candidate = cand_idx
                best_gain = current_gain
                break
            
            if best_candidate == -1:
                # No valid candidates left, but uncovered_count > 0
                # This shouldn't happen, but break to avoid infinite loop
                break
            
            # Select best candidate
            picked.add(best_candidate)
            code.append(list(all_words[best_candidate]))
            
            # Update covered status and gains
            for idx in spheres[best_candidate]:
                if not covered[idx]:
                    covered[idx] = True
                    uncovered_count -= 1
                    
                    # Update gain for all words whose sphere contains idx
                    # This is expensive: for each word j, if idx in spheres[j], gain[j]--
                    # We need an efficient way to do this
                    # Precompute reverse: for each word, which spheres contain it
                    # But that's the same as spheres reversed
                    # Alternative: just recompute gain for affected words
                    # Actually, we can maintain a list of words affected by each uncovered word
                    pass
            
            # The above update is O(total_words * sphere_size) per iteration
            # Which is too slow for 729 words with ~73 iterations
            # Let's use a different approach: recompute gains periodically or use a different data structure
            
            # Actually, let's simplify: just do the greedy scan each time
            # But that's what the original does...
            
            # Let me try a hybrid: use the heap but update gains efficiently
            # For each uncovered word idx, maintain a list of candidate words that cover it
            # When idx becomes covered, decrement gain for all candidates in that list
            
            # This requires precomputing: for each word, which candidates cover it
            # That's just the reverse of spheres
            # Let's precompute this
        
        # The above approach is getting complex. Let me use a simpler but still improved approach:
        # Use the original greedy scan but with better tie-breaking and early termination
        
        # Reset and use a cleaner approach
        covered = [False] * total_words
        uncovered_count = total_words
        code = []
        picked = set()
        
        # Precompute reverse spheres: for each word, which codewords cover it
        # This is the same as spheres but transposed
        # spheres[i] = list of words covered by codeword i
        # reverse_spheres[j] = list of codewords that cover word j
        reverse_spheres = [[] for _ in range(total_words)]
        for i in range(total_words):
            for j in spheres[i]:
                reverse_spheres[j].append(i)
        
        # For each candidate, track how many uncovered words it covers
        # gain[i] = number of uncovered words in spheres[i]
        gain = [0] * total_words
        for i in range(total_words):
            gain[i] = len(spheres[i])
        
        # Use a priority queue with lazy deletion
        # heap entries: (-gain, symmetry_key, word_idx)
        # symmetry_key: use the word itself for consistent tie-breaking
        heap = []
        for i in range(total_words):
            heap.append((-gain[i], all_words[i], i))
        heapq.heapify(heap)
        
        # Track version to detect stale entries
        # When gain[i] changes, old heap entries become stale
        # We'll check if the gain matches when we pop
        
        while uncovered_count > 0:
            if time.time() > deadline:
                break
            
            # Find the best candidate using the heap
            best_candidate = -1
            best_gain = 0
            
            while heap:
                neg_gain, word_tuple, cand_idx = heapq.heappop(heap)
                
                # Check if this entry is stale
                if -neg_gain != gain[cand_idx]:
                    continue
                
                # Check if already picked
                if cand_idx in picked:
                    continue
                
                best_candidate = cand_idx
                best_gain = -neg_gain
                break
            
            if best_candidate == -1:
                break
            
            # Select this candidate
            picked.add(best_candidate)
            code.append(list(all_words[best_candidate]))
            
            # Update: mark all words in spheres[best_candidate] as covered
            # For each newly covered word, decrement gain for all candidates that cover it
            for idx in spheres[best_candidate]:
                if not covered[idx]:
                    covered[idx] = True
                    uncovered_count -= 1
                    
                    # Decrement gain for all candidates that cover idx
                    for cand in reverse_spheres[idx]:
                        if cand not in picked:
                            gain[cand] -= 1
                            # Push updated gain to heap
                            heapq.heappush(heap, (-gain[cand], all_words[cand], cand))
        
        return code
    else:
        # Fallback: Random or simple construction for other params
        # For this task, we expect q=3, n=6, radius=1.
        # But to be safe, return a valid code (even if large)
        return [list(t) for t in itertools.product(range(q), repeat=n)]
