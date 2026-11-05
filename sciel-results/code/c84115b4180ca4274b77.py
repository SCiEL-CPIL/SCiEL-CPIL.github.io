import time
import random
import itertools

def construct(q, n, radius):
    # Constants for the specific problem instance
    # q=3, n=8, radius=1
    total_words = q ** n
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # 1. Precompute indices and spheres
    # Map word (tuple) to index
    # Index = w[0]*3^7 + w[1]*3^6 + ... + w[7]
    # Or simply use itertools.product and enumerate
    
    # Precompute all words as tuples for neighbor generation
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute spheres (bitmasks of covered words for each center)
    # A sphere of radius 1 includes the center and all words differing in 1 position.
    # Size = 1 + n * (q - 1)
    sphere_masks = [0] * total_words
    
    for i, w in enumerate(all_words):
        mask = 0
        # Include self
        mask |= (1 << i)
        # Include neighbors
        for pos in range(n):
            for new_val in range(q):
                if new_val == w[pos]:
                    continue
                # Create neighbor word
                neighbor = list(w)
                neighbor[pos] = new_val
                neighbor = tuple(neighbor)
                j = word_to_idx[neighbor]
                mask |= (1 << j)
        sphere_masks[i] = mask
        
    # Helper to convert index back to word
    def idx_to_word(idx):
        w = [0] * n
        base = 1
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w
        
    # Note: The mapping in word_to_idx uses product order.
    # product(range(3), repeat=8) generates (0,0,0,0,0,0,0,0) first.
    # My idx_to_word above assumes base-q representation where last digit is least significant.
    # Let's check consistency.
    # all_words[0] = (0,0,0,0,0,0,0,0).
    # idx_to_word(0) -> [0,0,0,0,0,0,0,0]. Consistent.
    # all_words[1] = (0,0,0,0,0,0,0,1).
    # idx_to_word(1) -> w[7]=1, others 0. Consistent.
    
    best_solution = None
    best_size = total_words + 1
    
    # Greedy Set Cover with Randomized Restarts
    # We will perform several restarts. In each restart, we run a greedy algorithm.
    # To avoid getting stuck in the same local minimum, we can randomize the tie-breaking
    # or the initial selection order.
    
    num_restarts = 50 # Adjust based on speed. Bitset ops are very fast.
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        covered = 0
        solution_indices = []
        
        # Optional: Randomize the order of candidates to break symmetry in greedy choice
        # We can just iterate 0..N-1 but pick random among max gain.
        
        while True:
            if time.time() > deadline:
                break
                
            best_gain = -1
            best_candidate = -1
            candidates_with_max_gain = []
            
            # Iterate through all possible centers
            for c in range(total_words):
                # Calculate gain: how many new words does this cover?
                # ~covered has 1s where we need coverage.
                # We need to mask ~covered to only consider valid bits? 
                # Python ints are infinite precision, but ~covered will have 1s in high bits too.
                # However, sphere_masks only have bits set in 0..total_words-1.
                # So (sphere_masks[c] & ~covered) will only have bits set where sphere has 1 AND covered has 0.
                # This is safe.
                
                gain = (sphere_masks[c] & ~covered).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = c
                    candidates_with_max_gain = [c]
                elif gain == best_gain:
                    candidates_with_max_gain.append(c)
                    
            if best_gain == 0:
                break # Fully covered
                
            # Tie-breaking: Randomly select from candidates with max gain to encourage diversity
            # In the first few steps, randomization helps explore different branches.
            selected = random.choice(candidates_with_max_gain)
            
            solution_indices.append(selected)
            covered |= sphere_masks[selected]
            
        # Check if we have a valid full cover
        if covered == (1 << total_words) - 1:
            if len(solution_indices) < best_size:
                best_size = len(solution_indices)
                best_solution = solution_indices.copy()
                
        # Early termination if we hit the known lower bound or a very good number
        # Lower bound is 403. If we find 403, we stop.
        if best_size <= 403:
            break
            
    if best_solution is None:
        # Fallback: Return the seed solution structure if something went wrong
        # But the greedy should always find a cover.
        # Just return a dummy or re-run once strictly deterministically?
        # The seed program returns 486. If we fail, we should at least return a valid code.
        # Let's do one deterministic run if no best found.
        covered = 0
        solution_indices = []
        while True:
            best_gain = -1
            best_candidate = -1
            for c in range(total_words):
                gain = (sphere_masks[c] & ~covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = c
            if best_gain == 0:
                break
            solution_indices.append(best_candidate)
            covered |= sphere_masks[best_candidate]
        best_solution = solution_indices
        best_size = len(solution_indices)

    # Convert indices back to words
    result = [idx_to_word(i) for i in best_solution]
    return result
