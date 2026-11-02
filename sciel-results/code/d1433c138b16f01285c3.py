"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(q, n, radius):
    # Specific case: q=3, n=6, radius=1
    # We use Simulated Annealing to minimize the number of codewords while maintaining coverage.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: direct sum of tetracode (if applicable) or simple greedy
        # For now, return a valid but potentially suboptimal code for non-standard params
        # to satisfy the interface.
        # Simple greedy for general case
        all_words = list(np.ndindex(*([q] * n)))
        # Convert tuples to lists for return
        # This is slow, so only for fallback
        def dist(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)
        
        covered = set()
        code = []
        # Precompute neighbors for speed if n is small, but for general fallback just iterate
        # Actually, for the specific task q=3, n=6 is the only one that matters for score.
        # We will just return the tetracode-based 81 code for non-matching params as a safe fallback.
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in np.ndindex(*([3] * (n - 4)))]
        else:
            # Brute force greedy for tiny cases
            uncovered = set(all_words)
            while uncovered:
                best_c = None
                best_gain = -1
                for c in all_words:
                    if c in code:
                        continue
                    gain = 0
                    for u in uncovered:
                        if dist(c, u) <= radius:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_c = c
                if best_c is None:
                    break
                code.append(best_c)
                uncovered = {u for u in uncovered if dist(best_c, u) > radius}
            return code

    # --- Specific Optimization for q=3, n=6, r=1 ---
    
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    N = 3 ** 6  # 729 total words
    
    # Precompute neighbors for each word (including itself)
    # neighbors[i] is a numpy array of indices j such that dist(i, j) <= 1
    # dist(i, j) <= 1 means they are the same or differ in exactly 1 position.
    
    # Convert index to base-3 vector
    def idx_to_vec(idx):
        vec = np.zeros(6, dtype=np.int8)
        temp = idx
        for k in range(6):
            vec[k] = temp % 3
            temp //= 3
        return vec

    # Precompute neighbor lists as numpy arrays for fast intersection/union if needed,
    # but for SA, we need to know which words are covered by a specific codeword.
    # Let's precompute a list of lists of neighbors.
    neighbors = [None] * N
    for i in range(N):
        v = idx_to_vec(i)
        nbrs = [i]
        for pos in range(6):
            orig = v[pos]
            for new_val in range(3):
                if new_val != orig:
                    v[pos] = new_val
                    # Convert back to index
                    j = 0
                    for k in range(6):
                        j += v[k] * (3 ** k)
                    nbrs.append(j)
            v[pos] = orig
        neighbors[i] = np.array(nbrs, dtype=np.int32)
        
    # Initialize with a greedy solution to have a valid starting point
    # Greedy: pick word covering most uncovered
    uncovered = np.ones(N, dtype=bool)
    code = []
    
    # To speed up greedy, we can just pick random words or use a simple heuristic.
    # Actually, let's start with a random set of size 81 (known valid upper bound) or just random 75.
    # Better: Start with the tetracode construction (81 words) which is guaranteed valid.
    # Tetracode construction:
    # C = { (a, b, a+b, a+2b) mod 3 } x {0,1,2} x {0,1,2}
    tetracode_part = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode_part:
        for x in range(3):
            for y in range(3):
                initial_code.append(h + [x, y])
                
    # Convert to indices
    def vec_to_idx(vec):
        idx = 0
        for k in range(6):
            idx += vec[k] * (3 ** k)
        return idx

    current_code = [vec_to_idx(v) for v in initial_code]
    current_code_set = set(current_code)
    
    # Calculate initial coverage count
    # cover_count[j] = number of codewords c in current_code such that j is in neighbors[c]
    cover_count = np.zeros(N, dtype=np.int32)
    for c in current_code:
        cover_count[neighbors[c]] += 1
        
    # Check if valid
    is_valid = np.all(cover_count > 0)
    
    # Simulated Annealing
    # State: set of codewords
    # Energy: number of uncovered words (we want 0). If 0, minimize size.
    # Actually, we want to minimize size subject to coverage.
    # Let's define Cost = size + M * (number of uncovered words).
    # If we maintain coverage, Cost = size.
    # Moves:
    # 1. Remove a codeword. If coverage remains, size decreases (good). If coverage lost, cost increases by M.
    # 2. Add a codeword. Size increases (bad), but might allow removals later.
    # 3. Swap a codeword.
    
    # Strategy: Start with valid code of size 81.
    # Try to remove words. If removal causes uncovered words, we can accept with probability exp(-DeltaE/T).
    # But we want to end up with a valid code.
    # So, we should only accept moves that keep the code valid, OR use a penalty method where we allow invalid states but strongly penalize them.
    
    # Let's use a penalty method.
    # E = len(code) + lambda * (num_uncovered)
    # lambda is large, e.g., 1000.
    # We want to find a state with num_uncovered=0 and minimal len(code).
    
    # Initialize SA
    T = 10.0
    cooling = 0.995
    min_T = 0.01
    
    best_code = current_code[:]
    best_size = len(current_code)
    best_uncovered = 0
    
    # We will track the actual set of indices for the current code.
    # Using a list is fine for size ~80.
    
    rng = np.random.default_rng(42)
    
    # Precompute for speed:
    # For each word i, we have neighbors[i].
    
    # Helper to get current energy
    def get_energy(code_list, cover_count):
        uncovered_count = np.sum(cover_count == 0)
        return len(code_list) + 1000 * uncovered_count

    current_energy = get_energy(current_code, cover_count)
    
    # SA Loop
    steps = 0
    while time.time() < deadline and T > min_T:
        # Choose a move
        # 70% chance remove, 30% chance add (or swap)
        # If size is small, maybe just remove/swap.
        
        move_type = rng.random()
        if move_type < 0.7 and len(current_code) > 1:
            # Remove
            idx_to_remove = rng.integers(0, len(current_code))
            c = current_code[idx_to_remove]
            
            # Calculate change in cover_count
            # New cover_count = cover_count - 1 for neighbors[c]
            # Uncovered words increase if any neighbor had count 1.
            
            nbrs_c = neighbors[c]
            # How many words become uncovered?
            # Words in nbrs_c with cover_count == 1 will become 0.
            newly_uncovered = np.sum(cover_count[nbrs_c] == 1)
            
            # How many words were previously uncovered?
            # If c was the only one covering some word, it was uncovered? No, if cover_count was 0, it was uncovered.
            # Removing c doesn't make uncovered words covered.
            
            # Change in uncovered count = newly_uncovered
            # Change in size = -1
            # Delta E = -1 + 1000 * newly_uncovered
            
            delta_E = -1 + 1000 * newly_uncovered
            
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                # Accept
                # Update state
                current_code.pop(idx_to_remove)
                cover_count[nbrs_c] -= 1
                
                current_energy += delta_E
                
                if np.all(cover_count > 0):
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = current_code[:]
                        best_uncovered = 0
                else:
                    # Track best valid? No, we track best energy.
                    pass
                    
        elif move_type < 0.9:
            # Add
            # Pick a random word not in code? Or one that covers many uncovered words.
            # To be efficient, pick a random candidate.
            # Let's pick a random word from 0..728.
            c = rng.integers(0, N)
            if c in current_code_set:
                continue # Skip if already in code, or pick another
            
            # Calculate change
            nbrs_c = neighbors[c]
            # Words in nbrs_c with cover_count == 0 will become covered.
            newly_covered = np.sum(cover_count[nbrs_c] == 0)
            
            # Change in uncovered count = -newly_covered
            # Change in size = +1
            # Delta E = 1 - 1000 * newly_covered
            
            delta_E = 1 - 1000 * newly_covered
            
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                # Accept
                current_code.append(c)
                current_code_set.add(c)
                cover_count[nbrs_c] += 1
                
                current_energy += delta_E
                
                if np.all(cover_count > 0):
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = current_code[:]
                        best_uncovered = 0
                        
        else:
            # Swap
            if len(current_code) < 2:
                continue
            idx_to_remove = rng.integers(0, len(current_code))
            c_out = current_code[idx_to_remove]
            
            # Pick c_in
            # Ideally one that covers uncovered words.
            # Random pick for simplicity, or weighted by uncovered coverage.
            c_in = rng.integers(0, N)
            if c_in in current_code_set:
                # Pick another
                c_in = (c_in + 1) % N
                if c_in in current_code_set:
                    continue
                    
            nbrs_out = neighbors[c_out]
            nbrs_in = neighbors[c_in]
            
            # Calculate delta cover_count
            # cover_count_new = cover_count - 1[nbrs_out] + 1[nbrs_in]
            # We need to count how many words go from 0->1 (covered) and 1->0 (uncovered).
            
            # This is tricky with overlapping sets.
            # Let's just simulate the change on a copy of cover_count for the affected indices?
            # Or compute directly.
            
            # Words affected are in union(nbrs_out, nbrs_in).
            # Let's just update a temporary array? No, too slow.
            
            # Let's calculate newly_uncovered and newly_covered carefully.
            # For each word w:
            # old_count = cover_count[w]
            # new_count = old_count - (1 if w in nbrs_out else 0) + (1 if w in nbrs_in else 0)
            
            # We can iterate over the unique words in nbrs_out and nbrs_in.
            # Size of nbrs is 13. Union is at most 26.
            
            unique_nbrs = np.unique(np.concatenate([nbrs_out, nbrs_in]))
            
            newly_uncovered = 0
            newly_covered = 0
            
            for w in unique_nbrs:
                old_c = cover_count[w]
                in_out = (w in nbrs_out) # This is slow due to numpy array search
                # Better to use a boolean mask or set
                # Let's precompute sets? No, numpy arrays are faster for sum.
                
            # Alternative:
            # Let's just use sets for the neighbor lists for fast membership?
            # No, let's stick to numpy but compute delta efficiently.
            
            # delta_count[w] = -1 if w in nbrs_out else 0 + 1 if w in nbrs_in else 0
            
            # We can create a temporary array of zeros, but that's allocation.
            # Let's just loop over the small lists.
            
            # Convert to sets for this small operation? 26 elements.
            set_out = set(nbrs_out.tolist())
            set_in = set(nbrs_in.tolist())
            
            for w in set_out.union(set_in):
                old_c = cover_count[w]
                new_c = old_c
                if w in set_out:
                    new_c -= 1
                if w in set_in:
                    new_c += 1
                
                if old_c == 0 and new_c > 0:
                    newly_covered += 1
                elif old_c > 0 and new_c == 0:
                    newly_uncovered += 1
            
            # Change in size = 0
            # Change in uncovered = newly_uncovered - newly_covered
            # Delta E = 1000 * (newly_uncovered - newly_covered)
            
            delta_E = 1000 * (newly_uncovered - newly_covered)
            
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                # Accept
                # Update state
                current_code[idx_to_remove] = c_in
                current_code_set.remove(c_out)
                current_code_set.add(c_in)
                
                # Update cover_count
                cover_count[nbrs_out] -= 1
                cover_count[nbrs_in] += 1
                
                current_energy += delta_E
                
                if np.all(cover_count > 0):
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = current_code[:]
                        best_uncovered = 0

        # Cool down
        T *= cooling
        steps += 1
        
        # Periodic check for time
        if time.time() > deadline - 5:
            break

    # Final verification and return
    # We want to return the best valid code found.
    # If best_code is not valid (shouldn't happen if we tracked correctly), we need to fix it.
    # But our tracking updates best_code only when valid.
    
    # However, if we never found a valid code smaller than 81, we return the initial one.
    # But wait, the initial code IS valid. So best_code is at least the initial one.
    
    # Let's double check the best_code validity just in case.
    final_cover = np.zeros(N, dtype=np.int32)
    for c in best_code:
        final_cover[neighbors[c]] += 1
        
    if not np.all(final_cover > 0):
        # Fallback to initial valid code
        best_code = current_code if np.all(cover_count > 0) else [vec_to_idx(v) for v in initial_code]
        
    # Convert indices back to vectors
    result = [idx_to_vec(c).tolist() for c in best_code]
    
    return result

# EVOLVE-BLOCK-END
