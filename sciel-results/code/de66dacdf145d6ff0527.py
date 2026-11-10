import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    # Parameters are fixed by the problem description effectively, 
    # but we should handle the general case or at least the specific one robustly.
    # The prompt says "evaluator passes this task's alphabet size q, length n and covering radius".
    # However, the task description specifies 8 matches, ternary, radius 1.
    # I will implement it generally for q=3, n=8, r=1 but allow parameters.
    
    if q != 3 or n != 8 or radius != 1:
         # Fallback to a simple construction for other cases to avoid errors, 
         # though the evaluator likely only calls with 3, 8, 1.
         # Direct sum for general q, n, r=1?
         # For now, return the seed solution if params don't match, 
         # or a simple greedy if they do.
         # Given the specific task contract, it's 3, 8, 1.
         pass

    N = q ** n
    deadline = time.time() + 115.0 # 5s safety margin
    
    # Precompute neighbors
    # List of all words
    # Convert 0..N-1 to base q digits?
    # It's faster to generate neighbors on the fly or precompute once.
    
    # Precompute neighbors for all N words
    # Shape: (N, 1 + 2*n)
    neighbors = np.zeros((N, 1 + 2 * n), dtype=np.int32)
    
    for w in range(N):
        idx = 0
        neighbors[w, idx] = w
        idx += 1
        
        # Get digits
        temp = w
        digits = [0] * n
        for i in range(n):
            digits[i] = temp % q
            temp //= q
            
        for i in range(n):
            d = digits[i]
            # change digit i to (d+1)%q
            w1 = w + ((d + 1) % q - d) * (q ** i)
            # change digit i to (d+2)%q
            w2 = w + ((d + 2) % q - d) * (q ** i)
            
            neighbors[w, idx] = w1
            idx += 1
            neighbors[w, idx] = w2
            idx += 1
            
    # Seed solution for comparison
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    seed_size = len(seed_code)
    
    # Convert seed code to integer representation
    def list_to_int(word):
        val = 0
        for i in range(n):
            val += word[i] * (q ** i)
        return val

    seed_code_ints = [list_to_int(w) for w in seed_code]
    
    best_code = seed_code_ints[:]
    best_size = len(best_code)
    
    rng = np.random.default_rng(42)
    
    # Strategy: Simulated Annealing / 1-Swap Local Search
    # Start with the seed code.
    # State: current_code (list of ints)
    # Move: Remove one codeword, add one new codeword.
    # Validity: The new code must still be a covering code.
    # Cost: len(code). We want to minimize this.
    # Since we always maintain a valid code, cost is just size.
    # A move is a "swap" (remove c_old, add c_new).
    # If c_new is already in code, it's a pure removal (if c_old is redundant).
    # If c_old is not redundant, we must add c_new that covers the "essential" neighbors of c_old.
    
    def get_essential_neighbors(code_set, c):
        """
        Return the set of neighbors of c that are ONLY covered by c.
        If this set is empty, c is redundant.
        """
        # Count coverage of neighbors of c by other codewords
        # This is expensive if done naively.
        # Optimization: Maintain a global coverage count array.
        # But in SA, we do many moves. Maintaining global counts is best.
        pass

    # We will maintain `cov_counts` globally for the current state.
    # cov_counts[w] = number of codewords in current_code that are within distance 1 of w.
    
    def compute_cov_counts(code):
        counts = np.zeros(N, dtype=np.int32)
        for c in code:
            counts[neighbors[c]] += 1
        return counts

    current_code = best_code[:]
    cov_counts = compute_cov_counts(current_code)
    
    # To check if a codeword c is removable:
    # min(cov_counts[neighbors[c]]) > 1 ?
    
    # To check if we can swap c_old with c_new:
    # 1. Temporarily remove c_old: cov_counts[neighbors[c_old]] -= 1
    # 2. Check if any neighbor of c_old has count 0. If yes, c_old was essential.
    # 3. If essential, c_new MUST cover all neighbors of c_old that now have count 0.
    #    i.e., for all w in neighbors[c_old] where cov_counts[w] == 0 (after removal),
    #    w must be in neighbors[c_new].
    # 4. If not essential (all counts >= 1), any c_new is valid (even one that doesn't cover anything new? 
    #    Actually, if c_old is redundant, we can just remove it. But the move is "swap", so we add c_new.
    #    If we add c_new, size stays same. We want to reduce size.
    #    So, if c_old is redundant, we should just remove it (size -1).
    #    If c_old is essential, we must swap with a c_new that covers the essential set. Size stays same.
    #    Wait, if we swap, size is constant. How do we reduce size?
    #    By finding a redundant codeword and removing it.
    #    So the "move" is:
    #    - If any codeword is redundant, remove it. (Size decreases)
    #    - Else, perform a swap: remove essential c_old, add c_new that covers its essential neighbors. (Size constant)
    #    
    #    This is a standard local search for minimum set cover.
    #    However, pure removal gets stuck. Swaps allow escaping.
    #    
    #    Let's implement:
    #    1. Try to find a redundant codeword. If found, remove it. Update best if smaller.
    #    2. If no redundant codeword, pick a random essential codeword c_old.
    #       Identify its essential neighbors E (where count becomes 0 after removal).
    #       Pick a random candidate c_new from the universe.
    #       Check if c_new covers E.
    #       If yes, perform swap.
    
    start_time = time.time()
    T = 10.0 # Initial temperature (not really used for cost since cost is size, but for acceptance of "neutral" or "worse" moves? 
             # Actually, if we maintain validity, size never increases. 
             # So we only accept moves that reduce size or maintain size (swaps).
             # Swaps that maintain size are "neutral". We accept them with prob exp(-0/T) = 1? 
             # Or we can use T to decide how many random swaps to try.
             # Let's just do random swaps if no reduction is possible.
             # To avoid cycling, we can keep a history or just rely on randomness.
    
    while time.time() < deadline:
        # Phase 1: Try to remove redundant codewords
        # This is fast if we just scan.
        removed_any = False
        # Iterate through a copy of indices
        # We need to be careful with cov_counts updates.
        
        # Find all redundant codewords
        redundant_indices = []
        for i, c in enumerate(current_code):
            if np.min(cov_counts[neighbors[c]]) > 1:
                redundant_indices.append(i)
        
        if redundant_indices:
            # Remove one (or all? Removing all at once is risky if they share neighbors? 
            # No, if min count > 1 for c, removing it makes counts >= 1 for its neighbors.
            # But if two redundant codewords share a neighbor, and that neighbor is covered ONLY by them and one other?
            # cov_counts[w] = 2. Remove c1 -> 1. Remove c2 -> 0. Invalid!
            # So we must remove one at a time and update counts.
            # Pick one randomly
            idx = rng.choice(redundant_indices)
            c = current_code[idx]
            
            # Remove c
            cov_counts[neighbors[c]] -= 1
            current_code.pop(idx)
            removed_any = True
            
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = current_code[:]
            
            # Continue loop to check for more removals
            continue # Re-evaluate redundancy immediately
        
        # Phase 2: No redundant codewords. Try a swap.
        # Pick a random codeword c_old to remove.
        idx_old = rng.integers(0, len(current_code))
        c_old = current_code[idx_old]
        
        # Find essential neighbors of c_old (those with count 1)
        # These are the words that will become uncovered if we remove c_old.
        neigh_old = neighbors[c_old]
        essential_mask = (cov_counts[neigh_old] == 1)
        essential_set = neigh_old[essential_mask]
        
        # If no essential neighbors, it was redundant? 
        # Wait, we checked for redundancy above. 
        # If min(cov_counts[neighbors[c]]) > 1, it's redundant.
        # So here, min(cov_counts[neighbors[c]]) == 1.
        # So essential_set is not empty.
        
        # We need to find a c_new such that essential_set is a subset of neighbors[c_new].
        # Also, c_new should preferably cover other words to potentially make other codewords redundant later.
        
        # Strategy: Pick a random word from essential_set, say w.
        # Any c_new that covers w is a candidate.
        # But we need c_new to cover ALL of essential_set.
        
        # Efficient way:
        # 1. Pick a random w from essential_set.
        # 2. Get all candidates that cover w.
        #    How to get candidates covering w? 
        #    We precomputed neighbors[c] for each c. 
        #    We need the inverse: for each w, which c cover it?
        #    Let's precompute `coverers` list.
        
        # Precompute coverers: coverers[w] = list of c such that w in neighbors[c]
        # This is the transpose of the neighbor matrix.
        # N=6561, neighbors per node = 17. Total edges = 6561 * 17 ~ 111k.
        # coverers will have ~111k entries total.
        
        # Let's build coverers once.
        # coverers = [[] for _ in range(N)]
        # for c in range(N):
        #     for w in neighbors[c]:
        #         coverers[w].append(c)
        
        # This takes time to build, but once is fine.
        # Or, we can just pick a random c_new from the whole universe and check if it covers essential_set.
        # Probability of covering a specific set of k words is low.
        # Better: Pick w from essential_set. Pick c_new from coverers[w]. Check if it covers rest of essential_set.
        
        # Let's build coverers.
        # We can do this lazily or at start.
        # Since N is small, let's just build it at start of this block if not already done.
        
        # Actually, let's just use the precomputed neighbors and do a check.
        # If essential_set is small (usually 1 or 2), checking is fast.
        
        # Pick a random w from essential_set
        w_target = essential_set[rng.integers(0, len(essential_set))]
        
        # We need a candidate c_new that covers w_target.
        # Instead of building coverers, let's just pick a random c_new from 0..N-1?
        # No, that's 1/17 chance to cover w_target.
        # Let's precompute `coverers` for efficiency.
        
        # I'll add coverers precomputation at the top of the function.
        
        # For now, let's assume we have coverers.
        # Let's refactor: Precompute coverers at the start.
        
        # --- Refactor: Precompute coverers ---
        # I will insert this in the code below.
        
        # For this specific step, let's just try a few random candidates from the universe?
        # No, that's too slow.
        # Let's pick c_new from a small pool.
        # The "essential" neighbors of c_old are close to c_old.
        # A good c_new is likely close to c_old or close to essential_set.
        
        # Let's just pick a random c_new from the entire space and check.
        # If it fails, try another.
        # But how many tries?
        
        # Better approach:
        # Pick w_target.
        # We know c_old covers w_target.
        # Let's pick c_new randomly from the set of all words?
        # No.
        
        # Let's stick to the "Swap" idea but make it efficient.
        # 1. Identify essential_set.
        # 2. If essential_set is large, it's hard to cover.
        # 3. Usually essential_set is small.
        
        # Let's precompute `coverers` properly in the final code.
        
        # For now, in this thought process, I'll design the code to use coverers.
        
        # --- Final Code Structure ---
        # 1. Precompute neighbors and coverers.
        # 2. Initialize code.
        # 3. Loop:
        #    a. Find redundant, remove one.
        #    b. If none, pick c_old, find essential_set.
        #    c. Pick w from essential_set.
        #    d. Pick c_new from coverers[w].
        #    e. Check if c_new covers all essential_set.
        #    f. If yes, swap.
        #    g. If no, try another c_new from coverers[w] (maybe up to K tries).
        #    h. If all fail, try a different c_old? Or just move on.
        
        # To avoid infinite loops of failed swaps, we can limit tries per c_old.
        
        # Let's implement this.

    # The code above is a sketch. I will now write the actual Python code.
    
    # --- ACTUAL IMPLEMENTATION ---
    
    # Re-run precomputation if needed? No, I'll put it at the top.
    
    # I will rewrite the whole function to be clean.
    
    pass # Placeholder

# REWRITING THE FUNCTION PROPERLY

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback
        return []

    N = q ** n
    deadline = time.time() + 115.0
    
    # Precompute neighbors
    neighbors = np.zeros((N, 1 + 2 * n), dtype=np.int32)
    # Precompute coverers: list of lists
    coverers = [[] for _ in range(N)]
    
    for w in range(N):
        idx = 0
        neighbors[w, idx] = w
        idx += 1
        coverers[w].append(w)
        
        temp = w
        digits = [0] * n
        for i in range(n):
            digits[i] = temp % q
            temp //= q
            
        for i in range(n):
            d = digits[i]
            w1 = w + ((d + 1) % q - d) * (q ** i)
            w2 = w + ((d + 2) % q - d) * (q ** i)
            
            neighbors[w, idx] = w1
            idx += 1
            neighbors[w, idx] = w2
            idx += 1
            
            coverers[w1].append(w)
            coverers[w2].append(w)
            
    # Seed solution
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    def list_to_int(word):
        val = 0
        for i in range(n):
            val += word[i] * (q ** i)
        return val

    current_code = [list_to_int(w) for w in seed_code]
    best_code = current_code[:]
    best_size = len(current_code)
    
    # Compute initial coverage counts
    cov_counts = np.zeros(N, dtype=np.int32)
    for c in current_code:
        cov_counts[neighbors[c]] += 1
        
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # 1. Try to remove redundant codewords
        # Find indices of redundant codewords
        # A codeword c is redundant if min(cov_counts[neighbors[c]]) > 1
        
        # Optimization: Instead of checking all, we can just pick a random one and check?
        # No, we want to remove as many as possible quickly.
        # But checking all is O(N_code * 17). N_code ~ 400. 400*17 = 6800 ops. Very fast.
        
        redundant_indices = []
        for i, c in enumerate(current_code):
            if np.min(cov_counts[neighbors[c]]) > 1:
                redundant_indices.append(i)
                
        if redundant_indices:
            # Remove one
            idx = redundant_indices[rng.integers(0, len(redundant_indices))]
            c = current_code[idx]
            cov_counts[neighbors[c]] -= 1
            current_code.pop(idx)
            
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = current_code[:]
            continue # Re-check immediately
        
        # 2. No redundant codewords. Try a swap.
        # Pick a random codeword to remove
        idx_old = rng.integers(0, len(current_code))
        c_old = current_code[idx_old]
        
        neigh_old = neighbors[c_old]
        # Essential neighbors: those with count 1
        essential_mask = (cov_counts[neigh_old] == 1)
        essential_set = neigh_old[essential_mask]
        
        if len(essential_set) == 0:
            # Should not happen if redundancy check was correct, but just in case
            # Remove it
            cov_counts[neigh_old] -= 1
            current_code.pop(idx_old)
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = current_code[:]
            continue
            
        # We need to find c_new that covers all words in essential_set.
        # Pick a random word w from essential_set to constrain candidates
        w_target = essential_set[rng.integers(0, len(essential_set))]
        
        # Candidates are in coverers[w_target]
        candidates = coverers[w_target]
        
        # Try up to K candidates
        found_swap = False
        # Shuffle candidates? Or just pick random indices.
        # Let's try a few random ones
        for _ in range(20): # Max 20 tries per swap attempt
            if not candidates:
                break
            c_new = candidates[rng.integers(0, len(candidates))]
            
            # Check if c_new covers all essential_set
            # c_new covers w if w is in neighbors[c_new]
            # We need: essential_set is subset of neighbors[c_new]
            
            # Efficient check:
            # For each w in essential_set, is w in neighbors[c_new]?
            # We can precompute a boolean matrix? No, too big.
            # Just check. essential_set is small (usually 1-3).
            
            valid = True
            for w in essential_set:
                # Check if w is in neighbors[c_new]
                # neighbors[c_new] is an array.
                # We can use np.isin? Or just loop.
                # Since essential_set is small, loop is fine.
                if w not in neighbors[c_new]:
                    valid = False
                    break
            
            if valid:
                # Perform swap
                # Remove c_old
                cov_counts[neighbors[c_old]] -= 1
                # Add c_new
                cov_counts[neighbors[c_new]] += 1
                
                # Update code list
                current_code[idx_old] = c_new
                
                found_swap = True
                break
        
        # If no swap found, we just continue the loop.
        # To avoid getting stuck in the same state, the randomness in idx_old and candidates helps.
        
        # Optional: If we are stuck for a long time, we could restart?
        # But the swap mechanism should allow exploration.
        
    # Convert best_code back to lists
    final_code = []
    for w in best_code:
        word = []
        temp = w
        for _ in range(n):
            word.append(temp % q)
            temp //= q
        final_code.append(word)
        
    return final_code
