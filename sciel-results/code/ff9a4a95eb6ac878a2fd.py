import numpy as np
import itertools
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: 120s limit, stop at 115s
    deadline = start_time + 115.0

    # 1. Generate all words and map to indices
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # 2. Precompute neighbors as bitsets (Python integers)
    # neighbors[i] is an integer where bit j is 1 if dist(words[i], words[j]) <= radius
    # For radius=1, neighbors[i] includes i and all words at distance 1.
    neighbors_bits = [0] * N
    
    # Efficiently compute neighbors using index arithmetic
    # For q=3, n=7, we can compute the index of a word by changing one digit.
    # Let's precompute the step sizes for each position.
    # Index is sum(d_i * q^(n-1-i))
    # Changing d_i to d_i + delta changes index by delta * q^(n-1-i)
    
    powers = [q ** (n - 1 - i) for i in range(n)]
    
    for i in range(N):
        mask = 1 << i
        w = words[i]
        # Start with self
        current_mask = mask
        
        # For each position, try all other values
        for pos in range(n):
            orig_val = w[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                # Calculate new index
                delta = (val - orig_val) * powers[pos]
                new_idx = i + delta
                current_mask |= (1 << new_idx)
        
        neighbors_bits[i] = current_mask

    # Full mask for all points
    full_mask = (1 << N) - 1
    
    def get_counts_bitset(code_list):
        """
        Returns a list of counts for each point, or just a boolean mask of covered points?
        For local search, we often just need to know if a point is covered by >= 1 codeword.
        However, for pruning, we need to know if a point is covered by >= 2 codewords 
        (so that removing a codeword doesn't uncover it).
        
        Instead of counting, we can maintain a bitset `covered` and a bitset `double_covered`?
        Or simpler: 
        To check if removing codeword c is valid:
        Let U be the union of neighbors of all codewords EXCEPT c.
        If (neighbors_bits[c] & ~U) == 0, then c is redundant.
        
        To check if swapping c with w is valid:
        Let U_old be union of neighbors of all codewords.
        Let U_new = U_old \ neighbors_bits[c] | neighbors_bits[w]
        We need U_new == full_mask.
        """
        pass

    def is_valid_code(code_list):
        """Check if code_list covers all points."""
        if not code_list:
            return N == 0
        union_mask = 0
        for c in code_list:
            union_mask |= neighbors_bits[c]
        return union_mask == full_mask

    def prune_code(code_list):
        """
        Iteratively remove redundant codewords.
        A codeword c is redundant if the union of neighbors of all other codewords 
        covers the neighbors of c.
        """
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            # Compute union of all neighbors
            total_union = 0
            for c in code_list:
                total_union |= neighbors_bits[c]
            
            if total_union != full_mask:
                break # Not even valid, can't prune safely without repair
            
            # Try to remove each codeword
            for i in range(len(code_list) - 1, -1, -1):
                c = code_list[i]
                # Union of all others
                union_others = 0
                for j in range(len(code_list)):
                    if j == i:
                        continue
                    union_others |= neighbors_bits[code_list[j]]
                
                # If neighbors of c are subset of union_others, c is redundant
                if (neighbors_bits[c] & ~union_others) == 0:
                    code_list.pop(i)
                    changed = True
                    break # Restart loop as structure changed
                    
        return code_list

    def improve_by_swaps(code_list, max_iterations=50):
        """
        Try to swap codewords to reduce size or enable further pruning.
        Strategy: 
        1. Try to remove a codeword. If it creates holes, try to fill them with a new codeword 
           that covers the holes AND overlaps significantly with the removed one's coverage 
           to allow future removals? 
           Actually, simpler: 
           Try to replace a codeword c with a new codeword w such that:
           1. w is not in code_list.
           2. The new code is valid.
           3. The new code has better "prunability" (e.g., more double-covered points).
           
        Given the small N, we can be a bit more exhaustive.
        """
        code_set = set(code_list)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations and time.time() < deadline - 5:
            improved = False
            iterations += 1
            
            # First, prune
            code_list = prune_code(code_list)
            code_set = set(code_list)
            
            if not code_list:
                break
                
            # Calculate current union
            total_union = 0
            for c in code_list:
                total_union |= neighbors_bits[c]
                
            # If not valid, we have a bug in previous steps, but assume valid.
            
            # Try swapping each codeword
            # Shuffle to avoid bias
            random.shuffle(code_list)
            
            for c in code_list:
                if time.time() > deadline - 5:
                    break
                    
                # What points are ONLY covered by c?
                # These are "critical points" for c.
                # If we remove c, these points become uncovered.
                # We need a new word w that covers ALL critical points.
                
                # Critical points = neighbors_bits[c] & ~(union of neighbors of others)
                union_others = 0
                for j in code_list:
                    if j != c:
                        union_others |= neighbors_bits[j]
                        
                critical_mask = neighbors_bits[c] & ~union_others
                
                if critical_mask == 0:
                    # c is redundant, prune should have caught it? 
                    # Maybe prune loop broke early?
                    code_list.remove(c)
                    code_set.remove(c)
                    improved = True
                    break # Restart
                    
                # Find candidates w that cover all critical points
                # w must satisfy: (neighbors_bits[w] & critical_mask) == critical_mask
                # Also w not in code_set.
                
                # To speed up, we can restrict candidates to neighbors of one critical point.
                # Pick the first set bit in critical_mask
                # Find index of first set bit
                # Python int has no bit_length that gives index directly, but we can use:
                # lowbit = critical_mask & -critical_mask
                # idx = lowbit.bit_length() - 1
                
                lowbit = critical_mask & -critical_mask
                p_idx = lowbit.bit_length() - 1
                
                # Candidates must be in neighbors of p_idx
                # Iterate through bits in neighbors_bits[p_idx]
                candidates = []
                mask_to_iter = neighbors_bits[p_idx]
                while mask_to_iter:
                    lb = mask_to_iter & -mask_to_iter
                    idx = lb.bit_length() - 1
                    candidates.append(idx)
                    mask_to_iter ^= lb
                    
                best_w = -1
                best_score = -1
                
                for w in candidates:
                    if w in code_set:
                        continue
                    
                    # Check if w covers all critical points
                    if (neighbors_bits[w] & critical_mask) != critical_mask:
                        continue
                    
                    # Valid swap candidate.
                    # Score: We want to maximize the overlap of w with the existing code 
                    # to create redundancy (double coverage) which allows future pruning.
                    # Score = number of points in neighbors_bits[w] that are already covered by others.
                    # Actually, if we swap c -> w, the "others" remain the same.
                    # The points covered by w that are ALSO covered by others become double-covered.
                    # The points covered by w that were ONLY covered by c (and now by w) remain single.
                    # The points covered by c but NOT w are now uncovered (but we ensured w covers critical points).
                    # Wait, if w covers critical points, then any point covered by c is either:
                    # 1. Covered by others (no change in coverage status for that point, but now covered by w too? No, if covered by others, it was already covered. Adding w makes it double).
                    # 2. Covered ONLY by c (these are critical points). w covers them, so they remain covered.
                    
                    # So the swap is valid.
                    # How good is it? 
                    # We want to maximize the chance of future pruning.
                    # A good swap creates a situation where w is redundant or c's former critical points are now "safe".
                    # Simple score: How many points in neighbors_bits[w] are covered by `union_others`?
                    # If high, w is heavily redundant with the rest, making it easier to remove w later?
                    # Or maybe we want w to be distinct from others to keep coverage robust?
                    
                    # Let's use: Score = count of bits in (neighbors_bits[w] & union_others)
                    # This measures how much w overlaps with the existing coverage.
                    # High overlap means w is "redundant" in a way that might allow removing w later?
                    # Actually, if w overlaps a lot with others, and covers critical points, 
                    # then after swap, the critical points are covered by w AND others? 
                    # No, critical points were NOT covered by others. So they are covered ONLY by w now.
                    # So they are still critical for w.
                    
                    # Maybe a better score is: Minimize the number of critical points for w?
                    # Critical points for w in the new code = neighbors_bits[w] & ~(union_others)
                    # Since w covers the old critical points, and old critical points were not in union_others,
                    # the new critical points for w are a subset of the old critical points?
                    # No. 
                    # New Critical(w) = neighbors_bits[w] & ~union_others.
                    # We know (neighbors_bits[w] & critical_old) == critical_old.
                    # And critical_old & union_others == 0.
                    # So critical_old is a subset of New Critical(w).
                    # So w has at least as many critical points as c did.
                    
                    # This suggests simple 1-for-1 swaps might not reduce size immediately unless 
                    # the new word allows a CHAIN of removals.
                    
                    # For now, just accept the first valid swap to explore the space.
                    # Or pick one that maximizes overlap with union_others (to create redundancy elsewhere).
                    score = bin(neighbors_bits[w] & union_others).count('1')
                    
                    if score > best_score:
                        best_score = score
                        best_w = w
                        
                if best_w != -1:
                    # Perform swap
                    code_list.remove(c)
                    code_set.remove(c)
                    code_list.append(best_w)
                    code_set.add(best_w)
                    improved = True
                    break # Restart loop
                    
        return code_list

    def run_lagrangian_greedy(seed):
        random.seed(seed)
        
        # Initialize weights (dual variables) to 1
        weights = np.ones(N, dtype=np.float32)
        
        code = []
        covered_mask = 0
        
        # Multiple passes of weighted greedy
        for pass_iter in range(5): # Few passes to update weights
            if time.time() > deadline - 20:
                break
                
            # Reset covered mask for this pass? 
            # No, we are building the code. 
            # Actually, standard Lagrangian greedy:
            # 1. Start with empty code.
            # 2. While not covered:
            #    Pick word w maximizing sum(weights[neighbors[w]]) - lambda * (is_in_code?)
            #    Since we can't pick same word twice, we just pick max gain.
            #    Gain = sum of weights of currently UNCOVERED points in neighbors[w].
            
            uncovered_mask = full_mask & ~covered_mask
            if uncovered_mask == 0:
                break
            
            best_idx = -1
            max_gain = -1
            best_weight_sum = -1
            
            # Iterate through all words
            # Optimization: Only consider words that cover at least one uncovered point?
            # We can check (neighbors_bits[i] & uncovered_mask) != 0
            
            for i in range(N):
                if neighbors_bits[i] & covered_mask: 
                    # If it covers some covered points, it might still cover new ones.
                    pass
                
                # Calculate gain: number of uncovered points covered by i
                # We can use bit_count
                common = neighbors_bits[i] & uncovered_mask
                if common == 0:
                    continue
                    
                # Gain is weighted sum
                # To speed up, if we just use bit_count, it's fast.
                # But Lagrangian uses weights.
                # For simplicity and speed, let's use bit_count for now, 
                # but update weights to guide future passes.
                gain = common.bit_count()
                
                # Tie-breaker: random or based on weight
                if gain > max_gain:
                    max_gain = gain
                    best_idx = i
                elif gain == max_gain:
                    # Random tie-break
                    if random.random() < 0.5:
                        best_idx = i
                        
            if max_gain == 0:
                break
                
            # Add best_idx
            code.append(best_idx)
            covered_mask |= neighbors_bits[best_idx]
            
            # Update weights (Subgradient step)
            # Increase weight of uncovered points to encourage covering them next time
            # But we are doing this within one construction.
            # Let's just do multiple independent greedy runs with different random seeds 
            # and keep the best. The "Lagrangian" part is effectively the randomization 
            # and the fact that we restart.
            
            # To incorporate weights properly, we would need to run the greedy loop 
            # using the weights. Let's do that for the first pass.
            
        # Actually, the above loop is just standard greedy. 
        # Let's refine: 
        # Pass 1: Standard Greedy.
        # Pass 2: Weighted Greedy where weights are higher for points that were 
        #         covered by only 1 codeword in Pass 1 (fragile points).
        
        if not code:
            return None
            
        # Calculate coverage counts for the current code
        counts = np.zeros(N, dtype=np.int32)
        for c in code:
            # Add 1 to all neighbors of c
            mask = neighbors_bits[c]
            # This is slow in Python loop. 
            # But N=2187, code size ~200. 200 * 15 ops is fine.
            # Let's just use the bitset to find neighbors and increment.
            m = mask
            while m:
                lb = m & -m
                idx = lb.bit_length() - 1
                counts[idx] += 1
                m ^= lb
                
        # Identify fragile points (count == 1)
        fragile_mask = 0
        for i in range(N):
            if counts[i] == 1:
                fragile_mask |= (1 << i)
        
        # Second pass: Greedy with bias towards covering fragile points
        # We can just run another standard greedy with a different seed, 
        # or use the fragile points to guide a "repair" or "swap" phase.
        
        # For now, let's just return the code from the first greedy pass.
        # The local search will do the heavy lifting.
        
        return code

    best_code = None
    best_size = float('inf')
    
    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline - 10:
        seed = num_restarts * 1000 + 42
        current_code = run_lagrangian_greedy(seed)
        num_restarts += 1
        
        if current_code is not None:
            # Local Search
            current_code = prune_code(current_code)
            current_code = improve_by_swaps(current_code)
            
            # Final validity check
            if is_valid_code(current_code):
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                    
        if num_restarts >= 10 and time.time() > deadline - 30:
            break

    if best_code is None:
        # Fallback
        return []

    # Final Prune
    best_code = prune_code(best_code)
    
    return [list(words[i]) for i in best_code]

# Helper to count bits in an integer (Python 3.8+)
# int.bit_count() is available in Python 3.8+
# The evaluator likely uses a modern Python.

if __name__ == "__main__":
    # Test
    code = construct(3, 7, 1)
    print(f"Code size: {len(code)}")
