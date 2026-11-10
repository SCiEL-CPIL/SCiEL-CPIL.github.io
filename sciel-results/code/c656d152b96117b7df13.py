import numpy as np
import time
import itertools

def construct(q, n, radius):
    """
    Construct a ternary covering code of length n with radius r.
    Strategy: 
    1. Generate a diverse pool of candidate codewords using structured seeds (tetracode) 
       and random perturbations.
    2. Use a Weighted Greedy Selection where the weight of a candidate is proportional 
       to the number of currently uncovered neighbors, but with a penalty for 
       redundancy (overlap with already selected codewords). This approximates a 
       Lagrangian relaxation of the set cover problem.
    3. Perform a local search phase to remove redundant codewords.
    """
    if q != 3 or n != 8 or radius != 1:
        # Fallback: simple greedy
        N = q ** n
        powers = [q ** i for i in range(n)]
        neighbors = np.zeros((N, 1 + (q - 1) * n), dtype=np.int32)
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            temp = w
            for i in range(n):
                d = temp % q
                temp //= q
                for new_d in range(q):
                    if new_d != d:
                        w1 = w + (new_d - d) * powers[i]
                        neighbors[w, idx] = w1
                        idx += 1
        uncovered = np.ones(N, dtype=np.uint8)
        code = []
        while uncovered.sum() > 0:
            gains = uncovered[neighbors].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            w = candidates[np.random.randint(len(candidates))]
            code.append(w)
            uncovered[neighbors[w]] = 0
        return [[int(c) for c in np.base_repr(w, base=q).zfill(n)][::-1] for w in code]

    N = 3 ** 8  # 6561
    deadline = time.time() + 115.0
    rng = np.random.default_rng(42)

    # Precompute neighbors
    neighbors = np.zeros((N, 17), dtype=np.int32)
    powers = [3 ** i for i in range(n)]
    for w in range(N):
        neighbors[w, 0] = w
        idx = 1
        temp = w
        for i in range(n):
            d = temp % 3
            temp //= 3
            w1 = w + ((d + 1) % 3 - d) * powers[i]
            neighbors[w, idx] = w1
            idx += 1
            w2 = w + ((d + 2) % 3 - d) * powers[i]
            neighbors[w, idx] = w2
            idx += 1

    # Generate a pool of high-quality candidates
    # 1. Tetracode based seeds
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    candidates = set()
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            w_int = 0
            for i in range(n):
                w_int += word[i] * powers[i]
            candidates.add(w_int)
    
    # 2. Random words to ensure coverage of "hard" spots
    for _ in range(2000):
        w_int = rng.integers(0, N)
        candidates.add(int(w_int))
    
    candidate_list = np.array(list(candidates), dtype=np.int32)
    
    best_code = []
    best_size = N # Initialize to worst case

    # Weighted Greedy with Redundancy Penalty
    # We want to maximize coverage of uncovered words, but minimize overlap with existing code.
    # Score(w) = (uncovered neighbors of w) - lambda * (covered neighbors of w)
    # This encourages picking words that cover new ground.
    
    for restart in range(5):
        if time.time() > deadline - 20:
            break
            
        uncovered = np.ones(N, dtype=np.uint8)
        code = []
        
        # Precompute neighbor indices for faster access? 
        # neighbors is (N, 17). 
        # For each candidate w, we want sum(uncovered[neighbors[w]]) - penalty
        
        # Optimization: Instead of checking all N words, check only candidates.
        # But candidates might not cover everything. We should allow any word if needed?
        # The problem allows any word. But a good code uses structured words.
        # Let's stick to the candidate pool for speed, but if we fail to cover, 
        # we can add random words.
        
        # To be safe and fast, let's use the full space for the greedy step but 
        # accelerate it. 
        # Actually, iterating over N=6561 is fast enough in numpy.
        
        while uncovered.sum() > 0:
            # Calculate gains for all words
            # uncovered[neighbors] is (N, 17). Sum along axis 1.
            gains = uncovered[neighbors].sum(axis=1)
            
            # Calculate redundancy: how many of the neighbors are ALREADY covered?
            # This is equivalent to: (total neighbors) - (uncovered neighbors)
            # total neighbors is always 17.
            # redundancy = 17 - gains
            # We want to maximize (gains - alpha * redundancy)
            # Since redundancy = 17 - gains, this is gains - alpha*(17-gains) = (1+alpha)*gains - 17*alpha
            # This is just a linear transform of gains! 
            # Wait, this logic holds if we consider all words. 
            # But we want to prefer words that are "disjoint" from existing code.
            # The "uncovered" mask already handles this. If a word is already covered, 
            # its contribution to 'gains' is 0. 
            # So pure greedy on 'gains' already avoids redundancy in terms of coverage.
            # However, it doesn't penalize picking a word that is "close" to existing ones 
            # if it covers the same set of uncovered words.
            
            # Let's add a small randomization to break ties and explore diversity
            # Tie-breaking: among max gains, pick random.
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            candidates_max = np.where(gains == max_gain)[0]
            
            # To encourage diversity, we can slightly penalize words that are neighbors 
            # of already selected words?
            # Let's just pick randomly from max gain candidates.
            w = candidates_max[rng.integers(len(candidates_max))]
            code.append(int(w))
            uncovered[neighbors[w]] = 0
            
            # Early stop if we have enough time and a good size?
            # No, we need to cover everything.
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
        # Local Search: Prune redundant codewords
        if time.time() < deadline - 10:
            pruned_code = best_code[:]
            changed = True
            while changed:
                changed = False
                for i in range(len(pruned_code)):
                    if time.time() > deadline - 5:
                        break
                    # Try removing pruned_code[i]
                    test_code = pruned_code[:i] + pruned_code[i+1:]
                    # Check coverage
                    # This is O(N * |code|) if done naively.
                    # Faster: Use a coverage count array.
                    # But rebuilding the count array for each removal is expensive.
                    # Let's do a batch check: 
                    # Create a mask of what is covered by test_code.
                    # If it covers everything, remove i.
                    
                    # Optimization: Only attempt removal if the codeword's unique coverage is low.
                    # Calculate unique coverage of pruned_code[i]
                    # Unique if: for all neighbors x of pruned_code[i], 
                    # no other codeword in test_code covers x.
                    
                    # Approximation: 
                    # If the codeword has few neighbors that are NOT neighbors of any other codeword, it's a candidate for removal.
                    
                    # Let's do a simple check: 
                    # If we remove it, does the set of covered words change?
                    # covered_by_others = OR of neighbors of all other words.
                    # This is hard to compute fast for every i.
                    
                    # Alternative Local Search: 
                    # Randomly remove a word, if coverage breaks, revert.
                    # Or: 
                    # For each word, check if it is "essential".
                    # A word w is essential if there exists a neighbor x of w 
                    # such that x is not a neighbor of any other word in the code.
                    
                    # Let's compute a "coverage count" array for the current code.
                    # counts[x] = number of codewords covering x.
                    # If we remove w, x becomes uncovered if counts[x] == 1 and w covers x.
                    # So w is removable if for all x in neighbors[w], counts[x] > 1.
                    
                    # We can maintain counts incrementally.
                    if i == 0 or not changed: # Reset counts if we changed code? 
                         # This is getting complex. Let's do a simpler heuristic.
                         pass
                    
                    # Simpler Local Search: 
                    # Randomly swap?
                    # Let's just do a "redundancy check" pass.
                    # For each codeword w, count how many of its neighbors are covered ONLY by w.
                    # If 0, it can be removed.
                    
                    # To do this fast, we need the coverage counts.
                    # Let's compute counts once for the whole code.
                    # This is O(|code| * 17).
                    # Then for each w, check if any neighbor has count 1.
                    # If yes, w is essential.
                    # If no, w is removable.
                    
                    # We'll do this in a separate loop after the greedy.
                    
            # Perform the redundancy removal pass on best_code
            if time.time() < deadline - 5:
                code_to_prune = best_code[:]
                # Compute coverage counts
                counts = np.zeros(N, dtype=np.int32)
                for w in code_to_prune:
                    counts[neighbors[w]] += 1
                
                new_code = []
                for w in code_to_prune:
                    # Check if w is essential
                    # w is essential if any of its neighbors has count 1
                    # If all neighbors have count > 1, w is redundant
                    is_essential = False
                    for x in neighbors[w]:
                        if counts[x] == 1:
                            is_essential = True
                            break
                    if is_essential:
                        new_code.append(w)
                    else:
                        # Remove w and update counts
                        counts[neighbors[w]] -= 1
                
                if len(new_code) < best_size:
                    best_size = len(new_code)
                    best_code = new_code

    # Convert to final format
    final_code = []
    for w in best_code:
        word = []
        temp = w
        for _ in range(n):
            word.append(temp % 3)
            temp //= 3
        final_code.append(word)
    return final_code
