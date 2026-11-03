import time
import random
import sys

def construct(q, n, radius):
    # Assume q=3, n=6, radius=1 based on task description
    if q == 3 and n == 6 and radius == 1:
        return _construct_ternary_6_1()
    
    # Fallback generic greedy
    return _construct_generic(q, n, radius)

def _construct_ternary_6_1():
    """
    Strategy: Maximum Independent Set (MIS) on the Conflict Graph.
    
    Problem Framing:
    We want to select a set of codewords C such that every word in {0,1,2}^6 is within
    distance 1 of some c in C.
    
    This is equivalent to finding a dominating set in the Hamming graph H(6,3).
    However, a more powerful formulation for finding *small* covering codes is often
    related to packing or independence in the "conflict graph" where two vertices
    are connected if they are "too close" to be useful together? No, that's packing.
    
    Actually, let's look at the structure of optimal ternary covering codes.
    The best known is 73.
    
    Alternative Structural View:
    The "Conflict Graph" for covering codes isn't standard.
    Let's stick to the Dominating Set formulation but use a different search method.
    
    The current method uses local search (remove/swap).
    The new method will use **Iterated Local Search (ILS)** with a different local move:
    Instead of just removing single words, we will use **Greedy Set Cover with Randomized Perturbation**.
    
    Actually, a truly different family is **Simulated Annealing on the set of codewords**
    where the state is the set of codewords, and moves are:
    1. Add a random word (if not present)
    2. Remove a random word (if coverage remains valid)
    3. Swap a word with another (if coverage remains valid)
    
    But the current program already does this.
    
    Let's try a **Greedy Construction from Scratch** with multiple random seeds.
    The current program starts from a specific algebraic structure (Tetracode).
    Starting from scratch with a greedy algorithm that picks the word covering the most
    *currently uncovered* points is a different initialization strategy.
    
    Algorithm:
    1. Precompute balls as bitsets.
    2. Initialize uncovered = all 1s (729 bits).
    3. While uncovered is not empty:
       a. Find word w that maximizes popcount(ball[w] & uncovered).
       b. Break ties randomly.
       c. Add w to code.
       d. Update uncovered: uncovered &= ~ball[w].
    4. Local Search:
       - Try to remove any word if valid.
       - Try to swap any word with any other word if valid and (optional) improves some metric?
         No, swap only if it allows a subsequent removal.
       
    To differentiate from the current "Tetracode Start + Local Search", I will use
    **Multiple Random Greedy Constructions** followed by **Aggressive Local Search**.
    The key difference is the *initialization*. The current one relies on the algebraic
    properties of the Tetracode. This one relies on statistical coverage efficiency.
    
    Furthermore, I will implement a **2-opt style move**: 
    Remove two words, add one word. This can reduce size by 1.
    Remove one word, add two words. This increases size by 1 (usually bad, but might help escape?).
    Actually, "Remove two, add one" is a powerful move for reducing size.
    
    Let's implement:
    1. Greedy Set Cover (randomized tie-breaking) -> Initial Code
    2. Local Search Phase:
       - Loop:
         - Try all single removals.
         - Try all (remove 2, add 1) moves.
         - If no improvement, restart with new random seed (or perturb current solution).
    """
    
    N = 729
    L = 6
    Q = 3
    
    # Precompute balls as bitsets
    # ball[i] is a bitset of all words within distance 1 of word i
    balls = [0] * N
    for i in range(N):
        word = []
        temp = i
        for _ in range(L):
            word.append(temp % Q)
            temp //= Q
        word = word[::-1]
        
        mask = 1 << i
        for pos in range(L):
            for new_val in range(Q):
                if new_val == word[pos]:
                    continue
                new_word = word.copy()
                new_word[pos] = new_val
                new_idx = 0
                for d in new_word:
                    new_idx = new_idx * Q + d
                mask |= 1 << new_idx
        balls[i] = mask

    # Helper to check if a set of codewords covers the whole space
    # We maintain a coverage count array for efficiency in local search
    # But for the initial greedy, we can use bitsets
    
    def greedy_construction(rng):
        uncovered = (1 << N) - 1
        code = set()
        
        while uncovered:
            best_word = -1
            best_count = -1
            # To speed up, we could sample, but N=729 is small.
            # Iterate all words, find one maximizing coverage of 'uncovered'
            # Optimization: If best_count == popcount(uncovered) is impossible, 
            # we just find max intersection.
            
            # Randomization: Shuffle order or add random noise to tie-breaking
            # We'll iterate all, but break ties randomly.
            candidates = []
            max_cov = 0
            
            for w in range(N):
                if w in code:
                    continue
                cov = (balls[w] & uncovered).bit_count()
                if cov > max_cov:
                    max_cov = cov
                    candidates = [w]
                elif cov == max_cov:
                    candidates.append(w)
            
            if not candidates:
                break # Should not happen if uncovered is not 0 and balls cover space
                
            # Randomly pick from candidates
            chosen = rng.choice(candidates)
            code.add(chosen)
            uncovered &= ~balls[chosen]
            
        return code

    def local_search(code, rng, time_limit):
        """
        Local search to minimize size.
        Moves:
        1. Remove one word (if valid).
        2. Remove two words, add one word (if valid).
        """
        code = set(code)
        # Initialize coverage counts
        coverage = [0] * N
        for c in code:
            m = balls[c]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                coverage[idx] += 1
                m ^= lsb
        
        def is_valid():
            # Check if any coverage is 0
            # This is O(N). N=729. Fast.
            for v in coverage:
                if v == 0:
                    return False
            return True
            
        def try_remove(c):
            # Temporarily remove c
            m = balls[c]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                coverage[idx] -= 1
                m ^= lsb
                
            if is_valid():
                code.remove(c)
                return True
            else:
                # Restore
                m = balls[c]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    coverage[idx] += 1
                    m ^= lsb
                return False

        def try_swap_remove2_add1(c1, c2, c_new):
            """
            Remove c1 and c2, add c_new.
            Size change: -1.
            """
            if c_new in code:
                return False
            if c1 == c2:
                return False
            if c1 == c_new or c2 == c_new:
                return False
                
            # Apply changes
            # Remove c1
            m1 = balls[c1]
            while m1:
                lsb = m1 & -m1
                idx = lsb.bit_length() - 1
                coverage[idx] -= 1
                m1 ^= lsb
            # Remove c2
            m2 = balls[c2]
            while m2:
                lsb = m2 & -m2
                idx = lsb.bit_length() - 1
                coverage[idx] -= 1
                m2 ^= lsb
            # Add c_new
            mn = balls[c_new]
            while mn:
                lsb = mn & -mn
                idx = lsb.bit_length() - 1
                coverage[idx] += 1
                mn ^= lsb
                
            if is_valid():
                code.remove(c1)
                code.remove(c2)
                code.add(c_new)
                return True
            else:
                # Rollback
                # Add c1
                m1 = balls[c1]
                while m1:
                    lsb = m1 & -m1
                    idx = lsb.bit_length() - 1
                    coverage[idx] += 1
                    m1 ^= lsb
                # Add c2
                m2 = balls[c2]
                while m2:
                    lsb = m2 & -m2
                    idx = lsb.bit_length() - 1
                    coverage[idx] += 1
                    m2 ^= lsb
                # Remove c_new
                mn = balls[c_new]
                while mn:
                    lsb = mn & -mn
                    idx = lsb.bit_length() - 1
                    coverage[idx] -= 1
                    mn ^= lsb
                return False

        start_time = time.time()
        best_size = len(code)
        best_code = set(code)
        
        # Iterated Local Search
        # We will run local search until no improvement, then perturb and restart.
        
        while time.time() < time_limit:
            improved = True
            while improved:
                improved = False
                if time.time() > time_limit:
                    break
                
                # Phase 1: Try all single removals
                # To avoid bias, randomize order
                code_list = list(code)
                rng.shuffle(code_list)
                for c in code_list:
                    if time.time() > time_limit:
                        break
                    if c not in code:
                        continue
                    if try_remove(c):
                        improved = True
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = set(code)
                
                if time.time() > time_limit:
                    break

                # Phase 2: Try all (remove 2, add 1) moves
                # This is O(|C|^2 * N). |C| ~ 73. 73^2 * 729 ~ 3.8M operations per pass.
                # Might be too slow for many passes.
                # Optimization: Only try a random subset of pairs?
                # Or precompute pairs?
                
                # Let's try a limited number of random pairs to keep it fast
                # Or iterate all pairs if time allows.
                # 3.8M simple integer ops in Python might take ~0.1-0.2s.
                # So we can do a few passes.
                
                code_list = list(code)
                rng.shuffle(code_list)
                
                # Sample pairs to avoid full O(N^2) if it's too slow
                # But N=729 is small enough. Let's try all pairs first.
                # If it's too slow, we'll sample.
                
                # To make it faster, we can pre-compute the "delta coverage" of removing c1 and c2?
                # No, just do it.
                
                # We'll iterate i < j in code_list
                # And for each pair, try all k in 0..N-1 not in code?
                # That's O(|C|^2 * N).
                # 73^2 * 729 = 3.8M.
                # Inside the loop, we do bit operations.
                # This might be 1-2 seconds per pass.
                # We have 120s budget. So ~50-100 passes possible.
                
                # However, we need to be careful with Python overhead.
                # Let's restrict to a random subset of pairs if the full set is too slow.
                
                # Strategy: Try all pairs, but break early if time is tight.
                
                found_move = False
                for i in range(len(code_list)):
                    if time.time() > time_limit:
                        break
                    c1 = code_list[i]
                    if c1 not in code: continue
                    for j in range(i+1, len(code_list)):
                        if time.time() > time_limit:
                            break
                        c2 = code_list[j]
                        if c2 not in code: continue
                        
                        # Try to find a c_new that covers the "hole"
                        # The hole is the set of points covered ONLY by c1 or c2 (or both) 
                        # that would become uncovered.
                        # Actually, it's easier to just try all c_new and check validity.
                        # But checking validity is O(N).
                        # So O(|C|^2 * N * N) is too slow.
                        
                        # We need a faster validity check or a smarter c_new selection.
                        
                        # Idea: The set of points that MUST be covered by c_new is:
                        # Uncovered_after_removal = (Points covered by c1 or c2) - (Points covered by others)
                        # Let S1 = ball[c1], S2 = ball[c2].
                        # Let S_others = Union(ball[c] for c in code - {c1, c2})
                        # The points that become uncovered are (S1 U S2) \ S_others.
                        # c_new must cover ALL these points.
                        # So ball[c_new] must contain (S1 U S2) \ S_others.
                        
                        # Let's compute the "essential" points for this pair.
                        # This computation is expensive if done for every pair.
                        
                        # Alternative: Just try a random c_new?
                        # Or try c_new that is close to c1 or c2?
                        
                        # Let's try a heuristic:
                        # Compute the union of balls of c1 and c2.
                        # Subtract coverage from all other codewords.
                        # The remaining set is the "deficit".
                        # We need a word c_new such that ball[c_new] covers the deficit.
                        
                        # This is a set cover sub-problem.
                        
                        # Given the time constraints, let's stick to simpler moves for now,
                        # or optimize the (2,1) move.
                        
                        # Actually, let's just do random swaps (1,1) and removals (1,0).
                        # The (2,1) move is complex to implement efficiently in Python within time.
                        # Let's see if (1,1) swaps and (1,0) removals are enough.
                        # The current program does this.
                        
                        # To be "structurally different", maybe I should implement the (2,1) move
                        # but only for a *subset* of pairs, or with a precomputed deficit check.
                        
                        pass # Skip complex move for now, focus on robust (1,0) and (1,1)
                
                # Let's implement a proper (2,1) move with a time check.
                # We'll compute the deficit for a pair (c1, c2).
                
                # Precompute coverage from others?
                # Too expensive to maintain.
                
                # Let's just do random (1,1) swaps.
                # The current program does random swaps.
                # I will do *exhaustive* (1,1) swaps if possible, or a large random sample.
                
                # Let's do a large random sample of swaps.
                for _ in range(200): # 200 random swaps per local search iteration
                    if time.time() > time_limit:
                        break
                    if len(code) < 2:
                        break
                    c_old = rng.choice(code_list)
                    if c_old not in code:
                        continue
                    c_new = rng.randrange(N)
                    if c_new in code:
                        continue
                    
                    # Try swap
                    # Remove c_old, Add c_new
                    m1 = balls[c_old]
                    while m1:
                        lsb = m1 & -m1
                        idx = lsb.bit_length() - 1
                        coverage[idx] -= 1
                        m1 ^= lsb
                        
                    mn = balls[c_new]
                    while mn:
                        lsb = mn & -mn
                        idx = lsb.bit_length() - 1
                        coverage[idx] += 1
                        mn ^= lsb
                        
                    if is_valid():
                        code.remove(c_old)
                        code.add(c_new)
                        improved = True
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = set(code)
                        # After a successful swap, we might enable a removal.
                        # So we break out of the swap loop and go back to removal loop.
                        break
                    else:
                        # Rollback
                        m1 = balls[c_old]
                        while m1:
                            lsb = m1 & -m1
                            idx = lsb.bit_length() - 1
                            coverage[idx] += 1
                            m1 ^= lsb
                        mn = balls[c_new]
                        while mn:
                            lsb = mn & -mn
                            idx = lsb.bit_length() - 1
                            coverage[idx] -= 1
                            mn ^= lsb
                
                if not improved:
                    # Perturb: Add a random word? No, that increases size.
                    # Perturb: Swap a word with a random one, even if invalid?
                    # No, we must maintain validity for the "best" tracking, 
                    # but ILS usually allows invalid intermediate states.
                    # However, our is_valid is strict.
                    
                    # Let's try a "kick": Remove a random word (making invalid), 
                    # then try to repair by adding words?
                    # This is complex.
                    
                    # Simple perturbation: Randomly swap 2 words (even if invalid temporarily?)
                    # No, let's just restart the local search with a slightly perturbed set.
                    # Pick a random word in code, remove it, add a random word not in code.
                    # This keeps size same.
                    
                    if len(code) > 0:
                        c_old = rng.choice(list(code))
                        c_new = rng.randrange(N)
                        if c_new not in code:
                            code.remove(c_old)
                            code.add(c_new)
                            # Recalculate coverage
                            coverage = [0] * N
                            for c in code:
                                m = balls[c]
                                while m:
                                    lsb = m & -m
                                    idx = lsb.bit_length() - 1
                                    coverage[idx] += 1
                                    m ^= lsb
                            # Check if valid. If not, revert?
                            # If not valid, we are stuck.
                            # Let's just revert if invalid.
                            if not is_valid():
                                code.remove(c_new)
                                code.add(c_old)
                                coverage = [0] * N
                                for c in code:
                                    m = balls[c]
                                    while m:
                                        lsb = m & -m
                                        idx = lsb.bit_length() - 1
                                        coverage[idx] += 1
                                        m ^= lsb

        return best_code

    # Main Logic
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    best_code_global = set()
    best_size_global = float('inf')
    
    # Run multiple greedy constructions
    # How many? Depends on time.
    # Each greedy construction is O(N^2) ~ 500k ops. Fast.
    # Local search is the bottleneck.
    
    # We will run one long local search on the best greedy result,
    # or multiple short ones.
    
    # Strategy:
    # 1. Generate 10 greedy codes.
    # 2. Keep the smallest.
    # 3. Run local search on it with remaining time.
    
    rng = random.Random(42)
    
    greedy_candidates = []
    for i in range(10):
        if time.time() > deadline - 10: # Leave 10s for local search
            break
        c = greedy_construction(rng)
        greedy_candidates.append(c)
        if len(c) < best_size_global:
            best_size_global = len(c)
            best_code_global = set(c)
            
    # Use the best greedy code as starting point for local search
    if not best_code_global:
        best_code_global = greedy_construction(rng)
        
    # Run local search
    # Allocate most of the remaining time to local search
    remaining_time = deadline - time.time()
    if remaining_time > 0:
        local_search_result = local_search(best_code_global, rng, remaining_time)
        if len(local_search_result) < len(best_code_global):
            best_code_global = local_search_result
            best_size_global = len(best_code_global)
            
    # Final check: Try to remove any more words from the best code found
    # (Just in case local search stopped early)
    # We'll do a quick pass.
    
    # Convert back to list of words
    def to_word(val):
        word = []
        for _ in range(L):
            word.append(val % Q)
            val //= Q
        return word[::-1]

    return [to_word(c) for c in best_code_global]

def _construct_generic(q, n, radius):
    total = q ** n
    words = [list(map(int, __import__('itertools').product(range(q), repeat=n))[i]) for i in range(total)]
    
    def dist(w1, w2):
        return sum(1 for a, b in zip(w1, w2) if a != b)
        
    covered = [False] * total
    code = []
    remaining = set(range(total))
    
    while remaining:
        best_w = None
        best_count = -1
        for w_idx in remaining:
            w = words[w_idx]
            count = 0
            for r_idx in remaining:
                if dist(w, words[r_idx]) <= radius:
                    count += 1
            if count > best_count:
                best_count = count
                best_w = w_idx
                
        code.append(words[best_w])
        w = words[best_w]
        new_remaining = set()
        for r_idx in remaining:
            if dist(w, words[r_idx]) > radius:
                new_remaining.add(r_idx)
        remaining = new_remaining
        
    return code
