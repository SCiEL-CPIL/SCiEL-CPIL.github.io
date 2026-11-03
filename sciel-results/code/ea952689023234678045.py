import itertools
import random
import time

def construct(q, n, radius):
    # Specialized for K_3(7,1)
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters: Simple Greedy
        all_words = list(itertools.product(range(q), repeat=n))
        word_index = {w: i for i, w in enumerate(all_words)}
        N = len(all_words)
        balls = []
        for i, w in enumerate(all_words):
            neighbors = [i]
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    nw = list(w); nw[pos] = val
                    neighbors.append(word_index[tuple(nw)])
            balls.append(neighbors)
        
        uncovered = set(range(N))
        solution = []
        while uncovered:
            best = -1
            best_count = -1
            for i in range(N):
                if i in solution: continue
                count = sum(1 for nb in balls[i] if nb in uncovered)
                if count > best_count:
                    best_count = count
                    best = i
            if best == -1: break
            solution.append(best)
            for nb in balls[best]:
                uncovered.discard(nb)
        return [all_words[i] for i in solution]

    # Specific logic for q=3, n=7, radius=1
    N = 3**7  # 2187
    all_words = list(itertools.product(range(3), repeat=7))
    word_index = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls as bitsets (integers)
    # ball[i] is an integer where bit j is 1 if word j is within distance 1 of word i
    balls = [0] * N
    for i, w in enumerate(all_words):
        mask = 1 << i
        for pos in range(n):
            for val in range(3):
                if val == w[pos]: continue
                nw = list(w)
                nw[pos] = val
                j = word_index[tuple(nw)]
                mask |= (1 << j)
        balls[i] = mask
        
    # Total space mask
    full_mask = (1 << N) - 1
    
    # Helper: Local Search to minimize size
    # state: set of indices
    def local_search(state):
        state = set(state)
        # Precompute coverage count for each word in the current state
        # cover_count[w] = number of codewords in state that cover w
        cover_count = [0] * N
        for idx in state:
            mask = balls[idx]
            # Iterate over set bits in mask
            m = mask
            while m:
                lsb = m & -m
                pos = lsb.bit_length() - 1
                cover_count[pos] += 1
                m ^= lsb
                
        improved = True
        while improved:
            improved = False
            # 1. Try to remove redundant codewords
            for idx in list(state):
                mask = balls[idx]
                # Check if all neighbors are covered by others
                redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    pos = lsb.bit_length() - 1
                    if cover_count[pos] <= 1:
                        redundant = False
                        break
                    m ^= lsb
                if redundant:
                    state.remove(idx)
                    m = mask
                    while m:
                        lsb = m & -m
                        pos = lsb.bit_length() - 1
                        cover_count[pos] -= 1
                        m ^= lsb
                    improved = True
                    break # Restart loop as counts changed
                    
            if improved:
                continue
                
            # 2. Try Swap: Replace one codeword with another to enable removal or reduce size?
            # Actually, a swap that keeps size same but improves "potential" for removal is complex.
            # A simpler swap: Replace A with B. If B covers some words A didn't, maybe we can remove others?
            # Let's try: For each idx in state, for each candidate c not in state:
            #   Trial = state - {idx} + {c}
            #   If len(Trial) < len(state) is impossible since we swap 1 for 1.
            #   But maybe Trial allows removing more?
            #   This is expensive. 
            #   Alternative: Try to replace idx with c if c covers MORE uncovered words? 
            #   But we are always fully covered.
            #   
            #   Let's stick to Removal first. If no removal, try "Perturbation" externally.
            #   However, a pure removal local search can get stuck in local minima where 
            #   no single removal works, but a swap does.
            
            # Efficient Swap Check:
            # For each idx in state:
            #   Let U_idx be the set of words covered ONLY by idx (cover_count[w] == 1 for w in ball[idx]).
            #   If we remove idx, these words become uncovered.
            #   We need a candidate c not in state such that ball[c] covers ALL of U_idx.
            #   If such c exists, we can swap idx -> c. Size remains same.
            #   But does this help? Only if it enables further removals.
            #   This is a "move" that maintains validity.
            #   We can perform such swaps to explore the space of valid solutions of same size,
            #   hoping to find a configuration where a removal becomes possible.
            
            # To keep it fast, we limit swap attempts.
            for idx in list(state):
                # Find words uniquely covered by idx
                unique_mask = 0
                m = balls[idx]
                while m:
                    lsb = m & -m
                    pos = lsb.bit_length() - 1
                    if cover_count[pos] == 1:
                        unique_mask |= lsb
                    m ^= lsb
                
                if unique_mask == 0:
                    # Should have been caught by removal logic? 
                    # cover_count[w] <= 1 check in removal catches this.
                    # Wait, removal checks cover_count <= 1. 
                    # If cover_count == 1, it's NOT redundant (removing it would uncover w).
                    # So unique_mask contains words with count 1.
                    # If unique_mask is 0, it means all words in ball[idx] have count >= 2.
                    # Then idx is redundant. Why wasn't it removed?
                    # Ah, the removal loop breaks on first improvement. 
                    # If we are here, it means no idx was redundant.
                    # So unique_mask must be non-zero for all idx?
                    # No, it's possible that for every idx, there is at least one word with count 1.
                    
                    # So we look for a swap.
                    # We need c such that (balls[c] & unique_mask) == unique_mask.
                    # This means c covers all uniquely covered words.
                    
                    # Optimization: Precompute for each word w, the list of codewords covering it?
                    # Or just iterate candidates?
                    # N=2187. State size ~186.
                    # For each idx (186), check candidates (2000). 372k checks.
                    # Each check is bit ops. This is feasible per iteration.
                    
                    found_swap = False
                    # Iterate through all possible candidates not in state
                    # To speed up, we can just iterate all 0..N-1 and check if in state
                    # Or maintain a set of state for O(1) lookup.
                    state_set = state
                    for c in range(N):
                        if c in state_set:
                            continue
                        if (balls[c] & unique_mask) == unique_mask:
                            # Perform swap
                            state.remove(idx)
                            state.add(c)
                            
                            # Update cover counts
                            # Decrement for idx
                            m = balls[idx]
                            while m:
                                lsb = m & -m
                                pos = lsb.bit_length() - 1
                                cover_count[pos] -= 1
                                m ^= lsb
                            # Increment for c
                            m = balls[c]
                            while m:
                                lsb = m & -m
                                pos = lsb.bit_length() - 1
                                cover_count[pos] += 1
                                m ^= lsb
                            
                            improved = True
                            found_swap = True
                            break
                            
                    if found_swap:
                        break # Restart local search loop
            
        return state

    # Initial Solution: Greedy
    def greedy_init():
        uncovered = full_mask
        sol = set()
        while uncovered:
            best = -1
            best_count = -1
            for i in range(N):
                if i in sol: continue
                count = (balls[i] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best = i
            if best == -1: break
            sol.add(best)
            uncovered &= ~balls[best]
        return local_search(sol)

    start_time = time.time()
    max_time = 115.0
    
    # Initial best
    best_sol = greedy_init()
    best_size = len(best_sol)
    
    # Iterated Local Search
    current_sol = best_sol.copy()
    
    while time.time() - start_time < max_time:
        # Perturb current solution
        # Randomly swap a few elements
        perturb_sol = current_sol.copy()
        num_swaps = random.randint(5, 15)
        for _ in range(num_swaps):
            if not perturb_sol: break
            # Pick a random element from state to remove
            idx_to_remove = random.choice(list(perturb_sol))
            perturb_sol.remove(idx_to_remove)
            # Pick a random element not in state to add
            # To ensure validity isn't broken too badly, we'll rely on local_search to fix it?
            # No, local_search assumes valid input (or at least tries to remove redundant).
            # If we add random, it might be invalid (uncovered words).
            # My local_search above only removes redundant and swaps. It doesn't ADD to cover holes.
            # So I need a "repair" step or a perturbation that keeps validity?
            
            # Better Perturbation:
            # 1. Remove a random codeword.
            # 2. Find the word(s) it uniquely covered.
            # 3. Add a codeword that covers them (if exists, else add random and hope repair works? No, repair is expensive).
            
            # Let's change strategy:
            # Perturbation: 
            # 1. Pick k random codewords from state.
            # 2. Remove them.
            # 3. Find a set of codewords to cover the newly uncovered words.
            # 4. Add them.
            # 5. Run Local Search.
            
            # This is complex to implement efficiently inline.
            # Let's use a simpler perturbation that maintains validity:
            # "Swap" perturbation:
            # Pick random idx in state.
            # Pick random c not in state.
            # Swap them.
            # This maintains size. It might break validity if c doesn't cover what idx covered.
            # But if we just do random swaps, we might end up with an invalid set.
            # My local_search doesn't repair invalid sets.
            
            # I need a `repair` function for ILS.
            pass
        
        # Implement Repair
        def repair(sol_set):
            sol_set = set(sol_set)
            # Calculate uncovered
            covered = 0
            for idx in sol_set:
                covered |= balls[idx]
            uncovered = full_mask & ~covered
            
            while uncovered:
                # Greedy add
                best = -1
                best_count = -1
                for i in range(N):
                    if i in sol_set: continue
                    count = (balls[i] & uncovered).bit_count()
                    if count > best_count:
                        best_count = count
                        best = i
                if best == -1: break
                sol_set.add(best)
                uncovered &= ~balls[best]
            return sol_set

        # Redo Perturbation with Repair
        perturb_sol = current_sol.copy()
        num_moves = random.randint(5, 15)
        for _ in range(num_moves):
            if not perturb_sol: break
            # Random Swap
            idx_to_remove = random.choice(list(perturb_sol))
            perturb_sol.remove(idx_to_remove)
            # Pick random candidate
            c = random.randint(0, N-1)
            while c in perturb_sol:
                c = random.randint(0, N-1)
            perturb_sol.add(c)
        
        # Repair if invalid
        # Check validity quickly
        covered = 0
        for idx in perturb_sol:
            covered |= balls[idx]
        if covered != full_mask:
            perturb_sol = repair(perturb_sol)
            
        # Local Search
        new_sol = local_search(perturb_sol)
        
        if len(new_sol) < best_size:
            best_size = len(new_sol)
            best_sol = new_sol
            current_sol = new_sol.copy()
        else:
            # Even if not better, accept if it's different to escape plateau?
            # ILS usually accepts non-worsening or with probability.
            # Let's accept if it's same size but different (to explore), or slightly worse?
            # To save time, only keep if it leads to better later.
            # Standard ILS: Accept if better. If not, maybe accept with prob p to diversify.
            if random.random() < 0.1: # 10% chance to accept worse/same to diversify
                current_sol = new_sol.copy()
            else:
                current_sol = best_sol.copy() # Revert to best to stay in basin

    return [all_words[i] for i in best_sol]
