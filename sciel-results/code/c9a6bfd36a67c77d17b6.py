import time
import random
import heapq
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We target 140s to be safe.
    deadline = start_time + 140.0

    total_space = q ** n

    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Precompute neighbor lists (words within distance 1)
    # For q=3, n=7, total_space=2187. Each word has 1 + n*(q-1) = 1 + 14 = 15 neighbors.
    # Storing as lists is efficient for iteration.
    neighbors = [[] for _ in range(total_space)]
    for i in range(total_space):
        neighbors[i].append(i)
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                neighbors[i].append(i + diff * powers[pos])
    
    # Convert to numpy arrays for faster bitwise operations? 
    # Actually, for n=7, q=3, the space is small enough that Python lists with set operations 
    # or bitmasks are fine. But let's use bitmasks for the "covered" state to make 
    # gain calculation fast if we stick to the bitmask approach, OR use numpy for counts.
    
    # Strategy: Lazy Greedy + Simulated Annealing
    # Representation: 
    # 1. Bitmask for covered words (fast union/intersection)
    # 2. Numpy array for counts (fast increment/decrement for SA)
    
    masks = [0] * total_space
    for i in range(total_space):
        m = 0
        for nb in neighbors[i]:
            m |= (1 << nb)
        masks[i] = m

    full_mask = (1 << total_space) - 1

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    def polish_remove(selected):
        """Remove redundant codewords. Returns new list."""
        if not selected:
            return selected
        
        # Use numpy for counts
        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            m = masks[w]
            # Extract bits
            mm = m
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit

        changed = True
        while changed:
            changed = False
            # Shuffle to avoid order bias
            indices = list(range(len(selected)))
            random.shuffle(indices)
            
            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                m = masks[w]
                
                # Check if all covered words have count > 1
                removable = True
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        removable = False
                        break
                    mm ^= lowbit

                if removable:
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    selected.pop(idx)
                    changed = True

        return selected

    def run_lazy_greedy():
        """
        Lazy Submodular Greedy Set Cover.
        Uses a max-heap. Each entry is (-gain, version, word).
        Gain is updated lazily.
        """
        # Initial gains: size of mask (all 15)
        # Since all masks have same size initially, we can just pick any.
        # But for generality:
        
        # To speed up, we can precompute the list of all words.
        all_words = list(range(total_space))
        
        # Heap items: (-gain, last_checked_version, word)
        # We need a way to know if a gain is stale.
        # A simpler "lazy" approach for small N:
        # Just recompute gain for candidates in the top K of the heap?
        # Or simply: standard greedy is O(N^2 * bitops). N=2187.
        # Lazy greedy reduces constant factors by not recomputing gains for words whose gain hasn't changed.
        
        # Let's implement a standard greedy first as baseline, then optimize if needed.
        # Actually, with N=2187, standard greedy taking max over 2187 items ~ 2187 * 2187 * (bitops) 
        # Bitops on 2187-bit ints are fast.
        # 2187^2 ~ 4.7M operations. Each op is a bit_count on a 2187-bit int.
        # This should run in < 1s.
        
        covered = 0
        selected = []
        remaining = full_mask
        
        while remaining and time.time() < deadline - 20:
            best_gain = -1
            best_word = -1
            
            # Optimization: Skip words that are already selected? 
            # No, we just check gain on remaining.
            
            for w in range(total_space):
                # Quick check: if mask[w] & remaining is 0, gain is 0.
                # But we need max.
                gain = (masks[w] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            
            if best_word == -1:
                break
                
            selected.append(best_word)
            covered |= masks[best_word]
            remaining = full_mask & ~covered
            
        return selected

    def simulated_annealing(initial_solution, time_limit):
        """
        Local search using Simulated Annealing.
        Moves: Add, Remove, Swap.
        Objective: Minimize |selected|.
        Constraint: Full coverage.
        
        State: List of selected words, and a coverage count array.
        """
        if not initial_solution:
            return initial_solution
        
        selected = initial_solution[:]
        # Build counts
        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            m = masks[w]
            mm = m
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit
        
        # Verify full coverage
        if np.any(counts == 0):
             # Fallback to greedy if initial is bad?
             # But initial from greedy should be full.
             pass

        current_size = len(selected)
        best_size = current_size
        best_selected = selected[:]
        
        T = 1.0 # Temperature
        T_min = 0.01
        cooling = 0.9995 # Slow cooling
        
        # Precompute list of all words for random selection
        all_words = list(range(total_space))
        
        while time.time() < time_limit and T > T_min:
            # Choose a move type
            # Probabilities: Remove (0.4), Add (0.2), Swap (0.4)
            r = random.random()
            
            # 1. Try Remove
            if r < 0.4 and len(selected) > 1:
                idx = random.randint(0, len(selected) - 1)
                w = selected[idx]
                m = masks[w]
                
                # Check if removal is valid (all counts > 1)
                removable = True
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        removable = False
                        break
                    mm ^= lowbit
                
                if removable:
                    # Execute
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    selected.pop(idx)
                    new_size = len(selected)
                    
                    # Cost change: -1 (improvement)
                    delta_cost = -1
                    # Accept (always accept improvements)
                    if new_size < best_size:
                        best_size = new_size
                        best_selected = selected[:]
                    # Update T?
                    T *= cooling
                    continue # Restart loop for next move? Or just continue.
                else:
                    # Cannot remove. Try to swap? Or just reject?
                    # In SA, we might accept bad moves. But removing a critical word breaks coverage.
                    # We should only consider valid moves or penalize heavily.
                    # Let's stick to valid moves for simplicity and speed, or use a penalty.
                    # If we use penalty, we need to track uncovered words.
                    # Let's stick to valid moves for Remove.
                    pass
            
            # 2. Try Add
            elif r < 0.6:
                w = random.choice(all_words)
                if w not in selected:
                    m = masks[w]
                    # Gain: number of 0s in counts covered by m
                    # This is expensive to compute exactly.
                    # Approximation: Just add it? 
                    # Adding increases size by 1. We only accept if it leads to a swap later?
                    # Or accept with probability exp(-1/T).
                    
                    # Execute Add
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] += 1
                        mm ^= lowbit
                    selected.append(w)
                    new_size = len(selected)
                    
                    delta_cost = 1
                    # Acceptance probability
                    if random.random() < np.exp(-delta_cost / T):
                        if new_size < best_size: # Won't happen for add
                            best_size = new_size
                            best_selected = selected[:]
                    else:
                        # Reject: Undo
                        mm = m
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit
                        selected.pop()
                    T *= cooling
                    continue
            
            # 3. Try Swap
            else:
                if len(selected) == 0:
                    continue
                idx = random.randint(0, len(selected) - 1)
                w_out = selected[idx]
                w_in = random.choice(all_words)
                
                if w_in == w_out or w_in in selected:
                    continue
                
                m_out = masks[w_out]
                m_in = masks[w_in]
                
                # Check validity of swap:
                # After removing w_out, adding w_in, are all counts >= 1?
                # New count for pos = counts[pos] - 1_{pos in m_out} + 1_{pos in m_in}
                # We need new_count >= 1 => counts[pos] + 1_{pos in m_in} - 1_{pos in m_out} >= 1
                
                valid = True
                # Iterate over union of masks? Or just check where it might fail.
                # Failure only if counts[pos] == 1 and pos in m_out and pos NOT in m_in.
                
                # Check m_out bits
                mm = m_out
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        # Check if pos is in m_in
                        if not (m_in & (1 << pos)):
                            valid = False
                            break
                    mm ^= lowbit
                
                if valid:
                    # Execute Swap
                    # Remove w_out
                    mm = m_out
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    # Add w_in
                    mm = m_in
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] += 1
                        mm ^= lowbit
                    
                    selected[idx] = w_in
                    # Size unchanged
                    T *= cooling
                    continue
                else:
                    # Invalid swap. 
                    # Could accept with penalty? 
                    # For now, just reject.
                    pass
            
            T *= cooling
            
        return best_selected

    best_size = total_space
    best_selected = list(range(total_space))

    # Phase 1: Initial Greedy
    if time.time() < deadline - 30:
        sol = run_lazy_greedy()
        if len(sol) < best_size:
            best_size = len(sol)
            best_selected = sol[:]
            # Quick polish
            polished = polish_remove(best_selected[:])
            if len(polished) < best_size:
                best_size = len(polished)
                best_selected = polished[:]

    # Phase 2: Simulated Annealing / Local Search
    # Run SA for remaining time
    if time.time() < deadline - 5:
        # Give SA most of the remaining time
        sa_limit = deadline - 2
        sa_result = simulated_annealing(best_selected[:], sa_limit)
        if len(sa_result) < best_size:
            best_size = len(sa_result)
            best_selected = sa_result[:]
        
        # Final Polish
        if time.time() < deadline - 1:
            polished = polish_remove(best_selected[:])
            if len(polished) < best_size:
                best_size = len(polished)
                best_selected = polished[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
