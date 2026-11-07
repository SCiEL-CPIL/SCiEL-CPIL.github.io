import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Precompute comparability
    # comp[i] is bitset of j such that i and j are comparable (strict subset in either direction)
    comp = [0] * total_masks
    for i in range(total_masks):
        ci = 0
        for j in range(total_masks):
            if i != j:
                if (i & j) == i or (i & j) == j:
                    ci |= (1 << j)
        comp[i] = ci

    def calc_score(sel_list, sel_mask):
        score = 0
        for s in sel_list:
            score += (comp[s] & sel_mask).bit_count()
        return score // 2

    def local_search(selected_list, selected_mask, current_score, deadline):
        """
        Perform hill climbing with single swaps.
        Returns (best_score, best_selection, current_score, selected_mask)
        """
        best_score = current_score
        best_selection = list(selected_list)
        indices = list(range(m))
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in indices:
                if time.time() >= deadline:
                    break
                old_mask = selected_list[idx]
                sel_without_old_mask = selected_mask & ~(1 << old_mask)
                loss = (comp[old_mask] & sel_without_old_mask).bit_count()
                
                best_new = -1
                best_delta = 0
                
                # Scan all possible new masks
                for cand in range(total_masks):
                    if (selected_mask >> cand) & 1:
                        continue
                    gain = (comp[cand] & sel_without_old_mask).bit_count()
                    delta = gain - loss
                    if delta > best_delta:
                        best_delta = delta
                        best_new = cand
                
                if best_new != -1 and best_delta > 0:
                    selected_list[idx] = best_new
                    selected_mask = sel_without_old_mask | (1 << best_new)
                    current_score += best_delta
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_selection = list(selected_list)
        
        return best_score, best_selection, current_score, selected_mask

    def simulated_annealing_polish(selected_list, selected_mask, current_score, deadline):
        """
        Final polish with simulated annealing to escape local optima.
        """
        best_score = current_score
        best_selection = list(selected_list)
        temp = 10.0
        cooling = 0.95
        
        while time.time() < deadline and temp > 0.1:
            idx = random.randint(0, m - 1)
            old_mask = selected_list[idx]
            sel_without_old_mask = selected_mask & ~(1 << old_mask)
            loss = (comp[old_mask] & sel_without_old_mask).bit_count()
            
            cand = random.randint(0, total_masks - 1)
            if (selected_mask >> cand) & 1:
                continue
                
            gain = (comp[cand] & sel_without_old_mask).bit_count()
            delta = gain - loss
            
            if delta > 0 or random.random() < math.exp(delta / temp):
                selected_list[idx] = cand
                selected_mask = sel_without_old_mask | (1 << cand)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_selection = list(selected_list)
            
            temp *= cooling
            
        return best_score, best_selection, current_score, selected_mask

    def greedy_initialization(seed):
        """
        Greedy initialization: pick 40 masks one by one, each time choosing
        the mask that maximizes comparability with already-picked masks.
        """
        rng = random.Random(seed)
        selected = []
        sel_mask = 0
        current_score = 0
        
        # Start with a few random or structural seeds to break symmetry
        # Try to start with empty set and full set for strong anchoring
        # But let's be more diverse: pick a random chain start
        
        # Strategy: 
        # 1. Pick a random permutation of bits
        # 2. Build a chain from empty to full
        # 3. Then greedily add remaining sets
        
        perm = list(range(n))
        rng.shuffle(perm)
        
        # Build initial chain
        chain = [0]
        mask = 0
        for bit in perm:
            mask |= (1 << bit)
            chain.append(mask)
        
        # We have 13 sets from the chain. We need 40 total.
        # Add them to selected
        for s in chain:
            if s not in selected:
                selected.append(s)
                sel_mask |= (1 << s)
        
        # Recalculate score for the chain
        current_score = calc_score(selected, sel_mask)
        
        # Now greedily add remaining 27 sets
        remaining = m - len(selected)
        for _ in range(remaining):
            best_cand = -1
            best_gain = -1
            
            # To speed up, we can sample candidates or scan all
            # Scanning all 4096 is feasible
            for cand in range(total_masks):
                if (sel_mask >> cand) & 1:
                    continue
                gain = (comp[cand] & sel_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_cand = cand
            
            if best_cand != -1:
                selected.append(best_cand)
                sel_mask |= (1 << best_cand)
                current_score += best_gain // 2  # Approximate, will be corrected in local search
        
        return selected, sel_mask, current_score

    best_overall_score = -1
    best_overall_selection = []
    
    # Multiple restarts with different greedy initializations
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        seed = 42 + restart * 100
        selected, sel_mask, current_score = greedy_initialization(seed)
        
        # Ensure exactly 40 distinct sets
        seen = set(selected)
        if len(seen) < m:
            for mask in range(total_masks):
                if mask not in seen:
                    selected.append(mask)
                    seen.add(mask)
                    sel_mask |= (1 << mask)
                    if len(selected) == m:
                        break
        elif len(seen) > m:
            selected = selected[:m]
            sel_mask = 0
            for s in selected:
                sel_mask |= (1 << s)
        
        # Recalculate score accurately
        current_score = calc_score(selected, sel_mask)
        
        # Local search
        ls_score, ls_selection, ls_current, ls_mask = local_search(selected, sel_mask, current_score, deadline - 2.0)
        
        if ls_score > best_overall_score:
            best_overall_score = ls_score
            best_overall_selection = ls_selection
        
        # If time remains, do simulated annealing polish on the best local solution
        if time.time() < deadline - 1.0 and ls_score == best_overall_score:
            sa_score, sa_selection, sa_current, sa_mask = simulated_annealing_polish(
                ls_selection, ls_mask, ls_current, deadline - 0.5
            )
            if sa_score > best_overall_score:
                best_overall_score = sa_score
                best_overall_selection = sa_selection

    # Final safety check: if we didn't find anything, fall back to a simple construction
    if not best_overall_selection:
        # Fallback: 40 random distinct masks
        best_overall_selection = random.sample(range(total_masks), m)

    # Convert to lists of integers
    result = []
    for mask in best_overall_selection:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
