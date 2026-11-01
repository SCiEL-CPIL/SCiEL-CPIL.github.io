import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Returns a list of codewords (each a list of ints in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (150s limit, leave buffer)

    if n == 0:
        return [[]]
    
    # Handle small cases or specific known perfect codes
    if q == 3 and radius == 1 and n == 4:
        return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for all words as bitmasks (integers) for fast coverage operations
    # neighbors_mask[i] is a bitmask where bit j is set if word j is within radius of word i
    neighbors_mask = [0] * total_words
    for i, w in enumerate(all_words):
        mask = 0
        mask |= (1 << i)  # Include self
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_w = list(w)
                new_w[pos] = val
                j = word_to_idx[tuple(new_w)]
                mask |= (1 << j)
        neighbors_mask[i] = mask
    
    # Also store neighbor lists for iterating when needed
    neighbors_list = []
    for i in range(total_words):
        nb = [i]
        w = all_words[i]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_w = list(w)
                new_w[pos] = val
                nb.append(word_to_idx[tuple(new_w)])
        neighbors_list.append(nb)
    
    # Precompute for each word, the set of codewords that cover it (reverse neighbors)
    # reverse_neighbors[j] = bitmask of codewords i such that j is in neighbors_mask[i]
    reverse_neighbors = [0] * total_words
    for i in range(total_words):
        mask = neighbors_mask[i]
        while mask:
            lowbit = mask & (-mask)
            j = lowbit.bit_length() - 1
            reverse_neighbors[j] |= (1 << i)
            mask ^= lowbit
    
    # Greedy Initialization using bitmasks
    def greedy_init(seed=0, time_budget=None):
        if time_budget is None:
            time_budget = deadline - time.time()
        rng = random.Random(seed)
        uncovered_mask = (1 << total_words) - 1  # All words uncovered
        code_indices = []
        code_set_mask = 0
        
        while uncovered_mask and time.time() < deadline - 5:
            best_idx = -1
            best_count = -1
            for i in range(total_words):
                if (code_set_mask >> i) & 1:
                    continue
                # Count how many uncovered words this codeword covers
                cnt = bin(neighbors_mask[i] & uncovered_mask).count('1')
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            
            if best_idx == -1 or best_count == 0:
                break
                
            code_indices.append(best_idx)
            code_set_mask |= (1 << best_idx)
            uncovered_mask &= ~neighbors_mask[best_idx]
                
        return code_indices

    # Direct Sum Initialization (for q=3, n=6, radius=1)
    def direct_sum_init():
        if q == 3 and n == 6 and radius == 1:
            tetracode = [(a, b, (a + b) % 3, (a + 2 * b) % 3) for a in range(3) for b in range(3)]
            code = []
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=2):
                    c = h + tail
                    code.append(word_to_idx[c])
            return code
        return greedy_init(0)
    
    # Check if a set of codewords (as bitmask) covers all words
    def is_covering(code_mask):
        # A word j is covered if there exists a codeword i in code_mask such that j is in neighbors_mask[i]
        # Equivalently, word j is uncovered if for all i in code_mask, j is NOT in neighbors_mask[i]
        # So j is covered if reverse_neighbors[j] & code_mask != 0
        # All covered means: for all j, reverse_neighbors[j] & code_mask != 0
        # Uncovered words: those j where reverse_neighbors[j] & code_mask == 0
        uncovered = 0
        for j in range(total_words):
            if (reverse_neighbors[j] & code_mask) == 0:
                uncovered |= (1 << j)
        return uncovered
    
    # Local Search using bitmask representation
    def local_search(init_code, seed=0, time_budget=None):
        if time_budget is None:
            time_budget = deadline - time.time()
        rng = random.Random(seed)
        active = list(init_code)
        active_set_mask = 0
        for c in active:
            active_set_mask |= (1 << c)
        
        # Compute coverage counts using bit operations
        # coverage[j] = number of codewords covering word j
        coverage = np.zeros(total_words, dtype=np.int32)
        for c in active:
            mask = neighbors_mask[c]
            while mask:
                lowbit = mask & (-mask)
                j = lowbit.bit_length() - 1
                coverage[j] += 1
                mask ^= lowbit
        
        # Ensure valid - add codewords if needed
        def repair():
            nonlocal active, active_set_mask, coverage
            uncovered_mask = is_covering(active_set_mask)
            while uncovered_mask and time.time() < deadline - 3:
                best_idx = -1
                best_count = -1
                for i in range(total_words):
                    if (active_set_mask >> i) & 1:
                        continue
                    cnt = bin(neighbors_mask[i] & uncovered_mask).count('1')
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = i
                if best_idx == -1 or best_count == 0:
                    break
                active.append(best_idx)
                active_set_mask |= (1 << best_idx)
                for nb in neighbors_list[best_idx]:
                    coverage[nb] += 1
                uncovered_mask = is_covering(active_set_mask)
        
        repair()
        
        best_size = len(active)
        best_code = list(active)
        best_mask = active_set_mask
        
        # Precompute: for each codeword, which words does it uniquely cover?
        # We'll compute this on the fly
        
        while time.time() < deadline - 3:
            improved = False
            
            # Phase 1: Removal - try to remove a codeword if it doesn't uncover any words
            # A codeword c can be removed if for all nb in neighbors[c], coverage[nb] > 1
            removal_found = False
            indices = list(range(len(active)))
            rng.shuffle(indices)
            
            for idx in indices:
                c = active[idx]
                can_remove = True
                for nb in neighbors_list[c]:
                    if coverage[nb] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove it
                    for nb in neighbors_list[c]:
                        coverage[nb] -= 1
                    active.pop(idx)
                    active_set_mask &= ~(1 << c)
                    improved = True
                    removal_found = True
                    break
            
            if removal_found:
                if len(active) < best_size:
                    best_size = len(active)
                    best_code = list(active)
                    best_mask = active_set_mask
                continue
            
            # Phase 2: Swaps
            # For each codeword c, find words uniquely covered by c
            # Then find a replacement c' that covers all those unique words
            swap_found = False
            indices = list(range(len(active)))
            rng.shuffle(indices)
            
            for idx in indices:
                c = active[idx]
                # Find words uniquely covered by c
                unique_words = []
                for nb in neighbors_list[c]:
                    if coverage[nb] == 1:
                        unique_words.append(nb)
                
                if not unique_words:
                    # This should have been caught by removal phase
                    continue

                # Find a candidate c' that covers all unique_words
                best_candidate = -1
                best_score = -1
                
                for c_prime in range(total_words):
                    if (active_set_mask >> c_prime) & 1:
                        continue
                    
                    # Check if c_prime covers all unique_words
                    covers_all = True
                    for w in unique_words:
                        if not ((neighbors_mask[c_prime] >> w) & 1):
                            covers_all = False
                            break
                            
                    if not covers_all:
                        continue
                    
                    # Score: number of neighbors of c_prime that are already well-covered
                    score = 0
                    for nb in neighbors_list[c_prime]:
                        if coverage[nb] > 1:
                            score += 1
                            
                    if score > best_score:
                        best_score = score
                        best_candidate = c_prime

                if best_candidate != -1:
                    # Perform swap
                    for nb in neighbors_list[c]:
                        coverage[nb] -= 1
                    active.pop(idx)
                    active_set_mask &= ~(1 << c)
                    
                    active.append(best_candidate)
                    active_set_mask |= (1 << best_candidate)
                    for nb in neighbors_list[best_candidate]:
                        coverage[nb] += 1
                    
                    improved = True
                    swap_found = True
                    break
            
            if not improved:
                # Stuck in local optimum. Perturb.
                if time.time() < deadline - 8:
                    perturb_count = rng.randint(1, 3)
                    removed = []
                    for _ in range(perturb_count):
                        if not active:
                            break
                        idx = rng.randint(0, len(active) - 1)
                        c = active.pop(idx)
                        active_set_mask &= ~(1 << c)
                        for nb in neighbors_list[c]:
                            coverage[nb] -= 1
                        removed.append(c)
                    
                    # Repair
                    repair()
                    
                    if len(active) < best_size:
                        best_size = len(active)
                        best_code = list(active)
                        best_mask = active_set_mask
                    else:
                        # Revert if not improved
                        # Actually, let's keep the perturbed state and continue searching
                        # But if it's worse, we might want to revert
                        pass
            
            if len(active) < best_size:
                best_size = len(active)
                best_code = list(active)
                best_mask = active_set_mask

        return best_code

    # Main Logic
    best_code = None
    best_size = float('inf')
    
    # Strategy 1: Direct Sum
    if q == 3 and n == 6 and radius == 1:
        ds_code = direct_sum_init()
        result = local_search(ds_code, seed=42)
        if len(result) < best_size:
            best_size = len(result)
            best_code = result
    
    # Strategy 2: Greedy with different seeds
    for seed in [0, 1, 2, 3, 4, 5]:
        if time.time() > deadline - 20:
            break
        g_code = greedy_init(seed)
        result = local_search(g_code, seed=seed * 100)
        if len(result) < best_size:
            best_size = len(result)
            best_code = result
    
    # Strategy 3: Multiple local search restarts from best found
    if best_code is not None and time.time() < deadline - 15:
        for seed in [10, 20, 30]:
            if time.time() > deadline - 10:
                break
            result = local_search(best_code, seed=seed)
            if len(result) < best_size:
                best_size = len(result)
                best_code = result
    
    if best_code is None:
        best_code = greedy_init(0)

    return [list(all_words[idx]) for idx in best_code]
