import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    N = 128
    TARGETS_RANGE = np.arange(128, 257)
    NUM_TARGETS = 129
    
    def evaluate(mask):
        """
        Returns (max_mult, sum_sq, is_complete)
        mask: boolean numpy array of length 129
        """
        conv = np.convolve(mask, mask, mode='full')
        # conv[k] is count of ordered pairs (a,b) with a+b=k
        # We care about k in [128, 256]
        target_conv = conv[128:257]
        
        i_vals = TARGETS_RANGE
        even_mask = (i_vals % 2 == 0)
        half_vals = i_vals // 2
        
        # D(i) = 1 if i is even and i/2 is in A
        D = np.zeros(NUM_TARGETS)
        D[even_mask] = mask[half_vals[even_mask]]
        
        # Unordered count C_unord = (C_ord + D) / 2
        mults = (target_conv + D) / 2.0
        
        if np.any(mults < 1):
            return float('inf'), float('inf'), False
        
        max_mult = np.max(mults)
        sum_sq = np.sum(mults * mults)
        
        return max_mult, sum_sq, True

    def greedy_repair(mask):
        """
        Add elements to mask until it is complete.
        Returns the modified mask.
        """
        current_mask = mask.copy()
        _, _, is_complete = evaluate(current_mask)
        
        while not is_complete:
            if time.time() > deadline - 10:
                break
                
            # Find missing targets
            conv = np.convolve(current_mask, current_mask, mode='full')
            target_conv = conv[128:257]
            i_vals = TARGETS_RANGE
            even_mask = (i_vals % 2 == 0)
            half_vals = i_vals // 2
            D = np.zeros(NUM_TARGETS)
            D[even_mask] = current_mask[half_vals[even_mask]]
            mults = (target_conv + D) / 2.0
            
            missing_indices = np.where(mults < 1)[0]
            if len(missing_indices) == 0:
                break
                
            missing_targets = i_vals[missing_indices]
            
            best_gain = -1
            best_elem = None
            
            # Try adding each x not in A
            for x in range(N + 1):
                if current_mask[x]:
                    continue
                    
                gain = 0
                for t in missing_targets:
                    y = t - x
                    if 0 <= y <= N and current_mask[y]:
                        gain += 1
                        
                if gain > best_gain:
                    best_gain = gain
                    best_elem = x
                    
            if best_elem is not None and best_gain > 0:
                current_mask[best_elem] = True
            else:
                # Fallback: add random element
                x = random.randint(0, N)
                current_mask[x] = True
                
            _, _, is_complete = evaluate(current_mask)
            
        return current_mask

    def local_search(mask, deadline):
        """
        Simulated Annealing / Local Search to minimize max multiplicity.
        """
        current_mask = mask.copy()
        current_max, current_sq, is_complete = evaluate(current_mask)
        
        if not is_complete:
            current_mask = greedy_repair(current_mask)
            current_max, current_sq, is_complete = evaluate(current_mask)
            if not is_complete:
                return current_mask, current_max, current_sq
                
        best_mask = current_mask.copy()
        best_max = current_max
        best_sq = current_sq
        
        # SA Parameters
        T_start = 5.0
        T_end = 0.1
        total_time = deadline - start_time
        elapsed = time.time() - start_time
        remaining = max(0, total_time - elapsed)
        
        while time.time() < deadline - 1:
            # Calculate Temperature
            frac = max(0, 1.0 - (time.time() - start_time) / total_time)
            T = T_end + (T_start - T_end) * frac
            
            # Generate Move
            move_type = random.random()
            new_mask = current_mask.copy()
            
            if move_type < 0.4: # 1-Swap
                a_idx = np.where(current_mask)[0]
                b_candidates = np.where(~current_mask)[0]
                if len(a_idx) == 0 or len(b_candidates) == 0:
                    continue
                a = a_idx[random.randint(0, len(a_idx)-1)]
                b = b_candidates[random.randint(0, len(b_candidates)-1)]
                new_mask[a] = False
                new_mask[b] = True
                
            elif move_type < 0.7: # 2-Swap
                a_idx = np.where(current_mask)[0]
                b_candidates = np.where(~current_mask)[0]
                if len(a_idx) < 2 or len(b_candidates) < 2:
                    continue
                a1 = a_idx[random.randint(0, len(a_idx)-1)]
                a2 = a_idx[random.randint(0, len(a_idx)-1)]
                if a1 == a2:
                    continue
                b1 = b_candidates[random.randint(0, len(b_candidates)-1)]
                b2 = b_candidates[random.randint(0, len(b_candidates)-1)]
                if b1 == b2:
                    continue
                new_mask[a1] = False
                new_mask[a2] = False
                new_mask[b1] = True
                new_mask[b2] = True
                
            elif move_type < 0.85: # Add
                b_candidates = np.where(~current_mask)[0]
                if len(b_candidates) == 0:
                    continue
                b = b_candidates[random.randint(0, len(b_candidates)-1)]
                new_mask[b] = True
                
            else: # Remove
                a_idx = np.where(current_mask)[0]
                if len(a_idx) <= 1:
                    continue
                a = a_idx[random.randint(0, len(a_idx)-1)]
                new_mask[a] = False
                
            new_max, new_sq, is_complete = evaluate(new_mask)
            
            if not is_complete:
                continue
                
            # Acceptance
            if new_max < current_max:
                current_mask = new_mask
                current_max = new_max
                current_sq = new_sq
                if current_max < best_max:
                    best_max = current_max
                    best_sq = current_sq
                    best_mask = current_mask.copy()
            elif new_max == current_max and new_sq < current_sq:
                current_mask = new_mask
                current_max = new_max
                current_sq = new_sq
                if current_sq < best_sq:
                    best_sq = current_sq
                    best_mask = current_mask.copy()
            else:
                # SA Acceptance
                delta = (new_max - current_max) * 100 + (new_sq - current_sq)
                if delta < 0:
                    current_mask = new_mask
                    current_max = new_max
                    current_sq = new_sq
                    if current_max < best_max:
                        best_max = current_max
                        best_sq = current_sq
                        best_mask = current_mask.copy()
                else:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        current_mask = new_mask
                        current_max = new_max
                        current_sq = new_sq
                        if current_max < best_max:
                            best_max = current_max
                            best_sq = current_sq
                            best_mask = current_mask.copy()
                            
        return best_mask, best_max, best_sq

    best_mask = None
    best_max = float('inf')
    best_sq = float('inf')
    
    # Generate Seeds
    seeds = []
    
    # 1. Random sets
    for _ in range(8):
        k = random.randint(25, 35)
        idx = random.sample(range(129), k)
        m = np.zeros(129, dtype=bool)
        m[idx] = True
        seeds.append(m)
        
    # 2. Structured: Low + High
    m = np.zeros(129, dtype=bool)
    m[0:20] = True
    m[108:129] = True
    seeds.append(m)
    
    # 3. Structured: Grid
    m = np.zeros(129, dtype=bool)
    for i in range(129):
        if i % 4 == 0 or i % 4 == 1:
            m[i] = True
    seeds.append(m)
    
    # 4. Structured: Arithmetic Progression
    m = np.zeros(129, dtype=bool)
    m[0:129:3] = True
    seeds.append(m)
    
    # Process Seeds
    for seed in seeds:
        if time.time() > deadline - 5:
            break
            
        # Repair
        m_repaired = greedy_repair(seed)
        mx, sq, _ = evaluate(m_repaired)
        if mx == float('inf'):
            continue
            
        # Optimize
        m_opt, mx_opt, sq_opt = local_search(m_repaired, deadline)
        
        if mx_opt < best_max:
            best_max = mx_opt
            best_sq = sq_opt
            best_mask = m_opt
        elif mx_opt == best_max and sq_opt < best_sq:
            best_sq = sq_opt
            best_mask = m_opt

    if best_mask is None:
        # Fallback: A simple valid set
        # [0, 128] is valid?
        # 128 = 0+128. 256 = 128+128.
        # What about 129? 1+128.
        # If A = {0, 1, ..., 128}, max mult is high.
        # Let's return a set that is likely valid.
        return list(range(0, 129, 2))
        
    return list(np.where(best_mask)[0])
