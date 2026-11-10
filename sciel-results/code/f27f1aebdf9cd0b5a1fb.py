"""Simulated annealing with bitmasks for 3-AP maximization under 4-AP constraint."""
import time
import numpy as np

N = 256
DEADLINE = 0.88  # fraction of 90s

def construct():
    start = time.time()
    deadline = start + 90 * DEADLINE
    
    # Precompute all 3-APs as bitmasks
    three_aps = []  # list of (a, b, c) where a<b<c and b-a==c-b
    for a in range(1, N+1):
        for d in range(1, (N - a + 1) // 2 + 1):
            b = a + d
            c = a + 2*d
            if c <= N:
                three_aps.append((a, b, c))
    
    # For each element x, precompute which 3-APs it participates in
    x_to_3ap_partners = {x: [] for x in range(1, N+1)}
    for (a, b, c) in three_aps:
        x_to_3ap_partners[a].append((b, c))
        x_to_3ap_partners[b].append((a, c))
        x_to_3ap_partners[c].append((a, b))
    
    # Precompute 4-AP constraints
    x_to_4ap_constraints = {x: [] for x in range(1, N+1)}
    for x in range(1, N+1):
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            x_to_4ap_constraints[x].append((x+d, x+2*d, x+3*d))
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            x_to_4ap_constraints[x].append((x-d, x+d, x+2*d))
        max_d = min((x - 1) // 2, N - x)
        for d in range(1, max_d + 1):
            x_to_4ap_constraints[x].append((x-2*d, x-d, x+d))
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            x_to_4ap_constraints[x].append((x-3*d, x-2*d, x-d))
    
    def mask_to_set(m):
        return [i+1 for i in range(N) if (m >> i) & 1]
    
    def count_3aps_from_mask(mask):
        count = 0
        for (a, b, c) in three_aps:
            if (mask >> (a-1)) & 1 and (mask >> (b-1)) & 1 and (mask >> (c-1)) & 1:
                count += 1
        return count
    
    def has_4ap_from_mask(mask, x):
        for (y, z, w) in x_to_4ap_constraints[x]:
            if ((mask >> (y-1)) & 1) and ((mask >> (z-1)) & 1) and ((mask >> (w-1)) & 1):
                return True
        return False
    
    def new_3aps_with_x_from_mask(mask, x):
        count = 0
        for (y, z) in x_to_3ap_partners[x]:
            if ((mask >> (y-1)) & 1) and ((mask >> (z-1)) & 1):
                count += 1
        return count
    
    def is_valid_4ap_free(mask):
        """Check if mask has no 4-AP."""
        for x in range(1, N+1):
            if (mask >> (x-1)) & 1:
                for (y, z, w) in x_to_4ap_constraints[x]:
                    if ((mask >> (y-1)) & 1) and ((mask >> (z-1)) & 1) and ((mask >> (w-1)) & 1):
                        return False
        return True
    
    # Greedy construction for initial solutions
    def greedy_construct(seed):
        rng = np.random.default_rng(seed)
        mask = 0
        order = list(range(1, N+1))
        rng.shuffle(order)
        
        for x in order:
            if has_4ap_from_mask(mask, x):
                continue
            gain = new_3aps_with_x_from_mask(mask, x)
            if gain > 0 or mask.bit_count() < 30:
                mask |= (1 << (x - 1))
        return mask
    
    # Simulated annealing
    def simulated_annealing(initial_mask, deadline):
        rng = np.random.default_rng()
        mask = initial_mask
        current_score = count_3aps_from_mask(mask)
        best_mask = mask
        best_score = current_score
        
        T_start = 10.0
        T_end = 0.1
        cooling_rate = 0.9995
        
        while time.time() < deadline:
            T = T_start * (T_end / T_start) ** (0.9995 ** int((time.time() - start) * 100))
            if T < T_end:
                T = T_end
            
            # Choose a random move
            move_type = rng.integers(0, 3)
            new_mask = mask
            
            if move_type == 0:  # Add
                candidates = [x for x in range(1, N+1) if not ((mask >> (x-1)) & 1) and not has_4ap_from_mask(mask, x)]
                if candidates:
                    x = candidates[rng.integers(0, len(candidates))]
                    new_mask = mask | (1 << (x-1))
            elif move_type == 1:  # Remove
                present = [x for x in range(1, N+1) if (mask >> (x-1)) & 1]
                if present:
                    x = present[rng.integers(0, len(present))]
                    new_mask = mask & ~(1 << (x-1))
            else:  # Swap
                present = [x for x in range(1, N+1) if (mask >> (x-1)) & 1]
                absent = [x for x in range(1, N+1) if not ((mask >> (x-1)) & 1) and not has_4ap_from_mask(mask & ~(1 << (x-1)), x)]
                if present and absent:
                    x_out = present[rng.integers(0, len(present))]
                    x_in = absent[rng.integers(0, len(absent))]
                    new_mask = (mask & ~(1 << (x_out-1))) | (1 << (x_in-1))
            
            if new_mask == mask:
                continue
            
            new_score = count_3aps_from_mask(new_mask)
            delta = new_score - current_score
            
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 0.01)):
                mask = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask
            
            # Cool down
            T *= cooling_rate
        
        return best_mask, best_score
    
    # Try multiple restarts with different greedy seeds
    best_mask = None
    best_score = -1
    
    # Phase 1: Greedy + local improvement
    for seed in range(20):
        if time.time() > deadline:
            break
        m = greedy_construct(seed)
        score = count_3aps_from_mask(m)
        if score > best_score:
            best_score = score
            best_mask = m
    
    # Phase 2: Simulated annealing from best greedy solution
    if best_mask is not None:
        sa_mask, sa_score = simulated_annealing(best_mask, deadline)
        if sa_score > best_score:
            best_score = sa_score
            best_mask = sa_mask
    
    # Phase 3: More SA restarts if time permits
    if time.time() < deadline - 5:
        for seed in range(10):
            if time.time() > deadline:
                break
            m = greedy_construct(100 + seed)
            sa_mask, sa_score = simulated_annealing(m, deadline)
            if sa_score > best_score:
                best_score = sa_score
                best_mask = sa_mask
    
    if best_mask is None or best_mask == 0:
        return [1]
    
    return mask_to_set(best_mask)
