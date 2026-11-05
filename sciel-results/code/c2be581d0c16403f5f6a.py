import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin

    N = 256
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Build adjacency bitmasks
    adj = [0] * N
    forbidden = [False] * N
    
    for i in range(N):
        v = i + 1
        if 2 * v in squares:
            forbidden[i] = True
            continue
        for sq in squares:
            u = sq - v
            if 1 <= u <= N:
                j = u - 1
                if not forbidden[j]:
                    adj[i] |= (1 << j)
                    
    candidates = [i for i in range(N) if not forbidden[i]]
    cand_mask = 0
    for i in candidates:
        cand_mask |= (1 << i)
        
    def greedy_init():
        available_mask = cand_mask
        selected_mask = 0
        while available_mask:
            best_v = -1
            min_deg = N + 1
            m = available_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                deg = (adj[v] & available_mask).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            if best_v == -1:
                break
            selected_mask |= (1 << best_v)
            available_mask &= ~((1 << best_v) | (adj[best_v] & available_mask))
        return selected_mask

    def local_search(mask):
        """
        Improves the independent set using 1-0, 1-2, 2-1, and 2-2 moves.
        Uses bitmask operations for speed.
        """
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > DEADLINE:
                break
                
            # 1. Try to add a vertex (1-0 move)
            not_in_mask = cand_mask & ~mask
            m = not_in_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                if (adj[v] & mask) == 0:
                    mask |= lsb
                    improved = True
                    break
            
            if improved:
                continue

            # 2. Try 1-2 moves: Remove u, add v, w
            m_u = mask
            found_move = False
            while m_u and not found_move:
                lsb_u = m_u & -m_u
                u = lsb_u.bit_length() - 1
                m_u ^= lsb_u
                
                mask_without_u = mask ^ lsb_u
                
                # Find candidates v not in mask such that adj[v] & mask_without_u == 0
                # Use bitmask intersection to speed this up
                valid_v_mask = cand_mask & ~mask
                # Filter: v must not be adjacent to any node in mask_without_u
                # adj[v] & mask_without_u == 0
                # We need to find all v where this holds
                # Iterate through valid_v_mask
                m_v = valid_v_mask
                valid_v_list = []
                while m_v:
                    lsb_v = m_v & -m_v
                    v = lsb_v.bit_length() - 1
                    m_v ^= lsb_v
                    if (adj[v] & mask_without_u) == 0:
                        valid_v_list.append(v)
                        
                if len(valid_v_list) < 2:
                    continue
                    
                n_valid = len(valid_v_list)
                for i in range(n_valid):
                    v = valid_v_list[i]
                    for j in range(i + 1, n_valid):
                        w = valid_v_list[j]
                        if (adj[v] & (1 << w)) == 0:
                            mask = mask_without_u | (1 << v) | (1 << w)
                            improved = True
                            found_move = True
                            break
                    if found_move:
                        break
                            
            if not found_move:
                # 3. Try 2-1 moves: Remove u, v from mask, add w
                # This keeps size same but might enable future improvements
                m_u = mask
                found_swap = False
                while m_u and not found_swap:
                    lsb_u = m_u & -m_u
                    u = lsb_u.bit_length() - 1
                    m_u ^= lsb_u
                    
                    m_v = mask & ~lsb_u
                    while m_v and not found_swap:
                        lsb_v = m_v & -m_v
                        v = lsb_v.bit_length() - 1
                        m_v ^= lsb_v
                        
                        mask_without_uv = mask ^ lsb_u ^ lsb_v
                        
                        # Find w not in mask such that adj[w] & mask_without_uv == 0
                        # and w is not adjacent to u or v (already removed, so just check mask_without_uv)
                        m_w = cand_mask & ~mask
                        while m_w and not found_swap:
                            lsb_w = m_w & -m_w
                            w = lsb_w.bit_length() - 1
                            m_w ^= lsb_w
                            if (adj[w] & mask_without_uv) == 0:
                                # Check that w is not u or v (already excluded by ~mask)
                                mask = mask_without_uv | lsb_w
                                improved = True
                                found_swap = True
                                break
            
            if not found_swap:
                # 4. Try 2-2 moves: Remove u, v, add w, x
                # This is expensive, so only try a limited number of combinations
                m_u = mask
                found_22 = False
                count_22 = 0
                while m_u and not found_22 and count_22 < 50:
                    lsb_u = m_u & -m_u
                    u = lsb_u.bit_length() - 1
                    m_u ^= lsb_u
                    count_22 += 1
                    
                    m_v = mask & ~lsb_u
                    while m_v and not found_22:
                        lsb_v = m_v & -m_v
                        v = lsb_v.bit_length() - 1
                        m_v ^= lsb_v
                        
                        mask_without_uv = mask ^ lsb_u ^ lsb_v
                        
                        # Find candidates w, x not in mask, compatible with mask_without_uv
                        m_w = cand_mask & ~mask
                        valid_w_list = []
                        while m_w:
                            lsb_w = m_w & -m_w
                            w = lsb_w.bit_length() - 1
                            m_w ^= lsb_w
                            if (adj[w] & mask_without_uv) == 0:
                                valid_w_list.append(w)
                                
                        if len(valid_w_list) < 2:
                            continue
                        
                        n_valid = len(valid_w_list)
                        for i in range(n_valid):
                            w = valid_w_list[i]
                            for j in range(i + 1, n_valid):
                                x = valid_w_list[j]
                                if (adj[w] & (1 << x)) == 0:
                                    mask = mask_without_uv | (1 << w) | (1 << x)
                                    improved = True
                                    found_22 = True
                                    break
                            if found_22:
                                break
            
            # 5. If no improvement, try a random 1-1 swap to escape local optimum
            if not improved and random.random() < 0.3:
                m_u = mask
                if m_u:
                    lsb_u = m_u & -m_u
                    u = lsb_u.bit_length() - 1
                    mask_without_u = mask ^ lsb_u
                    m_v = cand_mask & ~mask
                    if m_v:
                        lsb_v = m_v & -m_v
                        v = lsb_v.bit_length() - 1
                        if (adj[v] & mask_without_u) == 0:
                            mask = mask_without_u | lsb_v

        return mask

    def perturb(mask, k):
        """Remove k random vertices"""
        m = mask
        for _ in range(k):
            if not m:
                break
            count = m.bit_count()
            r = random.randint(1, count)
            cnt = 0
            temp = m
            while temp:
                lsb = temp & -temp
                cnt += 1
                if cnt == r:
                    m ^= lsb
                    break
                temp ^= lsb
        return m

    # Main ILS Loop
    best_mask = greedy_init()
    best_mask = local_search(best_mask)
    best_size = best_mask.bit_count()
    
    current_mask = best_mask
    stagnant_iterations = 0
    
    while time.time() - start_time < DEADLINE:
        # Perturb with adaptive size
        if stagnant_iterations > 10:
            k_remove = random.randint(3, 6)
        else:
            k_remove = random.randint(2, 4)
        current_mask = perturb(current_mask, k_remove)
        
        current_mask = local_search(current_mask)
        
        if current_mask.bit_count() > best_size:
            best_size = current_mask.bit_count()
            best_mask = current_mask
            stagnant_iterations = 0
        else:
            stagnant_iterations += 1
            
        # Occasionally, try a fresh greedy to explore different basins
        if random.random() < 0.15:
            fresh = greedy_init()
            fresh = local_search(fresh)
            if fresh.bit_count() > best_size:
                best_size = fresh.bit_count()
                best_mask = fresh
                current_mask = fresh
                stagnant_iterations = 0

    # Convert to list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        result.append(v + 1)
        m ^= lsb
        
    return sorted(result)

# EVOLVE-BLOCK-END
