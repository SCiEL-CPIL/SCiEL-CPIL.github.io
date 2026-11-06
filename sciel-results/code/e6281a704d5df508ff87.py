"""Bitmask-accelerated simulated annealing for equal-LCM-triple-free subset of [1,128]."""
import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85
    N = 128
    elements = list(range(1, N + 1))
    FULL_MASK = (1 << N) - 1
    
    # Precompute LCM table
    lcm_table = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_table[(i, j)] = i * j // gcd(i, j)
    
    # Find all bad triples and store as bitmasks
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[(a, b)]
            for c in range(b + 1, N + 1):
                if lcm_table[(a, c)] == lab and lcm_table[(b, c)] == lab:
                    bad_triples.append((1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)))
    
    num_bad = len(bad_triples)
    
    # For each node, list of bad triple indices it participates in
    triples_by_node = [[] for _ in range(N)]
    for idx, mask in enumerate(bad_triples):
        m = mask
        while m:
            lsb = m & (-m)
            node = lsb.bit_length() - 1
            triples_by_node[node].append(idx)
            m ^= lsb
    
    # Precompute for each node and each triple it's in: the "others" mask
    # others_mask[node][tidx] = the mask of the other two nodes in that triple
    node_others = [None] * N
    for node in range(N):
        node_others[node] = [0] * len(triples_by_node[node])
        for i, tidx in enumerate(triples_by_node[node]):
            tm = bad_triples[tidx]
            node_others[node][i] = tm & ~(1 << node)
    
    # Count violations in a bitmask
    def count_violations(mask):
        cnt = 0
        for tm in bad_triples:
            if (mask & tm) == tm:
                cnt += 1
        return cnt
    
    # Delta for adding node to mask: how many new violations created
    def delta_add(node, mask):
        cnt = 0
        others_list = node_others[node]
        tlist = triples_by_node[node]
        for i in range(len(tlist)):
            if (mask & others_list[i]) == others_list[i]:
                cnt += 1
        return cnt
    
    # Delta for removing node from mask: how many violations broken
    def delta_remove(node, mask):
        cnt = 0
        others_list = node_others[node]
        tlist = triples_by_node[node]
        for i in range(len(tlist)):
            if (mask & others_list[i]) == others_list[i]:
                cnt += 1
        return cnt
    
    best_mask = 0
    best_size = 0
    
    def update_best(mask):
        nonlocal best_mask, best_size
        sz = mask.bit_count()
        if sz > best_size:
            best_size = sz
            best_mask = mask
    
    # Greedy fill: add nodes in order of fewest conflicts if safe
    def greedy_fill(mask, order):
        for node in order:
            if not (mask >> node) & 1:
                if delta_add(node, mask) == 0:
                    mask |= (1 << node)
        return mask
    
    # Simulated annealing with bitmask
    def simulated_annealing(start_mask, time_limit):
        mask = start_mask
        current_E = count_violations(mask)
        
        T = 1.0
        cooling = 0.999
        min_T = 1e-4
        
        while time.time() < time_limit and T > min_T:
            move_type = random.random()
            
            if move_type < 0.4:
                # Add random node
                unselected = FULL_MASK & ~mask
                if unselected:
                    lsb = unselected & (-unselected)
                    node = lsb.bit_length() - 1
                    d = delta_add(node, mask)
                    new_E = current_E + d
                    
                    if new_E <= current_E:
                        mask |= (1 << node)
                        current_E = new_E
                    else:
                        if random.random() < 2.71828 ** (-d / T):
                            mask |= (1 << node)
                            current_E = new_E
                            
            elif move_type < 0.8:
                # Remove random node
                if mask:
                    lsb = mask & (-mask)
                    node = lsb.bit_length() - 1
                    d = delta_remove(node, mask)
                    new_E = current_E - d
                    
                    if new_E < current_E:
                        mask &= ~(1 << node)
                        current_E = new_E
                        
            else:
                # Swap
                if mask:
                    lsb_in = mask & (-mask)
                    x = lsb_in.bit_length() - 1
                    unselected = FULL_MASK & ~mask
                    if unselected:
                        lsb_out = unselected & (-unselected)
                        y = lsb_out.bit_length() - 1
                        
                        mask_temp = mask & ~(1 << x)
                        d_rem = delta_remove(x, mask)
                        d_add = delta_add(y, mask_temp)
                        d = d_rem + d_add
                        new_E = current_E + d
                        
                        if new_E < current_E:
                            mask = mask_temp | (1 << y)
                            current_E = new_E
                        elif new_E == current_E:
                            if random.random() < 0.5:
                                mask = mask_temp | (1 << y)
                                current_E = new_E
                        else:
                            if random.random() < 2.71828 ** (-d / T):
                                mask = mask_temp | (1 << y)
                                current_E = new_E
                    
            T *= cooling
            
            # If valid, try to greedily add more
            if current_E == 0:
                changed = True
                while changed:
                    changed = False
                    for node in range(N):
                        if not (mask >> node) & 1:
                            if delta_add(node, mask) == 0:
                                mask |= (1 << node)
                                changed = True
                update_best(mask)
        
        update_best(mask)
        return mask
    
    random.seed(42)
    
    # Initial greedy
    S0 = 0
    order = sorted(range(N), key=lambda x: len(triples_by_node[x]))
    S0 = greedy_fill(S0, order)
    update_best(S0)
    
    if time.time() < deadline - 10:
        simulated_annealing(S0, deadline - 10)
        
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        size = random.randint(30, 60)
        sample = random.sample(range(N), size)
        S_start = 0
        for node in sample:
            S_start |= (1 << node)
        simulated_annealing(S_start, deadline - 5)
        
    # Final polish: try to add any remaining safe nodes
    if time.time() < deadline - 2:
        S_best = best_mask
        for node in range(N):
            if not (S_best >> node) & 1:
                if delta_add(node, S_best) == 0:
                    S_best |= (1 << node)
        update_best(S_best)
        
    if best_mask == 0:
        best_mask = 1  # element 1
        
    # Convert bitmask to list
    result = []
    m = best_mask
    while m:
        lsb = m & (-m)
        node = lsb.bit_length() - 1
        result.append(node + 1)
        m ^= lsb
        
    return result
