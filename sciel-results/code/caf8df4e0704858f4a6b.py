"""Optimized solver for large reciprocal weight under lcm separation."""
# EVOLVE-BLOCK-START
def construct():
    from math import lcm
    from fractions import Fraction
    import sys

    N = 256
    nums = list(range(2, 257))
    n = len(nums)  # 255

    # Build adjacency bitsets for graph G where edge exists if lcm > 256
    adj = [0] * n
    for i in range(n):
        ai = nums[i]
        for j in range(i + 1, n):
            bj = nums[j]
            if lcm(ai, bj) > N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    weights = [Fraction(1, nums[i]) for i in range(n)]

    # Sort nodes by weight descending
    order = list(range(n))
    order.sort(key=lambda x: weights[x], reverse=True)
    
    # Remap to sorted order
    new_adj = [0] * n
    new_weights = [Fraction(0, 1)] * n
    pos = [0] * n
    for new_idx, old_idx in enumerate(order):
        pos[old_idx] = new_idx
        new_weights[new_idx] = weights[old_idx]
        old_mask = adj[old_idx]
        new_mask = 0
        tmp = old_mask
        while tmp:
            lsb = tmp & (-tmp)
            old_j = lsb.bit_length() - 1
            new_j = pos[old_j]
            new_mask |= (1 << new_j)
            tmp ^= lsb
        new_adj[new_idx] = new_mask

    best_weight = Fraction(0, 1)
    best_clique = []
    
    # Greedy coloring bound
    def get_bound(candidates_mask):
        cands = []
        tmp = candidates_mask
        while tmp:
            lsb = tmp & (-tmp)
            idx = lsb.bit_length() - 1
            cands.append(idx)
            tmp ^= lsb
        
        if not cands:
            return Fraction(0, 1)
        
        color_of = [-1] * n
        max_w_per_color = {}
        
        # cands are in increasing index order, which is descending weight order
        for v in cands:
            neigh_mask = new_adj[v] & candidates_mask
            used_colors = set()
            tmp_n = neigh_mask
            while tmp_n:
                lsb = tmp_n & (-tmp_n)
                u = lsb.bit_length() - 1
                if color_of[u] != -1:
                    used_colors.add(color_of[u])
                tmp_n ^= lsb
            
            c = 0
            while c in used_colors:
                c += 1
            color_of[v] = c
            if c not in max_w_per_color or new_weights[v] > max_w_per_color[c]:
                max_w_per_color[c] = new_weights[v]
                
        return sum(max_w_per_color.values())

    def bnb(candidates_mask, current_weight, current_clique):
        nonlocal best_weight, best_clique
        
        if not candidates_mask:
            if current_weight > best_weight:
                best_weight = current_weight
                best_clique = current_clique[:]
            return
        
        bound = get_bound(candidates_mask)
        if current_weight + bound <= best_weight:
            return
            
        tmp = candidates_mask
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            tmp ^= lsb
            
            # Include v
            new_weight = current_weight + new_weights[v]
            new_candidates = candidates_mask & new_adj[v]
            
            # Prune if possible
            # We can check bound of new_candidates
            # But get_bound is expensive. 
            # Let's just call bnb.
            bnb(new_candidates, new_weight, current_clique + [v])
            
            # Exclude v (handled by loop)
            # Check if remaining candidates can beat best
            if tmp:
                remaining_bound = get_bound(tmp)
                if current_weight + remaining_bound <= best_weight:
                    break

    init_mask = (1 << n) - 1
    bnb(init_mask, Fraction(0, 1), [])

    result = [nums[order[i]] for i in best_clique]
    return result
# EVOLVE-BLOCK-END
