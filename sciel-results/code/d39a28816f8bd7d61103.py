import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    n = 256
    # 1. Identify forbidden nodes (2*a is a perfect square)
    forbidden = set()
    for a in range(1, n + 1):
        s = 2 * a
        r = int(math.isqrt(s))
        if r * r == s:
            forbidden.add(a)
            
    valid_nodes = [i for i in range(1, n + 1) if i not in forbidden]
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    num_nodes = len(valid_nodes)
    
    # 2. Build conflict masks
    # conflict_masks[i] is a bitmask of indices j such that valid_nodes[i] + valid_nodes[j] is a square
    # Note: j != i is handled by the logic, but we include all conflicts.
    # If i==j, it's already filtered out by 'forbidden'.
    conflict_masks = [0] * num_nodes
    
    # Precompute squares
    max_sum = 2 * n
    squares = set()
    k = 1
    while k * k <= max_sum:
        squares.add(k * k)
        k += 1
        
    for i in range(num_nodes):
        a = valid_nodes[i]
        mask = 0
        for s in squares:
            b = s - a
            if b in node_to_idx:
                j = node_to_idx[b]
                # We only care about j > i to avoid double counting in some contexts, 
                # but for bitmask intersection, we need the full mask of conflicts for i.
                # However, for checking validity of set S:
                # For each i in S, (S & conflict_masks[i]) must be 0 (excluding i itself if self-loop, but no self-loops here).
                # Actually, if i is in S, we check if any OTHER node in S is in conflict_masks[i].
                # So conflict_masks[i] should contain all j != i that conflict with i.
                if j != i:
                    mask |= (1 << j)
        conflict_masks[i] = mask
        
    # 3. Initial Solution: Greedy
    # Try multiple random orders or specific heuristics
    best_set_mask = 0
    best_size = 0
    
    # Heuristic 1: Mod 3 class 1
    # This is known to be 86.
    # Let's try a greedy approach on the graph.
    
    def greedy_select(order):
        mask = 0
        # order is a list of indices
        for idx in order:
            # Check if idx conflicts with current mask
            # (mask & conflict_masks[idx]) == 0
            if (mask & conflict_masks[idx]) == 0:
                mask |= (1 << idx)
        return mask

    # Try a few random permutations
    random.seed(42)
    indices = list(range(num_nodes))
    
    for _ in range(10):
        random.shuffle(indices)
        m = greedy_select(indices)
        size = bin(m).count('1')
        if size > best_size:
            best_size = size
            best_set_mask = m
            
    # Also try the mod 3 strategy explicitly to ensure we have a good baseline
    # Nodes v where v % 3 == 1
    mod3_mask = 0
    for v in valid_nodes:
        if v % 3 == 1:
            idx = node_to_idx[v]
            mod3_mask |= (1 << idx)
    # Verify validity of mod3_mask
    is_valid = True
    temp_mask = mod3_mask
    while temp_mask:
        lsb = temp_mask & -temp_mask
        idx = lsb.bit_length() - 1
        if (mod3_mask & conflict_masks[idx]) != 0:
             # Check if the conflict is with itself? No, self-conflicts removed.
             # Check if there is ANY conflict.
             # Wait, (mod3_mask & conflict_masks[idx]) includes idx? No, conflict_masks[idx] has j != i.
             # So if intersection is non-zero, there is a conflict.
             is_valid = False
             break
        temp_mask ^= lsb
    if is_valid:
        size = bin(mod3_mask).count('1')
        if size > best_size:
            best_size = size
            best_set_mask = mod3_mask

    # 4. Local Search with 1-for-2 swaps and polish
    
    def count_conflicts(mask):
        count = 0
        temp = mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            count += bin(mask & conflict_masks[idx]).count('1')
            temp ^= lsb
        return count // 2

    def is_valid_mask(mask):
        return count_conflicts(mask) == 0

    def polish(mask):
        """Try to add any node that doesn't conflict with current set."""
        changed = True
        while changed:
            changed = False
            for idx in range(num_nodes):
                bit = 1 << idx
                if not (mask & bit) and (mask & conflict_masks[idx]) == 0:
                    mask |= bit
                    changed = True
        return mask

    best_mask_final = best_set_mask
    best_size_final = bin(best_mask_final).count('1')
    
    # First, polish the initial solution
    best_mask_final = polish(best_mask_final)
    best_size_final = bin(best_mask_final).count('1')
    
    # Now do local search with 1-for-2 swaps
    current_mask = best_mask_final
    current_size = best_size_final
    
    # Precompute for each node, its neighbors
    # neighbor_lists[i] = list of indices j that conflict with i
    neighbor_lists = []
    for i in range(num_nodes):
        mask = conflict_masks[i]
        lst = []
        temp = mask
        while temp:
            lsb = temp & -temp
            j = lsb.bit_length() - 1
            lst.append(j)
            temp ^= lsb
        neighbor_lists.append(lst)
    
    T = 50.0
    T_min = 0.1
    alpha = 0.995
    
    while time.time() - start_time < deadline - 5:
        move_type = random.random()
        
        if move_type < 0.4:
            # 1-for-2 swap: remove one node, try to add two non-conflicting nodes
            # Pick a random node to remove
            if current_size == 0:
                T *= alpha
                continue
            bits = []
            temp = current_mask
            while temp:
                lsb = temp & -temp
                bits.append(lsb.bit_length() - 1)
                temp ^= lsb
            remove_idx = random.choice(bits)
            remove_bit = 1 << remove_idx
            
            # Try all pairs of nodes that could be added
            # Nodes that conflict with remove_idx but not with each other
            candidates = []
            for idx in range(num_nodes):
                bit = 1 << idx
                if not (current_mask & bit) and (current_mask & conflict_masks[idx]) == 0:
                    candidates.append(idx)
            
            # Also consider nodes that DO conflict with current_mask but might not after removal
            for idx in range(num_nodes):
                bit = 1 << idx
                if not (current_mask & bit):
                    # Check if idx conflicts with current_mask minus remove_idx
                    new_base = current_mask & ~remove_bit
                    if (new_base & conflict_masks[idx]) == 0:
                        if idx not in candidates:
                            candidates.append(idx)
            
            # Try pairs from candidates
            found_improvement = False
            for i in range(len(candidates)):
                if time.time() - start_time > deadline:
                    break
                for j in range(i + 1, len(candidates)):
                    a = candidates[i]
                    b = candidates[j]
                    # Check if a and b conflict with each other
                    if (conflict_masks[a] & (1 << b)) or (conflict_masks[b] & (1 << a)):
                        continue
                    # Check if a and b conflict with current_mask minus remove_idx
                    new_base = current_mask & ~remove_bit
                    if (new_base & conflict_masks[a]) != 0 or (new_base & conflict_masks[b]) != 0:
                        continue
                    # This swap is valid and improves size by 1
                    new_mask = new_base | (1 << a) | (1 << b)
                    current_mask = new_mask
                    current_size += 1
                    found_improvement = True
                    break
                if found_improvement:
                    break
            
            if found_improvement:
                if current_size > best_size_final:
                    best_size_final = current_size
                    best_mask_final = current_mask
                continue
        
        elif move_type < 0.7:
            # Simple add
            idx = random.randint(0, num_nodes - 1)
            bit = 1 << idx
            if not (current_mask & bit) and (current_mask & conflict_masks[idx]) == 0:
                current_mask |= bit
                current_size += 1
                if current_size > best_size_final:
                    best_size_final = current_size
                    best_mask_final = current_mask
        
        elif move_type < 0.85:
            # Simple remove (to escape local optima)
            if current_size > 0:
                bits = []
                temp = current_mask
                while temp:
                    lsb = temp & -temp
                    bits.append(lsb.bit_length() - 1)
                    temp ^= lsb
                idx = random.choice(bits)
                bit = 1 << idx
                current_mask &= ~bit
                current_size -= 1
        
        else:
            # Random restart with small perturbation
            if current_size > 5:
                # Remove 2-3 random nodes
                for _ in range(random.randint(2, 3)):
                    bits = []
                    temp = current_mask
                    while temp:
                        lsb = temp & -temp
                        bits.append(lsb.bit_length() - 1)
                        temp ^= lsb
                    if not bits:
                        break
                    idx = random.choice(bits)
                    current_mask &= ~(1 << idx)
                    current_size -= 1
                # Polish
                current_mask = polish(current_mask)
                current_size = bin(current_mask).count('1')
                if current_size > best_size_final:
                    best_size_final = current_size
                    best_mask_final = current_mask
        
        T *= alpha
    
    # Final polish
    best_mask_final = polish(best_mask_final)
    best_size_final = bin(best_mask_final).count('1')
    
    # Convert best_mask_final to list
    result = []
    temp = best_mask_final
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        result.append(valid_nodes[idx])
        temp ^= lsb
        
    if not result:
        result = [1]
        
    return result
