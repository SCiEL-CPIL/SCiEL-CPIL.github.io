import time
import random

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 55
    
    N = 256
    conflict_mask = [0] * (N + 1)
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
    
    def popcount(x):
        return bin(x).count('1')
    
    def bits(x):
        res = []
        while x:
            lsb = x & -x
            res.append(lsb.bit_length() - 1)
            x ^= lsb
        return res

    all_mask = (1 << (N + 1)) - 2
    
    def greedy_construct():
        # Randomized greedy: pick node with min/max degree in remaining
        selected_mask = 0
        remaining_mask = all_mask
        while remaining_mask:
            # Pick node with min degree in remaining to allow more future picks?
            # Or max degree to eliminate many conflicts?
            # For MIS, min degree in remaining is usually better for size.
            best_node = -1
            best_deg = N + 1
            r = remaining_mask
            while r:
                lsb = r & -r
                node = lsb.bit_length() - 1
                deg = popcount(conflict_mask[node] & remaining_mask)
                if deg < best_deg:
                    best_deg = deg
                    best_node = node
                r ^= lsb
            
            if best_node == -1:
                break
                
            selected_mask |= (1 << best_node)
            remaining_mask &= ~(1 << best_node)
            remaining_mask &= ~conflict_mask[best_node]
            
        return selected_mask

    def local_search(start_mask):
        cur_mask = start_mask
        cur_size = popcount(cur_mask)
        removed_mask = all_mask & ~cur_mask
        
        # 1. Add back non-conflicting
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            r = removed_mask
            while r:
                lsb = r & -r
                node = lsb.bit_length() - 1
                if (conflict_mask[node] & cur_mask) == 0:
                    cur_mask |= lsb
                    removed_mask &= ~lsb
                    cur_size += 1
                    improved = True
                r ^= lsb
        
        # 2. Try 1-for-1 swaps
        # We iterate through removed nodes. If a removed node r conflicts with exactly one selected node s,
        # and r does not conflict with any other selected node, we can swap s out and r in.
        # This might allow other nodes to be added later? No, size stays same.
        # But it might enable *other* swaps or add-backs if the graph structure changes?
        # Actually, a swap keeps size constant. It's useful if it allows *subsequent* add-backs?
        # No, add-backs are maximal in step 1.
        # So swaps are only useful if we do multi-step swaps or if we use swaps to reach a local optimum 
        # that allows a *different* set of add-backs? 
        # Wait, if we are at a maximal independent set, we can't add anything.
        # A swap changes the set. The new set might not be maximal, allowing add-backs.
        # So: Swap -> Add-back loop.
        
        # Let's try a few random swaps to escape local optima
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            
            # Pick a random removed node r
            if removed_mask == 0:
                break
            r_bits = bits(removed_mask)
            if not r_bits:
                break
            r_node = random.choice(r_bits)
            r_bit = 1 << r_node
            
            # Find conflicts with cur_mask
            conflicts = conflict_mask[r_node] & cur_mask
            if conflicts == 0:
                # Should have been added in step 1
                cur_mask |= r_bit
                removed_mask &= ~r_bit
                cur_size += 1
                continue
                
            # If exactly one conflict, try swapping
            if conflicts & (conflicts - 1) == 0: # power of 2
                s_bit = conflicts
                s_node = s_bit.bit_length() - 1
                
                # Check if r conflicts with cur_mask without s
                if (conflict_mask[r_node] & (cur_mask ^ s_bit)) == 0:
                    # Perform swap
                    cur_mask ^= s_bit
                    cur_mask |= r_bit
                    removed_mask ^= s_bit
                    removed_mask ^= r_bit
                    # cur_size unchanged
                    
                    # Now try to add back non-conflicting nodes again
                    improved = True
                    while improved and time.time() < deadline - 2:
                        improved = False
                        rem = removed_mask
                        while rem:
                            lsb = rem & -rem
                            node = lsb.bit_length() - 1
                            if (conflict_mask[node] & cur_mask) == 0:
                                cur_mask |= lsb
                                removed_mask &= ~lsb
                                cur_size += 1
                                improved = True
                            rem ^= lsb
        
        return cur_mask, cur_size

    # Initial greedy
    best_mask = greedy_construct()
    best_size = popcount(best_mask)
    
    # Refine
    best_mask, best_size = local_search(best_mask)
    
    # Random restarts
    for i in range(200):
        if time.time() > deadline - 2:
            break
        
        # Perturb
        cur_mask = best_mask
        k = random.randint(1, 5)
        nodes = bits(cur_mask)
        if len(nodes) > k:
            to_remove = random.sample(nodes, k)
            for n in to_remove:
                cur_mask &= ~(1 << n)
        
        refined_mask, refined_size = local_search(cur_mask)
        
        if refined_size > best_size:
            best_size = refined_size
            best_mask = refined_mask
            
    return bits(best_mask)

# EVOLVE-BLOCK-END
