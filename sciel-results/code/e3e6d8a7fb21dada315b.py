import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe within 60s
    
    # 1. Determine valid candidates (diagonal check)
    # 2a must not be a perfect square
    squares = set(k*k for k in range(1, 23)) # max sum 512, sqrt(512) ~ 22.6
    
    candidates = []
    for i in range(1, 257):
        if 2 * i not in squares:
            candidates.append(i)
            
    N = len(candidates)
    
    # 2. Build conflict masks
    # mask[i] is a bitset where bit j is 1 if candidates[i] + candidates[j] is a square
    masks = [0] * N
    for i in range(N):
        ci = candidates[i]
        m = 0
        for j in range(N):
            if i == j:
                continue
            cj = candidates[j]
            if ci + cj in squares:
                m |= (1 << j)
        masks[i] = m
    
    # Precompute degree for each node
    degrees = [masks[i].bit_count() for i in range(N)]
    
    # 3. Greedy initialization with min-degree heuristic
    def greedy_solve():
        remaining = (1 << N) - 1
        selected = 0
        while remaining:
            # Pick node with min degree in remaining
            best_node = -1
            min_deg = float('inf')
            r = remaining
            while r:
                lsb = r & (-r)
                idx = lsb.bit_length() - 1
                deg = (masks[idx] & remaining).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_node = idx
                r ^= lsb
                
            if best_node == -1:
                break
                
            selected |= (1 << best_node)
            # Remove best_node and its neighbors
            remaining &= ~(masks[best_node] | (1 << best_node))
            
        return selected, selected.bit_count()

    best_set, best_size = greedy_solve()
    
    # 4. Local Search with better move selection
    def local_search(current_set, current_size):
        nonlocal best_set, best_size
        
        current = current_set
        current_sz = current_size
        best_local = current
        best_local_sz = current_size
        
        # Phase 1: Try to add any valid node (greedy addition)
        while True:
            if time.time() > deadline:
                break
            
            # Find all nodes that can be added
            addable = []
            for i in range(N):
                bit = 1 << i
                if not (current & bit) and (masks[i] & current) == 0:
                    addable.append(i)
            
            if not addable:
                break
            
            # Add the node with minimum degree (to preserve future options)
            best_add = min(addable, key=lambda i: (masks[i] & ~current).bit_count())
            current |= (1 << best_add)
            current_sz += 1
            
            if current_sz > best_local_sz:
                best_local = current
                best_local_sz = current_sz
            if current_sz > best_size:
                best_size = current_sz
                best_set = current
        
        # Phase 2: Try swaps to improve - more aggressive with 2-swap moves
        max_iterations = 800
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Try to find a beneficial swap
            found_swap = False
            
            # First try single swap: remove v, add u (net 0)
            for _ in range(30):
                # Pick a node not in current
                u_idx = random.randint(0, N - 1)
                u_bit = 1 << u_idx
                if current & u_bit:
                    continue
                
                # Find all v in current such that removing v allows adding u
                r = current
                while r:
                    lsb = r & (-r)
                    v_idx = lsb.bit_length() - 1
                    v_bit = 1 << v_idx
                    
                    if (masks[u_idx] & (current & ~v_bit)) == 0:
                        # Valid swap
                        current = (current & ~v_bit) | u_bit
                        found_swap = True
                        break
                    r ^= lsb
                
                if found_swap:
                    break
            
            if not found_swap:
                # Try 2-swap: remove v1, v2, add u1, u2 (net 0)
                found_2swap = False
                for _ in range(15):
                    # Pick two nodes not in current
                    u1_idx = random.randint(0, N - 1)
                    u1_bit = 1 << u1_idx
                    if current & u1_bit:
                        continue
                    
                    u2_idx = random.randint(0, N - 1)
                    u2_bit = 1 << u2_idx
                    if current & u2_bit or u2_idx == u1_idx:
                        continue
                    
                    # Check if u1 and u2 can coexist
                    if masks[u1_idx] & u2_bit:
                        continue
                    
                    # Find two nodes v1, v2 in current to remove
                    r1 = current
                    while r1 and not found_2swap:
                        lsb1 = r1 & (-r1)
                        v1_idx = lsb1.bit_length() - 1
                        v1_bit = 1 << v1_idx
                        
                        r2 = current & ~v1_bit
                        while r2 and not found_2swap:
                            lsb2 = r2 & (-r2)
                            v2_idx = lsb2.bit_length() - 1
                            v2_bit = 1 << v2_idx
                            
                            new_current = current & ~v1_bit & ~v2_bit
                            # Check if both u1 and u2 can be added
                            if (masks[u1_idx] & new_current) == 0 and (masks[u2_idx] & new_current) == 0:
                                current = new_current | u1_bit | u2_bit
                                found_2swap = True
                                found_swap = True
                            r2 ^= lsb2
                        r1 ^= lsb1
                
                if not found_2swap:
                    # Try double removal + single add (net -1, might enable future gains)
                    found_double = False
                    for _ in range(10):
                        u_idx = random.randint(0, N - 1)
                        u_bit = 1 << u_idx
                        if current & u_bit:
                            continue
                        
                        # Try removing two nodes
                        r1 = current
                        while r1 and not found_double:
                            lsb1 = r1 & (-r1)
                            v1_idx = lsb1.bit_length() - 1
                            v1_bit = 1 << v1_idx
                            
                            r2 = current & ~v1_bit
                            while r2 and not found_double:
                                lsb2 = r2 & (-r2)
                                v2_idx = lsb2.bit_length() - 1
                                v2_bit = 1 << v2_idx
                                
                                new_current = current & ~v1_bit & ~v2_bit
                                if (masks[u_idx] & new_current) == 0:
                                    # Accept with some probability to explore
                                    if random.random() < 0.4:
                                        current = new_current | u_bit
                                        current_sz -= 1
                                        found_double = True
                                r2 ^= lsb2
                            r1 ^= lsb1
                    
                    if not found_double:
                        break
            
            # After any swap, try to greedily add more
            while True:
                if time.time() > deadline:
                    break
                
                addable = []
                for i in range(N):
                    bit = 1 << i
                    if not (current & bit) and (masks[i] & current) == 0:
                        addable.append(i)
                
                if not addable:
                    break
                
                best_add = min(addable, key=lambda i: (masks[i] & ~current).bit_count())
                current |= (1 << best_add)
                current_sz += 1
                
                if current_sz > best_local_sz:
                    best_local = current
                    best_local_sz = current_sz
                if current_sz > best_size:
                    best_size = current_sz
                    best_set = current
        
        # Phase 3: Final greedy addition
        while True:
            if time.time() > deadline:
                break
            
            addable = []
            for i in range(N):
                bit = 1 << i
                if not (current & bit) and (masks[i] & current) == 0:
                    addable.append(i)
            
            if not addable:
                break
            
            best_add = min(addable, key=lambda i: (masks[i] & ~current).bit_count())
            current |= (1 << best_add)
            current_sz += 1
            
            if current_sz > best_local_sz:
                best_local = current
                best_local_sz = current_sz
            if current_sz > best_size:
                best_size = current_sz
                best_set = current
        
        return best_local, best_local_sz

    # Run local search
    best_set, best_size = local_search(best_set, best_size)
    
    # 5. Improved Branch and Bound with tighter upper bound
    if time.time() < deadline - 3.0:
        import sys
        sys.setrecursionlimit(10000)
        
        # Greedy coloring-based upper bound for independent set
        def color_bound(remaining):
            """
            Use greedy coloring on the conflict graph to get an upper bound.
            The number of colors needed gives an upper bound on the independence number.
            """
            if remaining == 0:
                return 0
            
            # Greedy coloring: assign each node the smallest color not used by its neighbors
            color_of = {}
            num_colors = 0
            r = remaining
            while r:
                lsb = r & (-r)
                idx = lsb.bit_length() - 1
                # Find colors used by already-colored neighbors
                neighbor_colors = set()
                for c in color_of:
                    if c != idx and (masks[idx] & (1 << c)):
                        neighbor_colors.add(color_of[c])
                
                # Assign smallest available color
                c = 0
                while c in neighbor_colors:
                    c += 1
                color_of[idx] = c
                num_colors = max(num_colors, c + 1)
                r ^= lsb
            
            return num_colors
        
        def branch_and_bound(remaining, current_size, current_set, depth=0):
            nonlocal best_set, best_size
            
            if time.time() > deadline:
                return
            
            # Pruning with tighter bound
            ub = color_bound(remaining)
            if current_size + ub <= best_size:
                return
            
            if remaining == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set
                return
            
            # Heuristic: Pick a node to branch on - use max degree in remaining
            best_node = -1
            max_deg = -1
            r = remaining
            while r:
                lsb = r & (-r)
                idx = lsb.bit_length() - 1
                deg = (masks[idx] & remaining).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_node = idx
                r ^= lsb
                
            if best_node == -1:
                return
            
            bit = 1 << best_node
            
            # Branch 1: Include best_node (usually more promising)
            new_remaining_1 = remaining & ~(masks[best_node] | bit)
            branch_and_bound(new_remaining_1, current_size + 1, current_set | bit, depth + 1)
            
            # Branch 2: Exclude best_node
            if time.time() > deadline:
                return
            new_remaining_2 = remaining & ~bit
            branch_and_bound(new_remaining_2, current_size, current_set, depth + 1)
        
        # Start branch and bound from the current best set as a starting point
        initial_remaining = (1 << N) - 1
        branch_and_bound(initial_remaining, 0, 0)
    
    # 6. Multiple restarts of local search to escape local optima
    if time.time() < deadline - 5.0:
        for restart in range(5):
            if time.time() > deadline - 2.0:
                break
            
            # Perturb the current best set
            perturbed = best_set
            # Randomly remove some nodes
            num_remove = max(1, best_size // 8)
            for _ in range(num_remove):
                if perturbed == 0:
                    break
                pos = random.randint(0, N - 1)
                if perturbed & (1 << pos):
                    perturbed &= ~(1 << pos)
            
            perturbed_size = perturbed.bit_count()
            
            # Run local search from perturbed state
            perturbed, perturbed_size = local_search(perturbed, perturbed_size)
            
            # If we found something better, update
            if perturbed_size > best_size:
                best_set = perturbed
                best_size = perturbed_size
    
    # 7. Final polish: try to add any remaining valid nodes
    if time.time() < deadline - 1.0:
        for i in range(N):
            bit = 1 << i
            if not (best_set & bit) and (masks[i] & best_set) == 0:
                best_set |= bit
                best_size += 1
    
    # Convert best_set bitset back to list of values
    result = []
    for i in range(N):
        if best_set & (1 << i):
            result.append(candidates[i])
            
    return result

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
