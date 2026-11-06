import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin
    
    N = 256
    
    # 1. Precomputation
    
    # Generate all 3-APs and 4-APs as bitmasks
    masks_3ap = []
    masks_4ap = []
    
    # three_aps_containing[x] will be a list of masks for 3-APs containing x
    # four_aps_containing[x] will be a list of masks for 4-APs containing x
    three_aps_containing = [[] for _ in range(N + 1)]
    four_aps_containing = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # 3-AP: a, b, c
            if c <= N:
                mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
                masks_3ap.append(mask)
                three_aps_containing[a].append(mask)
                three_aps_containing[b].append(mask)
                three_aps_containing[c].append(mask)
            
            # 4-AP: a, b, c, e
            if e <= N:
                mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
                masks_4ap.append(mask)
                four_aps_containing[a].append(mask)
                four_aps_containing[b].append(mask)
                four_aps_containing[c].append(mask)
                four_aps_containing[e].append(mask)
                
            if a + 3 * d > N:
                break

    # Helper functions
    
    def count_3aps(S):
        count = 0
        for m in masks_3ap:
            if (S & m) == m:
                count += 1
        return count

    def is_4ap_free(S):
        for m in masks_4ap:
            if (S & m) == m:
                return False
        return True

    def gain_add(x, S):
        """Calculate increase in 3-AP count if x is added to S."""
        gain = 0
        bit_x = 1 << (x - 1)
        for m in three_aps_containing[x]:
            # Check if the other two elements are in S
            others = m ^ bit_x
            if (S & others) == others:
                gain += 1
        return gain

    def is_4ap_free_add(x, S):
        """Check if adding x to S creates a 4-AP."""
        bit_x = 1 << (x - 1)
        for m in four_aps_containing[x]:
            others = m ^ bit_x
            if (S & others) == others:
                return False
        return True

    def loss_remove(x, S):
        """Calculate decrease in 3-AP count if x is removed from S."""
        loss = 0
        bit_x = 1 << (x - 1)
        for m in three_aps_containing[x]:
            others = m ^ bit_x
            if (S & others) == others:
                loss += 1
        return loss

    def is_4ap_free_swap(x, y, S):
        """Check if swapping x (remove) and y (add) in S creates a 4-AP.
        S is assumed to be 4-AP-free.
        New set S' = S \ {x} U {y}.
        Any new 4-AP must contain y.
        """
        bit_y = 1 << (y - 1)
        bit_x = 1 << (x - 1)
        
        for m in four_aps_containing[y]:
            # m contains y. Let others = m \ {y}.
            others = m ^ bit_y
            # We need others to be a subset of S'.
            # S' contains S \ {x}.
            # So, (others & S) should be equal to others, 
            # UNLESS x is in others, in which case we need (others \ {x}) to be in S.
            
            if (others & bit_x):
                # x is part of the 4-AP
                # We need the other 2 elements (others without x and y) to be in S
                required = others ^ bit_x
                if (S & required) == required:
                    return False
            else:
                # x is NOT part of the 4-AP
                # We need all 3 elements of others to be in S
                if (S & others) == others:
                    return False
        
        return True

    # 2. Initial Construction (Greedy)
    
    def greedy_init():
        S = 0
        # Start with elements that are "central" or just iterate 1..N
        # Let's try a few random orders and pick the best
        best_S = 0
        best_score = 0
        
        for seed in range(10):
            rng = random.Random(seed)
            order = list(range(1, N + 1))
            rng.shuffle(order)
            S = 0
            for x in order:
                if time.time() - start_time > DEADLINE:
                    break
                if is_4ap_free_add(x, S):
                    S |= (1 << (x - 1))
            
            score = count_3aps(S)
            if score > best_score:
                best_score = score
                best_S = S
                
        return best_S

    S = greedy_init()
    current_score = count_3aps(S)

    # 3. Local Search
    
    def local_search(S, score):
        improved = True
        while improved:
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            
            # Try Additions
            best_gain = 0
            best_add_x = -1
            for x in range(1, N + 1):
                if (S & (1 << (x - 1))) == 0:
                    if is_4ap_free_add(x, S):
                        gain = gain_add(x, S)
                        if gain > best_gain:
                            best_gain = gain
                            best_add_x = x
            
            if best_add_x != -1:
                S |= (1 << (best_add_x - 1))
                score += best_gain
                improved = True
                continue # Restart loop with new S
            
            # Try Swaps
            best_net_gain = 0
            best_swap_x = -1
            best_swap_y = -1
            
            # Optimization: Only consider swaps if we can find a better one
            # Iterate over x in S
            list_x = []
            for x in range(1, N + 1):
                if (S & (1 << (x - 1))) != 0:
                    list_x.append(x)
            
            for x in list_x:
                if time.time() - start_time > DEADLINE:
                    break
                loss_x = loss_remove(x, S)
                
                # Iterate over y not in S
                for y in range(1, N + 1):
                    if (S & (1 << (y - 1))) == 0:
                        if is_4ap_free_swap(x, y, S):
                            gain_y = gain_add(y, S ^ (1 << (x - 1)))
                            net = gain_y - loss_x
                            if net > best_net_gain:
                                best_net_gain = net
                                best_swap_x = x
                                best_swap_y = y
            
            if best_swap_x != -1:
                S = (S ^ (1 << (best_swap_x - 1))) | (1 << (best_swap_y - 1))
                score += best_net_gain
                improved = True
                continue # Restart loop

        return S, score

    S, current_score = local_search(S, current_score)

    # Convert bitmask to list of elements
    result = []
    for x in range(1, N + 1):
        if (S & (1 << (x - 1))) != 0:
            result.append(x)
            
    return result
