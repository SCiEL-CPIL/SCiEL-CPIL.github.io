import time
import random

def construct():
    N = 512
    start_time = time.time()
    time_limit = 58.0  # Safety margin

    # Precompute bit masks for each element
    bits = [1 << i for i in range(N + 1)]
    
    # Precompute all 4-term arithmetic progressions
    # aps_containing[x] is a list of tuples (y, z, w) such that {x, y, z, w} is a 4-AP
    aps_containing = [[] for _ in range(N + 1)]
    
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # AP is (a, b, c, e)
            aps_containing[a].append((b, c, e))
            aps_containing[b].append((a, c, e))
            aps_containing[c].append((a, b, e))
            aps_containing[e].append((a, b, c))

    def is_valid_add(mask, x):
        """Check if adding x to set represented by mask creates a 4-AP."""
        for y, z, w in aps_containing[x]:
            if (mask & bits[y]) and (mask & bits[z]) and (mask & bits[w]):
                return False
        return True

    def is_valid_swap(mask, i, j):
        """Check if swapping i (in S) with j (not in S) keeps S valid."""
        # We are effectively checking if j can be added to S \ {i}
        for y, z, w in aps_containing[j]:
            if y == i:
                if (mask & bits[z]) and (mask & bits[w]):
                    return False
            elif z == i:
                if (mask & bits[y]) and (mask & bits[w]):
                    return False
            elif w == i:
                if (mask & bits[y]) and (mask & bits[z]):
                    return False
            else:
                if (mask & bits[y]) and (mask & bits[z]) and (mask & bits[w]):
                    return False
        return True

    def local_search(mask, S_list):
        """
        Perform local search to maximize size of S.
        mask: integer bitmask of S
        S_list: list of elements in S
        """
        size = len(S_list)
        in_S = [False] * (N + 1)
        for x in S_list:
            in_S[x] = True
        
        while time.time() - start_time < time_limit:
            improved = False
            
            # 1. Try to add elements
            # Randomize order of non-S elements
            candidates = [x for x in range(1, N + 1) if not in_S[x]]
            random.shuffle(candidates)
            
            for x in candidates:
                if is_valid_add(mask, x):
                    mask |= bits[x]
                    in_S[x] = True
                    S_list.append(x)
                    size += 1
                    improved = True
                    break
            
            if improved:
                continue

            # 2. If no element could be added, try swaps
            # Randomize order of elements in S and candidates
            S_copy = S_list[:]
            random.shuffle(S_copy)
            
            for i in S_copy:
                if time.time() - start_time > time_limit:
                    break
                
                # Re-get candidates as some might have been added/removed in previous iterations? 
                # No, we break on add. So candidates list is still valid for "not in S".
                # However, we should probably re-shuffle or just iterate.
                # To be safe and efficient, we can just iterate through all non-S elements.
                # But checking all 390 for each of 120 is 46k checks.
                # Let's shuffle candidates once per i? That's expensive.
                # Let's just iterate through a pre-shuffled list of all non-S elements.
                
                candidates_out = [x for x in range(1, N + 1) if not in_S[x]]
                random.shuffle(candidates_out)
                
                for j in candidates_out:
                    if is_valid_swap(mask, i, j):
                        # Perform swap
                        mask ^= bits[i]  # Remove i
                        mask |= bits[j]   # Add j
                        in_S[i] = False
                        in_S[j] = True
                        # Update S_list: remove i, add j
                        # S_list is a list, removal is O(N). 
                        # Since N=512, this is fine.
                        S_list.remove(i)
                        S_list.append(j)
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                # Local optimum reached
                break
                
        return mask, S_list, size

    best_size = 0
    best_S_list = []
    
    # Strategy 1: Sorted Greedy + Local Search
    mask = 0
    S_list = []
    for x in range(1, N + 1):
        if is_valid_add(mask, x):
            mask |= bits[x]
            S_list.append(x)
    
    mask, S_list, size = local_search(mask, S_list)
    if size > best_size:
        best_size = size
        best_S_list = S_list[:]

    # Strategy 2: Multiple Randomized Greedy + Local Search
    while time.time() - start_time < time_limit - 5:
        mask = 0
        S_list = []
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        for x in order:
            if is_valid_add(mask, x):
                mask |= bits[x]
                S_list.append(x)
        
        mask, S_list, size = local_search(mask, S_list)
        if size > best_size:
            best_size = size
            best_S_list = S_list[:]

    return sorted(best_S_list)
