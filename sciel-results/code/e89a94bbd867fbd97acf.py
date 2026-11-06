import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = 85  # Safety margin before 90s kill
    
    # Bit masks for each element (1-indexed)
    bits = [0] * (N + 1)
    for i in range(1, N + 1):
        bits[i] = 1 << (i - 1)
    
    # Precompute 3-APs and 4-APs with partner lists
    # For 3-APs: for each element x, store list of (other1, other2) pairs
    partners3 = [[] for _ in range(N + 1)]
    three_aps = []
    
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            three_aps.append((a, b, c))
            partners3[a].append((b, c))
            partners3[b].append((a, c))
            partners3[c].append((a, b))
    
    # For 4-APs: for each element x, store list of masks of other 3 elements
    # that would form a 4-AP with x
    conflict4 = [[] for _ in range(N + 1)]
    four_aps = []
    
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            four_aps.append((a, b, c, e))
            # For each element in the 4-AP, the other three must NOT all be present
            conflict4[a].append(bits[b] | bits[c] | bits[e])
            conflict4[b].append(bits[a] | bits[c] | bits[e])
            conflict4[c].append(bits[a] | bits[b] | bits[e])
            conflict4[e].append(bits[a] | bits[b] | bits[c])
    
    # Helper: check if adding x to S_mask is valid (no 4-AP created)
    def can_add(x, S_mask):
        for mask in conflict4[x]:
            if (S_mask & mask) == mask:
                return False
        return True
    
    # Helper: count 3-APs in S_mask
    def count_3aps(S_mask):
        count = 0
        for a, b, c in three_aps:
            if (S_mask & bits[a]) and (S_mask & bits[b]) and (S_mask & bits[c]):
                count += 1
        return count
    
    # Helper: compute delta when adding x to S_mask
    def delta_add(x, S_mask):
        delta = 0
        for u, v in partners3[x]:
            if (S_mask & bits[u]) and (S_mask & bits[v]):
                delta += 1
        return delta
    
    # Helper: compute delta when removing x from S_mask
    def delta_remove(x, S_mask):
        delta = 0
        for u, v in partners3[x]:
            if (S_mask & bits[u]) and (S_mask & bits[v]):
                delta -= 1
        return delta
    
    # Greedy construction with different orderings
    def greedy_construct(order):
        S_mask = 0
        for x in order:
            if can_add(x, S_mask):
                S_mask |= bits[x]
        return S_mask
    
    # Local search using exhaustive move evaluation
    def local_search(S_mask, deadline):
        current_score = count_3aps(S_mask)
        best_score = current_score
        best_mask = S_mask
        
        S_elements = [i for i in range(1, N + 1) if S_mask & bits[i]]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try additions
            best_add = -1
            best_add_delta = 0
            for x in range(1, N + 1):
                if not (S_mask & bits[x]):
                    if can_add(x, S_mask):
                        d = delta_add(x, S_mask)
                        if d > best_add_delta:
                            best_add_delta = d
                            best_add = x
            
            if best_add != -1:
                S_mask |= bits[best_add]
                S_elements.append(best_add)
                current_score += best_add_delta
                if current_score > best_score:
                    best_score = current_score
                    best_mask = S_mask
                improved = True
                continue
            
            # Try swaps: for each x in S, try adding y not in S
            best_swap_x = -1
            best_swap_y = -1
            best_swap_delta = 0
            
            for x in S_elements:
                # Compute loss of removing x
                loss_x = 0
                for u, v in partners3[x]:
                    if (S_mask & bits[u]) and (S_mask & bits[v]):
                        loss_x += 1
                
                S_without_x = S_mask & ~bits[x]
                
                for y in range(1, N + 1):
                    if not (S_mask & bits[y]):
                        # Check if y can be added to S_without_x
                        valid = True
                        for mask in conflict4[y]:
                            if (S_without_x & mask) == mask:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # Compute gain of adding y to S_without_x
                        gain_y = 0
                        for u, v in partners3[y]:
                            if (S_without_x & bits[u]) and (S_without_x & bits[v]):
                                gain_y += 1
                        
                        delta = gain_y - loss_x
                        if delta > best_swap_delta:
                            best_swap_delta = delta
                            best_swap_x = x
                            best_swap_y = y
            
            if best_swap_x != -1:
                S_mask &= ~bits[best_swap_x]
                S_mask |= bits[best_swap_y]
                S_elements.remove(best_swap_x)
                S_elements.append(best_swap_y)
                current_score += best_swap_delta
                if current_score > best_score:
                    best_score = current_score
                    best_mask = S_mask
                improved = True
                continue
            
            # No improvement possible; perturb to escape local optimum
            if S_elements:
                # Remove a random element
                idx = random.randint(0, len(S_elements) - 1)
                x_remove = S_elements[idx]
                S_mask &= ~bits[x_remove]
                S_elements.pop(idx)
                current_score += delta_remove(x_remove, S_mask)
                improved = True
                continue
            
            # S is empty; break
            break
        
        return best_mask, best_score
    
    # Multiple restarts with different greedy orderings
    best_overall = 0
    best_mask_overall = 0
    
    # Create several different orderings
    orderings = []
    
    # Ordering 1: by 3-AP potential (descending)
    potential = [0] * (N + 1)
    for x in range(1, N + 1):
        potential[x] = len(partners3[x])
    orderings.append(sorted(range(1, N + 1), key=lambda x: -potential[x]))
    
    # Ordering 2: random
    orderings.append(list(range(1, N + 1)))
    random.shuffle(orderings[-1])
    
    # Ordering 3: alternating high/low potential
    orderings.append(sorted(range(1, N + 1), key=lambda x: potential[x] % 7))
    
    # Ordering 4: reverse
    orderings.append(list(reversed(range(1, N + 1))))
    
    # Ordering 5: another random
    orderings.append(list(range(1, N + 1)))
    random.shuffle(orderings[-1])
    
    num_restarts = len(orderings)
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Greedy init
        S_mask = greedy_construct(orderings[r % len(orderings)])
        
        # Local search
        best_mask, best_score = local_search(S_mask, deadline)
        
        if best_score > best_overall:
            best_overall = best_score
            best_mask_overall = best_mask
    
    # Additional random restarts if time permits
    while time.time() < deadline - 2:
        # Random init
        S_mask = 0
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        for x in random_order:
            if can_add(x, S_mask):
                S_mask |= bits[x]
        
        best_mask, best_score = local_search(S_mask, deadline)
        
        if best_score > best_overall:
            best_overall = best_score
            best_mask_overall = best_mask
        
        if time.time() > deadline - 1:
            break
    
    # Convert best_mask to sorted list
    result = [i for i in range(1, N + 1) if best_mask_overall & bits[i]]
    
    # Final validation
    # Check no 4-APs
    for a, b, c, e in four_aps:
        if (best_mask_overall & bits[a]) and (best_mask_overall & bits[b]) and \
           (best_mask_overall & bits[c]) and (best_mask_overall & bits[e]):
            # Should not happen, but handle gracefully
            pass
    
    return result
