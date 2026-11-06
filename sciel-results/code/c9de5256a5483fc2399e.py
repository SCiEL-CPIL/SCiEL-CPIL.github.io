import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin for 90s limit
    
    n = 256
    
    # Precompute 3-APs and 4-APs
    # Represent sets as bitmasks for fast intersection checks
    # Element i (1-indexed) corresponds to bit (i-1)
    
    aps3 = []
    aps4 = []
    
    # Map element to list of AP indices
    elem_to_aps3 = [[] for _ in range(n+1)]
    elem_to_aps4 = [[] for _ in range(n+1)]
    
    # Generate 3-APs
    ap_id = 0
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            if c <= n:
                # Store as tuple of (a, b, c) and bitmask
                mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
                aps3.append((a, b, c, mask))
                elem_to_aps3[a].append(ap_id)
                elem_to_aps3[b].append(ap_id)
                elem_to_aps3[c].append(ap_id)
                ap_id += 1
                
    # Generate 4-APs
    ap_id = 0
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= n:
                mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
                aps4.append((a, b, c, e, mask))
                elem_to_aps4[a].append(ap_id)
                elem_to_aps4[b].append(ap_id)
                elem_to_aps4[c].append(ap_id)
                elem_to_aps4[e].append(ap_id)
                ap_id += 1

    def count_3aps(A_mask):
        count = 0
        for _, _, _, mask in aps3:
            if (A_mask & mask) == mask:
                count += 1
        return count
        
    def count_4aps(A_mask):
        count = 0
        for _, _, _, _, mask in aps4:
            if (A_mask & mask) == mask:
                count += 1
        return count

    def is_valid_add(A_mask, x):
        # Check if adding x creates a 4-AP
        # x is 1-indexed
        bit_x = 1 << (x-1)
        for ap_idx in elem_to_aps4[x]:
            a, b, c, d, mask = aps4[ap_idx]
            # Check if other 3 elements are in A
            others_mask = mask ^ bit_x
            if (A_mask & others_mask) == others_mask:
                return False
        return True

    def run_greedy(seed):
        rng = random.Random(seed)
        A_mask = 0
        A_set = set()
        candidates = list(range(1, n + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            if is_valid_add(A_mask, x):
                A_mask |= (1 << (x-1))
                A_set.add(x)
                
        return A_mask, A_set

    def local_search(A_mask, A_set, time_limit):
        # Steepest ascent local search
        # Moves: Swap (remove u, add v), Add (add v if valid)
        # We will scan all possible moves and pick the best one.
        
        best_score = count_3aps(A_mask)
        no_improvement_count = 0
        
        while time.time() < time_limit:
            best_delta = 0
            best_move = None
            best_u = None
            best_v = None
            best_type = None # 'swap' or 'add'
            
            # Check Add moves
            for v in range(1, n + 1):
                if v in A_set:
                    continue
                if not is_valid_add(A_mask, v):
                    continue
                
                # Calculate delta 3-APs
                bit_v = 1 << (v-1)
                gained = 0
                for ap_idx in elem_to_aps3[v]:
                    a, b, c, mask = aps3[ap_idx]
                    others_mask = mask ^ bit_v
                    if (A_mask & others_mask) == others_mask:
                        gained += 1
                
                if gained > best_delta:
                    best_delta = gained
                    best_move = 'add'
                    best_v = v
                    best_u = None
                    best_type = 'add'
            
            # Check Swap moves
            for u in A_set:
                bit_u = 1 << (u-1)
                for v in range(1, n + 1):
                    if v == u or v in A_set:
                        continue
                    
                    # Check 4-AP validity for swap
                    valid = True
                    bit_v = 1 << (v-1)
                    for ap_idx in elem_to_aps4[v]:
                        a, b, c, d, mask = aps4[ap_idx]
                        others_mask = mask ^ bit_v
                        # The other 3 elements must be in A\{u}
                        # i.e., in A, and not u
                        # (A_mask & others_mask) == others_mask AND (bit_u & others_mask) == 0
                        if (A_mask & others_mask) == others_mask:
                            if (bit_u & others_mask) == 0:
                                valid = False
                                break
                            else:
                                # One of the others is u, so it's removed.
                                # This 4-AP is not formed.
                                pass
                    
                    if not valid:
                        continue
                    
                    # Calculate 3-AP delta
                    lost = 0
                    for ap_idx in elem_to_aps3[u]:
                        a, b, c, mask = aps3[ap_idx]
                        others_mask = mask ^ bit_u
                        # Others must be in A\{u} (which they are, by definition of mask)
                        # But we need to ensure they are NOT v? No, v is not in A yet.
                        # So if others are in A, the AP is lost.
                        if (A_mask & others_mask) == others_mask:
                            lost += 1
                            
                    gained = 0
                    for ap_idx in elem_to_aps3[v]:
                        a, b, c, mask = aps3[ap_idx]
                        others_mask = mask ^ bit_v
                        # Others must be in A\{u}
                        # i.e., in A, and not u
                        if (A_mask & others_mask) == others_mask:
                            if (bit_u & others_mask) == 0:
                                gained += 1
                            else:
                                # One of the others is u, so it's removed.
                                # This 3-AP is not formed.
                                pass
                                
                    delta = gained - lost
                    if delta > best_delta:
                        best_delta = delta
                        best_move = 'swap'
                        best_v = v
                        best_u = u
                        best_type = 'swap'
            
            if best_move is None:
                break
                
            # Apply best move
            if best_type == 'add':
                A_mask |= (1 << (best_v-1))
                A_set.add(best_v)
            else:
                A_mask &= ~(1 << (best_u-1))
                A_mask |= (1 << (best_v-1))
                A_set.remove(best_u)
                A_set.add(best_v)
                
            best_score += best_delta
            no_improvement_count = 0
            
        return A_mask, A_set, best_score

    # Initial Greedy Construction
    best_mask = 0
    best_set = set()
    best_score = 0
    
    # Run a few initializations
    for i in range(20):
        A_mask, A_set = run_greedy(i)
        
        # Ensure maximality: try adding any remaining valid element
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if x not in A_set and is_valid_add(A_mask, x):
                    A_mask |= (1 << (x-1))
                    A_set.add(x)
                    changed = True
        
        score = count_3aps(A_mask)
        if score > best_score:
            best_score = score
            best_mask = A_mask
            best_set = set(A_set)
            
    # Local Search Phase
    # We will run local search starting from best_set
    A_mask, A_set, current_score = local_search(best_mask, best_set, deadline - 2.0)
    
    if current_score > best_score:
        best_score = current_score
        best_mask = A_mask
        best_set = set(A_set)
        
    # Final check
    # Ensure best_set is valid
    if count_4aps(best_mask) != 0:
        # Fallback to greedy if local search broke it (shouldn't happen)
        best_mask, best_set = run_greedy(0)
        best_score = count_3aps(best_mask)
        
    return sorted(best_set)

# EVOLVE-BLOCK-END
