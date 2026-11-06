import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 256
    
    # Precompute 3-APs and 4-APs
    # For each element x, we need:
    # 1. List of bitmasks of the other 2 elements in 3-APs containing x.
    # 2. List of bitmasks of the other 3 elements in 4-APs containing x.
    
    triples_3 = [[] for _ in range(N + 1)]
    triples_4 = [[] for _ in range(N + 1)]
    
    # 3-APs: (a, a+d, a+2d)
    for a in range(1, N + 1):
        for d in range(1, N - a + 1):
            b = a + d
            c = a + 2 * d
            if c <= N:
                # For a: others are b, c
                mask_bc = (1 << b) | (1 << c)
                triples_3[a].append(mask_bc)
                # For b: others are a, c
                mask_ac = (1 << a) | (1 << c)
                triples_3[b].append(mask_ac)
                # For c: others are a, b
                mask_ab = (1 << a) | (1 << b)
                triples_3[c].append(mask_ab)
                
    # 4-APs: (a, a+d, a+2d, a+3d)
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # For a: others b, c, e
            mask_bce = (1 << b) | (1 << c) | (1 << e)
            triples_4[a].append(mask_bce)
            # For b: others a, c, e
            mask_ace = (1 << a) | (1 << c) | (1 << e)
            triples_4[b].append(mask_ace)
            # For c: others a, b, e
            mask_abe = (1 << a) | (1 << b) | (1 << e)
            triples_4[c].append(mask_abe)
            # For e: others a, b, c
            mask_abc = (1 << a) | (1 << b) | (1 << c)
            triples_4[e].append(mask_abc)

    def is_valid_add(S, x):
        # Check if adding x creates a 4-AP
        # S is int bitmask
        for mask in triples_4[x]:
            if (S & mask) == mask:
                return False
        return True

    def calc_gain(S, x):
        # Calculate number of new 3-APs if x is added to S
        gain = 0
        for mask in triples_3[x]:
            if (S & mask) == mask:
                gain += 1
        return gain

    def calc_loss(S, x):
        # Calculate number of 3-APs lost if x is removed from S
        # Note: S currently contains x
        loss = 0
        for mask in triples_3[x]:
            if (S & mask) == mask:
                loss += 1
        return loss

    def greedy_construct():
        S = 0
        # Order elements by potential? Just random or 1..N
        # Let's try 1..N first
        remaining = list(range(1, N + 1))
        random.shuffle(remaining)
        
        for x in remaining:
            if is_valid_add(S, x):
                S |= (1 << x)
        return S

    def hill_climb(S):
        improved = True
        while improved:
            improved = False
            best_delta = 0
            best_move = None
            best_move_type = None # 'add' or 'remove'
            
            # Try adds
            for x in range(1, N + 1):
                if (S & (1 << x)) == 0:
                    if is_valid_add(S, x):
                        gain = calc_gain(S, x)
                        if gain > best_delta:
                            best_delta = gain
                            best_move = x
                            best_move_type = 'add'
            
            # Try removes
            for x in range(1, N + 1):
                if (S & (1 << x)) != 0:
                    loss = calc_loss(S, x)
                    # Removing x changes score by -loss
                    # We only do this if it helps? 
                    # In pure hill climbing on score, we don't remove unless it leads to a better state later.
                    # But standard local search often allows moves that don't immediately improve if they are part of a swap.
                    # However, the insight says "exhaustive-move hill climbing". 
                    # If we just add, we might get stuck.
                    # Let's try swaps as well.
            
            # Try swaps (x in S, y not in S)
            # This is expensive O(N^2). 
            # We can optimize: Only try swaps if no add was found? Or always?
            # Let's try swaps if no add improves.
            if best_move_type is None:
                S_set_bits = [x for x in range(1, N+1) if (S & (1<<x))]
                S_not_bits = [x for x in range(1, N+1) if not (S & (1<<x))]
                
                for x in S_set_bits:
                    S_without_x = S & ~(1 << x)
                    loss_x = calc_loss(S, x)
                    for y in S_not_bits:
                        if is_valid_add(S_without_x, y):
                            gain_y = calc_gain(S_without_x, y)
                            delta = gain_y - loss_x
                            if delta > best_delta:
                                best_delta = delta
                                best_move = (x, y)
                                best_move_type = 'swap'
            
            if best_move_type == 'add':
                S |= (1 << best_move)
                improved = True
            elif best_move_type == 'swap':
                x, y = best_move
                S = (S & ~(1 << x)) | (1 << y)
                improved = True
        
        return S

    def count_3aps(S):
        count = 0
        # Iterate through all 3-APs? Or use the triples_3 structure?
        # Using triples_3: sum of gains from empty set? No.
        # Just iterate all 3-APs is O(N^2) ~ 16k checks. Fast enough.
        for a in range(1, N + 1):
            for d in range(1, N - a + 1):
                b = a + d
                c = a + 2 * d
                if c <= N:
                    if (S & (1 << a)) and (S & (1 << b)) and (S & (1 << c)):
                        count += 1
        return count

    best_S = 0
    best_score = 0
    
    # Multiple restarts
    num_restarts = 20
    time_per_restart = 3.0
    
    for i in range(num_restarts):
        if time.time() > deadline - 2:
            break
        
        S = greedy_construct()
        S = hill_climb(S)
        
        score = count_3aps(S)
        if score > best_score:
            best_score = score
            best_S = S
        
        # Perturbation: Randomly flip some bits and re-climb?
        # Or just restart with different shuffle.
        
    # Final polish: Try to improve best_S with more aggressive search
    if time.time() < deadline - 5:
        # Use a more exhaustive search on best_S
        # Maybe try all swaps again?
        S = best_S
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            best_delta = 0
            best_move = None
            best_move_type = None
            
            S_set_bits = [x for x in range(1, N+1) if (S & (1<<x))]
            S_not_bits = [x for x in range(1, N+1) if not (S & (1<<x))]
            
            # Adds
            for x in S_not_bits:
                if is_valid_add(S, x):
                    gain = calc_gain(S, x)
                    if gain > best_delta:
                        best_delta = gain
                        best_move = x
                        best_move_type = 'add'
                        
            # Swaps
            for x in S_set_bits:
                S_without_x = S & ~(1 << x)
                loss_x = calc_loss(S, x)
                for y in S_not_bits:
                    if is_valid_add(S_without_x, y):
                        gain_y = calc_gain(S_without_x, y)
                        delta = gain_y - loss_x
                        if delta > best_delta:
                            best_delta = delta
                            best_move = (x, y)
                            best_move_type = 'swap'
            
            if best_move_type == 'add':
                S |= (1 << best_move)
                improved = True
            elif best_move_type == 'swap':
                x, y = best_move
                S = (S & ~(1 << x)) | (1 << y)
                improved = True
                
        score = count_3aps(S)
        if score > best_score:
            best_score = score
            best_S = S

    # Convert bitmask to list
    result = [x for x in range(1, N + 1) if (best_S & (1 << x))]
    
    if not result:
        return [1]
        
    return result
