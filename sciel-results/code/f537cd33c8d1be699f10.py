"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin for 90s limit
    
    n = 512
    
    # Precompute all 4-APs and conflict masks
    # For each x, conflict_triples[x] = list of bitmasks for the other 3 elements
    conflict_triples = [[] for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            term4 = a + 3 * d
            
            mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (term4 - 1))
            conflict_triples[a].append(mask_a)
            
            mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (term4 - 1))
            conflict_triples[b].append(mask_b)
            
            mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (term4 - 1))
            conflict_triples[c].append(mask_c)
            
            mask_d = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            conflict_triples[term4].append(mask_d)
    
    conflict_triples = [tuple(lst) for lst in conflict_triples]
    bit_positions = {i: 1 << (i - 1) for i in range(1, n + 1)}
    
    def can_add(x, S_mask):
        """Check if x can be added to S without forming a 4-AP."""
        for mask in conflict_triples[x]:
            if (mask & S_mask) == mask:
                return False
        return True
    
    def try_add_all(S_mask, S_list):
        """Try to add all possible elements. Returns (new_mask, new_list)."""
        for x in range(1, n + 1):
            if not (S_mask & bit_positions[x]):
                if can_add(x, S_mask):
                    S_mask |= bit_positions[x]
                    S_list.append(x)
        return S_mask, S_list
    
    def greedy_fill(order, S_mask=0, S_list=None):
        """Greedy fill following a given order."""
        if S_list is None:
            S_list = []
        for x in order:
            if not (S_mask & bit_positions[x]):
                if can_add(x, S_mask):
                    S_mask |= bit_positions[x]
                    S_list.append(x)
        return S_mask, S_list
    
    def local_search(S_mask, S_list, max_rounds=2000):
        """1-2 exchange local search."""
        no_improve = 0
        max_no_improve = 150
        
        for _ in range(max_rounds):
            if time.time() >= deadline:
                break
            
            if not S_list:
                break
            
            # Try a swap: remove random y from S, add random x not in S
            x = random.randint(1, n)
            if S_mask & bit_positions[x]:
                continue
            
            y = random.choice(S_list)
            
            S_mask_without_y = S_mask & ~bit_positions[y]
            
            if can_add(x, S_mask_without_y):
                S_mask = S_mask_without_y | bit_positions[x]
                idx = S_list.index(y)
                S_list[idx] = x
                
                # Try to add more after swap
                S_mask, S_list = try_add_all(S_mask, S_list)
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > max_no_improve:
                break
        
        return S_mask, S_list
    
    best_mask = 0
    best_size = 0
    best_list = []
    
    all_elements = list(range(1, n + 1))
    
    # Compute AP-count for each element (for ordering)
    ap_counts = [len(conflict_triples[i]) for i in range(n + 1)]
    
    # Define multiple orderings
    orderings = [
        all_elements[:],           # natural
        all_elements[::-1],        # reverse
        sorted(all_elements, key=lambda x: ap_counts[x]),  # fewest APs first
        sorted(all_elements, key=lambda x: -ap_counts[x]), # most APs first
    ]
    
    # Add some random shuffles
    for _ in range(10):
        shuffled = all_elements[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)
    
    # Run multiple rounds with different orderings
    while time.time() < deadline:
        for order in orderings:
            if time.time() >= deadline:
                break
            
            # Greedy fill
            S_mask, S_list = greedy_fill(order)
            
            # Exhaustive add pass
            S_mask, S_list = try_add_all(S_mask, S_list)
            
            current_size = len(S_list)
            if current_size > best_size:
                best_size = current_size
                best_mask = S_mask
                best_list = S_list[:]
            
            # Local search
            S_mask, S_list = local_search(S_mask, S_list)
            S_mask, S_list = try_add_all(S_mask, S_list)
            
            current_size = len(S_list)
            if current_size > best_size:
                best_size = current_size
                best_mask = S_mask
                best_list = S_list[:]
            
            # Generate new random orderings
            shuffled = all_elements[:]
            random.shuffle(shuffled)
            orderings.append(shuffled)
            
            # Keep orderings list manageable
            if len(orderings) > 30:
                orderings = orderings[-30:]
    
    # Final polish
    final_list = []
    for i in range(n):
        if best_mask & (1 << i):
            final_list.append(i + 1)
    
    if final_list:
        temp_mask = best_mask
        temp_list = final_list[:]
        temp_mask, temp_list = try_add_all(temp_mask, temp_list)
        if len(temp_list) > best_size:
            best_size = len(temp_list)
            best_mask = temp_mask
            final_list = temp_list
    
    if not final_list:
        return [1, 2]
        
    return sorted(final_list)
# EVOLVE-BLOCK-END
