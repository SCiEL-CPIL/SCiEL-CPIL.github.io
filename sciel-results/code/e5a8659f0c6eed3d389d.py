import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin for 90s limit
    
    n = 512
    
    # Precompute all 4-APs
    # For each x, store a list of masks representing the other 3 elements of APs containing x
    conflict_triples = [[] for _ in range(n + 1)]
    
    # Generate all 4-APs (a, a+d, a+2d, a+3d)
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            term4 = a + 3 * d
            
            # For each element in the AP, the other 3 form the conflict mask
            mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (term4 - 1))
            conflict_triples[a].append(mask_a)
            
            mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (term4 - 1))
            conflict_triples[b].append(mask_b)
            
            mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (term4 - 1))
            conflict_triples[c].append(mask_c)
            
            mask_d = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            conflict_triples[term4].append(mask_d)
            
    # Convert to tuples for faster iteration
    conflict_triples = [tuple(lst) for lst in conflict_triples]
    
    bit_positions = {i: 1 << (i - 1) for i in range(1, n + 1)}
    
    def can_add(x, S_mask):
        """Check if x can be added to S without forming a 4-AP."""
        for mask in conflict_triples[x]:
            if (mask & S_mask) == mask:
                return False
        return True

    def greedy_fill(S_mask, S_list, order):
        """Try to add all possible elements in the given order."""
        added_any = False
        for x in order:
            if not (S_mask & bit_positions[x]):
                if can_add(x, S_mask):
                    S_mask |= bit_positions[x]
                    S_list.append(x)
                    added_any = True
        return S_mask, S_list, added_any

    best_mask = 0
    best_size = 0
    all_elements = list(range(1, n + 1))
    
    # Run multiple restarts to find a good solution
    # Each restart: Greedy + Simulated Annealing
    while time.time() < deadline:
        # 1. Greedy Construction with random shuffle
        S_mask = 0
        S_list = []
        candidates = all_elements[:]
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, S_mask):
                S_mask |= bit_positions[x]
                S_list.append(x)
        
        current_size = len(S_list)
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
            
        # 2. Simulated Annealing Phase
        # Maintain a list of elements NOT in S for faster random selection
        not_in_S = [x for x in all_elements if not (S_mask & bit_positions[x])]
        
        T = 1.0  # Initial temperature
        T_min = 0.01
        cooling_rate = 0.995
        max_no_improvement = 5000
        no_improvement_count = 0
        best_in_sa = current_size
        best_mask_in_sa = S_mask
        
        while time.time() < deadline and T > T_min:
            # Try to add an element first (always accept if possible)
            if not_in_S:
                added = False
                for _ in range(5):
                    idx = random.randint(0, len(not_in_S) - 1)
                    x = not_in_S[idx]
                    if can_add(x, S_mask):
                        S_mask |= bit_positions[x]
                        S_list.append(x)
                        not_in_S.pop(idx)
                        current_size += 1
                        no_improvement_count = 0
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = S_mask
                        if current_size > best_in_sa:
                            best_in_sa = current_size
                            best_mask_in_sa = S_mask
                        added = True
                        break
                
                if added:
                    # Cool down and continue
                    T *= cooling_rate
                    continue
                
            # If we couldn't add, try a swap
            # Try a swap: remove y from S, add x from not_in_S
            if S_list and not_in_S:
                y_idx = random.randint(0, len(S_list) - 1)
                y = S_list[y_idx]
                
                # Pick random x not in S to add
                swap_made = False
                for _ in range(10):
                    x_idx = random.randint(0, len(not_in_S) - 1)
                    x = not_in_S[x_idx]
                    
                    # Check if x can be added to S \ {y}
                    S_mask_without_y = S_mask & ~bit_positions[y]
                    if can_add(x, S_mask_without_y):
                        # Perform swap
                        S_mask = S_mask_without_y | bit_positions[x]
                        S_list[y_idx] = x
                        not_in_S.pop(x_idx)
                        not_in_S.append(y)
                        swap_made = True
                        no_improvement_count = 0
                        break
                
                if not swap_made:
                    no_improvement_count += 1
                    if no_improvement_count > max_no_improvement:
                        break
            else:
                # No more elements to add or swap, break out of SA
                break
                
            # Cool down
            T *= cooling_rate
            
        # After SA, restore the best solution found during SA
        if best_in_sa > current_size:
            S_mask = best_mask_in_sa
            S_list = []
            for i in range(n):
                if S_mask & (1 << i):
                    S_list.append(i + 1)
            current_size = best_in_sa
            
        # Final add pass after local search
        S_mask, S_list, _ = greedy_fill(S_mask, S_list, all_elements)
        current_size = len(S_list)
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask

    # Final polish: try to add any remaining elements to the best solution
    final_list = []
    for i in range(n):
        if best_mask & (1 << i):
            final_list.append(i + 1)
    
    if not final_list:
        return [1, 2]
        
    return final_list
