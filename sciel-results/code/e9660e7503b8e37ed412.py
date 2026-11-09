import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 256
    # Bitmask representation: bit i corresponds to element i (1-indexed in problem, 0-indexed in bit ops usually, 
    # but here we can use bit i for element i+1 or just bit i for element i. 
    # Let's use bit i for element i (1..256). So bit 1 is element 1, bit 256 is element 256.
    # Python ints can handle this easily.
    
    # Precompute 3-APs
    # A 3-AP is (a, a+d, a+2d). 
    # Mask is (1 << a) | (1 << (a+d)) | (1 << (a+2d))
    three_ap_masks = []
    elem_3ap_masks = [[] for _ in range(N + 1)] # elem_3ap_masks[x] contains masks of 3-APs involving x
    
    for d in range(1, (N - 1) // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            m = (1 << a) | (1 << b) | (1 << c)
            three_ap_masks.append(m)
            elem_3ap_masks[a].append(m)
            elem_3ap_masks[b].append(m)
            elem_3ap_masks[c].append(m)
            
    # Precompute 4-APs
    # A 4-AP is (a, a+d, a+2d, a+3d)
    four_ap_masks = []
    elem_4ap_masks = [[] for _ in range(N + 1)] # elem_4ap_masks[x] contains masks of 4-APs involving x
    
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            m = (1 << a) | (1 << b) | (1 << c) | (1 << e)
            four_ap_masks.append(m)
            elem_4ap_masks[a].append(m)
            elem_4ap_masks[b].append(m)
            elem_4ap_masks[c].append(m)
            elem_4ap_masks[e].append(m)

    def count_3aps(mask):
        count = 0
        for m in three_ap_masks:
            if (mask & m) == m:
                count += 1
        return count

    def is_4ap_free(mask):
        for m in four_ap_masks:
            if (mask & m) == m:
                return False
        return True

    def get_3ap_gain(mask, x):
        """Calculate number of new 3-APs formed by adding x to mask."""
        bit_x = 1 << x
        gain = 0
        for m in elem_3ap_masks[x]:
            # Check if the other two elements of the 3-AP are in mask
            # m contains x. m ^ bit_x removes x.
            if (mask & (m ^ bit_x)) == (m ^ bit_x):
                gain += 1
        return gain

    def get_3ap_loss(mask, x):
        """Calculate number of 3-APs lost by removing x from mask."""
        bit_x = 1 << x
        loss = 0
        for m in elem_3ap_masks[x]:
            # Check if the 3-AP was complete in mask
            if (mask & m) == m:
                loss += 1
        return loss

    def is_valid_add(mask, x):
        """Check if adding x to mask creates a 4-AP."""
        bit_x = 1 << x
        new_mask = mask | bit_x
        for m in elem_4ap_masks[x]:
            if (new_mask & m) == m:
                return False
        return True

    def is_valid_swap(mask, remove_x, add_y):
        """Check if swapping remove_x for add_y maintains 4-AP-free property."""
        bit_x = 1 << remove_x
        bit_y = 1 << add_y
        new_mask = (mask & ~bit_x) | bit_y
        
        # We only need to check 4-APs that involve the changed elements: remove_x or add_y
        # Actually, removing an element cannot create a new 4-AP. 
        # Adding an element can only create 4-APs that involve that element.
        # So we only need to check 4-APs involving add_y.
        for m in elem_4ap_masks[add_y]:
            if (new_mask & m) == m:
                return False
        return True

    def greedy_seed(ordering):
        mask = 0
        for x in ordering:
            if is_valid_add(mask, x):
                mask |= (1 << x)
        return mask

    def local_search(mask):
        """Try to improve mask by single additions or swaps."""
        improved = True
        current_score = count_3aps(mask)
        
        while improved and time.time() < deadline - 5.0:
            improved = False
            best_move = None
            best_score = current_score
            best_new_mask = mask
            
            # 1. Try adding a single element
            for y in range(1, N + 1):
                if (mask & (1 << y)) == 0:
                    if is_valid_add(mask, y):
                        gain = get_3ap_gain(mask, y)
                        new_score = current_score + gain
                        if new_score > best_score:
                            best_score = new_score
                            best_move = ('add', y, 0)
                            best_new_mask = mask | (1 << y)
            
            # 2. Try swapping: remove x, add y
            # To speed up, we can limit the search or just do it all.
            # Number of elements in mask is roughly N/4 to N/3.
            # Number of candidates to add is N - len(mask).
            # Total pairs is ~ 60 * 200 = 12000 checks. 
            # Each check involves O(1) validity check (if optimized) or small loop.
            
            elems_in_mask = [x for x in range(1, N + 1) if (mask & (1 << x)) != 0]
            
            for x in elems_in_mask:
                # If removing x doesn't lose any 3-APs, or if we gain more by adding y, it's good.
                # Loss of 3-APs by removing x
                loss_x = get_3ap_loss(mask, x)
                
                for y in range(1, N + 1):
                    if (mask & (1 << y)) != 0:
                        continue
                    if y == x:
                        continue
                    
                    # Check validity of swap
                    if not is_valid_swap(mask, x, y):
                        continue
                    
                    gain_y = get_3ap_gain(mask & ~(1 << x), y)
                    # Note: gain_y is calculated on the mask WITHOUT x. 
                    # This is correct because the 3-APs formed by y might involve x, 
                    # but since x is removed, they can't be formed.
                    # However, get_3ap_gain checks if the *other two* are in the mask.
                    # If one of the other two was x, it's not in mask & ~(1<<x), so it won't count. Correct.
                    
                    new_score = current_score - loss_x + gain_y
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_move = ('swap', x, y)
                        best_new_mask = (mask & ~(1 << x)) | (1 << y)
            
            if best_move is not None:
                mask = best_new_mask
                current_score = best_score
                improved = True
        
        return mask, current_score

    best_mask = 0
    best_score = 0
    
    # Run multiple restarts with different random orderings
    num_restarts = 50 # Adjust based on time
    
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Create a random ordering
        elems = list(range(1, N + 1))
        random.shuffle(elems)
        
        # Greedy seed
        seed_mask = greedy_seed(elems)
        
        # Local search
        refined_mask, score = local_search(seed_mask)
        
        if score > best_score:
            best_score = score
            best_mask = refined_mask

    # Convert best_mask to sorted list
    result = [x for x in range(1, N + 1) if (best_mask & (1 << x)) != 0]
    
    return result

# EVOLVE-BLOCK-END
