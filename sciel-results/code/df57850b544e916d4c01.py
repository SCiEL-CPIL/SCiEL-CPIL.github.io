import time
import random
import sys

def construct():
    N = 256
    MAX_EACH = 64
    DEADLINE = 55.0  # Safety margin for 60s limit
    start_time = time.time()
    
    # Bitmask helpers
    # Sums range from 1+1=2 to 256+256=512. Bit i represents sum i.
    # Diffs range from 1 to 255. Bit i represents diff i.
    
    def get_sum_mask(elems):
        mask = 0
        for i in range(len(elems)):
            for j in range(i, len(elems)):
                s = elems[i] + elems[j]
                mask |= (1 << s)
        return mask

    def get_diff_mask(elems):
        mask = 0
        for i in range(len(elems)):
            for j in range(i+1, len(elems)):
                d = abs(elems[i] - elems[j])
                mask |= (1 << d)
        return mask

    def can_add(elem, elems, sum_mask, diff_mask, other_diff_mask):
        # Check if elem can be added to set 'elems' given current masks
        # 1. Check sum collisions with existing elems and itself
        if (sum_mask >> (elem + elem)) & 1:
            return False
        for e in elems:
            if (sum_mask >> (elem + e)) & 1:
                return False
        
        # 2. Check diff collisions
        # Internal diffs: must not collide with existing diffs in 'elems'
        # External diffs: must not collide with 'other_diff_mask'
        for e in elems:
            d = abs(elem - e)
            if (diff_mask >> d) & 1:
                return False
            if (other_diff_mask >> d) & 1:
                return False
        return True

    def remove_elem(elem, elems, sum_mask, diff_mask):
        # Returns new sum_mask, diff_mask after removing elem
        # Since Sidon sets have unique sums, we can just clear the bits involving elem
        # But wait, if the set was NOT Sidon, this would be wrong. 
        # We assume we only call this on valid Sidon sets.
        for e in elems:
            if e == elem:
                continue
            s = elem + e
            sum_mask &= ~(1 << s)
            d = abs(elem - e)
            diff_mask &= ~(1 << d)
        # Clear elem + elem
        sum_mask &= ~(1 << (elem + elem))
        return sum_mask, diff_mask

    def add_elem(elem, elems, sum_mask, diff_mask):
        # Returns new sum_mask, diff_mask after adding elem
        # Assumes can_add was True
        sum_mask |= (1 << (elem + elem))
        for e in elems:
            s = elem + e
            sum_mask |= (1 << s)
            d = abs(elem - e)
            diff_mask |= (1 << d)
        return sum_mask, diff_mask

    # Greedy Initialization
    A = []
    B = []
    sum_mask_A = 0
    diff_mask_A = 0
    sum_mask_B = 0
    diff_mask_B = 0
    used = set()
    
    # Try to build A and B greedily
    # Interleave or build one then the other?
    # Building one restricts the other. 
    # Let's try to build both simultaneously, picking the one with more "room" or just alternating.
    # Simple strategy: Try to add to A, if fail try B.
    
    all_elements = list(range(1, N + 1))
    random.shuffle(all_elements)
    
    for x in all_elements:
        if len(A) >= MAX_EACH and len(B) >= MAX_EACH:
            break
        # Try A
        if len(A) < MAX_EACH and can_add(x, A, sum_mask_A, diff_mask_A, diff_mask_B):
            A.append(x)
            sum_mask_A, diff_mask_A = add_elem(x, A, sum_mask_A, diff_mask_A)
            used.add(x)
            continue
        # Try B
        if len(B) < MAX_EACH and can_add(x, B, sum_mask_B, diff_mask_B, diff_mask_A):
            B.append(x)
            sum_mask_B, diff_mask_B = add_elem(x, B, sum_mask_B, diff_mask_B)
            used.add(x)
            continue

    best_size = len(A) + len(B)
    best_A = A[:]
    best_B = B[:]
    best_sum_A = sum_mask_A
    best_diff_A = diff_mask_A
    best_sum_B = sum_mask_B
    best_diff_B = diff_mask_B

    # Local Search / ILS
    # We will perform random swaps to explore the space.
    # If a swap increases size, we take it.
    # If a swap maintains size, we take it with some probability (or always, to shuffle).
    # Then we try to add elements.
    
    current_A = A[:]
    current_B = B[:]
    cur_sum_A = sum_mask_A
    cur_diff_A = diff_mask_A
    cur_sum_B = sum_mask_B
    cur_diff_B = diff_mask_B
    cur_used = used.copy()
    
    # To allow adding elements, we need to know which are NOT in current sets
    # But since we swap, the set of used elements changes.
    # It's easier to just check against [1, N] if not in current_A and not in current_B?
    # No, A and B can overlap? "A and B need not themselves be disjoint."
    # So an element can be in both?
    # "Pair sums inside each set... unique."
    # "Nonzero difference sets must be disjoint."
    # If x is in A and B, then x+x is a sum in A and in B.
    # Diffs: if x in A and y in A, |x-y| in Diff(A). If x in B and z in B, |x-z| in Diff(B).
    # If x is in both, it contributes to both.
    # The problem says "Two Sidon subsets... A and B need not themselves be disjoint."
    # So overlap is allowed.
    
    # However, my greedy above used a `used` set, implying disjointness.
    # Let's re-read: "A and B need not themselves be disjoint."
    # This means we CAN put an element in both.
    # But does it help?
    # If x is in A, it uses up diffs relative to other A elements.
    # If x is in B, it uses up diffs relative to other B elements.
    # If x is in both, it uses up diffs in both.
    # It seems unlikely to be beneficial to overlap, as it constrains both sets.
    # However, the constraint is on the *difference sets* being disjoint.
    # Diff(A) and Diff(B) must be disjoint.
    # If x is in A and B, and y is in A (not B), then |x-y| is in Diff(A).
    # If z is in B (not A), then |x-z| is in Diff(B).
    # These are different diffs.
    # So overlap is technically allowed.
    # But for maximizing size, disjoint sets are usually better because they don't "double spend" the element's capacity to form unique sums/diffs?
    # Actually, if x is in A, it adds 1 to |A|. If x is in B, it adds 1 to |B|.
    # Total size increases by 2.
    # But it adds constraints to both.
    # Let's assume disjointness is optimal for now, but allow overlap in the search if it helps?
    # The greedy above enforced disjointness via `used`.
    # Let's relax that. We can try to add x to A even if x is in B.
    
    # Let's restart the greedy without the `used` constraint, just checking validity.
    # But wait, if x is in A, and we try to add x to B, we check can_add(x, B, ..., diff_mask_A).
    # This checks if diffs of x with B elements collide with diffs in A.
    # This is correct.
    
    # Let's re-run greedy allowing overlap.
    A = []
    B = []
    sum_mask_A = 0
    diff_mask_A = 0
    sum_mask_B = 0
    diff_mask_B = 0
    
    for x in all_elements:
        if len(A) >= MAX_EACH and len(B) >= MAX_EACH:
            break
        # Try A
        if len(A) < MAX_EACH and can_add(x, A, sum_mask_A, diff_mask_A, diff_mask_B):
            A.append(x)
            sum_mask_A, diff_mask_A = add_elem(x, A, sum_mask_A, diff_mask_A)
            continue
        # Try B
        if len(B) < MAX_EACH and can_add(x, B, sum_mask_B, diff_mask_B, diff_mask_A):
            B.append(x)
            sum_mask_B, diff_mask_B = add_elem(x, B, sum_mask_B, diff_mask_B)
            continue

    best_size = len(A) + len(B)
    best_A = A[:]
    best_B = B[:]
    best_sum_A = sum_mask_A
    best_diff_A = diff_mask_A
    best_sum_B = sum_mask_B
    best_diff_B = diff_mask_B

    # Local Search
    # We will try to improve the size.
    # Strategy:
    # 1. Try to add a random element to A or B.
    # 2. If fail, try a random swap in A or B.
    # 3. If swap is valid, accept it.
    # 4. Repeat.
    
    current_A = A[:]
    current_B = B[:]
    cur_sum_A = sum_mask_A
    cur_diff_A = diff_mask_A
    cur_sum_B = sum_mask_B
    cur_diff_B = diff_mask_B
    
    # To speed up "try to add", we need a pool of candidates.
    # Candidates are 1..256.
    
    while time.time() - start_time < DEADLINE:
        # Try to increase size
        improved = False
        
        # Try adding to A
        if len(current_A) < MAX_EACH:
            # Pick a random candidate
            # To avoid O(N) scan every time, we can just pick random 1..256
            # and check if it's already in A? 
            # If it is in A, adding it again is not allowed (set).
            # We should pick from elements NOT in A.
            # Since N=256 is small, we can just try a few random ones.
            for _ in range(10):
                x = random.randint(1, N)
                if x in current_A:
                    continue
                if can_add(x, current_A, cur_sum_A, cur_diff_A, cur_diff_B):
                    current_A.append(x)
                    cur_sum_A, cur_diff_A = add_elem(x, current_A, cur_sum_A, cur_diff_A)
                    improved = True
                    break
        
        # Try adding to B
        if len(current_B) < MAX_EACH:
            for _ in range(10):
                x = random.randint(1, N)
                if x in current_B:
                    continue
                if can_add(x, current_B, cur_sum_B, cur_diff_B, cur_diff_A):
                    current_B.append(x)
                    cur_sum_B, cur_diff_B = add_elem(x, current_B, cur_sum_B, cur_diff_B)
                    improved = True
                    break
        
        if improved:
            if len(current_A) + len(current_B) > best_size:
                best_size = len(current_A) + len(current_B)
                best_A = current_A[:]
                best_B = current_B[:]
                best_sum_A = cur_sum_A
                best_diff_A = cur_diff_A
                best_sum_B = cur_sum_B
                best_diff_B = cur_diff_B
            continue
        
        # If no improvement, try a swap to shuffle the state
        # Swap in A
        if len(current_A) > 0:
            # Pick a random element to remove
            idx_a = random.randint(0, len(current_A) - 1)
            a_to_remove = current_A[idx_a]
            
            # Pick a random element to add
            # It must not be in A
            for _ in range(10):
                x_to_add = random.randint(1, N)
                if x_to_add in current_A:
                    continue
                
                # Try swap
                # 1. Remove a_to_remove
                new_sum_A, new_diff_A = remove_elem(a_to_remove, current_A, cur_sum_A, cur_diff_A)
                # 2. Check if x_to_add can be added to the rest of A
                rest_A = current_A[:idx_a] + current_A[idx_a+1:]
                if can_add(x_to_add, rest_A, new_sum_A, new_diff_A, cur_diff_B):
                    # Accept swap
                    current_A = rest_A + [x_to_add]
                    cur_sum_A, cur_diff_A = add_elem(x_to_add, current_A, new_sum_A, new_diff_A)
                    break
                else:
                    # Revert (masks are already local variables, so just discard)
                    pass
        
        # Swap in B
        if len(current_B) > 0:
            idx_b = random.randint(0, len(current_B) - 1)
            b_to_remove = current_B[idx_b]
            
            for _ in range(10):
                x_to_add = random.randint(1, N)
                if x_to_add in current_B:
                    continue
                
                new_sum_B, new_diff_B = remove_elem(b_to_remove, current_B, cur_sum_B, cur_diff_B)
                rest_B = current_B[:idx_b] + current_B[idx_b+1:]
                if can_add(x_to_add, rest_B, new_sum_B, new_diff_B, cur_diff_A):
                    current_B = rest_B + [x_to_add]
                    cur_sum_B, cur_diff_B = add_elem(x_to_add, current_B, new_sum_B, new_diff_B)
                    break
                else:
                    pass

    # Final check and return
    # Ensure best_A and best_B are valid
    # The logic above maintains validity.
    
    return [best_A, best_B]

# EVOLVE-BLOCK-END
