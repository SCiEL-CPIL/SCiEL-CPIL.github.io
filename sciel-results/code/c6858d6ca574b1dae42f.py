# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Construct two nonempty Sidon subsets A, B of [1, 256] with at most 64 elements each.
    Pair sums inside each set (including a+a) must be unique.
    Nonzero difference sets of A and B must be disjoint.
    Maximize |A|+|B|.
    """
    start_time = time.time()
    # Safety margin: 85s out of 90s
    DEADLINE = 85.0
    
    N = 256
    MAX_SIZE = 64

    # Bitmask for differences 1..255
    # Python integers can handle large bitmasks efficiently.
    
    def _check_add(x, elements, used_sums, diff_mask_self, diff_mask_other):
        """
        Check if x can be added to elements.
        elements: list of ints
        used_sums: set of sums
        diff_mask_self: bitmask of differences in current set
        diff_mask_other: bitmask of differences in the other set
        Returns True if valid.
        """
        # Check sums
        if 2 * x in used_sums:
            return False
        for e in elements:
            if e + x in used_sums:
                return False
            d = abs(e - x)
            # Check if this difference is already used in self
            if (diff_mask_self >> d) & 1:
                return False
            # Check if this difference is used in other
            if (diff_mask_other >> d) & 1:
                return False
        return True

    def _apply_add(x, elements, used_sums, diff_mask_self):
        """
        Add x to elements, update used_sums and diff_mask_self.
        Returns new state.
        """
        new_elements = elements + [x]
        new_sums = set(used_sums)
        new_sums.add(2 * x)
        new_diff = diff_mask_self
        for e in elements:
            new_sums.add(e + x)
            d = abs(e - x)
            new_diff |= (1 << d)
        return new_elements, new_sums, new_diff

    def _rebuild_state(elements):
        """Rebuild sums and diff mask from scratch."""
        sums = set()
        diff_mask = 0
        n = len(elements)
        for i in range(n):
            for j in range(i, n):
                sums.add(elements[i] + elements[j])
            for j in range(i + 1, n):
                d = abs(elements[i] - elements[j])
                diff_mask |= (1 << d)
        return sums, diff_mask

    def _greedy_construct(order_a, order_b):
        """
        Greedy construction.
        """
        A = []
        B = []
        sums_a = set()
        sums_b = set()
        diff_a = 0
        diff_b = 0
        
        # Interleave to balance sizes? Or sequential?
        # Let's try sequential A then B, but with a check on remaining capacity.
        # Actually, interleaved often works better to avoid one set blocking the other's diffs.
        
        ia, ib = 0, 0
        while (ia < len(order_a) or ib < len(order_b)) and \
              len(A) < MAX_SIZE and len(B) < MAX_SIZE:
            if time.time() - start_time > DEADLINE:
                break
            
            added_any = False
            
            # Try A
            if ia < len(order_a) and len(A) < MAX_SIZE:
                x = order_a[ia]
                ia += 1
                if _check_add(x, A, sums_a, diff_a, diff_b):
                    A, sums_a, diff_a = _apply_add(x, A, sums_a, diff_a)
                    added_any = True
            
            # Try B
            if ib < len(order_b) and len(B) < MAX_SIZE:
                x = order_b[ib]
                ib += 1
                if _check_add(x, B, sums_b, diff_b, diff_a):
                    B, sums_b, diff_b = _apply_add(x, B, sums_b, diff_b)
                    added_any = True
            
            if not added_any and ia >= len(order_a) and ib >= len(order_b):
                break
                
        return A, B

    def _local_search(A, B):
        """
        Local search to improve size.
        """
        A = list(A)
        B = list(B)
        
        # Initial state
        sums_a, diff_a = _rebuild_state(A)
        sums_b, diff_b = _rebuild_state(B)
        
        no_improve_count = 0
        max_no_improve = 100
        
        while no_improve_count < max_no_improve:
            if time.time() - start_time > DEADLINE:
                break
            
            improved = False
            
            # 1. Try to add elements
            # Randomize order of candidates to avoid bias
            candidates_a = [x for x in range(1, N + 1) if x not in A]
            candidates_b = [x for x in range(1, N + 1) if x not in B]
            random.shuffle(candidates_a)
            random.shuffle(candidates_b)
            
            # Try adding to A
            if len(A) < MAX_SIZE:
                for x in candidates_a:
                    if _check_add(x, A, sums_a, diff_a, diff_b):
                        A, sums_a, diff_a = _apply_add(x, A, sums_a, diff_a)
                        improved = True
                        break
                        
            # Try adding to B
            if not improved and len(B) < MAX_SIZE:
                for x in candidates_b:
                    if _check_add(x, B, sums_b, diff_b, diff_a):
                        B, sums_b, diff_b = _apply_add(x, B, sums_b, diff_b)
                        improved = True
                        break
                        
            # 2. Try Swap: Remove one, Add one
            if not improved:
                # Try swapping in A
                swap_found = False
                if len(A) > 0:
                    # Randomly pick an element to remove
                    idx = random.randint(0, len(A) - 1)
                    orig = A[idx]
                    A_temp = A[:idx] + A[idx+1:]
                    sums_a_temp, diff_a_temp = _rebuild_state(A_temp)
                    
                    # Try to find a replacement
                    candidates_a = [x for x in range(1, N + 1) if x not in A_temp]
                    random.shuffle(candidates_a)
                    for x in candidates_a:
                        if _check_add(x, A_temp, sums_a_temp, diff_a_temp, diff_b):
                            A = A_temp + [x]
                            sums_a, diff_a = _rebuild_state(A)
                            improved = True
                            swap_found = True
                            break
                            
                # Try swapping in B
                if not swap_found and len(B) > 0:
                    idx = random.randint(0, len(B) - 1)
                    orig = B[idx]
                    B_temp = B[:idx] + B[idx+1:]
                    sums_b_temp, diff_b_temp = _rebuild_state(B_temp)
                    
                    candidates_b = [x for x in range(1, N + 1) if x not in B_temp]
                    random.shuffle(candidates_b)
                    for x in candidates_b:
                        if _check_add(x, B_temp, sums_b_temp, diff_b_temp, diff_a):
                            B = B_temp + [x]
                            sums_b, diff_b = _rebuild_state(B)
                            improved = True
                            break

            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                
        return A, B

    # Main Loop
    best_A = []
    best_B = []
    best_total = 0
    
    random.seed(42)
    
    # Number of restarts depends on time budget
    # Estimate time per restart: greedy is fast, local search is slower.
    # Let's do a fixed number of high-quality restarts.
    
    for restart in range(1000):
        if time.time() - start_time > DEADLINE:
            break
            
        # Generate random orders
        order_a = list(range(1, N + 1))
        order_b = list(range(1, N + 1))
        random.shuffle(order_a)
        random.shuffle(order_b)
        
        A, B = _greedy_construct(order_a, order_b)
        
        if len(A) + len(B) > best_total:
            A_ls, B_ls = _local_search(A, B)
            total_ls = len(A_ls) + len(B_ls)
            if total_ls > best_total:
                best_A, best_B, best_total = A_ls, B_ls, total_ls
        else:
            # Even if greedy is worse, local search might improve it?
            # Usually local search improves from a good start.
            # Let's try local search on current if it's close to best?
            # To save time, only local search if greedy result is promising.
            if len(A) + len(B) >= best_total - 2:
                A_ls, B_ls = _local_search(A, B)
                total_ls = len(A_ls) + len(B_ls)
                if total_ls > best_total:
                    best_A, best_B, best_total = A_ls, B_ls, total_ls

    # Final polish on the best solution found
    if best_A and best_B:
        A_final, B_final = _local_search(best_A, best_B)
        if len(A_final) + len(B_final) > best_total:
            best_A, best_B = A_final, B_final

    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    return [best_A, best_B]
# EVOLVE-BLOCK-END
