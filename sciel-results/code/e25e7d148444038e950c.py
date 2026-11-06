# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Construct two nonempty Sidon subsets A, B of [1, 256] with at most 64 elements each.
    Pair sums inside each set (including a+a) must be unique.
    Nonzero difference sets of A and B must be disjoint.
    Maximize |A|+|B|.
    """
    start_time = time.time()
    DEADLINE = 50.0  # safety margin before 60s hard kill
    
    N = 256
    MAX_SIZE = 64
    
    def _can_add_to_set(elements, sums, diffs, other_diffs, x):
        """Fast check if x can be added to a Sidon set."""
        if (2 * x) in sums:
            return False
        for e in elements:
            if (e + x) in sums:
                return False
            d = abs(e - x)
            if d in diffs or d in other_diffs:
                return False
        return True
    
    def _add_to_set(elements, sums, diffs, x):
        """Add x to set, updating sums and diffs. Returns new structures."""
        new_elements = elements + [x]
        new_sums = set(sums)
        new_diffs = set(diffs)
        for e in elements:
            new_sums.add(e + x)
            new_diffs.add(abs(e - x))
        new_sums.add(2 * x)
        return new_elements, new_sums, new_diffs
    
    def _remove_from_set(elements, sums, diffs, x):
        """Remove x from set, returning new structures."""
        new_elements = [e for e in elements if e != x]
        new_sums = set()
        new_diffs = set()
        for i in range(len(new_elements)):
            for j in range(i, len(new_elements)):
                new_sums.add(new_elements[i] + new_elements[j])
            for j in range(i + 1, len(new_elements)):
                new_diffs.add(abs(new_elements[i] - new_elements[j]))
        return new_elements, new_sums, new_diffs
    
    def _greedy_interleave(order_a, order_b, balance_factor=1.0):
        """
        Interleaved greedy: alternately try to add to A and B.
        balance_factor > 1 favors A, < 1 favors B.
        """
        A = []
        B = []
        sums_a = set()
        sums_b = set()
        diffs_a = set()
        diffs_b = set()
        
        idx_a = 0
        idx_b = 0
        prefer_a = True
        
        while (idx_a < len(order_a) or idx_b < len(order_b)) and \
              len(A) < MAX_SIZE and len(B) < MAX_SIZE:
            if time.time() - start_time > DEADLINE:
                break
            
            if prefer_a:
                if idx_a < len(order_a) and len(A) < MAX_SIZE:
                    x = order_a[idx_a]
                    idx_a += 1
                    if _can_add_to_set(A, sums_a, diffs_a, diffs_b, x):
                        A, sums_a, diffs_a = _add_to_set(A, sums_a, diffs_a, x)
                    prefer_a = False
                else:
                    prefer_a = False
            else:
                if idx_b < len(order_b) and len(B) < MAX_SIZE:
                    x = order_b[idx_b]
                    idx_b += 1
                    if _can_add_to_set(B, sums_b, diffs_b, diffs_a, x):
                        B, sums_b, diffs_b = _add_to_set(B, sums_b, diffs_b, x)
                    prefer_a = True
                else:
                    prefer_a = True
        
        # Finish remaining elements
        while idx_a < len(order_a) and len(A) < MAX_SIZE:
            if time.time() - start_time > DEADLINE:
                break
            x = order_a[idx_a]
            idx_a += 1
            if _can_add_to_set(A, sums_a, diffs_a, diffs_b, x):
                A, sums_a, diffs_a = _add_to_set(A, sums_a, diffs_a, x)
        
        while idx_b < len(order_b) and len(B) < MAX_SIZE:
            if time.time() - start_time > DEADLINE:
                break
            x = order_b[idx_b]
            idx_b += 1
            if _can_add_to_set(B, sums_b, diffs_b, diffs_a, x):
                B, sums_b, diffs_b = _add_to_set(B, sums_b, diffs_b, x)
        
        return A, B, len(A) + len(B)
    
    def _greedy_sequential(order_a, order_b):
        """Sequential greedy: build A first, then B."""
        A = []
        B = []
        sums_a = set()
        sums_b = set()
        diffs_a = set()
        diffs_b = set()
        
        for x in order_a:
            if len(A) >= MAX_SIZE:
                break
            if time.time() - start_time > DEADLINE:
                break
            if _can_add_to_set(A, sums_a, diffs_a, diffs_b, x):
                A, sums_a, diffs_a = _add_to_set(A, sums_a, diffs_a, x)
        
        for x in order_b:
            if len(B) >= MAX_SIZE:
                break
            if time.time() - start_time > DEADLINE:
                break
            if _can_add_to_set(B, sums_b, diffs_b, diffs_a, x):
                B, sums_b, diffs_b = _add_to_set(B, sums_b, diffs_b, x)
        
        return A, B, len(A) + len(B)
    
    def _rebuild(elements):
        sums = set()
        diffs = set()
        for i in range(len(elements)):
            for j in range(i, len(elements)):
                sums.add(elements[i] + elements[j])
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                diffs.add(abs(elements[i] - elements[j]))
        return sums, diffs

    def _greedy_fill(A, B, sums_a, sums_b, diffs_a, diffs_b, max_rounds=50):
        """
        Greedy fill: repeatedly scan for addable elements to A or B.
        Adds ALL possible elements per pass, not just the first.
        Returns updated structures and total size.
        """
        A_set = set(A)
        B_set = set(B)
        for _ in range(max_rounds):
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            # Try adding to A
            if len(A) < MAX_SIZE:
                for x in range(1, N + 1):
                    if x in A_set or x in B_set:
                        continue
                    if _can_add_to_set(A, sums_a, diffs_a, diffs_b, x):
                        A.append(x)
                        A_set.add(x)
                        for e in A[:-1]:
                            sums_a.add(e + x)
                            diffs_a.add(abs(e - x))
                        sums_a.add(2 * x)
                        improved = True
            # Try adding to B
            if len(B) < MAX_SIZE:
                for x in range(1, N + 1):
                    if x in A_set or x in B_set:
                        continue
                    if _can_add_to_set(B, sums_b, diffs_b, diffs_a, x):
                        B.append(x)
                        B_set.add(x)
                        for e in B[:-1]:
                            sums_b.add(e + x)
                            diffs_b.add(abs(e - x))
                        sums_b.add(2 * x)
                        improved = True
            if not improved:
                break
        return A, B, sums_a, sums_b, diffs_a, diffs_b, len(A) + len(B)

    def _local_search(A, B, max_rounds=30):
        """
        Local search: greedy fill then try swaps to free up space.
        """
        A = list(A)
        B = list(B)
        sums_a, diffs_a = _rebuild(A)
        sums_b, diffs_b = _rebuild(B)
        
        # First, greedy fill
        A, B, sums_a, sums_b, diffs_a, diffs_b, total = _greedy_fill(
            A, B, sums_a, sums_b, diffs_a, diffs_b, max_rounds=max_rounds)
        
        # Then try swaps
        A_set = set(A)
        B_set = set(B)
        for round_num in range(max_rounds):
            if time.time() - start_time > DEADLINE:
                break
            
            improved = False
            # Try removing from A and re-filling
            if len(A) > 1:
                best_gain = 0
                best_idx = -1
                for i in range(len(A)):
                    A_temp = A[:i] + A[i+1:]
                    sums_a_temp, diffs_a_temp = _rebuild(A_temp)
                    A_temp_set = set(A_temp)
                    # Count addable
                    addable = 0
                    for x in range(1, N + 1):
                        if x in A_temp_set or x in B_set:
                            continue
                        if _can_add_to_set(A_temp, sums_a_temp, diffs_a_temp, diffs_b, x):
                            addable += 1
                    gain = addable - 1
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                if best_idx >= 0 and best_gain > 0:
                    A = A[:best_idx] + A[best_idx+1:]
                    A_set = set(A)
                    sums_a, diffs_a = _rebuild(A)
                    A, B, sums_a, sums_b, diffs_a, diffs_b, total = _greedy_fill(
                        A, B, sums_a, sums_b, diffs_a, diffs_b, max_rounds=5)
                    A_set = set(A)
                    B_set = set(B)
                    improved = True
            
            # Try removing from B and re-filling
            if not improved and len(B) > 1:
                best_gain = 0
                best_idx = -1
                for i in range(len(B)):
                    B_temp = B[:i] + B[i+1:]
                    sums_b_temp, diffs_b_temp = _rebuild(B_temp)
                    B_temp_set = set(B_temp)
                    addable = 0
                    for x in range(1, N + 1):
                        if x in A_set or x in B_temp_set:
                            continue
                        if _can_add_to_set(B_temp, sums_b_temp, diffs_b_temp, diffs_a, x):
                            addable += 1
                    gain = addable - 1
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                if best_idx >= 0 and best_gain > 0:
                    B = B[:best_idx] + B[best_idx+1:]
                    B_set = set(B)
                    sums_b, diffs_b = _rebuild(B)
                    A, B, sums_a, sums_b, diffs_a, diffs_b, total = _greedy_fill(
                        A, B, sums_a, sums_b, diffs_a, diffs_b, max_rounds=5)
                    A_set = set(A)
                    B_set = set(B)
                    improved = True
            
            if not improved:
                break
        
        return A, B, len(A) + len(B)
    
    # Main search loop with random restarts
    best_A = [1]
    best_B = [2]
    best_total = 2
    
    # First, do a deterministic greedy pass (sequential) with fill
    order_a = list(range(1, N + 1))
    order_b = list(range(1, N + 1))
    A, B, total = _greedy_sequential(order_a, order_b)
    if total > 0:
        sums_a, diffs_a = _rebuild(A)
        sums_b, diffs_b = _rebuild(B)
        A, B, sums_a, sums_b, diffs_a, diffs_b, total = _greedy_fill(
            A, B, sums_a, sums_b, diffs_a, diffs_b)
    if total > best_total:
        best_A, best_B, best_total = A, B, total
    
    # Also try interleaved with fill
    A, B, total = _greedy_interleave(order_a, order_b)
    if total > 0:
        sums_a, diffs_a = _rebuild(A)
        sums_b, diffs_b = _rebuild(B)
        A, B, sums_a, sums_b, diffs_a, diffs_b, total = _greedy_fill(
            A, B, sums_a, sums_b, diffs_a, diffs_b)
    if total > best_total:
        best_A, best_B, best_total = A, B, total
    
    # Random restarts with mixed strategies
    random.seed(42)
    restart_count = 0
    for restart in range(2000):
        if time.time() - start_time > DEADLINE:
            break
        restart_count += 1
        
        # Generate random orders with some structure
        order_a = list(range(1, N + 1))
        order_b = list(range(1, N + 1))
        
        # Mix of pure random and partially sorted
        if restart % 3 == 0:
            random.shuffle(order_a)
            random.shuffle(order_b)
        elif restart % 3 == 1:
            # Sort by some random hash
            key_a = {x: random.random() for x in range(1, N + 1)}
            key_b = {x: random.random() for x in range(1, N + 1)}
            order_a.sort(key=lambda x: key_a[x])
            order_b.sort(key=lambda x: key_b[x])
        else:
            # Block shuffle: divide into blocks and shuffle within/blocks
            block_size = random.randint(8, 32)
            blocks_a = [order_a[i:i+block_size] for i in range(0, N, block_size)]
            random.shuffle(blocks_a)
            order_a = [x for block in blocks_a for x in block]
            
            blocks_b = [order_b[i:i+block_size] for i in range(0, N, block_size)]
            random.shuffle(blocks_b)
            order_b = [x for block in blocks_b for x in block]
        
        # Choose strategy
        strategy = random.random()
        if strategy < 0.4:
            A, B, total = _greedy_sequential(order_a, order_b)
        elif strategy < 0.8:
            A, B, total = _greedy_interleave(order_a, order_b)
        else:
            # Try both and take best
            A1, B1, total1 = _greedy_sequential(order_a, order_b)
            A2, B2, total2 = _greedy_interleave(order_a, order_b)
            if total1 >= total2:
                A, B, total = A1, B1, total1
            else:
                A, B, total = A2, B2, total2
        
        if total > best_total or (total >= best_total - 2 and random.random() < 0.3):
            # Apply greedy fill then local search to improve
            if total > 0:
                sums_a, diffs_a = _rebuild(A)
                sums_b, diffs_b = _rebuild(B)
                A, B, sums_a, sums_b, diffs_a, diffs_b, total = _greedy_fill(
                    A, B, sums_a, sums_b, diffs_a, diffs_b)
                A_ls, B_ls, total_ls = _local_search(A, B)
                if total_ls > best_total:
                    best_A, best_B, best_total = A_ls, B_ls, total_ls
                elif total > best_total:
                    best_A, best_B, best_total = A, B, total
    
    # Final local search on best solution
    A_final, B_final, total_final = _local_search(best_A, best_B)
    if total_final > best_total:
        best_A, best_B, best_total = A_final, B_final, total_final
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
