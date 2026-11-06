import time
import random
import numpy as np

def construct():
    """
    Constructs a subset A of [0, 128] such that every integer in [128, 256]
    is a sum of two elements of A. Minimizes the largest representation multiplicity.
    
    Strategy:
    1. Use the efficient incremental count update machinery from the current program.
    2. Adopt the structural search approach from the inspiration: 
       - Greedy initialization to ensure coverage.
       - Local search with pruning (removing elements) and swaps.
       - Objective: Minimize Peak, then Sum of Squares (to spread out representations).
       - Acceptance criteria based on peak and sum_sq, with simulated annealing for swaps.
    3. Multiple restarts with different initial states.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    target_min = 128
    target_max = 256
    N = 129  # Range [0, 128]
    R = target_max - target_min + 1  # 129

    # State variables
    # mask: boolean array of length N
    # counts: integer array of length R (sums target_min..target_max)
    # A_list: list of active indices

    def initialize_state(mask_init):
        mask = mask_init.copy()
        A_list = np.where(mask == 1)[0].tolist()
        counts = np.zeros(R, dtype=np.int32)
        
        # Initialize counts
        for i_idx, a in enumerate(A_list):
            for b in A_list[i_idx:]: # b >= a
                s = a + b
                if target_min <= s <= target_max:
                    counts[s - target_min] += 1
                elif s > target_max:
                    # Optimization: if s > target_max, we don't care, but we must be careful
                    # Actually, the loop structure `for b in A_list[i_idx:]` assumes sorted A_list?
                    # No, A_list is just a list. 
                    # Wait, the current program's initialize_state does:
                    # for i_idx, a in enumerate(A_list):
                    #    for b in A_list[i_idx:]:
                    # This assumes A_list is sorted? 
                    # `np.where` returns sorted indices. So A_list is sorted.
                    # If we add/remove, we must maintain sorted order or the counting logic breaks.
                    # The current program's add/remove does NOT maintain sorted order.
                    # Let's check the current program's add_element:
                    # It iterates `for y in A_list`. If A_list is not sorted, does it double count?
                    # No, add_element adds pairs {x, y} for y in A_list.
                    # If A_list is not sorted, and we add x, we add (x,y) for all y.
                    # But we also add (x,x).
                    # The total count for sum s is the number of pairs (a,b) with a+b=s, a<=b.
                    # If we maintain a list, and we add x, we need to add 1 to count[x+y] for all y in A_list.
                    # And 1 to count[2x].
                    # This works regardless of order, AS LONG AS we don't double count.
                    # But `initialize_state` uses `for b in A_list[i_idx:]`. This implies sorted order for O(N^2/2) counting.
                    # If A_list becomes unsorted, `initialize_state` is wrong.
                    # The current program's `add_element` appends to A_list. It does not sort.
                    # So after the first add, A_list is unsorted.
                    # Then `initialize_state` would be wrong if called again? It's only called once per restart.
                    # But `get_score` and updates rely on `counts`.
                    # The updates in `add_element` are correct regardless of order, because they just add 1 for each existing y.
                    # The only issue is if we re-initialize.
                    # However, the `initialize_state` in the current program is only used to set up the initial state.
                    # The subsequent updates are incremental.
                    # So the initial state must be consistent.
                    # Since `np.where` returns sorted, the initial counts are correct.
                    # After modifications, counts are updated incrementally, so they remain correct.
                    # So we don't need to sort A_list for correctness of counts, only for the initial setup if we were to re-initialize.
                    # We don't re-initialize.
                    pass
        
        return mask, counts, A_list

    def get_metrics(counts):
        peak = int(np.max(counts))
        sum_sq = float(np.sum(counts * counts))
        return peak, sum_sq

    def is_valid(counts):
        return bool(np.all(counts > 0))

    def add_element(mask, counts, A_list, x):
        # Add pairs {x, y} for y in A_list
        for y in A_list:
            s = x + y
            if target_min <= s <= target_max:
                counts[s - target_min] += 1
        # Add pair {x, x}
        s = 2 * x
        if target_min <= s <= target_max:
            counts[s - target_min] += 1
        
        mask[x] = 1
        A_list.append(x)

    def remove_element(mask, counts, A_list, x):
        # Remove pairs {x, y} for y in A_list, y != x
        for y in A_list:
            if y == x:
                continue
            s = x + y
            if target_min <= s <= target_max:
                counts[s - target_min] -= 1
        # Remove pair {x, x}
        s = 2 * x
        if target_min <= s <= target_max:
            counts[s - target_min] -= 1
        
        mask[x] = 0
        A_list.remove(x)

    def greedy_construct():
        """
        Construct a valid set A using greedy addition.
        """
        mask = np.zeros(N, dtype=np.bool_)
        counts = np.zeros(R, dtype=np.int32)
        A_list = []
        
        for _ in range(100):  # Max 100 additions
            if is_valid(counts):
                break
            gaps = np.where(counts == 0)[0]
            if len(gaps) == 0:
                break
            
            best_y = None
            best_gain = -1
            
            for y in range(N):
                if mask[y]:
                    continue
                
                gain = 0
                # Check how many gaps this y would cover
                for g in gaps:
                    s = target_min + g
                    a = s - y
                    if 0 <= a < N and (mask[a] or a == y):
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_y = y
            
            if best_y is not None and best_gain > 0:
                add_element(mask, counts, A_list, best_y)
            else:
                # If no single element helps, pick a random one to break tie/stagnation?
                # Or break.
                break
        
        return mask, counts, A_list

    def local_search(mask, counts, A_list, time_budget):
        """
        Perform local search to minimize peak and sum_sq.
        Returns the best (mask, counts, A_list) found.
        """
        best_mask = mask.copy()
        best_counts = counts.copy()
        best_A_list = A_list.copy()
        
        cur_peak, cur_sum_sq = get_metrics(counts)
        best_peak = cur_peak
        best_sum_sq = cur_sum_sq
        
        T = 1.0
        cooling_rate = 0.995
        min_T = 0.01
        no_improve = 0
        max_no_improve = 1000
        
        while time.time() < start_time + time_budget and T > min_T:
            # Move 1: Remove element (Pruning)
            # Try to remove an element if it doesn't break validity
            if len(A_list) > 10:
                # Randomly pick a candidate to remove
                # To be efficient, we can just pick a random element
                x = random.choice(A_list)
                
                # Check if removing x keeps validity
                # We need to check if any count becomes 0
                # We can simulate removal
                new_counts = counts.copy()
                for y in A_list:
                    if y == x:
                        continue
                    s = x + y
                    if target_min <= s <= target_max:
                        new_counts[s - target_min] -= 1
                s = 2 * x
                if target_min <= s <= target_max:
                    new_counts[s - target_min] -= 1
                
                if is_valid(new_counts):
                    new_peak, new_sum_sq = get_metrics(new_counts)
                    if new_peak < best_peak or (new_peak == best_peak and new_sum_sq < best_sum_sq):
                        # Accept removal
                        remove_element(mask, counts, A_list, x)
                        cur_peak, cur_sum_sq = get_metrics(counts)
                        best_peak = cur_peak
                        best_sum_sq = cur_sum_sq
                        best_mask = mask.copy()
                        best_counts = counts.copy()
                        best_A_list = A_list.copy()
                        no_improve = 0
                        continue
                else:
                    # Removal breaks validity, don't accept
                    pass
            
            # Move 2: Swap
            if len(A_list) == 0:
                continue
                
            x = random.choice(A_list)
            # Pick y not in A
            candidates_out = np.where(mask == 0)[0]
            if len(candidates_out) == 0:
                continue
            y = random.choice(candidates_out)
            
            # Simulate swap
            new_counts = counts.copy()
            
            # Remove x
            for a in A_list:
                if a == x:
                    continue
                s = x + a
                if target_min <= s <= target_max:
                    new_counts[s - target_min] -= 1
            s = 2 * x
            if target_min <= s <= target_max:
                new_counts[s - target_min] -= 1
            
            # Add y
            for a in A_list:
                if a == x:
                    continue
                s = y + a
                if target_min <= s <= target_max:
                    new_counts[s - target_min] += 1
            s = 2 * y
            if target_min <= s <= target_max:
                new_counts[s - target_min] += 1
            
            if not is_valid(new_counts):
                no_improve += 1
                if no_improve > max_no_improve:
                    break
                continue
            
            new_peak, new_sum_sq = get_metrics(new_counts)
            
            accepted = False
            if new_peak < cur_peak:
                accepted = True
            elif new_peak == cur_peak:
                if new_sum_sq < cur_sum_sq:
                    accepted = True
                elif random.random() < 0.05:
                    # Allow some random walk
                    accepted = True
            else:
                # Simulated Annealing
                delta_peak = new_peak - cur_peak
                delta_sum_sq = new_sum_sq - cur_sum_sq
                # Weighted delta
                delta = delta_peak * 1000 + delta_sum_sq
                if delta > 0 and T > 0:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        accepted = True
            
            if accepted:
                # Perform swap
                remove_element(mask, counts, A_list, x)
                add_element(mask, counts, A_list, y)
                
                cur_peak, cur_sum_sq = get_metrics(counts)
                no_improve = 0
                
                if cur_peak < best_peak or (cur_peak == best_peak and cur_sum_sq < best_sum_sq):
                    best_peak = cur_peak
                    best_sum_sq = cur_sum_sq
                    best_mask = mask.copy()
                    best_counts = counts.copy()
                    best_A_list = A_list.copy()
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
            
            T *= cooling_rate

        return best_mask, best_counts, best_A_list

    # Main Loop
    best_global_peak = float('inf')
    best_global_sum_sq = float('inf')
    best_global_mask = None
    
    # Initializations
    # 1. Greedy
    mask_g, counts_g, A_list_g = greedy_construct()
    if is_valid(counts_g):
        m, c, l = local_search(mask_g, counts_g, A_list_g, time_budget=20.0)
        p, ss = get_metrics(c)
        if p < best_global_peak or (p == best_global_peak and ss < best_global_sum_sq):
            best_global_peak = p
            best_global_sum_sq = ss
            best_global_mask = m

    # 2. Random Sparse
    rng = np.random.default_rng(42)
    for i in range(5):
        if time.time() > deadline:
            break
        mask_r = np.zeros(N, dtype=np.bool_)
        size = random.randint(30, 50)
        indices = rng.choice(N, size=size, replace=False)
        mask_r[indices] = True
        
        mask, counts, A_list = initialize_state(mask_r)
        
        # Repair if invalid
        if not is_valid(counts):
            # Simple greedy repair
            for _ in range(50):
                if is_valid(counts):
                    break
                gaps = np.where(counts == 0)[0]
                if len(gaps) == 0:
                    break
                best_y = None
                best_gain = -1
                for y in range(N):
                    if mask[y]:
                        continue
                    gain = 0
                    for g in gaps:
                        s = target_min + g
                        a = s - y
                        if 0 <= a < N and (mask[a] or a == y):
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_y = y
                if best_y is not None and best_gain > 0:
                    add_element(mask, counts, A_list, best_y)
                else:
                    break
            
            if not is_valid(counts):
                continue
        
        # Local Search
        time_left = deadline - time.time()
        if time_left < 10:
            break
            
        budget = min(10.0, time_left)
        m, c, l = local_search(mask, counts, A_list, time_budget=budget)
        p, ss = get_metrics(c)
        if p < best_global_peak or (p == best_global_peak and ss < best_global_sum_sq):
            best_global_peak = p
            best_global_sum_sq = ss
            best_global_mask = m

    # Final Result
    if best_global_mask is None:
        # Fallback
        return list(range(64, 129))
        
    result_list = np.where(best_global_mask == 1)[0].tolist()
    
    if not result_list:
        return [128]
        
    return result_list
