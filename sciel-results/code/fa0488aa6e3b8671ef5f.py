import random
import time

def construct():
    n = 256
    best_size = 0
    best_sets = ([], [])
    
    def try_add(set_list, sum_mask, diff_mask_self, diff_mask_other, x):
        new_sums = 1 << (2 * x)
        new_diffs = 0
        
        for s in set_list:
            new_sums |= 1 << (x + s)
            d1 = x - s
            d2 = s - x
            new_diffs |= 1 << (d1 + 256)
            new_diffs |= 1 << (d2 + 256)
        
        if new_sums & sum_mask:
            return None
        if new_diffs & diff_mask_self:
            return None
        if new_diffs & diff_mask_other:
            return None
            
        return (set_list + [x], sum_mask | new_sums, diff_mask_self | new_diffs)

    def remove_from(set_list, sum_mask, diff_mask_self, diff_mask_other, x):
        # This is hard to do with bitmasks efficiently without rebuilding.
        # Rebuilding is O(k^2). k~30. 900 ops. Acceptable for local search.
        new_set = [s for s in set_list if s != x]
        # Rebuild masks
        new_sum_mask = 0
        new_diff_mask = 0
        for i, s in enumerate(new_set):
            new_sum_mask |= 1 << (2*s)
            for t in new_set[i+1:]:
                new_sum_mask |= 1 << (s+t)
            for t in new_set:
                if t != s:
                    d = s - t
                    new_diff_mask |= 1 << (d + 256)
        return new_set, new_sum_mask, new_diff_mask

    def greedy(order):
        A = []
        B = []
        sum_A = 0
        diff_A = 0
        sum_B = 0
        diff_B = 0
        
        for x in order:
            res = try_add(A, sum_A, diff_A, diff_B, x)
            if res:
                A, sum_A, diff_A = res
            res = try_add(B, sum_B, diff_B, diff_A, x)
            if res:
                B, sum_B, diff_B = res
                
        return A, B, sum_A, diff_A, sum_B, diff_B

    def local_search_1_for_2(A, B, sum_A, diff_A, sum_B, diff_B, time_limit):
        # Try to improve A
        # Get rejected candidates for A
        # We don't have them stored. We can re-derive them or just try all 1..256 not in A.
        
        # To save time, we only do a few iterations.
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try removing each a in A
            for i in range(len(A)):
                a = A[i]
                # Remove a
                new_A, new_sum_A, new_diff_A = remove_from(A, sum_A, diff_A, diff_B, a)
                
                # Try to add two new elements x, y from [1, 256] \ new_A
                # We need to find x, y such that they fit in new_A and each other.
                # And they were not in A? Ideally.
                
                # To be fast, pick random candidates?
                # Or iterate?
                
                # Let's try to add any x, then any y.
                # This is O(N^2) per removal. Too slow.
                
                # Heuristic: Try to add the smallest available numbers?
                # Or just try a random pair.
                
                # For now, skip this complex local search and focus on fast greedy with more restarts and better ordering.
                pass
                
        return A, B, sum_A, diff_A, sum_B, diff_B

    start_time = time.time()
    max_time = 55.0
    
    candidates = list(range(1, n + 1))
    
    while time.time() - start_time < max_time:
        r = random.random()
        if r < 0.2:
            order = sorted(candidates)
        elif r < 0.4:
            order = sorted(candidates, reverse=True)
        else:
            order = candidates[:]
            random.shuffle(order)
            
        A, B, sum_A, diff_A, sum_B, diff_B = greedy(order)
        total = len(A) + len(B)
        
        if total > best_size:
            best_size = total
            best_sets = (A, B)
            
    return [best_sets[0], best_sets[1]]
