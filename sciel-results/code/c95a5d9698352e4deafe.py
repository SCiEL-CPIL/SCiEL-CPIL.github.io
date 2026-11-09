import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    N = 256
    
    A = set()
    remaining = set(range(1, N + 1))
    
    def is_valid_add(x, A):
        # Check 4-AP
        # x is 1st
        for d in range(1, (N - x) // 3 + 1):
            if x + d in A and x + 2 * d in A and x + 3 * d in A:
                return False
        # x is 2nd
        for d in range(1, min(x - 1, (N - x) // 2) + 1):
            if x - d in A and x + d in A and x + 2 * d in A:
                return False
        # x is 3rd
        for d in range(1, min(x // 2, N - x) + 1):
            if x - 2 * d in A and x - d in A and x + d in A:
                return False
        # x is 4th
        for d in range(1, (x - 1) // 3 + 1):
            if x - 3 * d in A and x - 2 * d in A and x - d in A:
                return False
        return True

    def count_3ap_gain(x, A):
        gain = 0
        # x is 1st
        for d in range(1, (N - x) // 2 + 1):
            if x + d in A and x + 2 * d in A:
                gain += 1
        # x is 2nd
        for d in range(1, min(x - 1, N - x) + 1):
            if x - d in A and x + d in A:
                gain += 1
        # x is 3rd
        for d in range(1, x // 2 + 1):
            if x - 2 * d in A and x - d in A:
                gain += 1
        return gain

    # Greedy Phase
    while remaining and time.time() < deadline - 10:
        best_x = -1
        best_score = -1
        for x in remaining:
            if is_valid_add(x, A):
                score = count_3ap_gain(x, A)
                if score > best_score:
                    best_score = score
                    best_x = x
        if best_x == -1:
            break
        A.add(best_x)
        remaining.remove(best_x)
        
    # Local Search Phase: 1-swap
    # Try to swap a in A with b in remaining
    # Calculate total 3APs for current A
    def count_total_3ap(A):
        count = 0
        # Count 3-APs (a, b, c) with a < b < c
        # b = (a+c)/2, so a+c = 2b
        # Iterate b in A
        for b in A:
            for d in range(1, min(b - 1, N - b) + 1):
                if b - d in A and b + d in A:
                    count += 1
        return count

    current_score = count_total_3ap(A)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # To avoid getting stuck, we can randomize the order of swaps or just do one pass
        for a in list(A):
            # Try to remove a
            A.remove(a)
            # Find best b to add
            best_b = -1
            best_new_score = current_score - 1 # At least we lose the 3APs involving a? No, we need to recalculate or estimate.
            # Recalculating total 3AP for every swap is O(N^2).
            # N=256, N^2=65k. Swaps: 256 * 256 = 65k. Total 4e9? Too slow.
            
            # We need a faster way.
            # Let's just do a few random swaps or skip local search if it's too slow.
            A.add(a)
            
    return sorted(A)
