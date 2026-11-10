import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    N = 1000

    def build_counts(S):
        """Build sum-count array from set S."""
        counts = np.zeros(2 * N + 1, dtype=np.int32)
        S_sorted = sorted(S)
        for i in range(len(S_sorted)):
            a = S_sorted[i]
            counts[2 * a] += 1
            for j in range(i + 1, len(S_sorted)):
                b = S_sorted[j]
                counts[a + b] += 1
        return counts

    def can_add_fast(S, counts, x):
        """Check if x can be added to S without violating B2[2]."""
        if counts[2 * x] >= 2:
            return False
        for y in S:
            if counts[x + y] >= 2:
                return False
        return True

    def do_add(S, counts, x):
        """Add x to S and update counts."""
        S.add(x)
        counts[2 * x] += 1
        for y in S:
            if y != x:
                counts[x + y] += 1

    def do_remove(S, counts, x):
        """Remove x from S and update counts."""
        S.discard(x)
        counts[2 * x] -= 1
        for y in S:
            if y != x:
                counts[x + y] -= 1

    def greedy_fill(S, counts, order):
        """Fill S greedily in given order."""
        for x in order:
            if x not in S and can_add_fast(S, counts, x):
                do_add(S, counts, x)

    def local_search_vectorized(S, counts, time_limit):
        """
        Vectorized local search:
        1. Greedy add: Find all addable candidates at once, add the best (smallest index or random).
        2. Swap: If stuck, try removing one element and adding another.
        """
        S_list = sorted(S)
        n = len(S_list)
        
        # Precompute blocked status for all 1..N
        # A candidate x is blocked if counts[2x] >= 2 OR there exists y in S such that counts[x+y] >= 2
        
        while time.time() < time_limit:
            S_list = sorted(S)
            n = len(S_list)
            S_set = set(S_list)
            
            # 1. Try to add any element
            # Vectorized check:
            # For all x in 1..N not in S:
            #   blocked if counts[2*x] >= 2
            #   OR if any y in S has counts[x+y] >= 2
            
            # Efficient check:
            # Create an array of blocked flags
            blocked = np.zeros(N + 1, dtype=bool)
            
            # Condition 1: counts[2*x] >= 2
            # x ranges 1..N, so 2*x ranges 2..2N
            # We only care about x in 1..N
            blocked_candidates_1 = np.where(counts[2:2*N+1:2] >= 2)[0] + 1
            blocked[blocked_candidates_1] = True
            
            # Condition 2: exists y in S such that counts[x+y] >= 2
            # This is equivalent to: for each y in S, mark x such that counts[x+y] >= 2
            # x + y = s => x = s - y
            # s ranges from y+1 to y+N (since x<=N, x>=1)
            # Actually, s ranges from 2 to 2N.
            # For a fixed y, x is valid if 1 <= x <= N and x not in S.
            # We want to know if there EXISTS y in S such that counts[x+y] >= 2.
            
            # Instead of iterating y, let's use the fact that if x is addable, counts[x+y] < 2 for ALL y in S.
            # So x is blocked if there is ANY y in S with counts[x+y] >= 2.
            
            # Let's compute the set of blocked x's due to condition 2.
            # For each y in S, the x's that are blocked are those where counts[x+y] >= 2.
            # This is a bit slow if S is large (~700).
            # Alternative:
            # For each x in 1..N, check if max(counts[x+y] for y in S) >= 2.
            # This is O(N * |S|) which is 1000 * 700 = 700k ops. Acceptable in Python? Maybe.
            # But we can vectorize it.
            
            # Let's create a matrix M of shape (N, |S|) where M[i, j] = counts[S[j] + (i+1)]
            # Then blocked_2[i] = any(M[i, :] >= 2)
            
            if n > 0:
                S_arr = np.array(S_list, dtype=np.int32)
                X_arr = np.arange(1, N + 1, dtype=np.int32)
                # Sums shape (N, n)
                sums = X_arr[:, None] + S_arr[None, :]
                # counts[sums] shape (N, n)
                c_sums = counts[sums]
                # blocked_2[i] is True if any c_sums[i, :] >= 2
                blocked_2 = np.any(c_sums >= 2, axis=1)
                blocked[1:] = blocked | blocked_2
            
            # Exclude elements already in S
            for y in S_list:
                blocked[y] = True
            
            # Find addable candidates
            addable = np.where(~blocked)[0]
            addable = addable[addable > 0]
            
            if len(addable) > 0:
                # Add one (e.g., the first one, or random)
                # To maximize density, maybe adding smaller numbers first is better?
                # Or random to avoid bias.
                x = random.choice(addable.tolist())
                do_add(S, counts, x)
                continue
            
            # 2. Try a swap
            if n > 0:
                # Try removing one element and see if we can add more than 1 (net gain >= 1)
                # To keep it simple, try removing a random element and then greedily adding.
                # Or, identify "conflict" elements.
                
                # Simple heuristic: Remove a random element, try to add as many as possible.
                # If we can add > 1, keep the new set. Else revert.
                
                # To speed up, we can just try a few random removals.
                # But we need to be careful with time.
                
                # Let's try to remove an element that is "expensive" in terms of blocking others.
                # Or just random.
                
                x_remove = random.choice(S_list)
                do_remove(S, counts, x_remove)
                
                # Now try to add greedily
                added_count = 0
                while True:
                    S_list = sorted(S)
                    n = len(S_list)
                    S_set = set(S_list)
                    
                    blocked = np.zeros(N + 1, dtype=bool)
                    blocked_candidates_1 = np.where(counts[2:2*N+1:2] >= 2)[0] + 1
                    blocked[blocked_candidates_1] = True
                    
                    if n > 0:
                        S_arr = np.array(S_list, dtype=np.int32)
                        X_arr = np.arange(1, N + 1, dtype=np.int32)
                        sums = X_arr[:, None] + S_arr[None, :]
                        c_sums = counts[sums]
                        blocked_2 = np.any(c_sums >= 2, axis=1)
                        blocked[1:] = blocked | blocked_2
                    
                    for y in S_list:
                        blocked[y] = True
                    
                    addable = np.where(~blocked)[0]
                    addable = addable[addable > 0]
                    
                    if len(addable) == 0:
                        break
                    
                    x = random.choice(addable.tolist())
                    do_add(S, counts, x)
                    added_count += 1
                    
                    if time.time() > time_limit:
                        break
                
                # Check if we improved
                if added_count > 1:
                    # We added more than we removed (1), so size increased
                    pass
                else:
                    # Revert
                    do_add(S, counts, x_remove)

    best_set = set()
    best_size = 0

    # Strategy 1: Ascending greedy
    S1 = set()
    c1 = np.zeros(2 * N + 1, dtype=np.int32)
    greedy_fill(S1, c1, range(1, N + 1))
    if len(S1) > best_size:
        best_size = len(S1)
        best_set = set(S1)

    # Strategy 2: Descending greedy
    S2 = set()
    c2 = np.zeros(2 * N + 1, dtype=np.int32)
    greedy_fill(S2, c2, range(N, 0, -1))
    if len(S2) > best_size:
        best_size = len(S2)
        best_set = set(S2)

    # Strategy 3: Random order greedy (multiple trials)
    for trial in range(5):
        S3 = set()
        c3 = np.zeros(2 * N + 1, dtype=np.int32)
        order = list(range(1, N + 1))
        random.shuffle(order)
        greedy_fill(S3, c3, order)
        if len(S3) > best_size:
            best_size = len(S3)
            best_set = set(S3)

    # Strategy 4: Dense random init then local search
    S4 = set()
    c4 = np.zeros(2 * N + 1, dtype=np.int32)
    init_candidates = list(range(1, N + 1))
    random.shuffle(init_candidates)
    for x in init_candidates:
        if len(S4) >= 700:
            break
        if can_add_fast(S4, c4, x):
            do_add(S4, c4, x)
    for x in init_candidates:
        if x not in S4 and can_add_fast(S4, c4, x):
            do_add(S4, c4, x)
    if len(S4) > best_size:
        best_size = len(S4)
        best_set = set(S4)

    # Local search on the best solution found
    if time.time() < deadline - 10:
        S_ls = set(best_set)
        c_ls = build_counts(S_ls)
        local_search_vectorized(S_ls, c_ls, deadline - 5)
        if len(S_ls) > best_size:
            best_size = len(S_ls)
            best_set = set(S_ls)

    # If time permits, try local search on other good solutions too
    remaining = deadline - time.time()
    if remaining > 15 and len(S1) >= best_size - 5:
        S_ls2 = set(S1)
        c_ls2 = build_counts(S_ls2)
        local_search_vectorized(S_ls2, c_ls2, min(deadline - 5, time.time() + remaining / 3))
        if len(S_ls2) > best_size:
            best_size = len(S_ls2)
            best_set = set(S_ls2)

    if time.time() < deadline - 5:
        S_ls3 = set(S2)
        c_ls3 = build_counts(S_ls3)
        local_search_vectorized(S_ls3, c_ls3, min(deadline - 5, time.time() + 10))
        if len(S_ls3) > best_size:
            best_size = len(S_ls3)
            best_set = set(S_ls3)

    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(len(res))
