import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin

    def generate_sums(S):
        """Generate all unordered triple sums for set S."""
        sums = set()
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(i, n):
                b = S[j]
                for k in range(j, n):
                    c = S[k]
                    sums.add(a + b + c)
        return sums

    def can_add(x, S, sums):
        """Check if x can be added to S without creating duplicate triple sums."""
        # 1. x, x, x
        if 3 * x in sums:
            return False
        
        # 2. x, x, y
        for y in S:
            if 2 * x + y in sums:
                return False
        
        # 3. x, y, z
        n = len(S)
        for i in range(n):
            y = S[i]
            for j in range(i, n):
                z = S[j]
                if x + y + z in sums:
                    return False
        return True

    def add_element(x, S, sums):
        """Add x to S and update sums."""
        S.append(x)
        # Add 3x
        sums.add(3 * x)
        # Add 2x + y
        for y in S[:-1]:
            sums.add(2 * x + y)
        # Add x + y + z
        n_old = len(S) - 1
        for i in range(n_old):
            y = S[i]
            for j in range(i, n_old):
                z = S[j]
                sums.add(x + y + z)

    def greedy_insert(order, S_init, sums_init):
        """Try to insert elements from order into S_init."""
        S = list(S_init)
        sums = set(sums_init)
        S_set = set(S)
        
        for x in order:
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            
            if can_add(x, S, sums):
                add_element(x, S, sums)
                S_set.add(x)
                
        return S

    # 1. Create Base-4 Skeleton
    skeleton = [4**i for i in range(7) if 4**i <= 4096]
    # skeleton = [1, 4, 16, 64, 256, 1024, 4096]
    skeleton_sums = generate_sums(skeleton)
    
    best_set = skeleton
    best_size = len(skeleton)

    # 2. Try Ascending Insertion
    candidates_asc = [x for x in range(1, 4097) if x not in skeleton]
    if time.time() < deadline:
        res = greedy_insert(candidates_asc, skeleton, skeleton_sums)
        if len(res) > best_size:
            best_set = res
            best_size = len(res)

    # 3. Try Descending Insertion
    candidates_desc = sorted(candidates_asc, reverse=True)
    if time.time() < deadline:
        res = greedy_insert(candidates_desc, skeleton, skeleton_sums)
        if len(res) > best_size:
            best_set = res
            best_size = len(res)

    # 4. Try Randomized Insertion
    # We can afford many random restarts because the check is fast
    # Each pass is O(N * K^2) ~ 4000 * 1600 = 6.4M ops. In Python, maybe 1-2s per pass?
    # Let's estimate. 6.4M iterations with set lookups.
    # Python set lookup is fast. 6.4M might take ~0.5s - 1s.
    # We have ~50s left. So ~50 restarts.
    
    random.seed(42)
    for i in range(50):
        if time.time() > deadline - 2.0:
            break
        
        candidates_rand = candidates_asc[:]
        random.shuffle(candidates_rand)
        
        res = greedy_insert(candidates_rand, skeleton, skeleton_sums)
        if len(res) > best_size:
            best_set = res
            best_size = len(res)
            # If we find a significantly better set, we could try to "polish" it
            # by trying to add more elements in a different order, but for now
            # just keep updating best.

    return best_set

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
