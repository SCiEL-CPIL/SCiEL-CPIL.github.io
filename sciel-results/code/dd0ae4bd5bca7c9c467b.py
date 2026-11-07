import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill
    
    n = 128
    # Seed solution: all numbers from 65 to 128 (64 numbers)
    best_size = 64
    best_set = set(range(65, 129))
    
    # Strategy: Model as forbidden 3-uniform hypergraph
    # A triple {a,b,c} is forbidden if a+b, a+c, b+c are all in the set
    # We want maximum independent set (no forbidden triple fully contained)
    
    # Key insight: if we only use large numbers (say > n/2), then sums exceed n
    # So all sums of pairs from {65..128} are > 128, hence not in the universe
    # This means the seed solution is trivially valid.
    
    # To do better, we need to include some smaller numbers but exclude some larger
    # to avoid creating forbidden triples.
    
    # Approach: Branch and bound with bitset representation
    # Universe: {1, ..., 128}
    # Represent set as a 128-bit integer (or Python int with bits)
    
    def bit(i):
        return 1 << i
    
    def has_set(s, i):
        return (s >> i) & 1
    
    def add_to_set(s, i):
        return s | bit(i)
    
    def remove_from_set(s, i):
        return s & ~bit(i)
    
    def popcount(s):
        return bin(s).count('1')
    
    # Precompute for each pair (i,j) with i<j, the forbidden element k=i+j
    # A triple {a,b,c} is forbidden if a+b in S, a+c in S, b+c in S
    # Note: a,b,c are elements of S, and a+b, a+c, b+c must also be in S
    # So we need to check: for a,b,c in S (distinct), are a+b, a+c, b+c all in S?
    
    # Let's use a different approach: greedy + local search with fast validation
    
    def is_valid_set(s_set):
        """Check if a set (Python set) is valid - no forbidden triple"""
        for a in s_set:
            for b in s_set:
                if a >= b:
                    continue
                for c in s_set:
                    if b >= c:
                        continue
                    if (a + b) in s_set and (a + c) in s_set and (b + c) in s_set:
                        return False
        return True
    
    def is_valid_set_fast(s_set):
        """Faster validity check using bitset"""
        # For each pair (a,b) with a<b, if a+b is in set, then for any c > b,
        # we need to check that NOT (a+c in set AND b+c in set)
        # This is still O(n^3) worst case but with early termination
        s_list = sorted(s_set)
        n = len(s_list)
        for i in range(n):
            a = s_list[i]
            for j in range(i+1, n):
                b = s_list[j]
                ab = a + b
                if ab > 128 or ab not in s_set:
                    continue
                for k in range(j+1, n):
                    c = s_list[k]
                    ac = a + c
                    bc = b + c
                    if ac <= 128 and bc <= 128 and ac in s_set and bc in s_set:
                        return False
        return True
    
    # Greedy construction: start with all 128 numbers, remove elements that create conflicts
    def greedy_remove():
        s = set(range(1, 129))
        changed = True
        while changed:
            changed = False
            # Find a forbidden triple and remove one element
            s_list = sorted(s)
            n = len(s_list)
            for i in range(n):
                if time.time() > deadline:
                    break
                a = s_list[i]
                for j in range(i+1, n):
                    if time.time() > deadline:
                        break
                    b = s_list[j]
                    ab = a + b
                    if ab > 128 or ab not in s:
                        continue
                    for k in range(j+1, n):
                        c = s_list[k]
                        ac = a + c
                        bc = b + c
                        if ac <= 128 and bc <= 128 and ac in s and bc in s:
                            # Found forbidden triple {a,b,c}
                            # Remove the element that's part of the most other forbidden triples
                            # For now, just remove c (or the largest)
                            s.remove(c)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
        return s
    
    # Try greedy
    if time.time() < deadline:
        greedy_set = greedy_remove()
        if len(greedy_set) > best_size:
            best_size = len(greedy_set)
            best_set = greedy_set
    
    # Local search: start from best, try to add elements back or swap
    def local_search(s_set, max_iter=5000):
        s = set(s_set)
        current_size = len(s)
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() < deadline:
            improved = False
            iters += 1
            # Try adding each element not in s
            candidates = [x for x in range(1, 129) if x not in s]
            random.shuffle(candidates)
            for x in candidates:
                new_s = s | {x}
                if is_valid_set_fast(new_s):
                    s = new_s
                    current_size += 1
                    improved = True
                    break
        return s, current_size
    
    # Run local search from multiple starting points
    if time.time() < deadline:
        # Start from seed
        s, sz = local_search(set(range(65, 129)))
        if sz > best_size:
            best_size = sz
            best_set = s
    
    # Try different initializations
    random.seed(42)
    for trial in range(10):
        if time.time() > deadline - 5:
            break
        
        # Random initialization: take random subset of size ~70
        # Bias toward larger numbers
        candidates = list(range(1, 129))
        # Weight toward larger numbers
        weights = [i**2 for i in range(1, 129)]
        target_size = random.randint(60, 80)
        while len(candidates) > target_size:
            idx = random.randrange(len(candidates))
            candidates.pop(idx)
        
        init_set = set(candidates)
        # Validate and fix
        if not is_valid_set_fast(init_set):
            # Remove conflicts
            for x in sorted(init_set):
                if not is_valid_set_fast(init_set):
                    init_set.remove(x)
        
        s, sz = local_search(init_set)
        if sz > best_size:
            best_size = sz
            best_set = s
    
    # Final validation
    if not is_valid_set_fast(best_set):
        # Fall back to seed
        best_set = set(range(65, 129))
        best_size = 64
    
    result = sorted(best_set)
    
    # Ensure types are correct
    return [int(x) for x in result]
