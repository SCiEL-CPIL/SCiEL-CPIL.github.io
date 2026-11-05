import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Try to find a valid 5-coloring for n=26
    # D = {1, ..., 12}
    D = list(range(1, 13))
    
    def is_c4_free(A):
        # A is a set of integers mod 26
        # Check if C_26(A) is C4-free
        sums = {}
        # Pairs of distinct elements
        elements = list(A)
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                a = elements[i]
                b = elements[j]
                s = (a + b) % 26
                if s == 0:
                    continue
                if s not in sums:
                    sums[s] = []
                sums[s].append((a, b))
        
        for s, pairs in sums.items():
            if len(pairs) > 1:
                # Check if they are distinct pairs
                # Since we iterate i<j, pairs are unique
                # If len > 1, there are at least 2 pairs with same sum
                # This implies a C4
                return False
        return True

    def check_partition(partition):
        # partition is a list of 5 sets
        for k in range(5):
            S_k = partition[k]
            # A_k = S_k union (-S_k)
            A_k = set()
            for d in S_k:
                A_k.add(d)
                A_k.add((26 - d) % 26)
            if not is_c4_free(A_k):
                return False
        return True

    # Backtracking search for partition of D into 5 sets
    # We assign each d in D to a color 0..4
    # Symmetry breaking: Assign 1 to color 0
    # Assign 2 to color 0 or 1?
    
    colors = [set() for _ in range(5)]
    assignment = [-1] * 13 # index 1..12
    
    def backtrack(idx):
        if time.time() > deadline:
            return None
            
        if idx == 13:
            if check_partition(colors):
                return assignment[:]
            return None
        
        d = idx
        # Try assigning d to color 0..4
        # Symmetry: If color k is empty, we can only assign to color 0 if all previous are 0?
        # Simple symmetry breaking:
        # The set of used colors so far must be a prefix of 0..max_used
        max_used = -1
        for i in range(1, idx):
            if assignment[i] > max_used:
                max_used = assignment[i]
                
        for c in range(5):
            if c > max_used + 1:
                break
            assignment[d] = c
            colors[c].add(d)
            res = backtrack(idx + 1)
            if res is not None:
                return res
            colors[c].remove(d)
            
        assignment[d] = -1
        return None

    # Start backtracking
    # Fix 1 to color 0
    assignment[1] = 0
    colors[0].add(1)
    
    result = backtrack(2)
    
    if result is not None:
        # Construct the full coloring for n=26
        n = 26
        colors_list = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                # d is in 1..12 or -12..-1
                # We need the distance in 1..12
                dist = min(d, n - d)
                # Find which color dist is assigned to
                col = result[dist]
                colors_list.append(col)
        return {'n': n, 'colors': colors_list}
    else:
        # Fallback to n=11
        n = 11
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

