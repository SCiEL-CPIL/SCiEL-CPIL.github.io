"""Ambient integer sets with small largest Sidon subset."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _max_sidon_subset_size(ambient):
    """
    Exhaustively compute the largest subset of ambient with all unordered pair sums unique,
    INCLUDING diagonals (i.e., for a subset S, the multiset {s_i + s_j : i <= j} must have
    all distinct values).
    
    For a subset S of size k, the number of unordered pairs (including self-pairs) is k*(k+1)/2.
    We want the maximum k such that there exists a k-element subset of ambient where all these
    sums are distinct.
    """
    n = len(ambient)
    if n <= 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        # Two elements a, b: sums are a+a, a+b, b+b. Need a+a != a+b != b+b and a+a != b+b.
        # a+a != a+b iff a != b (true since distinct)
        # a+b != b+b iff a != b (true)
        # a+a != b+b iff 2a != 2b iff a != b (true)
        return 2
    
    # Try from largest possible size downward
    for k in range(n, 0, -1):
        # Check if any k-subset has all pair sums (including diagonals) distinct
        for subset in combinations(ambient, k):
            sums = set()
            valid = True
            for i in range(k):
                for j in range(i, k):
                    s = subset[i] + subset[j]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            if valid:
                return k
    return 1

def _greedy_construct_collision_maximizing():
    """
    Greedy heuristic: select 12 integers from [0,128] that maximize intra-set sum collisions.
    The idea: if many pairs in the ambient set share the same sum, then any large subset is likely
    to contain a collision, forcing the max Sidon subset to be small.
    
    Strategy: pick numbers that create many repeated pair sums.
    """
    # A strong collision structure: use an arithmetic progression-like structure
    # where many pairs have the same sum. For example, if we pick {a, b, c, d, ...}
    # such that a+d = b+c, etc.
    
    # Let's try: pick numbers in pairs that sum to the same value.
    # If we have pairs (x_i, M - x_i), then x_i + (M - x_i) = M for all i.
    # This creates many collisions at sum M.
    
    # But we also need to consider that the Sidon condition requires ALL pair sums distinct.
    # So if we have multiple pairs summing to M, any subset containing two elements from different
    # such pairs will have a collision at M.
    
    # Let's construct: pick 6 pairs (x_i, M - x_i) for i=0..5, with all x_i distinct and M - x_i distinct.
    # Choose M = 64. Pairs: (1,63), (2,62), (3,61), (4,60), (5,59), (6,58)
    # Ambient set: {1, 2, 3, 4, 5, 6, 58, 59, 60, 61, 62, 63}
    
    # Let's verify the max Sidon subset size for this set.
    # Any 3-element subset: if it contains two elements from the same pair, say x and M-x,
    # then x+(M-x)=M. If it contains a third element y, we need y+y != M, x+y != M, (M-x)+y != M.
    # y+y = M iff y = M/2 = 32. None of our elements is 32.
    # x+y = M iff y = M-x, which is the other element in the pair. So if the subset is {x, M-x, y} where y is not M-x,
    # then x+y != M and (M-x)+y != M. But we also need x+x, (M-x)+(M-x), x+y, (M-x)+y all distinct.
    # x+x = 2x, (M-x)+(M-x) = 2M-2x, x+y, (M-x)+y = M-x+y.
    # These are 4 values. Need them all distinct.
    # 2x != 2M-2x iff x != M/2 = 32. True.
    # 2x != x+y iff x != y. True (distinct elements).
    # 2x != M-x+y iff 3x != M+y.
    # 2M-2x != x+y iff 2M-3x != y.
    # 2M-2x != M-x+y iff M-x != y. True.
    # x+y != M-x+y iff x != M-x iff x != M/2. True.
    
    # So for a 3-subset {x, M-x, y} where y is from a different pair, the sums are:
    # 2x, 2(M-x), x+y, (M-x)+y, M (from x+(M-x))
    # Wait, that's 5 sums for 3 elements (3*4/2 = 6 pairs including diagonals).
    # Pairs: (x,x), (x,M-x), (x,y), (M-x,M-x), (M-x,y), (y,y)
    # Sums: 2x, M, x+y, 2M-2x, M-x+y, 2y
    # Need all 6 distinct.
    # M is unique (only from x+(M-x)).
    # 2x, 2M-2x, x+y, M-x+y, 2y need to be distinct from each other and from M.
    
    # This is getting complex. Let me just compute it.
    
    ambient = [1, 2, 3, 4, 5, 6, 58, 59, 60, 61, 62, 63]
    size = _max_sidon_subset_size(ambient)
    return ambient, size

def _local_search(ambient, max_iter=2000):
    """
    Local search: try swapping one element with another to reduce max Sidon subset size.
    """
    current = list(ambient)
    current_size = _max_sidon_subset_size(current)
    best = list(current)
    best_size = current_size
    
    all_candidates = set(range(129))
    
    for iteration in range(max_iter):
        improved = False
        for i in range(len(current)):
            for new_val in all_candidates - set(current):
                candidate = list(current)
                candidate[i] = new_val
                candidate.sort()
                if len(set(candidate)) != 12:
                    continue
                size = _max_sidon_subset_size(candidate)
                if size < best_size:
                    best = list(candidate)
                    best_size = size
                    current = list(candidate)
                    current_size = size
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best, best_size

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 75s deadline, 15s safety margin
    
    # Start with a collision-maximizing structure
    ambient, initial_size = _greedy_construct_collision_maximizing()
    best = list(ambient)
    best_size = initial_size
    
    # Try several different collision structures
    structures = []
    
    # Structure 1: pairs summing to 64
    structures.append([1, 2, 3, 4, 5, 6, 58, 59, 60, 61, 62, 63])
    
    # Structure 2: pairs summing to 128
    structures.append([1, 2, 3, 4, 5, 6, 122, 123, 124, 125, 126, 127])
    
    # Structure 3: mix of pairs summing to different values
    structures.append([1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15])
    
    # Structure 4: arithmetic progression
    structures.append(list(range(0, 128, 11))[:12])  # 0, 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121
    
    # Structure 5: clustered
    structures.append([0, 1, 2, 3, 4, 5, 64, 65, 66, 67, 68, 69])
    
    # Structure 6: two clusters with internal collisions
    structures.append([1, 3, 5, 7, 9, 11, 61, 63, 65, 67, 69, 71])
    
    for struct in structures:
        if time.time() > deadline:
            break
        struct = struct[:12]
        if len(set(struct)) != 12:
            continue
        size = _max_sidon_subset_size(struct)
        if size < best_size:
            best = list(struct)
            best_size = size
    
    # Local search from the best structure
    if time.time() < deadline - 10:
        best, best_size = _local_search(best, max_iter=500)
    
    return best

# EVOLVE-BLOCK-END
