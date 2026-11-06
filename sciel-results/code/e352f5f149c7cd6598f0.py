import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin before the 60s/90s kill

    n = 192
    max_size = 48

    # Precompute for each d in [1, 192], the set of "forbidden pairs" (u, v) with u < v
    # such that for some w (distinct from u, v, and d), d | (u + v + w).
    # Actually, the constraint is: for a set S, no d in S divides the sum of any 3 DISTINCT OTHER elements in S.
    # So if we have S, and we pick d in S, we look at S \ {d}. We need no triple {u, v, w} in S \ {d} such that d | (u+v+w).
    # This is equivalent to: for all d in S, and all pairs {u, v} in S \ {d}, there is no w in S \ {d, u, v} such that d | (u+v+w).
    # Or: for all d in S, and all pairs {u, v} in S \ {d}, the value (u+v) mod d must not be "covered" by any w in S \ {d, u, v}.
    # Specifically, if we fix d and a pair {u, v} from S \ {d}, the forbidden residues for w are those where (u+v+w) % d == 0.
    # i.e., w % d == -(u+v) % d.
    # So if S \ {d} contains two elements u, v and a third element w such that w % d == -(u+v) % d, we have a conflict.
    
    # A more efficient way to check validity of a set S:
    # For each d in S:
    #   Let T = S \ {d}.
    #   If |T| < 3, continue.
    #   Check if there exist distinct u, v, w in T such that (u+v+w) % d == 0.
    #   This can be done by checking all pairs (u, v) in T, and seeing if T contains an element w (distinct from u, v) 
    #   such that w % d == -(u+v) % d.
    #   To speed this up, we can maintain for each d, a count of residues of elements in T.
    #   But T changes as S changes.
    
    # Let's try a different approach: Simulated Annealing or a smarter local search.
    # The key bottleneck is the validity check.
    # Let's precompute a "conflict" structure.
    # For each d in [1, 192], and each pair (u, v) with u < v, u != d, v != d,
    # determine if there exists a w != d, w != u, w != v such that d | (u+v+w).
    # If such a w exists, then {d, u, v, w} is a "conflict tuple" where d divides the sum of u, v, w.
    # Actually, we don't need to precompute all w. We just need to know if a set S contains such a tuple.
    
    # Let's define a function is_valid(S) that is fast.
    # S is a set of integers.
    # For each d in S:
    #   T = S - {d}
    #   If len(T) < 3: continue
    #   residues = {}
    #   for x in T:
    #       r = x % d
    #       residues[r] = residues.get(r, 0) + 1
    #   # Now check if any pair (u, v) in T has a complement w in T.
    #   # Actually, it's easier to iterate over pairs (u, v) in T and check if T has an element w
    #   # such that w % d == (-(u+v)) % d.
    #   # To do this fast, we can precompute the set of residues present in T, and their counts.
    #   # But we need to ensure w is distinct from u and v.
    #   # If u % d == v % d == target, we need at least 3 elements with that residue? No, u, v, w are distinct elements.
    #   # Let's just iterate pairs. |T| is at most 47. Pairs ~ 1100.
    #   # For each pair, we do a dict lookup. 1100 * 48 = 52800 ops. This is very fast.
    
    def is_valid(S):
        S_list = list(S)
        for d in S_list:
            T = [x for x in S_list if x != d]
            L = len(T)
            if L < 3:
                continue
            
            # Count residues of elements in T modulo d
            # We need to check if there exist distinct u, v, w in T such that (u+v+w) % d == 0.
            # Iterate over pairs (u, v) in T.
            # For each pair, let target = (-(u+v)) % d.
            # We need to know if there is a w in T, w != u, w != v, such that w % d == target.
            
            # To make this O(L^2) instead of O(L^3), we can precompute residue counts.
            # But we must handle the case where w is u or v.
            # Let's use a dictionary of residues.
            res_count = {}
            for x in T:
                r = x % d
                res_count[r] = res_count.get(r, 0) + 1
            
            for i in range(L):
                u = T[i]
                for j in range(i + 1, L):
                    v = T[j]
                    target = (-(u + v)) % d
                    # We need a w in T with w % d == target, w != u, w != v.
                    # The number of elements in T with residue 'target' is res_count.get(target, 0).
                    # We must subtract 1 if u % d == target, and 1 if v % d == target.
                    count_w = res_count.get(target, 0)
                    if u % d == target:
                        count_w -= 1
                    if v % d == target:
                        count_w -= 1
                    if count_w > 0:
                        return False
        return True

    def can_add(S, x):
        # Check if adding x to S keeps it valid.
        # We don't need to check the whole set again, just the new conflicts involving x.
        # New conflicts:
        # 1. x divides sum of 3 elements in S.
        # 2. Some d in S divides sum of a triple that includes x.
        #    The triple is {x, u, v} where u, v in S, u != v.
        #    So we need to check if for any d in S, there exist u, v in S \ {d} such that (x+u+v) % d == 0.
        #    Note: u and v must be distinct from d, and distinct from each other.
        #    Also, the triple {x, u, v} must consist of distinct elements. Since x is new, x != u, x != v.
        #    We just need u != v and u, v != d.
        
        S_list = list(S)
        K = len(S_list)
        
        # Check 1: x | (u+v+w) for u, v, w in S.
        if K >= 3:
            # Iterate triples in S. K is small.
            for i in range(K):
                u = S_list[i]
                for j in range(i + 1, K):
                    v = S_list[j]
                    for k in range(j + 1, K):
                        w = S_list[k]
                        if (u + v + w) % x == 0:
                            return False
        
        # Check 2: d | (x+u+v) for d in S, u, v in S \ {d}.
        for d in S_list:
            # We need u, v in S \ {d} such that (x+u+v) % d == 0.
            # u + v = (-x) % d.
            # Let T = S \ {d}.
            # We need to check if T has a pair (u, v) with u+v = (-x) % d (mod d).
            # Actually, u+v+w is the sum. Here the sum is x+u+v.
            # So we need (x+u+v) % d == 0.
            # This is equivalent to u+v = (-x) % d.
            # Let's iterate pairs in T.
            T = [z for z in S_list if z != d]
            L = len(T)
            if L < 2:
                continue
            
            # We can use the same residue counting trick or just iterate.
            # Iterating is O(L^2).
            for i in range(L):
                u = T[i]
                for j in range(i + 1, L):
                    v = T[j]
                    if (x + u + v) % d == 0:
                        return False
        
        return True

    best_S = set()
    best_size = 0
    
    # Strategy: Start with a large set of "large" numbers, which are less likely to divide sums of others.
    # Then try to add smaller numbers or swap.
    
    # Initial construction: Greedy descending.
    S = set()
    for x in range(n, 0, -1):
        if can_add(S, x):
            S.add(x)
    
    if len(S) > best_size:
        best_size = len(S)
        best_S = set(S)
    
    # Local Search: Simulated Annealing style or Hill Climbing with swaps.
    # We will try to improve the set by:
    # 1. Adding an element.
    # 2. Swapping an element with another.
    
    current_S = set(S)
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try to add
        not_in_S = [x for x in range(1, n + 1) if x not in current_S]
        if not_in_S:
            random.shuffle(not_in_S)
            for x in not_in_S:
                if can_add(current_S, x):
                    current_S.add(x)
                    improved = True
                    if len(current_S) > best_size:
                        best_size = len(current_S)
                        best_S = set(current_S)
                    break
        
        # If no add, try swap
        if not improved and current_S:
            # Pick a random element to remove
            r = random.choice(list(current_S))
            current_S.remove(r)
            
            # Try to add as many as possible
            not_in_S = [x for x in range(1, n + 1) if x not in current_S]
            random.shuffle(not_in_S)
            added = []
            for x in not_in_S:
                if can_add(current_S, x):
                    current_S.add(x)
                    added.append(x)
            
            # If we added more than 1, size increased.
            # If we added 1, size is same.
            # If we added 0, size decreased.
            # We should keep the change if size increased, or if size is same but we want to explore?
            # For now, only keep if size increased.
            if len(added) > 1:
                if len(current_S) > best_size:
                    best_size = len(current_S)
                    best_S = set(current_S)
            else:
                # Revert
                for x in added:
                    current_S.discard(x)
                current_S.add(r)
                
                # If we are stuck, try a random restart or a more aggressive swap.
                # Let's try a random restart with a bias towards large numbers.
                if random.random() < 0.1:
                    new_S = set()
                    candidates = list(range(n, 0, -1))
                    random.shuffle(candidates)
                    # Keep the top 10 large numbers as a seed?
                    for x in candidates:
                        if can_add(new_S, x):
                            new_S.add(x)
                    if len(new_S) > len(current_S):
                        current_S = new_S
                        if len(current_S) > best_size:
                            best_size = len(current_S)
                            best_S = set(current_S)
        
        # Check time
        if time.time() - start_time > time_limit:
            break
    
    return list(best_S)

if __name__ == "__main__":
    res = construct()
    print(len(res))
