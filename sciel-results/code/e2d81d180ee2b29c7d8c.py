# EVOLVE-BLOCK-START
import sys
import time

def construct():
    n = 192
    # We want to maximize the size of S subset of {1..192}
    # Constraint: for all x in S, for all distinct a,b,c in S\{x}, x does not divide a+b+c
    
    # Precompute nothing heavy, but we can use bitsets for S
    # Since n=192, we can use a Python integer as a bitset if we map values to indices?
    # Or just a set of integers.
    
    best_S = set()
    best_size = 0
    start_time = time.time()
    TIME_LIMIT = 55.0

    def is_valid_addition(a, S):
        """
        Check if a can be added to S.
        S is a set of integers.
        """
        # 1. Check if a divides sum of any 3 distinct elements in S
        # Sum of 3 elements a,b,c in S.
        # We need to check if there exist x,y,z in S distinct such that (x+y+z) % a == 0
        if a == 1:
            if len(S) >= 3:
                return False
        else:
            # Iterate pairs in S
            # To speed up, we can check if there is any z in S such that z % a == (-x-y) % a
            # We can precompute residues? No, S changes.
            # But S is small (< 100).
            items = list(S)
            len_S = len(items)
            for i in range(len_S):
                x = items[i]
                for j in range(i + 1, len_S):
                    y = items[j]
                    target_res = (-x - y) % a
                    # Check if any z in S, z != x, z != y, has z % a == target_res
                    # We can iterate k
                    for k in range(len_S):
                        z = items[k]
                        if z == x or z == y:
                            continue
                        if z % a == target_res:
                            return False
        
        # 2. Check if any b in S divides sum of (a + x + y) for distinct x,y in S\{b}
        for b in S:
            if b == 1:
                # 1 divides everything.
                # Need to ensure no distinct x,y in S\{1} such that triple (a,x,y) exists.
                # Triple elements are a, x, y. Distinct from b=1.
                # So x,y must be distinct from 1.
                # If there are at least 2 elements in S that are not 1, and a != 1,
                # then we can pick x,y from S\{1} (and a is not 1).
                # Wait, the triple is (a, x, y).
                # The elements are a, x, y.
                # They must be distinct from b=1.
                # So a != 1, x != 1, y != 1.
                # And a, x, y must be distinct from each other?
                # "distinct OTHER selected integers".
                # The summands are 3 distinct integers from S \ {b}.
                # So we need 3 distinct integers from S \ {1} that include a?
                # No, the condition is: b does NOT divide sum of ANY 3 distinct OTHER selected integers.
                # The "other" integers are from S \ {b}.
                # So we need to check if there exist 3 distinct elements u,v,w in S \ {b} such that b | (u+v+w).
                # One of u,v,w is the new element a.
                # So we need 2 other elements x,y in S \ {b, a} such that b | (a+x+y).
                # If b=1, 1 divides everything.
                # So we just need to check if there exist 2 distinct x,y in S \ {1, a} such that x,y are in S.
                # i.e., is |S \ {1, a}| >= 2?
                count = 0
                for v in S:
                    if v != 1 and v != a:
                        count += 1
                        if count >= 2:
                            return False
            else:
                # Check if there exist distinct x, y in S \ {b, a} such that b | (a + x + y)
                # (a + x + y) % b == 0 => (x + y) % b == (-a) % b
                target_res = (-a) % b
                # Iterate pairs in S \ {b, a}
                # To optimize, we can just iterate S and skip b and a
                found = False
                # We need to be careful: x and y must be distinct from each other and from b and a.
                # The summands are 3 distinct elements from S \ {b}.
                # One is a. The other two are x, y from S \ {b, a}.
                # So x, y are distinct from each other.
                items = [v for v in S if v != b and v != a]
                len_items = len(items)
                for i in range(len_items):
                    x = items[i]
                    for j in range(i + 1, len_items):
                        y = items[j]
                        if (x + y) % b == target_res:
                            found = True
                            break
                    if found:
                        break
                if found:
                    return False
        
        return True

    # Greedy initialization
    S_greedy = set()
    for a in range(n, 0, -1):
        if is_valid_addition(a, S_greedy):
            S_greedy.add(a)
    
    best_S = S_greedy.copy()
    best_size = len(S_greedy)

    # Backtracking search
    # We try to add elements that were skipped or swap.
    # Simple DFS with pruning.
    # Order: descending.
    
    def dfs(idx, current_S, current_size):
        nonlocal best_S, best_size
        
        if time.time() - start_time > TIME_LIMIT:
            return
            
        # Pruning
        # Max possible size is current_size + (n - idx)
        if current_size + (n - idx) <= best_size:
            return
            
        if idx == n:
            if current_size > best_size:
                best_S = current_S.copy()
                best_size = current_size
            return
            
        a = n - idx
        
        # Try to include a
        if is_valid_addition(a, current_S):
            current_S.add(a)
            dfs(idx + 1, current_S, current_size + 1)
            current_S.remove(a)
            
        # Try to skip a
        dfs(idx + 1, current_S, current_size)

    # To make DFS more effective, we can start from the greedy solution and try to improve?
    # Or just run DFS from scratch with good pruning.
    # The greedy solution is a strong lower bound.
    
    # Let's try to run DFS.
    # If it's too slow, we might not improve.
    # But the pruning `current_size + remaining <= best_size` is strong if best_size is close to optimal.
    
    dfs(0, set(), 0)

    return list(best_S)
# EVOLVE-BLOCK-END
