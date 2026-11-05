import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s margin before 60s kill
    
    # Precompute LCM of 1..24
    L = 1
    for i in range(1, 25):
        L = L * i // gcd(L, i)
        
    # Map element k to its scaled value v_k = L / k
    # v_k are integers
    vals = [0] * 25
    for k in range(1, 25):
        vals[k] = L // k
        
    # We process elements in descending order of value (ascending order of k? No, descending k means ascending v? No.)
    # k=1 -> v=L
    # k=24 -> v=L/24
    # Larger k means smaller v.
    # The condition "u is subsumed by S" means v_u = sum(v_j).
    # If we pick elements in descending order of k (24, 23, ... 1), we are adding smaller v's first.
    # Wait, the previous logic was:
    # "Small elements depend on Large elements" -> k_small has v_large.
    # If we add k=24 (v_small) first, it doesn't subsume anyone (v_small < v_u for all u<24).
    # But can k=24 be subsumed by others? No, because others are smaller k (larger v), so sum(v_j) > v_j > v_24.
    # So adding large k (small v) first is safe from being subsumed by *future* smaller k (larger v).
    # However, adding a small k (large v) later might be subsumed by the sum of previous large k (small v)?
    # No, sum of small v's can equal a large v.
    # Example: 1/6 = 1/7 + ... no.
    # 1/1 = 1/2+1/3+1/6.
    # v_1 = L. v_2=L/2, v_3=L/3, v_6=L/6.
    # L = L/2 + L/3 + L/6.
    # If we process 24, 23, ..., 6, 3, 2, 1.
    # When we reach 1, we check if v_1 is in sums of {2,3,6,...}.
    # Yes, if 2,3,6 are present.
    # So we MUST check if v_x is in current_sums.
    
    # What about the reverse?
    # When we add x, does it subsume an existing u?
    # v_u = sum(S) + v_x ?
    # If u was added earlier, u > x (since we go 24->1).
    # So v_u < v_x.
    # Can v_u = S + v_x?
    # v_u < v_x, so S must be negative? No, S is sum of positive v's.
    # So v_u < v_x implies v_u cannot be formed by adding v_x to a positive sum.
    # Therefore, if we process in descending order of k (ascending order of v? No, descending k -> ascending v is FALSE.
    # k=1, v=L. k=24, v=L/24.
    # Descending k: 24, 23, ..., 1.
    # v values: L/24, L/23, ..., L.
    # So v is INCREASING.
    # We add small v's first.
    # Existing elements have smaller v.
    # New element x has larger v.
    # Can existing u (v_u < v_x) be subsumed by S + v_x?
    # v_u = S + v_x. Since S >= 0, v_u >= v_x. Contradiction.
    # So adding a larger v (smaller k) NEVER subsumes an existing smaller v (larger k).
    # We only need to check if the NEW element x is subsumed by the CURRENT set.
    # i.e., is v_x in current_sums?
    
    # This simplifies the problem massively!
    # We just need to ensure that when we add x, v_x is not in the subset sums of the previously added elements.
    
    best_set = []
    
    def dfs_search(current_set, sums, next_k):
        nonlocal best_set, deadline
        
        if time.time() > deadline:
            return
            
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
        # Pruning: even if we take all remaining, can we beat best?
        remaining = next_k - 1  # if next_k is 1, remaining 0. k goes down to 1.
        # Actually, available are next_k-1, ..., 1. Count is next_k - 1.
        if len(current_set) + (next_k - 1) <= len(best_set):
            return
            
        if next_k < 1:
            return
            
        # Try to include next_k
        k = next_k
        v_k = vals[k]
        
        # Check if v_k is in sums
        if v_k not in sums:
            # Add k
            new_sums = set(sums)
            for s in sums:
                new_sums.add(s + v_k)
            # Also add v_k itself (subset {k})
            new_sums.add(v_k)
            
            dfs_search(current_set + [k], new_sums, k - 1)
            
        # Try to exclude next_k
        dfs_search(current_set, sums, k - 1)

    # Start with empty set, sums={0}, next_k=24
    dfs_search([], {0}, 24)
    
    return best_set
