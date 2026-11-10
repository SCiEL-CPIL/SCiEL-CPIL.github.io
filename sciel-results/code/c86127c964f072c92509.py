import time
import random
from math import gcd
from itertools import combinations

# Constants
N = 128
RANGE = list(range(1, N + 1))

# Precompute GCDs
gcds = [[0] * (N + 1) for _ in range(N + 1)]
for i in RANGE:
    for j in RANGE:
        gcds[i][j] = gcd(i, j)

# Precompute bad triples
# A triple (a, b, c) is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)
bad_triples = set()
# To save memory/time, we might not need to store all triples if we check on the fly, 
# but N=128 means C(128,3) ~ 340k triples. Storing them is fine.
# However, checking if a new element x forms a bad triple with existing set S:
# We need to check all pairs (a,b) in S.
# If we store bad triples as a set of frozensets or sorted tuples, we can check membership.
# Let's store them as a set of sorted tuples (a,b,c) with a<b<c.
for a, b, c in combinations(RANGE, 3):
    if gcds[a][b] == gcds[a][c] == gcds[b][c]:
        bad_triples.add((a, b, c))

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    best_set = set()
    
    # Helper to check if adding x to S is valid
    def can_add(x, S):
        # Check all pairs in S
        # If S is large, this is O(|S|^2)
        # We can optimize by pre-computing conflicts? 
        # For |S| < 100, O(100^2) = 10,000 checks is fast.
        # Check if (a, b, x) is a bad triple for any a, b in S
        # We need to check if (min(a,b,x), mid, max) is in bad_triples
        s_list = list(S)
        for i in range(len(s_list)):
            a = s_list[i]
            for j in range(i + 1, len(s_list)):
                b = s_list[j]
                # Sort a, b, x to check in bad_triples
                t = sorted((a, b, x))
                if tuple(t) in bad_triples:
                    return False
        return True

    # Greedy Construction
    def greedy_construct(order):
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S

    # Local Search Improvement
    def local_search(S):
        S = set(S)
        improved = True
        while improved:
            improved = False
            # Try to add elements
            for x in RANGE:
                if x not in S:
                    if can_add(x, S):
                        S.add(x)
                        improved = True
            # Try to remove elements if it allows adding more
            # This is harder. Let's try a simple swap/remove heuristic.
            # Calculate "cost" of each element: number of elements it blocks.
            # An element x blocks y if y cannot be added to S \ {x} but can be added to S? No.
            # If y is not in S, and can_add(y, S) is False, but can_add(y, S \ {x}) is True, then x blocks y.
            # If x blocks many y's, it might be worth removing x if we can add more than 1 y.
            
            # This O(N * |S| * |S|^2) might be slow if done naively.
            # Let's stick to a simpler improvement: 
            # Randomly shuffle and re-greedy? Or just try adding all possible, then try removing one and re-adding all.
            
            # Let's try: For each x in S, temporarily remove it, try to add all possible, if size increases, keep it.
            for x in list(S):
                S_temp = S - {x}
                # Try to add any element not in S_temp
                added = False
                for y in RANGE:
                    if y not in S_temp:
                        if can_add(y, S_temp):
                            S_temp.add(y)
                            added = True
                if len(S_temp) > len(S):
                    S = S_temp
                    improved = True
                    break # Restart loop as S changed significantly
            
        return S

    # Main Loop
    # Try multiple random orders
    best_size = 0
    
    # 1. Standard Greedy (1 to 128)
    S1 = greedy_construct(RANGE)
    S1 = local_search(S1)
    if len(S1) > best_size:
        best_size = len(S1)
        best_set = S1
        
    if time.time() > deadline:
        return sorted(best_set)

    # 2. Reverse Greedy (128 to 1)
    S2 = greedy_construct(list(reversed(RANGE)))
    S2 = local_search(S2)
    if len(S2) > best_size:
        best_size = len(S2)
        best_set = S2
        
    if time.time() > deadline:
        return sorted(best_set)

    # 3. Randomized Greedy + Local Search
    random.seed(42)
    for _ in range(50): # Number of iterations
        if time.time() > deadline - 5.0:
            break
            
        order = RANGE[:]
        random.shuffle(order)
        S = greedy_construct(order)
        S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            # If we found a very large set, we might want to polish it further
            # But local_search already does a decent job.
            
    # 4. Try a "Max Conflict Removal" strategy from a full set
    # Start with all numbers, remove those in most bad triples
    S_full = set(RANGE)
    # Count conflicts for each element
    # This is complex to do incrementally. 
    # Instead, let's try to improve the best_set found so far with a more aggressive local search
    # if time permits.
    
    # Aggressive Local Search: Simulated Annealing style
    # We might skip this if the simple local search is fast enough and effective.
    # Given N=128, the simple local search is likely sufficient to find near-optimal solutions.
    
    return sorted(best_set)

if __name__ == "__main__":
    res = construct()
    print(len(res))
