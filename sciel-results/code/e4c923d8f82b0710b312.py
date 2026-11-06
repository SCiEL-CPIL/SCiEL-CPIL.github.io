import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s safety margin
    
    N = 512
    limit = N
    
    # Precompute compatibility matrix or helper
    # comp[i][j] = 1 if lcm(i+1, j+1) <= limit else 0
    # Using 0-indexed for array access, values 1..512
    # We can just compute lcm on the fly or precompute a set of compatible pairs for each i.
    
    compatible = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            # lcm(i, j) = i * j // gcd(i, j)
            g = gcd(i, j)
            lcm_val = (i // g) * j
            if lcm_val <= limit:
                compatible[i].add(j)
                if i != j:
                    compatible[j].add(i)
            # else: not compatible

    def is_compatible_with_set(v, S):
        for u in S:
            if v not in compatible[u]:
                return False
        return True

    def greedy_init():
        S = set()
        # Try adding in increasing order
        for i in range(1, N + 1):
            if is_compatible_with_set(i, S):
                S.add(i)
        return S

    def local_search(S, time_limit):
        S = set(S)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try adding
            for v in range(1, N + 1):
                if v not in S:
                    if is_compatible_with_set(v, S):
                        S.add(v)
                        improved = True
            # Try swapping
            # To speed up, we can break early if time is tight, but N is small.
            # We iterate over all u in S and v not in S.
            # This is O(|S| * (N - |S|)) checks.
            # Worst case ~ 256 * 256 = 65k checks. Fast.
            if time.time() > time_limit:
                break
                
            # Collect candidates for swap
            # A swap (u, v) is valid if v is compatible with S \ {u}
            # And we assume we always swap if it doesn't reduce size? 
            # Wait, swap keeps size same. We only swap if we are stuck and want to explore different configurations?
            # Actually, if we can add, we add. If we can't add, we might swap to enable a future add.
            # Standard local search for Max Independent Set:
            # 1. Add if possible.
            # 2. If no add possible, try swaps that lead to a "better" position. 
            #    How to define better? Maybe just random swaps to escape local minima.
            
            # Let's do: If no add, pick a random u in S and random v not in S.
            # If v is compatible with S\{u}, perform swap.
            # This is a random walk on the space of max-cardinality sets (if we only swap when valid).
            # But we might get stuck in a local optimum where no single swap helps add a new node.
            # To improve, we can also try removing a node and then greedily adding back?
            
            # Simpler: If no add, do a random valid swap.
            if not improved:
                # Check if time allows a swap
                if time.time() < time_limit - 0.1:
                    u_list = list(S)
                    if u_list:
                        u = random.choice(u_list)
                        # Find a random v not in S
                        v_candidates = [v for v in range(1, N + 1) if v not in S]
                        if v_candidates:
                            # Try a few random v's
                            for _ in range(10):
                                v = random.choice(v_candidates)
                                # Check if v compatible with S - {u}
                                temp_S = S - {u}
                                if is_compatible_with_set(v, temp_S):
                                    S = temp_S | {v}
                                    improved = True # Mark as changed to continue loop
                                    break
                            if not improved:
                                # Try a different u? Or just break and restart?
                                # Let's try a few more u's
                                for _ in range(10):
                                    u = random.choice(u_list)
                                    for _ in range(10):
                                        v = random.choice(v_candidates)
                                        temp_S = S - {u}
                                        if is_compatible_with_set(v, temp_S):
                                            S = temp_S | {v}
                                            improved = True
                                            break
                                    if improved:
                                        break
        return S

    best_S = greedy_init()
    best_size = len(best_S)
    
    # Run multiple restarts with randomization
    # The greedy above is deterministic. Let's vary the order.
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Randomized Greedy
        S = set()
        # Shuffle candidates
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        # To bias towards "good" nodes, we can sort by degree or just random.
        # Let's try sorting by number of compatible neighbors (descending) to pack tightly?
        # Or just random.
        
        for v in candidates:
            if time.time() > deadline:
                break
            if is_compatible_with_set(v, S):
                S.add(v)
        
        # Local Search
        S = local_search(S, deadline - 0.5)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
            
    return list(best_S)
