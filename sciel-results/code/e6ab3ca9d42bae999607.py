import time
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
    
    # For each y, define:
    #   low[y] = min(y, inv[y])
    #   high[y] = max(y, inv[y])
    # An interval [L, R] is valid iff for all y in [L, R], inv[y] is NOT in [L, R].
    # Equivalently, for all y in [L, R], either inv[y] < L or inv[y] > R.
    #
    # For a fixed L, the interval [L, R] becomes invalid at the smallest R >= L
    # such that there exists y in [L, R] with L <= inv[y] <= R.
    # Such a y exists when max(y, inv[y]) <= R and min(y, inv[y]) >= L.
    # So R_conflict(L) = min{ max(y, inv[y]) : y in [L, N], min(y, inv[y]) >= L }
    # The maximum valid R for start L is R_conflict(L) - 1.
    
    low = [0] * (N + 1)
    high = [0] * (N + 1)
    for y in range(1, N + 1):
        low[y] = min(y, inv[y])
        high[y] = max(y, inv[y])
    
    # Bucket by low[y] value: bucket[v] contains all y with low[y] == v
    bucket = [[] for _ in range(N + 2)]
    for y in range(1, N + 1):
        bucket[low[y]].append(y)
    
    # Sweep L from N down to 1.
    # At step L, add all y with low[y] == L to the heap (they become active for this and smaller L).
    # The heap contains high[y] for all active y (i.e., y with low[y] >= L).
    # R_conflict(L) = min(heap) if heap non-empty, else N+1.
    # Max valid R for start L = R_conflict(L) - 1 (capped at N).
    
    heap = []
    best_start = 1
    best_length = 0
    
    for L in range(N, 0, -1):
        if time.time() > deadline:
            break
        
        # Add y's that become active at this L
        for y in bucket[L]:
            heapq.heappush(heap, high[y])
        
        if heap:
            r_conflict = heap[0]
            max_R = min(r_conflict - 1, N)
        else:
            max_R = N
        
        if max_R >= L:
            length = max_R - L + 1
            if length > best_length:
                best_length = length
                best_start = L
    
    if best_length == 0:
        # Fallback: return a trivial valid interval
        # Self-inverse residues mod 1009 are 1 and 1008 (since 1*1=1, 1008*1008 = 1016064, 1016064 mod 1009 = 1)
        # So any x not in {1, 1008} is valid as a singleton.
        best_start = 2
        best_length = 1
    
    return [best_start, best_length]
