import time
import sys

N = 256
sys.setrecursionlimit(10000)

# Precompute reciprocals and suffix sums
recips = [0.0] + [1.0 / i for i in range(1, N + 1)]
suffix_sum = [0.0] * (N + 2)
for i in range(N, 0, -1):
    suffix_sum[i] = suffix_sum[i + 1] + recips[i]

best_sum = 0.0
best_set = []
start_time = 0.0
TIME_LIMIT = 80.0  # Safety margin for 90s limit

def check_conflict(S_set, x):
    """
    Check if adding x to S_set creates a 3-term AP.
    S_set is a set of integers.
    """
    for a in S_set:
        # Case 1: a, x, c => c = 2x - a
        c = 2 * x - a
        if c in S_set and c != x:
            return True
        
        # Case 2: a, b, x => b = (a+x)/2
        if (a + x) % 2 == 0:
            b = (a + x) // 2
            if b in S_set and b != a and b != x:
                return True
    return False

def greedy_init():
    """
    Generate a good initial solution using a greedy approach.
    """
    S = []
    S_set = set()
    current_sum = 0.0
    for x in range(1, N + 1):
        if not check_conflict(S_set, x):
            S.append(x)
            S_set.add(x)
            current_sum += recips[x]
    return S, S_set, current_sum

def branch_and_bound(x, S_list, S_set, current_sum):
    global best_sum, best_set, start_time
    
    # Time check
    if time.time() - start_time > TIME_LIMIT:
        return
        
    if x > N:
        if current_sum > best_sum:
            best_sum = current_sum
            best_set = list(S_list)
        return

    # Upper Bound Calculation (Optimistic: ignore conflicts among remaining)
    ub = current_sum + suffix_sum[x]
    if ub <= best_sum:
        return

    # Branch 1: Include x
    if not check_conflict(S_set, x):
        S_list.append(x)
        S_set.add(x)
        branch_and_bound(x + 1, S_list, S_set, current_sum + recips[x])
        S_list.pop()
        S_set.remove(x)

    # Branch 2: Exclude x
    branch_and_bound(x + 1, S_list, S_set, current_sum)

def construct():
    global best_sum, best_set, start_time
    start_time = time.time()
    
    # 1. Get initial solution
    init_list, init_set, init_sum = greedy_init()
    best_sum = init_sum
    best_set = init_list
    
    # 2. Run Branch and Bound
    branch_and_bound(1, [], set(), 0.0)
    
    return best_set
