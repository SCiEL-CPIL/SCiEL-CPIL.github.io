import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    N = 256

    # Precompute reciprocal weights as Fractions
    weights = [Fraction(0)] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = Fraction(1, i)

    # Suffix sums for pruning (exact)
    suffix = [Fraction(0)] * (N + 2)
    for i in range(N, 0, -1):
        suffix[i] = suffix[i + 1] + weights[i]

    # Precompute AP pairs: for each x, list of (a, b) such that a, b, x form a 3-AP
    # i.e., b - a = x - b => 2b = a + x
    ap_pairs = [[] for _ in range(N + 1)]
    for x in range(3, N + 1):
        for d in range(1, x // 2 + 1):
            a = x - 2 * d
            b = x - d
            if a >= 1:
                ap_pairs[x].append((a, b))

    # Greedy initialization to get a good starting bound
    in_S = [False] * (N + 1)
    S_list = []
    mask = 0
    for i in range(1, N + 1):
        valid = True
        for a, b in ap_pairs[i]:
            if in_S[a] and in_S[b]:
                valid = False
                break
        if valid:
            in_S[i] = True
            S_list.append(i)
            mask |= (1 << (i - 1))

    best_score = sum(weights[x] for x in S_list) if S_list else Fraction(0)
    best_mask = mask

    # Branch and bound DFS
    # We process elements in order 1..N
    # State: idx (next element to consider), current_sum, mask
    # Optimization: 
    # 1. Use bitmasks for fast membership tests
    # 2. Prune using suffix sums
    # 3. Order of exploration: try including x first if it's "cheap" to include?
    #    Actually, since weights are 1/i, smaller i have larger weights.
    #    We want to include small numbers if possible.
    #    Standard B&B: explore include first to raise the bound quickly.

    # To speed up validity check, we can precompute for each x, the pairs.
    # The check is: for all (a,b) in ap_pairs[x], NOT (a in mask AND b in mask)
    
    def is_valid_add(x, mask):
        for a, b in ap_pairs[x]:
            if (mask & (1 << (a - 1))) and (mask & (1 << (b - 1))):
                return False
        return True

    # Iterative DFS with explicit stack to avoid recursion limit and overhead?
    # Recursion depth is 256, which is fine for Python if we increase limit, but recursion overhead is high.
    # Let's try recursion first with optimizations.
    
    import sys
    sys.setrecursionlimit(300)

    def dfs(idx, current_sum, mask):
        nonlocal best_score, best_mask
        
        if time.time() > deadline:
            return

        # Pruning
        # If current_sum + remaining_possible <= best_score, prune
        # remaining_possible is suffix[idx]
        if current_sum + suffix[idx] <= best_score:
            return

        if idx > N:
            if current_sum > best_score:
                best_score = current_sum
                best_mask = mask
            return

        x = idx

        # Option 1: Include x
        if is_valid_add(x, mask):
            new_mask = mask | (1 << (x - 1))
            dfs(idx + 1, current_sum + weights[x], new_mask)

        # Option 2: Exclude x
        dfs(idx + 1, current_sum, mask)

    dfs(1, Fraction(0), 0)

    # Reconstruct the best set
    best_set = []
    for i in range(1, N + 1):
        if best_mask & (1 << (i - 1)):
            best_set.append(i)

    # Fallback if something went wrong (shouldn't happen)
    if not best_set:
        best_set = [1]

    return best_set
