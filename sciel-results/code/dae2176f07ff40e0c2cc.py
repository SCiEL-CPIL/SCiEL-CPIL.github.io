import sys

# Increase recursion limit just in case, though depth is only 256
sys.setrecursionlimit(10000)

def construct():
    N = 256
    recips = [0.0] * (N + 1)
    for i in range(1, N + 1):
        recips[i] = 1.0 / i
    
    # Suffix sums for upper bound pruning
    # suffix_sum[i] = sum of recips[i] to recips[N]
    suffix_sum = [0.0] * (N + 2)
    for i in range(N, 0, -1):
        suffix_sum[i] = suffix_sum[i + 1] + recips[i]
    
    best_score = 0.0
    best_set = []
    
    # Use a list for current set to allow easy removal
    current_set = []
    current_set_set = set()
    
    def is_safe(x, S_set):
        # Check if adding x creates a 3-AP
        # 1. x is middle: a, x, c -> 2x = a+c -> c = 2x-a
        for a in S_set:
            if (2 * x - a) in S_set:
                return False
        # 2. x is endpoint (max or min): a, b, x or x, b, a -> 2b = a+x -> a = 2b-x
        for b in S_set:
            if (2 * b - x) in S_set:
                return False
        return True

    def dfs(idx, current_sum):
        nonlocal best_score, best_set
        
        # Pruning: if current sum + max possible remaining sum <= best, stop
        if current_sum + suffix_sum[idx] <= best_score:
            return
            
        if idx > N:
            if current_sum > best_score:
                best_score = current_sum
                best_set = list(current_set)
            return
        
        x = idx
        
        # Branch 1: Try to include x
        if is_safe(x, current_set_set):
            current_set.append(x)
            current_set_set.add(x)
            dfs(idx + 1, current_sum + recips[x])
            current_set.pop()
            current_set_set.remove(x)
            
        # Branch 2: Exclude x
        dfs(idx + 1, current_sum)
        
    dfs(1, 0.0)
    
    return best_set

