import time
import random

def _has_perfect_matching(m, L):
    """
    Checks if there is a perfect matching in the bipartite graph
    between indices 1..12 and numbers (m, m+L] such that i divides b_i.
    """
    n = 12
    # candidates are m+1, ..., m+L
    # match[i] will store the candidate index (0-based relative to m+1) matched to i
    # We use 0-based indexing for candidates: 0 corresponds to m+1, L-1 corresponds to m+L
    match = [-1] * L
    
    def dfs(i, visited):
        """
        Tries to find an augmenting path for index i.
        i is 0-based index (0..11) representing the number i+1.
        """
        idx = i + 1
        for k in range(L):
            candidate_val = m + 1 + k
            if visited[k]:
                continue
            if candidate_val % idx == 0:
                visited[k] = True
                # If candidate k is not matched, or if the index currently matched to k can find another candidate
                if match[k] == -1 or dfs(match[k], visited):
                    match[k] = i
                    return True
        return False
    
    # We need to match all 12 indices.
    # We iterate through indices 0..11
    for i in range(n):
        visited = [False] * L
        if not dfs(i, visited):
            return False
    return True

def _compute_f(m):
    """
    Computes f(12, m): the least positive L such that (m, m+L] 
    contains distinct b_1..b_12 with i | b_i.
    """
    # Binary search for L
    # L must be at least 12 (pigeonhole) and at most 144 (given in task description)
    lo, hi = 1, 144
    best_L = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if _has_perfect_matching(m, mid):
            best_L = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return best_L

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s safety margin
    
    best_m = 0
    best_f = 0
    
    # Random search
    # The range is 0..1000000
    # We can't exhaustively search, so we sample.
    # To ensure we cover the space well, we can use a mix of random and structured sampling.
    
    # First, check a few "obvious" candidates or just random.
    # Given the "worst interval" nature, high values of f(12,m) are rare spikes.
    # Random search is the best bet.
    
    # To improve efficiency, we can pre-check some necessary conditions?
    # Not really, the matching is the core.
    
    # Let's do random sampling
    # Number of samples depends on speed. 
    # _compute_f is fast. Let's aim for ~50,000 to 100,000 samples if possible.
    
    # Use a fixed seed for reproducibility in local testing, but random for evaluator?
    # The evaluator runs the code. It doesn't matter if it's random or fixed, 
    # but random helps avoid missing specific peaks if the distribution is uniform.
    
    # We will sample random integers.
    # We can also add a local refinement step at the end.
    
    samples = 0
    while time.time() < deadline:
        m = random.randint(0, 1000000)
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
            samples += 1
            # If we find a very high value, we might want to refine locally
            # But we don't know the max possible. Max is 144.
            # If we find 144, we can stop? 
            # "Maximize this independently computed L." 
            # If we hit the theoretical max, we are done.
            if best_f == 144:
                break
        
        # Check time every 1000 samples to avoid overhead
        if samples % 1000 == 0 and time.time() > deadline - 10:
            break
    
    # Local refinement around best_m
    # Check neighbors of best_m
    # The function f(12, m) is likely to have small plateaus or spikes.
    # Checking +/- 100 around best_m is cheap.
    
    if time.time() < deadline - 2:
        for delta in range(-100, 101):
            m_local = best_m + delta
            if 0 <= m_local <= 1000000:
                f_local = _compute_f(m_local)
                if f_local > best_f:
                    best_f = f_local
                    best_m = m_local

    return int(best_m)

