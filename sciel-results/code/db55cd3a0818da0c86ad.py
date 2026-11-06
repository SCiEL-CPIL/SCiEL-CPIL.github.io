<<<<<<< SEARCH
def construct():
    # Greedy to find a good initial solution
    # Try adding from 24 down to 1
    S_greedy = []
    for k in range(24, 0, -1):
        # Check if adding k is valid
        if _is_valid(S_greedy + [k]):
            S_greedy.append(k)
            
    max_size = len(S_greedy)
    best_set = S_greedy[:]
=======
def _greedy_with_order(order):
    """Greedy construction using a specific element order."""
    S = []
    for k in order:
        if _is_valid(S + [k]):
            S.append(k)
    return S

def _local_search(S, deadline):
    """Try to improve set S by swap operations (remove one, add another)."""
    S = sorted(S)
    improved = True
    while improved and _time_remaining() > 3:
        improved = False
        for i in range(len(S)):
            for j in range(1, 25):
                if j in S:
                    continue
                new_S = S[:i] + S[i+1:] + [j]
                if _is_valid(new_S):
                    S = sorted(new_S)
                    improved = True
                    break
            if improved:
                break
    return S

def construct():
    # Multiple greedy orders to find a good initial solution
    orders = [
        list(range(24, 0, -1)),  # descending
        list(range(1, 25)),      # ascending
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [24, 12, 23, 11, 22, 10, 21, 9, 20, 8, 19, 7, 18, 6, 17, 5, 16, 4, 15, 3, 14, 2, 13, 1],
    ]
    
    best_greedy = []
    for order in orders:
        S_g = _greedy_with_order(order)
        if len(S_g) > len(best_greedy):
            best_greedy = S_g
    
    # Local search to try to improve
    best_greedy = _local_search(best_greedy, _DEADLINE)
    
    max_size = len(best_greedy)
    best_set = best_greedy[:]
>>>>>>> REPLACE

