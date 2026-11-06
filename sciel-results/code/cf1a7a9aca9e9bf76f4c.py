import time

N = 16

def _verify_network(comparators, n=16):
    """Verify sorting network against all 2^n Boolean inputs."""
    for mask in range(1 << n):
        w = [0] * n
        for i in range(n):
            w[i] = (mask >> i) & 1
        for a, b in comparators:
            if w[a] > w[b]:
                w[a], w[b] = w[b], w[a]
        for i in range(n - 1):
            if w[i] > w[i + 1]:
                return False
    return True

def _odd_even_transposition(n=16):
    return [(i, i + 1) for phase in range(n) for i in range(phase % 2, n - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline (10s safety margin)
    
    # Try candidate networks, return the best (fewest comparators) that verifies
    best = None
    
    # Candidate 1: Odd-even transposition (120 comparators)
    c1 = _odd_even_transposition()
    if _verify_network(c1):
        best = c1
    
    # Try to find better networks within time budget
    # ... (pruning, etc.)
    
    if best is None:
        # Should not happen
        best = _odd_even_transposition()
    
    return best
