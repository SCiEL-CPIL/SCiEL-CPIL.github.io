# ... existing imports ...

def _als_solve(rank, n, m, p, max_iters=500, restarts=100, timeout=100):
    # ... existing code ...
    pass

def construct():
    n, m, p = DIMENSIONS
    deadline = 115
    
    # Try the known rank-12 structure first (if we had it hardcoded, we'd put it here)
    # Since we don't have the hardcoded tensors, we rely on search.
    # CRITICAL CHANGE: Start search from rank 12, not 19.
    # The rank of M_{2,3,4} is known to be 12.
    
    # We will try ranks 12, 13, ..., 23.
    # Lower ranks are harder to find but yield better scores.
    # We allocate more time to lower ranks initially.
    
    # Time budget allocation:
    # Try rank 12 for 30s
    # Try rank 13 for 20s
    # ...
    
    time_budgets = {
        12: 30,
        13: 20,
        14: 15,
        15: 10,
        16: 5,
        17: 5,
        18: 5,
        19: 5,
        20: 5,
        21: 5,
        22: 5,
        23: 5
    }
    
    current_time = time.time()
    start_time = current_time
    
    for r in range(12, 24):
        budget = time_budgets.get(r, 5)
        # Ensure we don't exceed the global deadline
        if time.time() - start_time > deadline - 5:
            break
            
        # Adjust restarts and iters based on budget
        # More restarts for lower ranks to escape local minima
        max_iters = 100
        restarts = max(10, int(budget * 2)) # ~20-60 restarts
        
        result = _als_solve(r, n, m, p, max_iters=max_iters, restarts=restarts, timeout=budget)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback
    return _schoolbook()
