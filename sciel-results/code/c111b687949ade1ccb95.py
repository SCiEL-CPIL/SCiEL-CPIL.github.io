import time
from math import gcd
import random

MAXIMUM = 128
DEADLINE = 55.0

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # The maximum number of disjoint residue classes with distinct moduli
    # is achieved by selecting a set of moduli that are pairwise non-coprime.
    # The largest such set is the set of all even numbers.
    moduli = [m for m in range(2, MAXIMUM + 1) if m % 2 == 0]
    n = len(moduli)
    
    if n == 0:
        return [[0, 2]]
        
    # Precompute GCDs for fast compatibility checks
    gcds = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g
            
    # assigned[i] will store the residue for moduli[i], or -1 if unassigned
    assigned = [-1] * n
    
    def is_compatible(idx, val):
        # Check if assigning val to moduli[idx] is compatible with all assigned variables
        for j in range(n):
            if assigned[j] != -1:
                g = gcds[idx][j]
                if val % g == assigned[j] % g:
                    return False
        return True
        
    def backtrack(idx):
        nonlocal assigned
        if time.time() > deadline:
            return False
            
        if idx == n:
            return True
            
        m_val = moduli[idx]
        
        # Heuristic: Try values that are "central" or random?
        # For now, try 0 to m_val-1
        # To speed up, we can shuffle the order of values or try specific patterns
        for val in range(m_val):
            if is_compatible(idx, val):
                assigned[idx] = val
                if backtrack(idx + 1):
                    return True
                assigned[idx] = -1
        return False
        
    # Try to find a solution for the full set of even moduli
    # We try a few different orders of variables to help backtracking
    orders = [
        list(range(n)),
        list(reversed(range(n))),
        sorted(range(n), key=lambda i: moduli[i]),
        sorted(range(n), key=lambda i: -moduli[i])
    ]
    
    # Add a few random orders
    for _ in range(5):
        rand_order = list(range(n))
        random.shuffle(rand_order)
        orders.append(rand_order)
        
    for order in orders:
        if time.time() > deadline:
            break
            
        # Re-index variables based on the order? 
        # No, the 'assigned' array is indexed by original index.
        # We need to modify the backtrack function to accept an order or 
        # simply permute the 'moduli' and 'gcds' arrays.
        # It's easier to just try the greedy approach first, then backtrack with original order.
        
        # Let's try a greedy approach first for each order
        assigned_greedy = [-1] * n
        success = True
        for idx in order:
            m_val = moduli[idx]
            found = False
            for val in range(m_val):
                # Check compatibility with already assigned in this greedy run
                ok = True
                for j in order:
                    if j >= idx: # Only check previously processed in this loop?
                        # Actually, we need to check against all assigned in 'assigned_greedy'
                        pass
                # Simpler: check against all assigned_greedy
                compatible = True
                for j in range(n):
                    if assigned_greedy[j] != -1:
                        g = gcds[idx][j]
                        if val % g == assigned_greedy[j] % g:
                            compatible = False
                            break
                if compatible:
                    assigned_greedy[idx] = val
                    found = True
                    break
            if not found:
                success = False
                break
                
        if success:
            # Construct the result
            result = []
            for i in range(n):
                result.append([assigned_greedy[i], moduli[i]])
            return result
            
    # If greedy failed, use backtracking on the original order (or best heuristic order)
    # Backtracking is more robust but slower.
    # We'll use the first order (natural) for backtracking.
    assigned_bt = [-1] * n
    
    def backtrack_impl(idx):
        nonlocal assigned_bt
        if time.time() > deadline:
            return False
            
        if idx == n:
            return True
            
        m_val = moduli[idx]
        
        # Try values
        for val in range(m_val):
            compatible = True
            for j in range(n):
                if assigned_bt[j] != -1:
                    g = gcds[idx][j]
                    if val % g == assigned_bt[j] % g:
                        compatible = False
                        break
            if compatible:
                assigned_bt[idx] = val
                if backtrack_impl(idx + 1):
                    return True
                assigned_bt[idx] = -1
        return False
        
    if backtrack_impl(0):
        result = []
        for i in range(n):
            result.append([assigned_bt[i], moduli[i]])
        return result
        
    # Fallback: Return a valid smaller set
    # Try to find a valid set of size 1
    return [[0, 2]]
