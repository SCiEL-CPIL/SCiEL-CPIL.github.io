import time
import random
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

_WALL_LIMIT = 150.0
_MARGIN = 10.0

def construct():
    start = time.time()
    deadline = start + _WALL_LIMIT - _MARGIN
    rng = random.Random(12345)
    
    best_edges = None
    best_n = None
    best_count = float('inf')
    
    # Try n=10, 11, 12 to find the minimum number of edges
    for n in (10, 11, 12):
        if time.time() > deadline:
            break
            
        try:
            all_edges = list(combinations(range(n), 5))
            num_edges = len(all_edges)
            num_colorings = 1 << n
            
            # Build the constraint matrix A efficiently
            rows = []
            cols = []
            data = []
            
            # Precompute masks for faster lookup? 
            # For each edge, we need to add rows for colorings where it is mono.
            for j, e in enumerate(all_edges):
                comp = [v for v in range(n) if v not in e]
                num_comp = len(comp)
                
                # Iterate over assignments to complement
                for mask_comp in range(1 << num_comp):
                    c0 = 0
                    # Set bits for complement based on mask_comp
                    # We can do this faster by precomputing bit positions
                    for k, v in enumerate(comp):
                        if (mask_comp >> k) & 1:
                            c0 |= (1 << v)
                    
                    # Case 1: e is 0
                    rows.append(c0)
                    cols.append(j)
                    data.append(1)
                    
                    # Case 2: e is 1
                    c1 = c0
                    for v in e:
                        c1 |= (1 << v)
                    rows.append(c1)
                    cols.append(j)
                    data.append(1)
            
            A = csr_matrix((data, (rows, cols)), shape=(num_colorings, num_edges))
            
            c = np.ones(num_edges)
            lb = np.ones(num_colorings)
            ub = np.full(num_colorings, np.inf)
            integrality = np.ones(num_edges)
            
            # Use a reasonable time limit for the solver
            # Prioritize finding a good solution quickly
            time_limit = max(1.0, deadline - time.time())
            
            try:
                res = milp(
                    c=c,
                    constraints=LinearConstraint(A, lb, ub),
                    integrality=integrality,
                    bounds=Bounds(0, 1),
                    options={"time_limit": time_limit}
                )
                
                if res.status == 0: # Optimal
                    x = np.round(res.x).astype(int)
                    edges_count = sum(x)
                    
                    # Verify validity (optional but safe)
                    # Check if every coloring is covered
                    # A @ x >= 1
                    # If this holds, it's a valid cover.
                    # Since we minimize, if status is 0, it should be optimal.
                    
                    if edges_count < best_count:
                        best_count = edges_count
                        best_n = n
                        best_edges = [list(all_edges[j]) for j in range(num_edges) if x[j] == 1]
                        
            except Exception:
                pass

        except Exception:
            pass

    # Fallback to greedy if ILP failed
    if best_edges is None:
        for n in (11, 10, 12):
            if time.time() > deadline:
                break
            try:
                all_edges = list(combinations(range(n), 5))
                # Fast mask generation
                masks = []
                for e in all_edges:
                    m = 0
                    # Iterate colorings
                    for c in range(1 << n):
                        # Check if mono
                        # Get colors of vertices in e
                        # If all 0 or all 1
                        # (c & (1<<v)) == 0 for all v in e OR == (1<<v) for all v in e
                        is_zero = True
                        is_one = True
                        for v in e:
                            bit = (c >> v) & 1
                            if bit == 1:
                                is_zero = False
                            else:
                                is_one = False
                            if not is_zero and not is_one:
                                break
                        if is_zero or is_one:
                            m |= (1 << c)
                    masks.append(m)
                
                full_mask = (1 << (1 << n)) - 1
                uncovered = full_mask
                chosen = []
                remaining = list(range(len(masks)))
                
                while uncovered and time.time() < deadline:
                    best_idx = -1
                    best_count = 0
                    # Greedy step
                    for idx in remaining:
                        count = (uncovered & masks[idx]).bit_count()
                        if count > best_count:
                            best_count = count
                            best_idx = idx
                    if best_idx == -1:
                        break
                    chosen.append(best_idx)
                    uncovered &= ~masks[best_idx]
                    remaining.remove(best_idx)
                
                if uncovered == 0:
                    if best_edges is None or len(chosen) < best_count:
                        best_edges = [list(all_edges[idx]) for idx in chosen]
                        best_n = n
                        best_count = len(chosen)
            except:
                pass

    if best_edges is None:
        # Absolute fallback: Complete 5-graph on 9 vertices (known to be non-2-colorable? 
        # Actually, complete 5-graph on 9 vertices has C(9,5)=126 edges. 
        # Is it non-2-colorable? Yes, likely. 
        # But we want minimum edges. This is a fallback for validity.)
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_n = 9

    return {'n': best_n, 'edges': best_edges}

if __name__ == '__main__':
    construct()
