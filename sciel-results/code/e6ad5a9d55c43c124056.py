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
    
    # Try n=11 and n=12
    for n in (11, 12):
        if time.time() > deadline:
            break
            
        try:
            all_edges = list(combinations(range(n), 5))
            num_edges = len(all_edges)
            num_colorings = 1 << n
            
            # Build the constraint matrix A
            # A[i, j] = 1 if coloring i is monochromatic on edge j
            # We can build this row by row or column by column.
            # Columns (edges) are easier to generate masks for.
            # But milp needs the matrix.
            
            # Let's build A as a sparse matrix.
            # Rows: colorings (0 to 2^n - 1)
            # Cols: edges (0 to num_edges - 1)
            
            rows = []
            cols = []
            data = []
            
            for j, e in enumerate(all_edges):
                # For each edge, find all colorings that are monochromatic on it.
                # There are 2 such colorings: all 0s on e, all 1s on e.
                # Wait.
                # A coloring is a vector of length n.
                # "Monochromatic on e" means all vertices in e have the same color.
                # If vertices in e are 0, the rest can be anything.
                # So for a fixed edge e, there are $2^{n-5}$ colorings where e is all 0, and $2^{n-5}$ where e is all 1.
                # Total $2 \cdot 2^{n-5} = 2^{n-4}$ colorings per edge.
                
                # We need to iterate through these colorings to add to A.
                # Or, we can iterate through all colorings and check edges?
                # $2^n$ colorings * $num\_edges$ checks.
                # For n=11: 2048 * 462 ~ 1M ops. Fast.
                # For n=12: 4096 * 792 ~ 3M ops. Fast.
                
                pass # We'll do the coloring loop for simplicity in building A, 
                # or we can do the edge loop and generate the specific colorings.
                # Generating specific colorings is faster.
                
                # Vertices in e: v1..v5
                # Vertices not in e: rest
                # Case 1: e is all 0.
                # The bits for v1..v5 are 0.
                # The bits for rest can be 0 or 1.
                # There are 2^(n-5) such colorings.
                # Case 2: e is all 1.
                # The bits for v1..v5 are 1.
                # The bits for rest can be 0 or 1.
                
                # To add to sparse matrix, we need (row, col) pairs.
                # col = j
                # row = coloring index
                
                # Iterate over all subsets of the complement of e?
                # No, just iterate over all 2^(n-5) assignments to the complement.
                
                comp = [v for v in range(n) if v not in e]
                num_comp = len(comp)
                
                for mask_comp in range(1 << num_comp):
                    # Construct coloring c0 where e is 0
                    c0 = 0
                    for k, v in enumerate(comp):
                        if (mask_comp >> k) & 1:
                            c0 |= (1 << v)
                    # e is 0, so no bits set for e
                    rows.append(c0)
                    cols.append(j)
                    data.append(1)
                    
                    # Construct coloring c1 where e is 1
                    c1 = c0
                    for v in e:
                        c1 |= (1 << v)
                    rows.append(c1)
                    cols.append(j)
                    data.append(1)
            
            # Create sparse matrix A
            A = csr_matrix((data, (rows, cols)), shape=(num_colorings, num_edges))
            
            # Objective: minimize sum x_j
            c = np.ones(num_edges)
            
            # Constraints: A x >= 1
            # milp uses LinearConstraint(A, lb, ub)
            # We want A x >= 1
            # So lb = np.ones(num_colorings), ub = np.inf
            lb = np.ones(num_colorings)
            ub = np.full(num_colorings, np.inf)
            
            # Variable bounds: 0 <= x_j <= 1
            # Integrality: 1 (binary)
            integrality = np.ones(num_edges)
            
            # Time limit for solver
            # milp options: time_limit is not directly an option in all versions?
            # Actually, `milp` has `options={'time_limit': ...}`?
            # Let's check. `scipy.optimize.milp` passes options to HiGHS.
            # HiGHS supports `time_limit`.
            
            try:
                res = milp(
                    c=c,
                    constraints=LinearConstraint(A, lb, ub),
                    integrality=integrality,
                    bounds=Bounds(0, 1),
                    options={"time_limit": max(0.1, deadline - time.time())}
                )
                
                if res.status == 0: # Optimal
                    # res.x is the solution
                    # Check if it's a valid cover (should be, but verify)
                    # x is binary
                    x = res.x
                    # Due to floating point, x might be 0.999 or 1.001
                    # Round to nearest int
                    x_int = np.round(x).astype(int)
                    
                    # Verify cover
                    # If sum(A[:, j] * x_int[j]) >= 1 for all i
                    # We can just check the objective value and status.
                    # If status is 0, it is optimal.
                    
                    edges_count = sum(x_int)
                    if edges_count < best_count:
                        best_count = edges_count
                        best_n = n
                        best_edges = [list(all_edges[j]) for j in range(num_edges) if x_int[j] == 1]
                        
            except Exception as e:
                # Fallback if milp fails
                pass

        except Exception:
            pass

    # Fallback to greedy if ILP failed or didn't run
    if best_edges is None:
        # Re-run the previous greedy logic as a fallback
        # ... (insert previous code here or simplified version)
        # For brevity in the edit, I will include a compact version of the previous logic
        # or just return a known valid construction if available.
        # But we should try to be robust.
        
        # Let's reuse the greedy logic from the original code but wrapped in a function or inline.
        # Since I am replacing the whole body, I will include the greedy logic as a fallback.
        
        for n in (12, 11, 10):
            if time.time() > deadline:
                break
            try:
                all_edges = list(combinations(range(n), 5))
                full_mask = (1 << (1 << n)) - 1
                masks = []
                for e in all_edges:
                    m = 0
                    for c in range(1 << n):
                        colors = [(c >> v) & 1 for v in e]
                        if len(set(colors)) == 1:
                            m |= (1 << c)
                    masks.append(m)
                
                # Greedy
                uncovered = full_mask
                chosen = []
                remaining = list(range(len(masks)))
                while uncovered:
                    best_idx = -1
                    best_count = 0
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
                    if time.time() > deadline:
                        break
                
                if uncovered == 0:
                    if best_edges is None or len(chosen) < len(best_edges):
                        best_edges = [list(all_edges[idx]) for idx in chosen]
                        best_n = n
            except:
                pass

    if best_edges is None:
        # Absolute fallback: Complete 5-graph on 9 vertices
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_n = 9

    return {'n': best_n, 'edges': best_edges}

if __name__ == '__main__':
    construct()
