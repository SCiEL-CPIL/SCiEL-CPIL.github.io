import time
import numpy as np
from fractions import Fraction
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

N = 256
DEADLINE_MARGIN = 5.0

def construct():
    start_time = time.time()
    deadline = start_time + 90.0 - DEADLINE_MARGIN
    
    # 1. Build the conflict graph
    # Nodes: 1..N
    # Edge (u, v) exists if there exists w such that {u, v, w} is a 3-AP.
    # This is equivalent to:
    #   2u = v + w  => w = 2u - v
    #   2v = u + w  => w = 2v - u
    #   2w = u + v  => w = (u + v) / 2 (must be integer)
    
    # For MWIS, we need a graph where edges represent conflicts.
    # Conflict between i and j if they cannot coexist in a 3-AP-free set.
    # They cannot coexist if there exists k such that {i, j, k} is a 3-AP.
    
    # Let's build the adjacency matrix A (0/1) where A[i,j] = 1 if i and j conflict.
    # Note: In the standard graph formulation for Independent Set, an edge means "cannot be together".
    
    # Efficient construction:
    # For each i < j:
    #   Check if (i, j, k) is AP for some k in [1, N].
    #   k = 2j - i. If 1 <= k <= N and k != i and k != j, then conflict.
    #   k = 2i - j. If 1 <= k <= N and k != i and k != j, then conflict.
    #   If (i + j) % 2 == 0: k = (i + j) / 2. If 1 <= k <= N and k != i and k != j, then conflict.
    
    # Actually, simpler:
    # A set S is 3-AP free if it contains no (a, b, c) with a+b=2c, a!=b!=c!=a.
    # So i and j conflict if there exists k such that {i, j, k} is a 3-AP.
    # The possible k values are:
    # 1. k = 2j - i (if i, j, k is AP with j middle? No, i, j, k => j-i = k-j => k = 2j - i. Here j is middle? No, i, j, k order doesn't matter for the set. The condition is a+b=2c.
    #    Pairs in a 3-AP {a, b, c} are (a,b), (a,c), (b,c).
    #    If we have i and j, they conflict if there is a k such that {i, j, k} is a 3-AP.
    #    This happens if:
    #    - i + j = 2k => k = (i+j)/2 (integer)
    #    - i + k = 2j => k = 2j - i
    #    - j + k = 2i => k = 2i - j
    
    # So, for each pair (i, j) with i < j:
    #   if (i + j) % 2 == 0 and 1 <= (i+j)//2 <= N and (i+j)//2 != i and (i+j)//2 != j: conflict
    #   if 1 <= 2*j - i <= N and 2*j - i != i and 2*j - i != j: conflict
    #   if 1 <= 2*i - j <= N and 2*i - j != i and 2*i - j != j: conflict
    
    # Note: Since i < j, 2*i - j < i < j, so if it's in range, it's distinct from i and j.
    # 2*j - i > j > i, so if it's in range, it's distinct from i and j.
    # (i+j)/2 is between i and j, so distinct.
    
    adj = np.zeros((N, N), dtype=int)
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            conflict = False
            # Case 1: k = (i + j) / 2
            if (i + j) % 2 == 0:
                k = (i + j) // 2
                if 1 <= k <= N:
                    conflict = True
            # Case 2: k = 2*j - i
            k2 = 2 * j - i
            if 1 <= k2 <= N:
                conflict = True
            # Case 3: k = 2*i - j
            k3 = 2 * i - j
            if 1 <= k3 <= N:
                conflict = True
            
            if conflict:
                adj[i-1][j-1] = 1
                adj[j-1][i-1] = 1
                
    # 2. Formulate ILP
    # Maximize sum(1/x * x_i)
    # Subject to:
    #   x_i + x_j <= 1 for all edges (i, j)
    #   x_i in {0, 1}
    
    # Weights
    weights = np.array([1.0 / x for x in range(1, N + 1)])
    
    # Objective for linprog (minimize -c^T x)
    c = -weights
    
    # Constraints: A_ub x <= b_ub
    # We need to construct A_ub from the adjacency matrix.
    # Only include edges where adj[i,j] == 1.
    
    rows = []
    cols = []
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i][j] == 1:
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
                
    if len(rows) > 0:
        A_ub = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(rows), N))
    else:
        A_ub = csr_matrix((0, N))
        
    b_ub = np.ones(len(rows))
    
    # Bounds: 0 <= x_i <= 1
    bounds = [(0, 1) for _ in range(N)]
    
    # Integrality: x_i in {0, 1}
    integrality = [1] * N
    
    # Solve
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, integrality=integrality, method='highs')
    except Exception as e:
        # Fallback to greedy if ILP fails
        return _greedy_fallback()
    
    if time.time() > deadline:
        return _greedy_fallback()
        
    if res.status == 0:
        # res.x is the solution
        x_sol = res.x
        # Threshold at 0.5 to get binary solution
        selected = [i + 1 for i in range(N) if x_sol[i] > 0.5]
        
        # Validate
        if _is_3ap_free(selected):
            return selected
        else:
            # If validation fails (shouldn't happen with exact solver), fallback
            return _greedy_fallback()
    else:
        return _greedy_fallback()

def _is_3ap_free(s):
    if not s:
        return True
    s_set = set(s)
    lst = list(s)
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            a, b = lst[i], lst[j]
            # Check if a, b, c is AP with c > b
            c = 2 * b - a
            if c > b and c in s_set:
                return False
    return True

def _greedy_fallback():
    """Greedy fallback in case ILP fails or times out."""
    s = []
    s_set = set()
    for x in range(1, N + 1):
        can_add = True
        for y in s:
            z1 = 2 * x - y
            if z1 in s_set:
                can_add = False
                break
            if (x + y) % 2 == 0:
                z2 = (x + y) // 2
                if z2 in s_set and z2 != x and z2 != y:
                    can_add = False
                    break
        if can_add:
            s.append(x)
            s_set.add(x)
    return s

if __name__ == "__main__":
    res = construct()
    print(res)
    score = sum(1.0/x for x in res)
    print(f"Score: {score}")
    print(f"Size: {len(res)}")
    print(f"Valid: {_is_3ap_free(res)}")
