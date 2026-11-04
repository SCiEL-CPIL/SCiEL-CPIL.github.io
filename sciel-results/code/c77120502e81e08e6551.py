import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Spectral Initialization
    # The QAP can be relaxed to a linear assignment problem using the principal
    # eigenvectors of A and B. Facilities with similar spectral signatures should
    # be assigned to locations with similar spectral signatures.
    
    # Compute the sum matrix to capture both flow and distance structure for initialization
    # A common heuristic is to use the eigenvectors of A and B separately or their sum.
    # Let's use the leading eigenvector of A and B to create a "position" vector.
    
    # For speed and robustness, we can use the row sums as a first approximation, 
    # but spectral is better. Let's compute the top eigenvector of A and B.
    # However, eigendecomposition of 150x150 is fast.
    
    # Use the principal eigenvector of A (flows) and B (distances)
    # We want to match high flow facilities to low distance locations? 
    # Actually, the spectral method for QAP usually involves minimizing the trace of P^T B P A.
    # A standard relaxation is to assign facility i to location j that minimizes 
    # sum_k A[i,k] * <v_k, u_j> where v_k are eigenvectors of A and u_j of B.
    
    # Simpler and effective: 
    # 1. Compute row sums of A (total flow out of i) and B (total distance from j).
    # 2. But spectral is more nuanced. Let's use the first few eigenvectors.
    
    # For n=150, full eigendecomposition is ~150^3 = 3.3M ops, very fast.
    w_a, v_a = np.linalg.eigh(a)
    w_b, v_b = np.linalg.eigh(b)
    
    # Take the top k eigenvectors (e.g., 5-10) to embed in low-dim space
    k = 10
    # Eigenvectors are columns. v_a[:, i] is the i-th eigenvector.
    # We want to project facilities and locations into this subspace.
    # Facility i is represented by its coordinates in the basis of v_a.
    # Location j is represented by its coordinates in the basis of v_b.
    
    # Actually, a common spectral initialization for QAP is:
    # Minimize sum_i,j A[i,j] B[p[i], p[j]]
    # Relax P to orthogonal matrix V. V = v_a v_b^T (if dimensions match and we use top eigenvectors).
    # Then round to permutation using Hungarian on a cost matrix derived from the relaxed solution.
    
    # Let's construct the cost matrix for Hungarian:
    # C[i, j] = - sum_k A[i, k] * (v_a_k^T v_b_j) ? No.
    
    # Standard approach:
    # 1. Compute eigendecomposition of A: A = V_a Lambda_a V_a^T
    # 2. Compute eigendecomposition of B: B = V_b Lambda_b V_b^T
    # 3. The relaxed solution is P* = V_a V_b^T (aligning eigenvectors).
    # 4. We need to round P* to a permutation. We can use the Hungarian algorithm on a cost matrix 
    #    that measures the "distance" between the assignment implied by P* and the actual assignment.
    #    Alternatively, we can simply use the row sums of the product of the spectral embeddings.
    
    # Let's use a simpler, robust spectral init:
    # Project A and B onto the top k eigenspaces.
    # A_k = V_a_k Lambda_a_k V_a_k^T
    # B_k = V_b_k Lambda_b_k V_b_k^T
    # We want to minimize trace(P^T B_k P A_k).
    # This is still a QAP.
    
    # Let's stick to the "rank-matching" but enhanced with spectral info.
    # Actually, the inspiration 3 mentions "spectral init + Hungarian".
    # Let's try:
    # Cost matrix C[i, j] = - (A_row_sums[i] * B_col_sums[j]) is the simple one.
    # Spectral: 
    # Let u_i be the i-th row of the top-k eigenvector matrix of A.
    # Let v_j be the j-th row of the top-k eigenvector matrix of B.
    # We want to assign i to j such that u_i and v_j are "compatible".
    # If A and B have similar structures, we might want to match high components to high components.
    # However, QAP minimizes sum A[i,j] B[p[i], p[j]].
    # If A[i,j] is large, we want B[p[i], p[j]] to be small.
    
    # Let's use the following heuristic for initialization:
    # 1. Compute the "center of mass" for each facility based on flows: c_a[i] = sum_j A[i,j] * x_j (where x_j is some reference).
    # Too complex.
    
    # Let's use the simple rank matching as a baseline, but improve it with a 2-opt search.
    # The rank matching is O(n log n). The 2-opt search is the main driver.
    
    # Re-evaluating: The current program is just rank matching. 
    # The key is the local search.
    
    # Initialize with rank matching (fast and decent)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Compute initial cost
    # Cost = sum_{i,j} A[i,j] B[p[i], p[j]]
    # We can compute this efficiently.
    # Let P be the permutation matrix. Cost = trace(P A P^T B) ? No.
    # Cost = sum_i sum_j A[i,j] B[p[i], p[j]]
    
    def compute_cost(p):
        # p[i] is the location of facility i
        # We need sum A[i,j] * B[p[i], p[j]]
        # Let's create a matrix B_p where B_p[i,j] = B[p[i], p[j]]
        # Then cost = sum A[i,j] * B_p[i,j] = np.sum(a * b[np.ix_(p, p)])
        return np.sum(a * b[np.ix_(p, p)])
    
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Local Search: 2-opt (swap two locations)
    # Delta cost for swapping locations of facilities i and j:
    # Let p_i = p[i], p_j = p[j].
    # New p'[i] = p_j, p'[j] = p_i, others same.
    # Old cost terms involving i or j:
    # sum_k A[i,k] B[p_i, p[k]] + sum_k A[k,i] B[p[k], p_i] 
    # + sum_k A[j,k] B[p_j, p[k]] + sum_k A[k,j] B[p[k], p_j]
    # - A[i,i] B[p_i, p_i] - A[j,j] B[p_j, p_j] (if i!=j, these are counted twice in the sums above? 
    # Wait, the sum is over all i,j.
    # Let's derive the delta carefully.
    
    # C = sum_{u,v} A[u,v] B[p[u], p[v]]
    # Terms changing are those where u=i, v=i; u=i, v=j; u=j, v=i; u=j, v=j; 
    # and all u where v=i or v=j, and all v where u=i or u=j.
    
    # Actually, it's easier to just compute the cost of the new permutation if we do it naively?
    # No, O(n^2) per swap is too slow for 150^2 * many iterations.
    # We need O(n) or O(1) delta.
    
    # Let's define:
    # F[i] = sum_k A[i,k] B[p[i], p[k]]  (Outflow cost from i to all others, based on current p)
    # G[i] = sum_k A[k,i] B[p[k], p[i]]  (Inflow cost from all others to i, based on current p)
    # Note: F[i] includes A[i,i]B[p[i],p[i]].
    
    # If we swap p[i] and p[j] (let x = p[i], y = p[j]):
    # New p'[i] = y, p'[j] = x.
    
    # Change in cost:
    # 1. Terms involving i as source:
    #    Old: sum_k A[i,k] B[x, p[k]]
    #    New: sum_k A[i,k] B[y, p'[k]]
    #    Note p'[k] = p[k] for k != i,j. p'[i]=y, p'[j]=x.
    #    So New_i_out = A[i,i]B[y,y] + A[i,j]B[y,x] + sum_{k!=i,j} A[i,k]B[y, p[k]]
    #    Old_i_out = A[i,i]B[x,x] + A[i,j]B[x, p[j]] + sum_{k!=i,j} A[i,k]B[x, p[k])
    #               = A[i,i]B[x,x] + A[i,j]B[x,y] + sum_{k!=i,j} A[i,k]B[x, p[k])
    
    # 2. Terms involving i as dest:
    #    Old: sum_k A[k,i] B[p[k], x]
    #    New: sum_k A[k,i] B[p'[k], y]
    #    New_i_in = A[i,i]B[y,y] + A[j,i]B[x,y] + sum_{k!=i,j} A[k,i]B[p[k], y]
    #    Old_i_in = A[i,i]B[x,x] + A[j,i]B[p[j], x] + sum_{k!=i,j} A[k,i]B[p[k], x]
    #               = A[i,i]B[x,x] + A[j,i]B[y,x] + sum_{k!=i,j} A[k,i]B[p[k], x]
    
    # 3. Similar for j.
    
    # This is getting complicated. Let's use a simpler incremental update strategy.
    # Precompute for each location l:
    # row_sum_A[l] ? No, A is facility-flow, B is location-distance.
    
    # Let's try a different local search: **Or-opt** or just **Random Swaps** with full cost re-evaluation?
    # For n=150, full cost re-evaluation is O(n^2) = 22,500 ops.
    # If we do 10,000 swaps, that's 225M ops. In NumPy, this might be feasible within 100s?
    # 225M ops in Python/NumPy... NumPy is C-speed. 225M simple multiply-adds is ~0.1-0.5 seconds.
    # So we can afford full re-evaluation for many swaps!
    
    # Strategy:
    # 1. Start with rank matching.
    # 2. Loop:
    #    a. Pick two random indices i, j.
    #    b. Swap p[i], p[j].
    #    c. Compute new cost.
    #    d. If new cost < current cost, accept.
    #    e. Else, revert.
    # 3. Repeat until time limit.
    
    # This is Simulated Annealing with T=0 (Greedy Hill Climbing).
    # To escape local minima, we can add a small probability of accepting worse moves (SA).
    
    # Let's implement Greedy Hill Climbing with Random Restarts if stuck?
    # Or just run for the time limit.
    
    # Optimization: 
    # Instead of full re-evaluation, we can cache the cost matrix B_p = B[np.ix_(p,p)].
    # When we swap p[i] and p[j], only rows i, j and cols i, j of B_p change?
    # No, B_p[i, k] = B[p[i], p[k]]. If p[i] changes, row i of B_p changes.
    # If p[j] changes, col j of B_p changes? No, B_p[k, j] = B[p[k], p[j]]. So col j changes.
    # So swapping p[i] and p[j] changes row i, col i, row j, col j of B_p.
    # We can update the cost by subtracting the old contributions of these rows/cols and adding the new ones.
    
    # Let's implement the O(n) delta update.
    
    # Precompute:
    # cost = sum_{u,v} A[u,v] B_p[u,v]
    
    # Let's maintain:
    # row_contrib[i] = sum_v A[i,v] B_p[i,v]
    # col_contrib[j] = sum_u A[u,j] B_p[u,j]
    # Note: cost = sum_i row_contrib[i] = sum_j col_contrib[j]
    # Also, the term A[i,i]B_p[i,i] is counted in both row_contrib[i] and col_contrib[i].
    # So sum_i row_contrib[i] = cost.
    
    # If we swap p[i] and p[j] (x=p[i], y=p[j]):
    # New B_p rows/cols:
    # B_p'[i, v] = B[y, p'[v]]
    # B_p'[v, i] = B[p'[v], y]
    # B_p'[j, v] = B[x, p'[v]]
    # B_p'[v, j] = B[p'[v], x]
    # For v not in {i,j}, p'[v] = p[v].
    
    # Delta cost = New Cost - Old Cost
    # New Cost = sum_{u,v} A[u,v] B_p'[u,v]
    
    # Let's compute the change in row_contrib and col_contrib.
    
    # This is still a bit tricky to get right without bugs.
    # Given the time limit and n=150, let's try the **Full Re-evaluation** first.
    # It is simpler and less prone to bugs.
    
    # Estimate:
    # 100 seconds / 0.005 seconds per iteration (optimistic) = 20,000 iterations.
    # 100 seconds / 0.05 seconds per iteration (pessimistic) = 2,000 iterations.
    # Even 2,000 random swaps might improve the rank-matching solution significantly.
    
    # Let's add a "kick" or "restart" mechanism if no improvement for X iterations.
    
    current_p = p.copy()
    current_cost = compute_cost(current_p)
    best_p = current_p.copy()
    best_cost = current_cost
    
    no_improve_count = 0
    max_no_improve = 1000  # Reset if stuck
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Random restart or perturbation
            # Swap 5 random pairs
            for _ in range(5):
                i, j = np.random.choice(n, 2, replace=False)
                current_p[i], current_p[j] = current_p[j], current_p[i]
            current_cost = compute_cost(current_p)
            no_improve_count = 0
            continue
            
        i, j = np.random.choice(n, 2, replace=False)
        
        # Swap
        current_p[i], current_p[j] = current_p[j], current_p[i]
        
        new_cost = compute_cost(current_p)
        
        if new_cost < current_cost:
            current_cost = new_cost
            no_improve_count = 0
            if new_cost < best_cost:
                best_cost = new_cost
                best_p = current_p.copy()
        else:
            # Revert
            current_p[i], current_p[j] = current_p[j], current_p[i]
            no_improve_count += 1
            
    return best_p.tolist()

# EVOLVE-BLOCK-END
