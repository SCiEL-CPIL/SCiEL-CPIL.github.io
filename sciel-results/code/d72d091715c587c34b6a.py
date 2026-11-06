import time
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import milp, LinearConstraint, Bounds
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s deadline
    
    n = 512
    
    # 1. Precompute 4-APs
    # List of tuples (a, b, c, d) 0-indexed
    ap_list = []
    # Incidence: list of lists, ap_inc[i] = list of indices in ap_list containing i
    ap_inc = [[] for _ in range(n)]
    
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            idxs = (a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1)
            ap_idx = len(ap_list)
            ap_list.append(idxs)
            for idx in idxs:
                ap_inc[idx].append(ap_idx)
    
    # 2. Local Search Function
    def local_search(init_mask, time_limit):
        """
        init_mask: numpy array of bools
        time_limit: seconds
        """
        mask = init_mask.copy()
        # counts[j] = number of selected elements in ap_list[j]
        counts = np.zeros(len(ap_list), dtype=np.int32)
        
        # Initialize counts
        for i in range(n):
            if mask[i]:
                for ap_idx in ap_inc[i]:
                    counts[ap_idx] += 1
        
        best_mask = mask.copy()
        best_size = np.sum(mask)
        
        # Track violations: indices of APs where counts == 4
        # We can just check counts == 4
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            # Check for violations
            violated = np.where(counts == 4)[0]
            
            if len(violated) > 0:
                # Remove an element from a random violated AP
                v_ap_idx = violated[random.randint(0, len(violated)-1)]
                a, b, c, d = ap_list[v_ap_idx]
                # Choose one to remove. Heuristic: random
                rem_idx = [a, b, c, d][random.randint(0, 3)]
                
                # Remove rem_idx
                mask[rem_idx] = False
                for ap_idx in ap_inc[rem_idx]:
                    counts[ap_idx] -= 1
            else:
                # No violations, try to add an element
                # Find candidates: not in mask
                candidates = np.where(~mask)[0]
                if len(candidates) == 0:
                    break
                
                # Pick a random candidate
                cand = candidates[random.randint(0, len(candidates)-1)]
                
                # Check if adding cand creates a violation
                # Temporarily add
                mask[cand] = True
                new_counts = counts.copy() # This is slow. Better to check incrementally.
                
                # Check incrementally
                can_add = True
                for ap_idx in ap_inc[cand]:
                    if counts[ap_idx] == 3:
                        can_add = False
                        break
                
                if can_add:
                    # Commit
                    for ap_idx in ap_inc[cand]:
                        counts[ap_idx] += 1
                    # Update best
                    cur_size = np.sum(mask)
                    if cur_size > best_size:
                        best_size = cur_size
                        best_mask = mask.copy()
                else:
                    # Revert
                    mask[cand] = False
            
            # Periodically check best
            if time.time() > end_time:
                break
                
        return best_mask

    # 3. Initial Solution: Ternary 0/1 set
    def ternary_01_set(n):
        result = np.zeros(n, dtype=bool)
        for mask in range(1, 1 << 6):
            val = 0
            for bit in range(6):
                if mask & (1 << bit):
                    val += 3 ** bit
            if 1 <= val <= n:
                result[val - 1] = True
        return result

    init_mask = ternary_01_set(n)
    
    # 4. Run Local Search for a short time to get a good LB
    ls_time = 5.0
    if time.time() < deadline - 5.0:
        ls_mask = local_search(init_mask, ls_time)
    else:
        ls_mask = init_mask
    
    # 5. Try ILP
    # Build ILP constraints
    # A @ x <= 3
    # A is sparse
    A = lil_matrix((len(ap_list), n), dtype=np.float64)
    b_ub = np.ones(len(ap_list)) * 3.0
    
    for i, ap in enumerate(ap_list):
        A[i, ap[0]] = 1.0
        A[i, ap[1]] = 1.0
        A[i, ap[2]] = 1.0
        A[i, ap[3]] = 1.0
        
    A_csr = A.tocsr()
    c = -np.ones(n)
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(n)
    constraint = LinearConstraint(A_csr, lb=-np.inf, ub=b_ub)
    
    ilp_mask = None
    remaining_time = deadline - time.time() - 2.0 # 2s safety margin
    
    if remaining_time > 5.0:
        try:
            # Use HiGHS time limit
            options = {"time_limit": remaining_time}
            res = milp(c, constraints=constraint, integrality=integrality, bounds=bounds, options=options)
            
            if res.status == 0: # Optimal
                x = res.x
                ilp_mask = x > 0.5
            elif res.status == 2: # Time limit reached, but maybe has a feasible solution?
                # milp doesn't guarantee a feasible solution on time limit in all cases
                # But we can check if res.x is valid
                if res.x is not None:
                    # Verify validity
                    x = res.x > 0.5
                    # Check 4-APs
                    # This check is O(E), which is fast
                    valid = True
                    for ap in ap_list:
                        if x[ap[0]] and x[ap[1]] and x[ap[2]] and x[ap[3]]:
                            valid = False
                            break
                    if valid:
                        ilp_mask = x
        except Exception:
            pass
    
    # 6. Decide which result to return
    # Compare sizes
    ls_size = np.sum(ls_mask)
    ilp_size = np.sum(ilp_mask) if ilp_mask is not None else 0
    
    if ilp_mask is not None and ilp_size > ls_size:
        return sorted([i+1 for i in range(n) if ilp_mask[i]])
    else:
        return sorted([i+1 for i in range(n) if ls_mask[i]])

# EVOLVE-BLOCK-END
