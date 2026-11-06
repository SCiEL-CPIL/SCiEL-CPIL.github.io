import time
from math import gcd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin to be safe against 60s/90s limits
    
    N = 256
    # Nodes are 2 to 256
    nodes = list(range(2, N + 1))
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Weights are 1/node
    c = np.array([-1.0 / node for node in nodes])  # Minimize negative sum
    
    # Build incompatibility pairs
    # Two nodes a, b are incompatible if lcm(a, b) <= 256
    # We need constraints x_a + x_b <= 1 for all such pairs
    A_rows = []
    A_cols = []
    b_vals = []
    
    # Precompute gcd or just compute on the fly
    # Number of pairs is ~32k, which is small
    for i in range(n):
        a = nodes[i]
        for j in range(i + 1, n):
            b = nodes[j]
            g = gcd(a, b)
            lcm_val = (a * b) // g
            if lcm_val <= N:
                # Incompatible
                A_rows.append(i)
                A_cols.append(j)
                b_vals.append(1)
    
    if not A_rows:
        # No constraints? Unlikely, but handle it
        # If no constraints, take all nodes
        return nodes
    
    # Create sparse matrix for constraints
    # Using scipy.sparse
    from scipy.sparse import csr_matrix
    m = len(b_vals)
    A = csr_matrix((b_vals, (A_rows, A_cols)), shape=(m, n))
    
    # Objective is to minimize c^T x, where c is -1/node
    # Constraints: A x <= b_vals
    # Bounds: x in [0, 1] and integer
    
    # Define linear constraints
    # milp expects constraints in form lb <= A @ x <= ub
    # Here we have A @ x <= 1
    constraints = LinearConstraint(A, lb=None, ub=b_vals)
    
    # Bounds
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality
    integrality = np.ones(n)  # All variables are binary/integer
    
    try:
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds
        )
        
        if res.status == 0:  # Optimal
            # res.x contains the solution
            # Due to floating point, we threshold at 0.5
            selected_indices = [i for i in range(n) if res.x[i] > 0.5]
            solution = [nodes[i] for i in selected_indices]
            
            # Verify solution validity just in case
            valid = True
            sol_set = set(solution)
            for i in range(len(solution)):
                a = solution[i]
                for j in range(i + 1, len(solution)):
                    b = solution[j]
                    g = gcd(a, b)
                    lcm_val = (a * b) // g
                    if lcm_val <= N:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and solution:
                return solution
            elif solution:
                # If invalid, it's a bug in our logic, but return what we got if non-empty
                # Or fallback
                pass
            else:
                # Empty solution, fallback
                pass
        else:
            # Solver failed
            pass
    except Exception as e:
        pass
    
    # Fallback: Use the set [129, 256] which is likely valid and has decent score
    # Or a simple greedy
    # Let's just return the large numbers as a safe bet
    fallback = list(range(129, N + 1))
    
    # Verify fallback
    valid = True
    for i in range(len(fallback)):
        a = fallback[i]
        for j in range(i + 1, len(fallback)):
            b = fallback[j]
            g = gcd(a, b)
            lcm_val = (a * b) // g
            if lcm_val <= N:
                valid = False
                break
        if not valid:
            break
    
    if valid:
        return fallback
    else:
        # If even this fails (unlikely), return a single element
        return [N]

# EVOLVE-BLOCK-END
