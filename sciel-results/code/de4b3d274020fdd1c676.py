"""ILP-based construction of a 3-AP-free subset of [1,256] maximizing sum of reciprocals."""
import time
from fractions import Fraction
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

N = 256
DEADLINE = 90.0  # wall-clock limit; we aim to finish well before

def _build_conflicts():
    """Return list of (i, j) index pairs (0-based, i<j) that form a 3-AP with some third element.
    Actually, for 3-AP-free, we need: no three distinct elements a, b, c in S such that
    a + c = 2b. Equivalently, for any pair (a, c) with a < c and (a+c) even, the midpoint
    b = (a+c)//2 cannot also be in S.
    
    So the constraint is: for each valid triple (a, b, c) with a < b < c and a+c=2b,
    we cannot have all three selected. This is a 3-variable constraint, not pairwise.
    
    For ILP, we can use: x_a + x_b + x_c <= 2 for each such triple.
    """
    triples = []
    for b in range(2, N + 1):  # b is the middle term, 1-indexed
        for d in range(1, min(b - 1, N - b) + 1):  # common difference
            a = b - d
            c = b + d
            if 1 <= a and c <= N:
                triples.append((a - 1, b - 1, c - 1))  # 0-based indices
    return triples

def _greedy_solution():
    """Greedy: pick elements in decreasing order of 1/x, skip if it creates a 3-AP."""
    selected = set()
    # Process from small x to large x? No, we want high reciprocal weight, so prefer small x.
    # Greedy by descending 1/x means ascending x.
    for x in range(1, N + 1):
        idx = x - 1
        # Check if adding x creates any 3-AP
        ok = True
        for y in selected:
            # Check if x, y, z form a 3-AP for some z in selected or z=x or z=y
            # We need to check all triples involving x and two from selected
            pass
        # More efficient: for each pair (a, b) in selected, check if x completes a 3-AP
        # Triple (a, b, c) with a+b=2c or a+c=2b or b+c=2a
        # If we add x, we need to check:
        # 1. For each a in selected: is there b in selected such that a+x=2b or a+b=2x or b+x=2a?
        # Let's just check all pairs
        sel_list = list(selected)
        for i in range(len(sel_list)):
            for j in range(i + 1, len(sel_list)):
                a, b = sel_list[i], sel_list[j]
                # Check if {a, b, x} forms a 3-AP
                s = sorted([a, b, x])
                if s[2] - s[1] == s[1] - s[0]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            selected.add(x)
    return selected

def construct():
    start_time = time.time()
    
    # First, get a quick greedy solution as a fallback
    greedy_set = _greedy_solution()
    greedy_score = sum(1.0 / x for x in greedy_set)
    
    # Try ILP
    try:
        triples = _build_conflicts()
        n = N
        m = len(triples)
        
        # Objective: maximize sum(1/x_i * x_i_var) for i in 1..256
        # In scipy milp, we minimize, so we negate
        c = np.array([-1.0 / x for x in range(1, n + 1)])
        
        # Constraints: for each triple (a, b, c), x_a + x_b + x_c <= 2
        # A matrix of shape (m, n)
        A = np.zeros((m, n), dtype=np.int8)
        for k, (a, b, c) in enumerate(triples):
            A[k, a] = 1
            A[k, b] = 1
            A[k, c] = 1
        
        # Bounds: each variable is 0 or 1
        integrality = np.ones(n, dtype=np.int8)
        lb = np.zeros(n)
        ub = np.ones(n)
        
        # Time limit for the solver
        time_remaining = DEADLINE - (time.time() - start_time) - 5.0  # 5s safety margin
        if time_remaining < 1.0:
            return list(greedy_set)
        
        res = milp(
            c=c,
            constraints=[LinearConstraint(A, ub=2)],
            integrality=integrality,
            bounds=Bounds(lb, ub),
            options={'time_limit': time_remaining, 'mip_gap': 0.01}
        )
        
        if res.status == 0 and res.x is not None:
            selected = [i + 1 for i in range(n) if res.x[i] > 0.5]
            if len(selected) > 0:
                # Verify no 3-AP
                sel_set = set(selected)
                valid = True
                for x in selected:
                    for y in selected:
                        if x == y:
                            continue
                        # Check if there exists z such that x, y, z form a 3-AP
                        # z = 2y - x or z = 2x - y or z = (x+y)/2
                        for z in [2*y - x, 2*x - y]:
                            if z in sel_set and z != x and z != y:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    return selected
                else:
                    # Fallback to greedy
                    return list(greedy_set)
            else:
                return list(greedy_set)
        else:
            return list(greedy_set)
            
    except Exception:
        return list(greedy_set)

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
