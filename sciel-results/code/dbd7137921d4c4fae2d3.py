import time
from math import gcd

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000

def _lcm(a, b):
    return a * b // gcd(a, b)

# Compute LCM of all divisors for periodicity
LCM_ALL = 1
for d in DIVISORS:
    LCM_ALL = _lcm(LCM_ALL, d)
# LCM_ALL = 30030

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # We search for the minimum width W.
    # Lower bound: 10 distinct integers in [L, L+W] implies W >= 9.
    for W in range(9, 500):
        if time.time() > deadline:
            break
            
        # Scan starting positions L.
        # The pattern repeats every LCM_ALL.
        for L in range(1, LCM_ALL + 1):
            if time.time() > deadline:
                break
                
            if L + W > MAX_VAL:
                break
            
            # 1. Fast Feasibility Check (Gap Check)
            # For each divisor d, check if there is a multiple in [L, L+W]
            # First multiple >= L is k*d where k = ceil(L/d)
            # We need k*d <= L+W
            feasible = True
            candidates = []
            for i in range(N):
                d = DIVISORS[i]
                # Calculate first multiple >= L
                # k = (L + d - 1) // d
                # val = k * d
                # If val > L + W, no candidate
                # If val <= L + W, candidates are val, val+d, ... <= L+W
                
                # Optimization: Calculate remainder to find distance to next multiple
                rem = L % d
                # Distance to next multiple of d
                # If L is multiple, distance is 0. Else distance is d - rem.
                # Wait, if L % d == 0, next multiple is L (dist 0).
                # If L % d != 0, next multiple is L + (d - L%d).
                
                if rem == 0:
                    first_val = L
                else:
                    first_val = L + (d - rem)
                
                if first_val > L + W:
                    feasible = False
                    break
                
                # Generate candidates for this divisor
                cand_list = []
                v = first_val
                while v <= L + W:
                    cand_list.append(v)
                    v += d
                candidates.append(cand_list)
            
            if not feasible:
                continue
            
            # 2. Solve SDR using Unit Propagation + Backtracking
            # State: list of candidate lists, some may be reduced
            # We work on a copy to allow backtracking if needed, 
            # but Unit Propagation is deterministic.
            
            # Deep copy candidates for the solver
            domains = [list(c) for c in candidates]
            
            # Unit Propagation
            # 1. Find all domains with size 1.
            # 2. Assign their values.
            # 3. If conflict (same value assigned to two domains), fail.
            # 4. Remove assigned values from other domains.
            # 5. Repeat.
            
            def solve_sdr(domains):
                n = len(domains)
                # assignment[i] = value or None
                assignment = [None] * n
                # active[i] = True if domain i is not yet assigned
                active = [True] * n
                
                # We need to track which values are used
                used_values = set()
                
                # Queue for unit propagation
                # We can just loop until no changes or failure
                changed = True
                while changed:
                    changed = False
                    # Check for failures: any domain empty?
                    for i in range(n):
                        if active[i] and not domains[i]:
                            return None
                        # If domain size 1, assign
                        if active[i] and len(domains[i]) == 1:
                            val = domains[i][0]
                            if val in used_values:
                                return None
                            assignment[i] = val
                            used_values.add(val)
                            active[i] = False
                            changed = True
                            
                    # Remove used values from other domains
                    if changed:
                        for i in range(n):
                            if active[i]:
                                # Filter out used values
                                new_domain = [v for v in domains[i] if v not in used_values]
                                if len(new_domain) < len(domains[i]):
                                    domains[i] = new_domain
                                    # If it became empty, we'll catch it in next loop
                                    # If it became 1, we'll catch it in next loop
                                    changed = True # Trigger another pass to catch new units
                
                # If all assigned, success
                if not any(active):
                    return assignment
                
                # If some remain, use backtracking
                # Sort remaining indices by domain size (most constrained first)
                remaining_indices = [i for i in range(n) if active[i]]
                remaining_indices.sort(key=lambda i: len(domains[i]))
                
                def backtrack(idx):
                    if idx == len(remaining_indices):
                        return True
                    i = remaining_indices[idx]
                    for val in domains[i]:
                        if val not in used_values:
                            # Assign
                            assignment[i] = val
                            used_values.add(val)
                            # Remove from other active domains
                            saved_domains = []
                            valid = True
                            for j in range(n):
                                if active[j] and j != i:
                                    saved_domains.append((j, list(domains[j])))
                                    new_d = [v for v in domains[j] if v != val]
                                    if not new_d:
                                        valid = False
                                        break
                                    domains[j] = new_d
                            
                            if valid and backtrack(idx + 1):
                                return True
                            
                            # Undo
                            used_values.remove(val)
                            assignment[i] = None
                            if valid:
                                for j, d in saved_domains:
                                    domains[j] = d
                    return False
                
                return backtrack(0)

            result = solve_sdr(domains)
            if result is not None:
                # Verify types and bounds
                # result is a list of ints
                return [int(x) for x in result]
    
    # Fallback: return a valid set if no small W found
    # e.g., x_i = d_i * (i+1) * 100
    # This might have a large range, but it's valid.
    return [int(d * (i + 1) * 100) for i, d in enumerate(DIVISORS)]
