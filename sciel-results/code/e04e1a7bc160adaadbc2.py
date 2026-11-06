import math
import time
import random

def _is_disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Key insight: two residue classes are disjoint iff gcd(m1,m2) does not divide (r1-r2).
    # If gcd(m1,m2)==1, they ALWAYS intersect. So we need pairwise non-coprime moduli.
    # Largest clique in gcd>1 graph on [2..128]: all even numbers (64 of them).
    # Strategy: use all even moduli, then solve CSP for residues.
    
    moduli = list(range(2, 129, 2))  # 64 even moduli
    n = len(moduli)
    
    # Precompute gcd matrix
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                gcds[i][j] = math.gcd(moduli[i], moduli[j])
    
    # For each pair (i,j) with gcd g, the constraint is: residue_i NOT congruent to residue_j mod g
    # This is a binary CSP. Use backtracking with forward checking.
    
    # Order: process moduli in order of increasing modulus (more constrained first since fewer residues)
    order = list(range(n))
    order.sort(key=lambda i: moduli[i])
    
    def solve_csp():
        if time.time() > deadline:
            return None
        
        residues = [None] * n
        # domains[i] = set of valid residues for variable i given current assignments
        domains = [set(range(moduli[i])) for i in range(n)]
        
        def get_candidates(i):
            """Get valid residues for variable i given current assignments."""
            cands = []
            for r in domains[i]:
                valid = True
                for j in range(n):
                    if j == i:
                        continue
                    if residues[j] is not None:
                        if not _is_disjoint(r, moduli[i], residues[j], moduli[j]):
                            valid = False
                            break
                if valid:
                    cands.append(r)
            return cands
        
        def forward_check(var, val):
            """After assigning var=val, check if any unassigned variable has empty domain."""
            for j in range(n):
                if j == var or residues[j] is not None:
                    continue
                # Check if any value in domains[j] is still valid
                has_valid = False
                for rj in domains[j]:
                    if _is_disjoint(rj, moduli[j], val, moduli[var]):
                        has_valid = True
                        break
                if not has_valid:
                    return False
            return True
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == n:
                return True
            
            i = order[idx]
            cands = get_candidates(i)
            
            if not cands:
                return False
            
            # Try candidates in a heuristic order: prefer residues that constrain others least
            random.shuffle(cands)
            
            for r in cands:
                residues[i] = r
                if forward_check(i, r):
                    if backtrack(idx + 1):
                        return True
                residues[i] = None
            
            return False
        
        if backtrack(0):
            return residues
        return None
    
    # Try solving the CSP
    result = solve_csp()
    
    if result is not None:
        return [[result[i], moduli[i]] for i in range(n)]
    
    # Fallback: try with a subset - use multiples of 2 up to some limit
    # Or try a different approach: greedy with restarts
    def greedy_with_restarts():
        best_result = None
        for attempt in range(200):
            if time.time() > deadline:
                break
            res = [None] * n
            assigned = [False] * n
            # Process in order of increasing modulus
            ok = True
            for idx in order:
                m_k = moduli[idx]
                valid_res = []
                for r in range(m_k):
                    valid = True
                    for j in range(n):
                        if assigned[j]:
                            if not _is_disjoint(r, m_k, res[j], moduli[j]):
                                valid = False
                                break
                    if valid:
                        valid_res.append(r)
                if not valid_res:
                    ok = False
                    break
                res[idx] = random.choice(valid_res)
                assigned[idx] = True
            if ok:
                return res
            # Partial: count how many we got
            cnt = sum(1 for a in assigned if a)
            if best_result is None or cnt > best_result[0]:
                best_result = (cnt, [res[i] if assigned[i] else None for i in range(n)])
        if best_result and best_result[0] > 0:
            # Return partial solution
            pairs = []
            for i in range(n):
                if best_result[1][i] is not None:
                    pairs.append([best_result[1][i], moduli[i]])
            return pairs
        return None
    
    result2 = greedy_with_restarts()
    if result2 is not None and len(result2) > 0:
        return result2
    
    # Absolute fallback
    return [[0, 2], [1, 4]]
