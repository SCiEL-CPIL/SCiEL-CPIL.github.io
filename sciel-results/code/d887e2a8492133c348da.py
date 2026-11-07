"""
Strategy:
1. Determine the LCM of the divisors to understand the periodicity of the problem.
2. We want to find 10 distinct integers x_i such that x_i % d_i == 0.
3. Let the interval be [L, L+W]. We want to minimize W.
4. For W=16 (known optimum), rigid divisors (d>16) have at most 1 multiple in window.
   Precompute all L mod 30030 where rigid assignments are valid (distinct offsets).
   Then for each valid L, quickly check if flexible divisors can be assigned.
   This avoids iterating all 30030 L values with full backtracking.
"""

import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    divisors = P['divisors']
    n = len(divisors)
    
    # Calculate LCM
    def lcm(a, b):
        return a * b // gcd(a, b)
    LCM = reduce(lcm, divisors)
    
    # Search W from 16 upwards (16 is known optimum from exhaustive search)
    best_solution = None
    
    for W in range(16, 100):
        if time.time() > deadline:
            break
            
        # Identify rigid and flexible divisors
        rigid_indices = []
        flexible_indices = []
        for i, d in enumerate(divisors):
            if d > W:
                rigid_indices.append(i)
            else:
                flexible_indices.append(i)
        
        rigid_divs = [divisors[i] for i in rigid_indices]
        n_rigid = len(rigid_divs)
        
        # For W=16: rigid divisors are 21,22,26,33,35,39
        # Each has exactly one multiple in [L, L+16] for any L (since d > W+1)
        # Offset = (d - (L % d)) % d, but if L%d==0, offset=0
        # Actually: the smallest multiple >= L is ceil(L/d)*d
        # offset = ceil(L/d)*d - L
        # This is in [0, d-1]. For it to be <= W, we need offset <= W.
        
        # Precompute for each rigid divisor, which L mod d give valid offsets
        # and what the offset is
        rigid_offset_info = []
        for d in rigid_divs:
            # For each residue r = L % d, compute offset
            # offset = (d - r) % d
            # Valid if offset <= W
            valid_residues = {}
            for r in range(d):
                offset = (d - r) % d
                if offset <= W:
                    valid_residues[r] = offset
            rigid_offset_info.append(valid_residues)
        
        # Now iterate L from 0 to LCM-1
        # But we can be smarter: for each rigid divisor, only certain L mod d are valid
        # Let's just iterate L and check quickly
        
        found = False
        for L in range(0, LCM):
            if time.time() > deadline:
                break
            
            # Check rigid divisors: get offsets, ensure distinct
            used_offsets = set()
            rigid_assignments = {}
            valid = True
            
            for ri, idx in enumerate(rigid_indices):
                d = rigid_divs[ri]
                r = L % d
                if r not in rigid_offset_info[ri]:
                    valid = False
                    break
                offset = rigid_offset_info[ri][r]
                if offset in used_offsets:
                    valid = False
                    break
                used_offsets.add(offset)
                rigid_assignments[idx] = L + offset
            
            if not valid:
                continue
            
            # Check flexible divisors
            assignments = [None] * n
            for idx, val in rigid_assignments.items():
                assignments[idx] = val
            
            sol = _assign_flexible_fast(divisors, flexible_indices, L, W, used_offsets, assignments)
            if sol is not None:
                best_solution = sol
                found = True
                break
        
        if found:
            break
    
    if best_solution is None:
        # Fallback: return seed
        return [int(d) for d in P['divisors']]
    
    # Validate and return
    result = []
    for i, val in enumerate(best_solution):
        v = int(val)
        if not (1 <= v <= 1000000) or v % divisors[i] != 0:
            return [int(d) for d in P['divisors']]
        result.append(v)
    
    if len(set(result)) != len(result):
        return [int(d) for d in P['divisors']]
        
    return result

def _assign_flexible_fast(divisors, flexible_indices, L, W, used_offsets, assignments):
    """
    Fast assignment for flexible divisors using backtracking with MRV.
    """
    if not flexible_indices:
        return assignments
    
    # Generate candidates for each flexible index
    candidates = []
    for i in flexible_indices:
        d = divisors[i]
        cands = []
        # First multiple >= L
        k = (L + d - 1) // d
        val = k * d
        while val <= L + W:
            offset = val - L
            if offset not in used_offsets:
                cands.append(offset)
            val += d
        candidates.append(cands)
    
    # If any flexible divisor has no candidates, fail
    for cands in candidates:
        if not cands:
            return None
    
    # Order by fewest candidates (MRV)
    order = sorted(range(len(flexible_indices)), key=lambda x: len(candidates[x]))
    
    def backtrack(idx):
        if idx == len(order):
            return True
        fi = order[idx]
        orig_idx = flexible_indices[fi]
        for offset in candidates[fi]:
            if offset not in used_offsets:
                used_offsets.add(offset)
                assignments[orig_idx] = L + offset
                if backtrack(idx + 1):
                    return True
                used_offsets.remove(offset)
                assignments[orig_idx] = None
        return False
        
    if backtrack(0):
        return assignments
    else:
        return None

# EVOLVE-BLOCK-END
