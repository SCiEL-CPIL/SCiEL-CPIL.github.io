"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    best_score = 0
    best_set = [0, 1, 3]  # seed fallback
    
    def compute_score(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        # Use sets for sum and diff
        sumset = set()
        diffset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
        
        n_sum = len(sumset)
        n_diff = len(diffset)
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score
    
    # Pre-compute seed
    best_score = compute_score([0, 1, 3])
    
    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if max(cand) == 0:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # Phase 1: Exhaustive search for small sets (size <= 6)
    # This is feasible for small max_u
    max_u_limit_small = 20
    for max_u in range(1, max_u_limit_small + 1):
        if time.time() > deadline - 5:
            break
        # Generate all subsets of {1, ..., max_u} of size 2 to 5
        # and add 0 and max_u (optional)
        # Actually, just iterate over all subsets of {0, ..., max_u}
        # But that's 2^max_u which is too big for max_u=20 (1M)
        # Instead, iterate over combinations of size 3 to 6 from {1, ..., max_u}
        # and always include 0.
        
        elements = list(range(1, max_u + 1))
        for size in range(3, 7):
            if len(elements) < size:
                continue
            for comb in combinations(elements, size):
                cand = sorted([0] + list(comb))
                try_set(cand)
    
    # Phase 2: Structured sets with larger max_u
    # Patterns:
    # 1. Two blocks: {0, ..., a} U {b, ..., b+a}
    # 2. Geometric: {0, 1, r, r^2, ...}
    # 3. Arithmetic: {0, d, 2d, ...}
    
    for max_u in range(10, 200):
        if time.time() > deadline - 5:
            break
        
        # Two blocks
        for a in range(1, 10): # size of first block
            for b in range(a + 1, max_u - a + 1):
                if time.time() > deadline - 5:
                    break
                block1 = list(range(0, a + 1))
                block2 = list(range(b, b + a + 1))
                cand = sorted(set(block1 + block2))
                if max(cand) <= max_u:
                    try_set(cand)
        
        # Geometric with small ratios
        for r in [2, 3, 4, 5]:
            geo = [0, 1]
            while geo[-1] * r <= max_u:
                geo.append(geo[-1] * r)
            try_set(geo)
            
        # Arithmetic progressions
        for d in range(1, 10):
            ap = []
            v = 0
            while v <= max_u:
                ap.append(v)
                v += d
            try_set(ap)
            
        # Mixed: {0, 1, ..., a} U {b}
        for a in range(1, 10):
            for b in range(a + 1, max_u + 1):
                cand = sorted(list(range(0, a + 1)) + [b])
                try_set(cand)
                
        # Mixed: {0, 1, ..., a} U {b, c}
        for a in range(1, 5):
            for b in range(a + 1, max_u + 1):
                for c in range(b + 1, max_u + 1):
                    cand = sorted(list(range(0, a + 1)) + [b, c])
                    try_set(cand)
                    
    # Phase 3: Local search around best set
    if time.time() < deadline - 10:
        current = list(best_set)
        no_improve = 0
        max_no_improve = 50
        
        while time.time() < deadline - 10 and no_improve < max_no_improve:
            improved = False
            
            # Try adding an element
            if len(current) < 15:
                max_cur = max(current)
                # Try elements near existing ones and at the end
                candidates = set()
                for x in current:
                    for dx in [-2, -1, 1, 2]:
                        candidates.add(x + dx)
                candidates.update(range(max_cur + 1, max_cur + 10))
                candidates.discard(0)
                
                for new_el in sorted(candidates):
                    if new_el in current:
                        continue
                    cand = sorted(current + [new_el])
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        break
            
            # Try replacing an element
            if len(current) >= 3:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    # Try replacing with nearby values
                    for new_el in range(max(1, current[i] - 5), current[i] + 6):
                        if new_el in current:
                            continue
                        cand = list(current)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand:
                            continue
                        s = compute_score(cand)
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                            current = list(cand)
                            improved = True
                            break
                    if improved:
                        break
            
            # Try removing an element
            if len(current) > 3:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    cand = [x for j, x in enumerate(current) if j != i]
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        break
            
            if not improved:
                no_improve += 1
            else:
                no_improve = 0
    
    return list(best_set)
# EVOLVE-BLOCK-END
