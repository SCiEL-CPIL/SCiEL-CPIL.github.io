import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Initial set of classes using powers of 2
    # Moduli: 2, 4, 8, 16, 32, 64, 128
    # Residues: 1, 3, 7, 15, 31, 63, 127
    initial_classes = [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    
    # Candidates are all moduli from 2 to 128 not in initial_classes
    used_moduli = {m for _, m in initial_classes}
    candidates = [m for m in range(2, 129) if m not in used_moduli]
    
    # Sort candidates to improve search efficiency?
    # Maybe sort by number of prime factors or just ascending?
    # Let's try ascending first.
    candidates.sort()
    
    best_result = initial_classes[:]
    
    def is_disjoint(r1, m1, r2, m2):
        g = math.gcd(m1, m2)
        return (r1 - r2) % g != 0
        
    def can_add(r, m, current_classes):
        for r_c, m_c in current_classes:
            if not is_disjoint(r, m, r_c, m_c):
                return False
        return True

    def get_valid_residues(m, current_classes):
        valid = []
        for r in range(m):
            if can_add(r, m, current_classes):
                valid.append(r)
        return valid

    def backtrack(current_classes, candidates, depth):
        nonlocal best_result
        
        if time.time() > deadline:
            return
            
        if len(current_classes) > len(best_result):
            best_result = current_classes[:]
            
        if not candidates:
            return
            
        # Try to add the first candidate
        m = candidates[0]
        remaining = candidates[1:]
        
        # Check if we can add m
        valid_residues = get_valid_residues(m, current_classes)
        
        if valid_residues:
            # Try each valid residue
            for r in valid_residues:
                new_classes = current_classes + [[r, m]]
                backtrack(new_classes, remaining, depth + 1)
                if time.time() > deadline:
                    return
        else:
            # Cannot add m, skip it
            backtrack(current_classes, remaining, depth + 1)
            
        # Also, we should consider NOT adding m?
        # The above logic forces a decision on candidates[0].
        # If we can't add it, we skip.
        # If we can, we try adding.
        # But what if adding m prevents adding 10 other moduli?
        # We need to also branch on "skipping m".
        
    # The above backtrack function only explores the "add" branch if possible,
    # and the "skip" branch if not. It doesn't explore "skip" if "add" is possible.
    # We need to explore both.
    
    def backtrack_v2(current_classes, candidates):
        nonlocal best_result
        
        if time.time() > deadline:
            return
            
        if len(current_classes) > len(best_result):
            best_result = current_classes[:]
            
        if not candidates:
            return
            
        m = candidates[0]
        remaining = candidates[1:]
        
        # Option 1: Skip m
        backtrack_v2(current_classes, remaining)
        
        # Option 2: Add m
        valid_residues = get_valid_residues(m, current_classes)
        if valid_residues:
            # To reduce branching, we can pick just one residue?
            # No, different residues might allow different future additions.
            # But the number of valid residues is usually small.
            for r in valid_residues:
                new_classes = current_classes + [[r, m]]
                backtrack_v2(new_classes, remaining)
                if time.time() > deadline:
                    return

    backtrack_v2(initial_classes, candidates)
    
    return best_result
