import time
from math import gcd

def construct():
        start_time = time.time()
        deadline = start_time + 110  # Safety margin
        
        # Candidate moduli: Multiples of 2 in [20, 200]
        candidates = list(range(20, 201, 2))
        
        # Try decreasing order first
        order = sorted(candidates, reverse=True)
        
        best_set = []
        best_score = 0.0
        
        # We will try to assign residues
        # assignment: dict m -> r
        assignment = {}
        
        def is_compatible(m, r, current_assignment):
            for m2, r2 in current_assignment.items():
                g = gcd(m, m2)
                if r % g == r2 % g:
                    return False
            return True
        
        # Greedy assignment
        for m in order:
            found = False
            # Try all residues
            for r in range(m):
                if is_compatible(m, r, assignment):
                    assignment[m] = r
                    found = True
                    break
            if not found:
                # Backtracking is too expensive for 90 nodes.
                # If we fail, we might need to restart with a different strategy.
                break
        
        if assignment:
            score = sum(1.0 / m for m in assignment.keys())
            if score > best_score:
                best_score = score
                best_set = [[m, r] for m, r in assignment.items()]
        
        # If decreasing failed or we want to try increasing
        if len(assignment) < len(candidates):
             # Try increasing order
             order = sorted(candidates)
             assignment = {}
             for m in order:
                 found = False
                 for r in range(m):
                     if is_compatible(m, r, assignment):
                         assignment[m] = r
                         found = True
                         break
                 if not found:
                     break
             if assignment:
                 score = sum(1.0 / m for m in assignment.keys())
                 if score > best_score:
                     best_score = score
                     best_set = [[m, r] for m, r in assignment.items()]
        
        # If we still have time, try random restarts with shuffled residue order
        while time.time() < deadline:
            assignment = {}
            order = sorted(candidates, reverse=True)
            success = True
            for m in order:
                found = False
                # Shuffle residues? No, just try 0..m-1.
                # To add randomness, we can rotate the start index
                start_r = int((time.time() * 1000) % m)
                for i in range(m):
                    r = (start_r + i) % m
                    if is_compatible(m, r, assignment):
                        assignment[m] = r
                        found = True
                        break
                if not found:
                    success = False
                    break
            
            if success:
                score = sum(1.0 / m for m in assignment.keys())
                if score > best_score:
                    best_score = score
                    best_set = [[m, r] for m, r in assignment.items()]
                # If we found a full set, we can't do better with this set of moduli
                # unless we change the set of moduli.
                # But Multiples of 2 is likely the max set.
                break
        
        if not best_set:
            # Fallback to seed
            return [[20 * j, j - 1] for j in range(1, 11)]
            
        return best_set
