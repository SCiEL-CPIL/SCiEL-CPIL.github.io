import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_solution = None
    best_diameter = float('inf')
    
    required_diffs = set(range(1, 31))
    
    # Get an initial bound using a simple greedy approach
    # This ensures best_diameter is not inf initially
    a_greedy = [0]
    used_g = set()
    covered_g = set()
    
    # Simple greedy to cover 1..30
    for d in range(1, 31):
        if d in covered_g:
            continue
        found = False
        a_greedy_sorted = sorted(a_greedy)
        # Try to add x = a_i + d
        for a_i in a_greedy_sorted:
            x = a_i + d
            if x in a_greedy:
                continue
            new_diffs = [x - y for y in a_greedy]
            collision = False
            for diff in new_diffs:
                if diff in used_g:
                    collision = True
                    break
            if not collision:
                a_greedy.append(x)
                used_g.update(new_diffs)
                covered_g.update(new_diffs)
                found = True
                break
        
        if not found:
            # Fallback
            x = 2 * max(a_greedy) + 3 * d + 1
            while True:
                new_diffs = [x - y for y in a_greedy]
                collision = False
                for diff in new_diffs:
                    if diff in used_g:
                        collision = True
                        break
                if not collision:
                    a_greedy.append(x)
                    used_g.update(new_diffs)
                    covered_g.update(new_diffs)
                    break
                x += 1
                
    if a_greedy:
        best_diameter = max(a_greedy)
        best_solution = list(a_greedy)
    
    # Backtracking search for better solutions
    for N in range(9, 33):
        if time.time() > deadline:
            break
            
        A = [0]
        used_diffs = set()
        covered_req = set()
        
        def backtrack(k):
            nonlocal best_solution, best_diameter, deadline
            
            if time.time() > deadline:
                return False
                
            if k == N:
                if required_diffs.issubset(covered_req):
                    diam = A[-1]
                    if diam < best_diameter:
                        best_diameter = diam
                        best_solution = list(A)
                return True
                
            # Pruning
            if A[-1] >= best_diameter:
                return False
                
            # Check if remaining slots can cover required diffs
            slots_left = (N * (N - 1) // 2) - (k * (k - 1) // 2)
            uncovered = len(required_diffs) - len(covered_req)
            if uncovered > slots_left:
                return False

            # Generate candidates
            candidates = set()
            
            # 1. Candidates from required diffs
            for d in required_diffs - covered_req:
                for a_i in A:
                    x = a_i + d
                    if x > A[-1]:
                        candidates.add(x)
                        
            # 2. Small increments from max
            for i in range(1, 10):
                candidates.add(A[-1] + i)
                
            # Sort by value
            for x in sorted(candidates):
                if x >= best_diameter:
                    break
                
                # Check validity
                new_diffs = []
                valid = True
                for a_i in A:
                    diff = x - a_i
                    if diff in used_diffs:
                        valid = False
                        break
                    new_diffs.append(diff)
                
                if not valid:
                    continue
                
                # Recurse
                A.append(x)
                used_diffs.update(new_diffs)
                covered_req.update(new_diffs)
                
                backtrack(k + 1)
                
                A.pop()
                used_diffs.difference_update(new_diffs)
                covered_req.difference_update(new_diffs)
                
        backtrack(1)
        
    return best_solution
