import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_result = []
    best_count = 0
    
    # Candidate sets
    # 1. Even numbers
    evens = [m for m in range(2, 129) if m % 2 == 0]
    # 2. Multiples of 3
    threes = [m for m in range(2, 129) if m % 3 == 0]
    # 3. Multiples of 5
    fives = [m for m in range(2, 129) if m % 5 == 0]
    
    # We will try to solve for each set
    # To increase diversity, we can shuffle candidates
    
    def is_valid(residue, modulus, current_moduli, current_residues):
        for m in current_moduli:
            r = current_residues[m]
            g = __import__('math').gcd(modulus, m)
            if residue % g == r % g:
                return False
        return True

    def search(candidates, current_moduli, current_residues, depth=0):
        nonlocal best_result, best_count, deadline
        
        if time.time() > deadline:
            return
        
        if len(current_moduli) > best_count:
            best_count = len(current_moduli)
            # Copy result
            best_result = [[r, m] for m, r in current_residues.items()]
        
        if not candidates:
            return
        
        # Pruning: if current + remaining <= best, stop
        if len(current_moduli) + len(candidates) <= best_count:
            return
        
        # Pick the first candidate
        m = candidates[0]
        rest = candidates[1:]
        
        # Try to include m
        # Find a valid residue
        found_residue = None
        # Heuristic: try residues that are "far" from existing?
        # Just iterate 0..m-1
        # Optimization: if m is large, we can sample? No, need exact.
        # But m <= 128, so max 128 checks.
        
        valid_residues = []
        for r in range(m):
            if is_valid(r, m, current_moduli, current_residues):
                valid_residues.append(r)
        
        if valid_residues:
            # Try one valid residue (or all? Backtracking needs all)
            # To save time, try just one? No, we need to explore.
            # But if we just want MAX count, picking any valid one might work?
            # Not necessarily, a bad choice might block others.
            # However, for this problem, symmetry suggests most residues are equivalent?
            # Not quite.
            # Let's try the first valid residue to keep it fast.
            # If this fails to produce a large set, we might need to backtrack.
            # Given the time, let's try to be greedy first, then backtrack if needed?
            # Actually, pure greedy (pick first valid) is very fast.
            # Let's try Greedy with multiple orderings.
            
            # Greedy Step
            r = valid_residues[0]
            current_moduli.append(m)
            current_residues[m] = r
            search(rest, current_moduli, current_residues, depth+1)
            
            # Backtrack
            current_moduli.pop()
            del current_residues[m]
            
            # Try to skip m
            search(rest, current_moduli, current_residues, depth+1)
            
            # If we have time, try other residues for m?
            # This increases branching factor significantly.
            # For now, stick to 1 residue + skip.
            # If score is low, we can expand.

    # Run search for different orderings
    orderings = [
        evens,
        sorted(evens, reverse=True),
        threes,
        sorted(threes, reverse=True),
        fives,
        sorted(fives, reverse=True)
    ]
    
    # Add random shuffles
    for _ in range(10):
        if time.time() > deadline:
            break
        shuffled = evens[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)
        
        shuffled = threes[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)

    for order in orderings:
        if time.time() > deadline:
            break
        search(order, [], {})

    # Fallback to seed if no better found
    if best_count <= 7:
        return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
        
    return best_result
