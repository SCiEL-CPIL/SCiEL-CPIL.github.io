import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Candidate moduli
    moduli = list(range(20, 201))
    # Sort by 1/m descending (i.e., m ascending)
    moduli.sort()
    
    best_score = 0.0
    best_classes = []
    
    # Greedy approach
    # We can try multiple times with different randomizations if needed, 
    # but deterministic greedy is a good start.
    
    classes = []
    for m in moduli:
        if time.time() > deadline:
            break
        found = False
        # Try residues. 
        # To be deterministic, just 0 to m-1.
        # We can randomize the order of residues to break ties or explore different paths in a randomized restart.
        # For now, simple sequential.
        for r in range(m):
            valid = True
            for m2, r2 in classes:
                g = math.gcd(m, m2)
                if r % g == r2 % g:
                    valid = False
                    break
            if valid:
                classes.append((m, r))
                found = True
                break
        
    # Calculate score
    score = sum(1.0 / m for m, r in classes)
    
    if score > best_score:
        best_score = score
        best_classes = [[m, r] for m, r in classes]
        
    # If we have time, try some random restarts or perturbations?
    # The greedy is very fast (< 0.1s). We can afford 1000 restarts.
    # Let's try randomizing the order of residues.
    
    for _ in range(500):
        if time.time() > deadline:
            break
        
        classes = []
        for m in moduli:
            found = False
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                valid = True
                for m2, r2 in classes:
                    g = math.gcd(m, m2)
                    if r % g == r2 % g:
                        valid = False
                        break
                if valid:
                    classes.append((m, r))
                    found = True
                    break
        score = sum(1.0 / m for m, r in classes)
        if score > best_score:
            best_score = score
            best_classes = [[m, r] for m, r in classes]
            
    return best_classes
