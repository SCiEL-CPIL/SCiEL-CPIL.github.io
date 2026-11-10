"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s to be safe within 90s
    
    candidates = list(range(2, 41))
    
    # Precompute: for a given set, check if all subset products are distinct
    def is_valid(selected):
        # Build all subset products
        prods = {1}
        for x in selected:
            new_prods = set()
            for p in prods:
                new_prods.add(p * x)
            if new_prods & prods:
                return False
            prods |= new_prods
        return True
    
    # Greedy with random restarts
    best = []
    
    # Try several random orderings
    for attempt in range(2000):
        if time.time() > deadline:
            break
        
        random.shuffle(candidates)
        current = []
        prods = {1}
        
        for x in candidates:
            # Check if adding x keeps all subset products distinct
            new_prods = set()
            for p in prods:
                new_prods.add(p * x)
            if not (new_prods & prods):
                current.append(x)
                prods |= new_prods
        
        if len(current) > len(best):
            best = current
            if len(best) >= 18:
                break
    
    # Verify best
    if best:
        prods = {1}
        for x in best:
            new_prods = set()
            for p in prods:
                new_prods.add(p * x)
            if new_prods & prods:
                # Should not happen, but fallback
                best = []
                break
            prods |= new_prods
    
    # Ensure return type is list of Python ints
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
