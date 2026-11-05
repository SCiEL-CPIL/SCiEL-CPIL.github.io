"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Generate all odd divisors of base
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()
    
    # Greedy: moduli in increasing order, each takes the class covering most uncovered residues
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    
    for m in divisors:
        if time.time() > deadline:
            break
        # Find residue that covers most uncovered positions
        best_r = 0
        best_count = -1
        # Vectorized: for each residue r, count uncovered positions r, r+m, r+2m, ...
        # We can compute this efficiently
        for r in range(m):
            count = covered[r::m].sum()
            uncovered = len(range(r, base, m)) - count
            if uncovered > best_count:
                best_count = uncovered
                best_r = r
        classes.append([m, best_r])
        covered[best_r::m] = 1
    
    # Local search: try swapping residues for each modulus to improve coverage
    if time.time() < deadline - 5:
        for _ in range(3):  # up to 3 passes
            if time.time() > deadline - 2:
                break
            improved = False
            for idx, (m, r) in enumerate(classes):
                if time.time() > deadline - 2:
                    break
                # Try all residues for this modulus
                best_r = r
                best_uncovered = (covered == 0).sum()
                # Temporarily remove this class
                old_vals = covered[r::m].copy()
                covered[r::m] = 0
                base_uncovered = (covered == 0).sum()
                
                for new_r in range(m):
                    if new_r == r:
                        continue
                    # Count how many currently uncovered positions would be covered by new_r
                    new_cover = covered[new_r::m]
                    would_cover = (new_cover == 0).sum()
                    new_uncovered = base_uncovered - would_cover
                    if new_uncovered < best_uncovered:
                        best_uncovered = new_uncovered
                        best_r = new_r
                
                # Restore
                covered[r::m] = old_vals
                
                if best_r != r:
                    classes[idx][1] = best_r
                    covered[r::m] = 0
                    covered[best_r::m] = 1
                    improved = True
            
            if not improved:
                break
    
    return classes
# EVOLVE-BLOCK-END
