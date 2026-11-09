"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer before 120s limit
    
    base = 51975
    # Generate all divisors of base greater than 1
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    # Process in decreasing order: larger moduli first
    divisors.sort(reverse=True)
    
    covered = bytearray(base)  # 0 = uncovered, 1 = covered
    classes = []
    
    for m in divisors:
        if time.time() > deadline:
            break
        
        # Find residue class r that covers most uncovered residues
        best_r = 0
        best_count = -1
        for r in range(m):
            # Count uncovered residues in class r mod m
            count = 0
            for i in range(r, base, m):
                if covered[i] == 0:
                    count += 1
            if count > best_count:
                best_count = count
                best_r = r
        
        classes.append([m, best_r])
        # Mark covered
        for i in range(best_r, base, m):
            covered[i] = 1
    
    return classes
# EVOLVE-BLOCK-END
