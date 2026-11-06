"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import time
    
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit to be safe
    
    target = Fraction(5, 31)
    best_solution = None
    best_max_denom = float('inf')
    
    # Search space for d1, d2, d3
    # d1 is small
    for d1 in range(7, 25):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        
        min_d2 = (r1.denominator + r1.numerator - 1) // r1.numerator
        start_d2 = max(min_d2, d1 + 1)
        
        for d2 in range(start_d2, 150):
            if time.time() > deadline:
                break
            r2 = r1 - Fraction(1, d2)
            if r2 <= 0:
                continue
            
            min_d3 = (r2.denominator + r2.numerator - 1) // r2.numerator
            start_d3 = max(min_d3, d2 + 1)
            
            # Limit d3 to 1000
            for d3 in range(start_d3, 1000):
                if time.time() > deadline:
                    break
                r3 = r2 - Fraction(1, d3)
                if r3 <= 0:
                    continue
                
                # Greedy completion
                remaining = r3
                next_min = d3 + 1
                terms = [d1, d2, d3]
                current_max = d3
                valid = True
                
                while remaining > 0:
                    min_d = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
                    d = max(min_d, next_min)
                    
                    if d >= best_max_denom:
                        valid = False
                        break
                        
                    terms.append(d)
                    remaining -= Fraction(1, d)
                    next_min = d + 1
                    current_max = d
                    
                    if len(terms) > 256:
                        valid = False
                        break
                
                if valid and remaining == 0:
                    if current_max < best_max_denom:
                        best_max_denom = current_max
                        best_solution = terms[:]
                        
    return best_solution
# EVOLVE-BLOCK-END
