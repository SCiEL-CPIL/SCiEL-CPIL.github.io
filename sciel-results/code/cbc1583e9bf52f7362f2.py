"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    
    target = Fraction(5, 31)
    best_max = float('inf')
    best_solution = None
    
    def greedy_complete(remaining, min_denom):
        if remaining == 0:
            return []
        if remaining < 0:
            return None
        result = []
        r = remaining
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d < min_denom:
                d = min_denom
            if d in result:
                d += 1
            result.append(d)
            r -= Fraction(1, d)
        return result
    
    def search(remaining, min_denom, depth, current_max, used):
        nonlocal best_max, best_solution
        
        if remaining == 0:
            if current_max < best_max:
                best_max = current_max
                best_solution = list(used)
            return
        
        if remaining < 0:
            return
        
        if depth >= 2:
            completion = greedy_complete(remaining, min_denom)
            if completion is not None:
                new_max = current_max
                for d in completion:
                    if d > new_max:
                        new_max = d
                if new_max < best_max:
                    best_max = new_max
                    best_solution = list(used) + completion
            return
        
        inv_rem = Fraction(1, 1) / remaining
        min_possible_d = max(min_denom, inv_rem.numerator // inv_rem.denominator + (1 if inv_rem.numerator % inv_rem.denominator != 0 else 0))
        
        max_d_to_try = min(300, int(1 / remaining) + 50) if remaining > 0 else 300
        
        for d in range(min_possible_d, max_d_to_try + 1):
            if d in used:
                continue
            if Fraction(1, d) > remaining:
                continue
            
            new_remaining = remaining - Fraction(1, d)
            new_max = d if current_max < d else current_max
            
            if new_max >= best_max:
                break
            
            search(new_remaining, d + 1, depth + 1, new_max, used | {d})
    
    search(target, 1, 0, 0, frozenset())
    
    if best_solution is not None:
        return best_solution
    
    r = target
    values = []
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
# EVOLVE-BLOCK-END
