from fractions import Fraction
import random

def construct():
    best_set = []
    best_size = 0
    
    # Deterministic orders
    orders = [
        list(range(24, 0, -1)),
        list(range(1, 25)),
        list(range(12, 0, -1)) + list(range(24, 12, -1)) # Mixed
    ]
    
    # Random orders
    random.seed(42) # For reproducibility if needed, though not required
    for _ in range(200):
        rand_order = list(range(1, 25))
        random.shuffle(rand_order)
        orders.append(rand_order)
        
    for order in orders:
        S = []
        Sums = {Fraction(0, 1)}
        
        for x in order:
            rx = Fraction(1, x)
            
            # Check if rx is in Sums
            if rx in Sums:
                continue
                
            # Check if adding x makes any existing a invalid
            valid_add = True
            for a in S:
                ra = Fraction(1, a)
                target = ra - rx
                if target > 0:
                    if target in Sums:
                        valid_add = False
                        break
                        
            if valid_add:
                S.append(x)
                new_sums = set(Sums)
                for s in Sums:
                    new_sums.add(s + rx)
                Sums = new_sums
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    return best_set
