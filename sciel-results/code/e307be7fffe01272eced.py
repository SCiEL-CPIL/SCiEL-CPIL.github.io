import numpy as np

def construct():
    max_val = 512
    max_size = 48
    max_sum = max_val * max_size
    
    # dp[k, s] = number of subsets of size k with sum s
    # We use integers to count, but we only care if > 0.
    # However, counts can get large. Use bool?
    # If we use bool, update is OR.
    # dp_new[k, s] = dp_old[k, s] OR dp_old[k-1, s-x]
    
    dp_main = np.zeros((max_size + 1, max_sum + 1), dtype=bool)
    dp_main[0, 0] = True
    
    selected = []
    # dp_excl[a] will be stored in a dict or list indexed by value?
    # Since values are up to 512, we can use a list of size 513.
    # But we only need to keep them for selected elements.
    # When we add x, we need dp_excl for all a in selected.
    # And we create dp_excl[x] = copy of dp_main (before update).
    
    dp_excl = {}
    
    for x in range(1, max_val + 1):
        if len(selected) >= max_size:
            break
            
        # Check validity of adding x
        
        # 1. Check if x is mean of subset of selected
        # Subset T of selected, size k >= 2, sum = k*x
        valid = True
        for k in range(2, len(selected) + 1):
            target_sum = k * x
            if target_sum > max_sum:
                break
            if dp_main[k, target_sum]:
                valid = False
                break
        
        if not valid:
            continue
            
        # 2. Check if any a in selected becomes mean of subset involving x
        # Subset T of (selected U {x}) \ {a}, size k >= 2, sum = k*a
        # T must contain x.
        # T = T' U {x}, T' subset of selected \ {a}
        # sum(T') + x = k*a  => sum(T') = k*a - x
        # size(T') = k - 1 = j
        # j >= 1
        # k = j + 1
        # sum(T') = (j+1)*a - x
        
        for a in selected:
            # We need dp of selected \ {a}
            # This is dp_excl[a]
            dp_a = dp_excl[a]
            
            for j in range(1, len(selected)): # j is size of T'
                k = j + 1
                target_sum = (j + 1) * a - x
                if target_sum < 0 or target_sum > max_sum:
                    continue
                if dp_a[j, target_sum]:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
            
        # Add x
        selected.append(x)
        
        # Save dp_excl[x] = current dp_main (which is DP of selected before x)
        dp_excl[x] = dp_main.copy()
        
        # Update dp_main to include x
        # Iterate k backwards
        for k in range(len(selected), 0, -1):
            # dp_main[k, s] |= dp_main[k-1, s-x]
            # Vectorized:
            # dp_main[k, x:] |= dp_main[k-1, :-x]
            if x <= max_sum:
                dp_main[k, x:] |= dp_main[k-1, :-x]
        
        # Update dp_excl for existing a in selected (excluding x)
        # For each a in selected[:-1]:
        # dp_excl[a] is DP of (selected \ {a})
        # New set is (selected \ {a}) U {x}
        # So update dp_excl[a] with x
        for a in selected[:-1]:
            dp_a = dp_excl[a]
            # Update dp_a with x
            # Size of set for dp_a is len(selected) - 1
            # New size is len(selected)
            for k in range(len(selected), 1, -1):
                # dp_a[k, s] |= dp_a[k-1, s-x]
                if x <= max_sum:
                    dp_a[k, x:] |= dp_a[k-1, :-x]
                    
    return selected
