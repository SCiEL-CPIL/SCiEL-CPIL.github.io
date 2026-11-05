"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def _compute_tails(values):
    """Return list of lists of block products for each position.
    tails[i] = [prod(values[i]), prod(values[i-1:i+1]), ...] (right-to-left suffixes)
    Actually we want: for position i, the products of consecutive blocks ending at i.
    """
    n = len(values)
    tails = [[] for _ in range(n)]
    for i in range(n):
        prod = 1
        for j in range(i, -1, -1):
            prod *= values[j]
            tails[i].append(prod)
    return tails

def _all_products(values):
    """Return set of all nonempty consecutive block products."""
    n = len(values)
    prods = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= values[j]
            prods.add(prod)
    return prods

def _is_valid(values):
    """Check strictly increasing and all block products distinct."""
    if not values:
        return True
    for i in range(1, len(values)):
        if values[i] <= values[i-1]:
            return False
    n = len(values)
    seen = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= values[j]
            if prod in seen:
                return False
            seen.add(prod)
    return True

def _greedy_from(start_val, values, products):
    """Continue greedy from start_val given current state. Returns (new_values, new_products)."""
    values = list(values)
    products = set(products)
    for x in range(start_val, 129):
        if x in values:
            continue
        # Check strictly increasing
        if values and x <= values[-1]:
            continue
        # Compute tails
        tails = [x]
        prod = x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values, products

def construct():
    deadline = time.time() + 85.0  # 5s safety margin
    
    # Initial greedy
    values, products = [], set()
    for x in range(2, 129):
        if time.time() > deadline:
            break
        tails = [x]
        prod = x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    
    best_len = len(values)
    best_values = list(values)
    
    # Local search: try swap-and-insert
    # For each position i, try replacing values[i] with a new value v
    # that maintains strictly increasing order, then re-greedy from there
    n = len(values)
    
    # Build a set of used values for quick lookup
    used_set = set(values)
    
    # Try each position
    for i in range(n):
        if time.time() > deadline:
            break
        
        # Try replacing values[i] with various candidates
        lo = (values[i-1] + 1) if i > 0 else 2
        hi = (values[i+1] - 1) if i < n - 1 else 128
        
        # Also try inserting a new value and removing one
        # Strategy: for each position i, remove values[i], try inserting new values
        removed = values[i]
        values_new = values[:i] + values[i+1:]
        products_new = _all_products(values_new)
        
        # Now try to add back removed and/or new values greedily
        # First, try to re-insert removed
        can_reinsert = True
        # Check if removed can be re-inserted at position i
        if values_new:
            if i < len(values_new) and removed >= values_new[i]:
                can_reinsert = False
            if i > 0 and removed <= values_new[i-1]:
                can_reinsert = False
        
        if can_reinsert:
            # Try inserting removed back
            tails = [removed]
            prod = removed
            for a in reversed(values_new[:i]):
                prod *= a
                tails.append(prod)
            if not products_new.intersection(tails):
                vals2 = values_new[:i] + [removed] + values_new[i:]
                prods2 = _all_products(vals2)
                # Now continue greedy from max(vals2)+1
                vals2, prods2 = _greedy_from(max(vals2)+1, vals2, prods2)
                if len(vals2) > best_len:
                    best_len = len(vals2)
                    best_values = vals2
                    values = list(vals2)
                    products = set(prods2)
                    used_set = set(values)
                    n = len(values)
        
        # Also try inserting a completely new value at position i
        for v in range(lo, hi + 1):
            if time.time() > deadline:
                break
            if v == removed:
                continue
            if v in used_set:
                continue
            
            # Try inserting v at position i into values_new
            tails = [v]
            prod = v
            for a in reversed(values_new[:i]):
                prod *= a
                tails.append(prod)
            if products_new.intersection(tails):
                continue
            
            vals2 = values_new[:i] + [v] + values_new[i:]
            prods2 = _all_products(vals2)
            # Continue greedy
            vals2, prods2 = _greedy_from(max(vals2)+1, vals2, prods2)
            if len(vals2) > best_len:
                best_len = len(vals2)
                best_values = vals2
                values = list(vals2)
                products = set(prods2)
                used_set = set(values)
                n = len(values)
    
    return best_values
# EVOLVE-BLOCK-END
