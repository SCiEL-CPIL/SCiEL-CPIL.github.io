"""Program for dense increasing sequences with unique block products."""
# EVOLVE-BLOCK-START
import math

def construct():
    """
    Return strictly increasing integers in [2,128] such that products of
    every nonempty consecutive block are distinct. Maximize length.
    
    Approach: Greedy with prime-based conflict avoidance.
    We build the sequence by trying each candidate x in [2,128] in order.
    For each x, we check if adding it would create a duplicate consecutive-block
    product. We track all block products seen so far.
    
    Key insight: The set of all consecutive block products grows quadratically.
    For a sequence of length n, there are n*(n+1)/2 block products. We need
    all to be distinct.
    
    Strategy: For each new candidate x, compute all new block products that
    would be created (those ending at x). If any conflicts with existing products,
    skip x. Otherwise, add x and update the product set.
    """
    values = []
    products = set()
    
    for x in range(2, 129):
        # Compute all new block products ending at x
        # These are: x, x*last, x*last*second_last, ...
        new_products = []
        prod = x
        for a in reversed(values):
            prod *= a
            new_products.append(prod)
        new_products.append(x)  # x itself
        
        # Check for conflicts
        conflict = False
        for p in new_products:
            if p in products:
                conflict = True
                break
        
        if not conflict:
            values.append(x)
            products.update(new_products)
    
    return values
# EVOLVE-BLOCK-END
