import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    candidates = list(range(2, 41))
    
    # Greedy selection
    selected = []
    products = {1}
    
    for c in candidates:
        # Check if adding c creates a collision
        # New products will be {p * c for p in products}
        # Collision if any p * c is already in products
        # Or if any two p1*c and p2*c are equal (impossible since p1!=p2)
        # So just check intersection with existing products
        
        new_products = set()
        valid = True
        for p in products:
            np = p * c
            if np in products:
                valid = False
                break
            new_products.add(np)
        
        if valid:
            selected.append(c)
            products.update(new_products)
            
        if time.time() > deadline:
            break
            
    # The greedy order matters. 
    # Trying smaller numbers first might be better for packing?
    # Or maybe trying primes first?
    # The candidates are 2..40 in increasing order.
    # This adds 2, 3, 4 (valid), 5, 6?
    # If we have 2, 3, 4. Products: 1, 2, 3, 4, 6, 8, 12, 24.
    # Try 5: 5, 10, 15, 20, 30, 40, 60, 120. None in old set. Add 5.
    # Try 6: 6 is in products (from 2*3). Collision. Reject 6.
    # Try 7: Add.
    # Try 8: 8 is in products (from 4*2? No, 4*2=8 is in products? 
    # Wait, if we have 2 and 4, product 8 is in set.
    # So 8 is rejected.
    # This greedy will find a valid set.
    # To maximize, we might need to be smarter, but this is a valid fix.
    
    return [int(x) for x in selected]
