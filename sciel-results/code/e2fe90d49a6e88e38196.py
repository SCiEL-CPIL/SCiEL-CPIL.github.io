def construct():
    # Generate candidates 2..40
    candidates = list(range(2, 41))
    
    # Sort by "norm" descending. 
    # The directive suggests "largest norm first". 
    # Using the value itself as norm is a reasonable interpretation.
    # Alternatively, sum of prime exponents. Let's stick to value for simplicity 
    # unless "norm" implies vector norm. 
    # Vector norm: sqrt(sum(e_i^2)).
    # Let's use the value as a proxy for "large".
    candidates.sort(reverse=True)
    
    products = {1}
    selected = []
    
    for x in candidates:
        new_products = set(products)
        for p in products:
            new_products.add(x * p)
        
        if len(new_products) == 2 * len(products):
            products = new_products
            selected.append(x)
            
    return selected
