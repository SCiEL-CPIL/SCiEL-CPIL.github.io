import random

def construct():
    candidates = list(range(2, 41))
    best_set = []
    
    # Run multiple random restarts to find a better local optimum
    # 50 restarts is fast enough for this problem size
    for _ in range(50):
        random.shuffle(candidates)
        selected = []
        products = {1}
        
        for x in candidates:
            new_products = {p * x for p in products}
            # Check for collision with existing products
            if not new_products.intersection(products):
                selected.append(x)
                products.update(new_products)
        
        if len(selected) > len(best_set):
            best_set = selected
            
    # Ensure return type is list of ints
    return [int(x) for x in best_set]
