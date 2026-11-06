import time
import sys

def is_valid(seq):
    """Check if all consecutive block products are distinct."""
    if not seq:
        return True
    products = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Phase 1: Reverse Greedy
    selected = []
    # We want strictly increasing, so we build the set and sort at the end
    # But is_valid needs the sorted order.
    # So we keep 'selected' as a sorted list.
    
    for x in range(128, 1, -1):
        if time.time() > deadline:
            break
        candidate = selected + [x]
        candidate.sort()
        if is_valid(candidate):
            selected = candidate
            
    # Phase 2: Fill in missing numbers
    # Try to add any number from 2 to 128 that is not in selected
    # Order: maybe try smaller ones first? Or random?
    # Let's try 2 to 128.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(2, 129):
            if x in selected:
                continue
            candidate = selected + [x]
            candidate.sort()
            if is_valid(candidate):
                selected = candidate
                improved = True
                break # Restart loop to check with new set
                
    return selected
