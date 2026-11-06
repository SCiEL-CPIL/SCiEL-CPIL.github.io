import time
import random
from itertools import combinations

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def get_products(seq):
    """Get all consecutive block products for a sorted sequence."""
    products = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            products.add(prod)
    return products

def is_valid(seq):
    """Check if all consecutive block products are distinct."""
    if not seq:
        return True
    n = len(seq)
    products = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True

def can_insert(seq, val):
    """Check if val can be inserted into sorted seq while maintaining validity.
    Only checks products that involve val."""
    n = len(seq)
    # Find insertion position
    pos = 0
    while pos < n and seq[pos] < val:
        pos += 1
    
    # Check all blocks that include val
    # Blocks are [i..j] where i <= pos <= j (in the new sequence)
    # In the old sequence, these correspond to blocks that start before/at pos and end after/at pos
    
    # First, get all existing products to check for conflicts
    existing_products = get_products(seq)
    
    # Check all new blocks that include val
    # A block [i..j] in new sequence includes val if i <= pos <= j
    # The product is (product of seq[i..pos-1]) * val * (product of seq[pos..j])
    
    new_products = set()
    for i in range(pos, -1, -1):
        left_prod = 1
        for k in range(i, pos):
            left_prod *= seq[k]
        for j in range(pos, n):
            right_prod = 1
            for k in range(pos, j):
                right_prod *= seq[k]
            block_prod = left_prod * val * right_prod
            if block_prod in existing_products or block_prod in new_products:
                return False
            new_products.add(block_prod)
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    primes = [p for p in range(2, 129) if is_prime(p)]
    composites = [c for c in range(2, 129) if not is_prime(c)]
    
    best_set = set(primes)
    best_len = len(primes)
    best_seq = sorted(best_set)
    
    # Strategy 1: Try adding composites one by one in different orders
    # Use multiple random permutations
    
    num_iterations = 0
    for iteration in range(5000):
        if time.time() > deadline:
            break
        num_iterations += 1
        
        current_set = set(primes)
        candidates = composites.copy()
        random.shuffle(candidates)
        
        for c in candidates:
            if time.time() > deadline:
                break
            
            # Try to insert c into current_set
            seq = sorted(current_set)
            if can_insert(seq, c):
                current_set.add(c)
        
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set
            best_seq = sorted(best_set)
    
    # Strategy 2: Local search - try swapping or removing elements
    # Try to improve by removing one element and adding two others
    for _ in range(100):
        if time.time() > deadline:
            break
        
        current_seq = list(best_seq)
        n = len(current_seq)
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            # Remove element i
            reduced = current_seq[:i] + current_seq[i+1:]
            removed_val = current_seq[i]
            
            # Try to add two composites
            added = []
            for c in composites:
                if c == removed_val or c in current_seq:
                    continue
                seq = sorted(reduced + added)
                if can_insert(seq, c):
                    added.append(c)
                    if len(added) == 2:
                        break
            
            if len(added) >= 2:
                new_seq = sorted(reduced + added)
                if is_valid(new_seq) and len(new_seq) > best_len:
                    best_len = len(new_seq)
                    best_seq = new_seq
                    best_set = set(new_seq)
    
    return best_seq
