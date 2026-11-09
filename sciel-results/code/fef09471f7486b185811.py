"""Iterated Local Search for dense increasing sequences with unique block products."""
import time
import random

def _rebuild_products(seq):
    """Rebuild the set of all consecutive block products for seq."""
    products = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            products.add(prod)
    return products

def _can_append(seq, x, products):
    """Check if x can be appended to seq without creating duplicate block products."""
    new_prods = []
    prod = x
    new_prods.append(prod)
    for a in reversed(seq):
        prod *= a
        new_prods.append(prod)
    for p in new_prods:
        if p in products:
            return False
    return True

def _greedy_fill(seq, products):
    """Greedily add missing numbers in ascending order. Returns (new_seq, new_products)."""
    seq_set = set(seq)
    new_seq = list(seq)
    new_products = set(products)
    for x in range(2, 129):
        if x in seq_set:
            continue
        if _can_append(new_seq, x, new_products):
            prod = x
            new_products.add(prod)
            for a in reversed(new_seq):
                prod *= a
                new_products.add(prod)
            new_seq.append(x)
            seq_set.add(x)
    return new_seq, new_products

def _local_search(seq, deadline):
    """
    Local search: repeatedly find the element whose removal allows the most new elements
    to be added, remove it, and greedily refill. Stop when no improvement or time runs out.
    """
    best_seq = list(seq)
    best_len = len(best_seq)
    
    while time.time() < deadline:
        n = len(best_seq)
        if n == 0:
            break
        
        best_gain = 0
        best_idx = -1
        best_new_seq = None
        
        # Try removing each element
        for i in range(n):
            if time.time() >= deadline:
                break
            
            candidate = best_seq[:i] + best_seq[i+1:]
            cand_products = _rebuild_products(candidate)
            
            # Greedily refill
            new_seq, new_products = _greedy_fill(candidate, cand_products)
            
            gain = len(new_seq) - best_len
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                best_new_seq = new_seq
        
        if best_new_seq is None or best_gain <= 0:
            break
        
        best_seq = best_new_seq
        best_len = len(best_seq)
    
    return best_seq

def _build_greedy_base():
    """Build initial sequence using ascending greedy."""
    values = []
    products = set()
    for x in range(2, 129):
        tails = []
        prod = x
        tails.append(prod)
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    random.seed(42)
    
    # Phase 1: Build greedy base
    seq = _build_greedy_base()
    best_seq = list(seq)
    best_len = len(best_seq)
    
    # Phase 2: Iterated Local Search
    while time.time() < deadline:
        # Perturbation: remove k random elements
        k = random.randint(1, min(3, len(best_seq)))
        if k > 0 and len(best_seq) > k:
            remove_indices = set()
            while len(remove_indices) < k:
                remove_indices.add(random.randint(0, len(best_seq) - 1))
            
            perturbed = [x for i, x in enumerate(best_seq) if i not in remove_indices]
            
            # Rebuild products for perturbed sequence
            perturbed_products = _rebuild_products(perturbed)
            
            # Local search from perturbed state
            candidate = _local_search(perturbed, deadline)
            
            if len(candidate) > best_len:
                best_seq = candidate
                best_len = len(best_seq)
        
        # Also try local search from current best without perturbation
        candidate = _local_search(best_seq, deadline)
        if len(candidate) > best_len:
            best_seq = candidate
            best_len = len(best_seq)
        
        if time.time() >= deadline:
            break
    
    return best_seq
