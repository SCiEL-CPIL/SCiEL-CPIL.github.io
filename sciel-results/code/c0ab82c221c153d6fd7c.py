import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    random.seed(42)
    
    def compute_products(seq):
        """Compute all consecutive block products for a sequence."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    return None  # collision
                products.add(prod)
        return products
    
    def is_valid(seq):
        """Check if a strictly increasing sequence has all distinct consecutive block products."""
        if len(seq) == 0:
            return True
        if not all(seq[i] < seq[i+1] for i in range(len(seq)-1)):
            return False
        if any(x < 2 or x > 128 for x in seq):
            return False
        return compute_products(seq) is not None
    
    def greedy_build():
        """Greedily build a valid sequence."""
        seq = []
        products = set()
        for x in range(2, 129):
            # Try adding x
            new_products = set(products)
            prod = x
            valid = True
            # Compute new products ending at x
            for i in range(len(seq)-1, -1, -1):
                prod *= seq[i]
                if prod in new_products:
                    valid = False
                    break
                new_products.add(prod)
            if valid:
                seq.append(x)
                products = new_products
        return seq
    
    def local_search(seq, products, time_limit=10):
        """Improve sequence via local search (add, remove, swap operations)."""
        end_time = time.time() + time_limit
        best_seq = list(seq)
        best_len = len(seq)
        best_products = set(products)
        
        def try_insert(x, seq, prods):
            """Try inserting x into seq at the correct position."""
            pos = 0
            for i, v in enumerate(seq):
                if v < x:
                    pos = i + 1
                else:
                    break
            new_seq = seq[:pos] + [x] + seq[pos:]
            # Check strictly increasing
            if not all(new_seq[j] < new_seq[j+1] for j in range(len(new_seq)-1)):
                return None, None
            # Check validity
            new_prods = set(prods)
            valid = True
            # Compute products ending at the new element
            for i in range(pos-1, -1, -1):
                prod = x
                for j in range(i, pos):
                    prod *= seq[j]
                if prod in new_prods:
                    valid = False
                    break
                new_prods.add(prod)
            # Also need to check products that span the new element
            if valid:
                for i in range(pos):
                    prod = 1
                    for j in range(i, pos):
                        prod *= seq[j]
                    prod *= x
                    for j in range(pos, len(new_seq)):
                        if prod in new_prods:
                            valid = False
                            break
                        new_prods.add(prod)
                        prod *= new_seq[j]
            if valid:
                return new_seq, new_prods
            return None, None
        
        # First, try adding elements
        improved = True
        while improved and time.time() < end_time:
            improved = False
            candidates = list(range(2, 129))
            random.shuffle(candidates)
            for x in candidates:
                if x in best_seq:
                    continue
                new_seq, new_prods = try_insert(x, best_seq, best_products)
                if new_seq is not None:
                    best_seq = new_seq
                    best_products = new_prods
                    best_len = len(best_seq)
                    improved = True
                    break
        
        # Then try 2-opt style moves: remove one element and try to add a different one
        improved = True
        while improved and time.time() < end_time:
            improved = False
            for i in range(len(best_seq)):
                for x in range(2, 129):
                    if x in best_seq:
                        continue
                    # Remove best_seq[i] and try to add x
                    temp_seq = best_seq[:i] + best_seq[i+1:]
                    temp_prods = set()
                    n = len(temp_seq)
                    valid = True
                    for a in range(n):
                        prod = 1
                        for b in range(a, n):
                            prod *= temp_seq[b]
                            if prod in temp_prods:
                                valid = False
                                break
                            temp_prods.add(prod)
                    if not valid:
                        continue
                    new_seq, new_prods = try_insert(x, temp_seq, temp_prods)
                    if new_seq is not None and len(new_seq) >= best_len:
                        best_seq = new_seq
                        best_products = new_prods
                        best_len = len(best_seq)
                        improved = True
                        break
                if improved:
                    break
        
        return best_seq, best_products, best_len
    
    def beam_search(width=200, max_time=20, diversify=False):
        """Beam search to find a good sequence."""
        end_time = time.time() + max_time
        
        # State: (seq_tuple, products_set, last_element)
        # Start with empty sequence
        beam = [((), frozenset(), 0)]
        best_seq = []
        best_len = 0
        
        for x in range(2, 129):
            if time.time() > end_time:
                break
            
            next_beam = []
            for seq, prods, last in beam:
                # Option 1: Skip x
                next_beam.append((seq, prods, last))
                
                # Option 2: Include x
                if last >= x:
                    continue
                    
                # Compute new products
                new_prods = set(prods)
                prod = x
                valid = True
                for i in range(len(seq)-1, -1, -1):
                    prod *= seq[i]
                    if prod in new_prods:
                        valid = False
                        break
                    new_prods.add(prod)
                
                if valid:
                    new_seq = seq + (x,)
                    next_beam.append((new_seq, frozenset(new_prods), x))
                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_seq = list(new_seq)
            
            # Deduplicate and prune
            unique = {}
            for seq, prods, last in next_beam:
                if seq not in unique:
                    unique[seq] = (prods, last)
            
            if diversify:
                # Use a more diverse scoring: length desc, then random tie-break
                items = list(unique.keys())
                random.shuffle(items)
                sorted_seqs = sorted(items, key=lambda s: (-len(s), random.random()))
            else:
                # Sort by length desc, then by sum asc (prefer smaller values)
                sorted_seqs = sorted(unique.keys(), key=lambda s: (-len(s), sum(s) if s else 0))
            
            beam = [(s, unique[s][0], unique[s][1]) for s in sorted_seqs[:width]]
        
        return best_seq
    
    # Strategy: Combine greedy, beam search, and local search
    
    # 1. Get greedy solution
    greedy_seq = greedy_build()
    greedy_products = compute_products(greedy_seq)
    best_seq = list(greedy_seq)
    best_len = len(greedy_seq)
    
    # 2. Multiple beam searches with different parameters
    beam_configs = [
        (150, 15, False),
        (300, 15, True),
        (200, 10, True),
        (100, 10, False),
    ]
    
    for width, max_time, diversify in beam_configs:
        if time.time() > deadline - 10:
            break
        beam_seq = beam_search(width=width, max_time=max_time, diversify=diversify)
        if len(beam_seq) > best_len:
            best_seq = beam_seq
            best_len = len(best_seq)
            best_products = compute_products(best_seq)
    
    # 3. Local search on the best found
    if time.time() < deadline - 5:
        remaining = deadline - 5 - time.time()
        if remaining > 5:
            ls_seq, ls_products, ls_len = local_search(best_seq, best_products, time_limit=min(remaining, 20))
            if ls_len > best_len:
                best_seq = ls_seq
                best_len = ls_len
                best_products = ls_products
    
    # 4. Final validation
    if not is_valid(best_seq):
        # Fallback to greedy
        best_seq = greedy_build()
    
    return best_seq

if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Length: {len(result)}")
    print(f"Time: {t1-t0:.2f}s")
    print(result[:20], "...")
