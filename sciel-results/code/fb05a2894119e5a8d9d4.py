import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin
    
    # 1. Greedy Initial Solution
    values, products, suffixes = [], set(), []
    for x in range(2, 129):
        if time.time() > deadline:
            break
        new_suffixes = [x]
        prod = x
        for a in reversed(values):
            prod *= a
            new_suffixes.append(prod)
        
        # Check conflict
        conflict = False
        for p in new_suffixes:
            if p in products:
                conflict = True
                break
        
        if not conflict:
            values.append(x)
            products.update(new_suffixes)
            suffixes = new_suffixes
        else:
            # No need to update suffixes if we don't add x
            pass

    best_seq = values
    best_len = len(values)
    
    # 2. Local Search (1-in, 1-out-1-in)
    # We maintain the current sequence and its products/suffixes
    current_seq = best_seq[:]
    current_products, current_suffixes = get_products(current_seq)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to insert any missing number
        for m in range(2, 129):
            if time.time() > deadline:
                return best_seq
            if m in current_seq:
                continue
            
            # Try direct insertion
            new_seq, new_products, new_suffixes, success = try_insert(current_seq, current_products, current_suffixes, m)
            if success:
                current_seq = new_seq
                current_products = new_products
                current_suffixes = new_suffixes
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq = current_seq[:]
                improved = True
                break # Restart scan with new state
            
            # If direct insertion failed, try 1-out-1-in
            # Identify candidates for removal
            # We just try removing each element
            for i in range(len(current_seq)):
                if time.time() > deadline:
                    return best_seq
                # Create seq without element i
                test_seq = current_seq[:i] + current_seq[i+1:]
                # We need products of test_seq to check insertion
                # Computing products is O(N^2)
                test_products, test_suffixes = get_products(test_seq)
                
                # Try inserting m into test_seq
                new_seq, new_products, new_suffixes, success = try_insert(test_seq, test_products, test_suffixes, m)
                if success:
                    current_seq = new_seq
                    current_products = new_products
                    current_suffixes = new_suffixes
                    if len(current_seq) > best_len:
                        best_len = len(current_seq)
                        best_seq = current_seq[:]
                    improved = True
                    break # Restart scan with new state
            
            if improved:
                break
                
    return best_seq

def get_products(seq):
    products = set()
    suffixes = []
    # Compute all subarray products
    # Efficient way: iterate end index, keep running product
    for i in range(len(seq)):
        prod = 1
        for j in range(i, len(seq)):
            prod *= seq[j]
            products.add(prod)
    # Compute suffixes for the whole sequence (needed for try_insert optimization? 
    # Actually try_insert only needs suffixes of the prefix before insertion point?
    # No, try_insert appends to the end? 
    # Wait, the sequence must be strictly increasing. 
    # If we insert m, it goes at the correct sorted position.
    # The "suffixes" logic in the seed program assumed appending to the end.
    # If we insert in the middle, the suffixes of the *whole* sequence change.
    # And the products of subarrays crossing the insertion point change.
    
    # My try_insert implementation below assumes we are APPENDING to the end.
    # This is WRONG if m is not the largest element.
    # The sequence must be sorted.
    # So if we have [2, 5, 7] and we want to insert 3, it becomes [2, 3, 5, 7].
    # The subarray products change significantly.
    # The "suffix" trick only works for appending to the end.
    
    # So, for general insertion, we must recompute ALL products.
    # This is O(N^2) per check.
    # N is small, so it's fine.
    
    return products, [] # suffixes not needed if we recompute all

def try_insert(seq, products, suffixes, m):
    # Insert m into sorted seq
    # Find position
    pos = 0
    while pos < len(seq) and seq[pos] < m:
        pos += 1
    
    new_seq = seq[:pos] + [m] + seq[pos:]
    
    # Compute new products
    new_products = set()
    n = len(new_seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= new_seq[j]
            new_products.add(prod)
    
    # Check if any product in new_products was in old products 
    # AND is not generated by subarrays that don't include m?
    # Actually, the condition is: ALL products in the new sequence must be distinct.
    # So we just need to check if new_products has any duplicates?
    # No, the products are generated by subarrays.
    # If len(new_products) == n*(n+1)/2, then all are distinct.
    # If len(new_products) < n*(n+1)/2, then there is a duplicate.
    
    expected_count = n * (n + 1) // 2
    if len(new_products) == expected_count:
        return new_seq, new_products, [], True
    else:
        return new_seq, new_products, [], False

# EVOLVE-BLOCK-END
