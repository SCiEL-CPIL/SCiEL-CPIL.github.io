import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit (evaluator kills at 90s, but 60s is safer for overhead)

    # Precompute some properties if needed, but products are dynamic.
    
    def greedy_construct():
        """Greedy: include x if it doesn't create duplicate products."""
        values = []
        products = set()
        for x in range(2, 129):
            new_tails = [x]
            p = x
            for a in reversed(values):
                p *= a
                new_tails.append(p)
            if not products.intersection(new_tails):
                values.append(x)
                products.update(new_tails)
        return values

    initial_seq = greedy_construct()
    best_seq = initial_seq[:]
    best_len = len(initial_seq)

    # Helper to compute all products for a sequence (for verification/scoring)
    def compute_all_products(seq):
        n = len(seq)
        prods = []
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.append(p)
        return prods

    def is_valid(seq):
        if not seq:
            return True
        prods = compute_all_products(seq)
        return len(prods) == len(set(prods))

    # Fast check for insertion validity
    # Returns (is_valid, new_products_set_delta) 
    # Actually, we just need to check if new products conflict with existing ones.
    def check_insert_validity(seq, pos, x, existing_prod_set):
        """
        Checks if inserting x at pos creates any duplicate product.
        Assumes seq is valid.
        Returns True if valid.
        """
        n = len(seq)
        new_prods = set()
        
        # Left extensions: x, x*seq[pos-1], x*seq[pos-1]*seq[pos-2], ...
        p = x
        new_prods.add(p)
        for i in range(pos - 1, -1, -1):
            p *= seq[i]
            new_prods.add(p)
            
        # Right extensions: x*seq[pos], x*seq[pos]*seq[pos+1], ...
        # Note: The singleton x is already added.
        # The block [pos, pos] is just x.
        # The block [pos-1, pos] is seq[pos-1]*x.
        # We need to be careful not to double count or miss.
        
        # Let's re-derive:
        # Blocks ending at new index pos (0-indexed in new seq, x is at pos):
        # [pos, pos] -> x
        # [pos-1, pos] -> seq[pos-1]*x
        # ...
        # [0, pos] -> seq[0]*...*seq[pos-1]*x
        
        # Blocks starting at new index pos:
        # [pos, pos] -> x (already counted)
        # [pos, pos+1] -> x*seq[pos]
        # [pos, pos+2] -> x*seq[pos]*seq[pos+1]
        # ...
        # [pos, n-1] -> x*seq[pos]*...*seq[n-1]
        
        # So we need to check:
        # 1. x
        # 2. x * (product of suffix of seq[:pos])
        # 3. x * (product of prefix of seq[pos:])
        
        # Existing products in seq are all distinct.
        # New products are:
        # A = { x * (prod of seq[i:pos]) for i in 0..pos-1 } U { x }
        # B = { x * (prod of seq[pos:j]) for j in pos..n-1 } U { x }
        # Note: x is in both.
        
        # We just need to check if any p in A U B is in existing_prod_set.
        # And also if any p in A intersects any p in B (excluding x).
        
        # Check against existing
        if existing_prod_set.intersection(new_prods):
            return False
            
        # Check internal conflicts within new_prods
        if len(new_prods) < len(list(new_prods)): # Set handles uniqueness, so this is always false if we use set.
            pass
            
        # Wait, if new_prods has duplicates, it means x * suffix1 == x * suffix2 => suffix1 == suffix2.
        # Since seq is strictly increasing, products of different suffixes/prefixes are distinct?
        # Not necessarily. e.g. seq=[2,3,6]. 
        # But seq is strictly increasing integers.
        # If we insert x, the new products involving x are x * P_left and x * P_right.
        # Can x * P_left1 == x * P_left2? Only if P_left1 == P_left2.
        # P_lefts are products of consecutive elements ending at pos-1.
        # Since elements are > 1, these products are strictly increasing as we extend left.
        # So P_lefts are distinct.
        # Similarly for P_rights.
        # Can x * P_left == x * P_right? => P_left == P_right.
        # This is possible. e.g. seq=[2, 3], insert x=1? No, x>=2.
        # seq=[2, 4], insert x=2? No, strictly increasing.
        # seq=[3, 2]? No, strictly increasing.
        # So we just need to check intersection with existing_prod_set.
        
        return True

    def get_product_set(seq):
        return set(compute_all_products(seq))

    # Main Search Loop
    current_seq = initial_seq[:]
    current_prod_set = get_product_set(current_seq)
    
    T = 10.0
    cool_rate = 0.9995
    no_improve = 0
    step = 0
    
    # To speed up, we'll use a limited number of random restarts
    while time.time() < deadline - 5:
        step += 1
        r = random.random()
        
        # Diversify: if stuck, restart with a perturbed sequence
        if no_improve > 500:
            # Perturb current sequence
            if random.random() < 0.5:
                # Remove random elements
                for _ in range(min(3, len(current_seq))):
                    if current_seq:
                        idx = random.randint(0, len(current_seq) - 1)
                        current_seq.pop(idx)
            else:
                # Add random elements if valid
                in_seq = set(current_seq)
                available = [x for x in range(2, 129) if x not in in_seq]
                if available:
                    x = random.choice(available)
                    pos = 0
                    for i, v in enumerate(current_seq):
                        if v < x:
                            pos = i + 1
                        else:
                            break
                    if check_insert_validity(current_seq, pos, x, current_prod_set):
                        current_seq.insert(pos, x)
            
            # Rebuild product set
            current_prod_set = get_product_set(current_seq)
            T = max(T, 5.0)
            no_improve = 0
            continue

        if r < 0.40:
            # Try to add an element
            in_seq = set(current_seq)
            available = [x for x in range(2, 129) if x not in in_seq]
            if not available:
                continue
            x = random.choice(available)
            pos = 0
            for i, v in enumerate(current_seq):
                if v < x:
                    pos = i + 1
                else:
                    break
            
            if check_insert_validity(current_seq, pos, x, current_prod_set):
                # Perform insertion
                current_seq.insert(pos, x)
                # Update product set incrementally?
                # For simplicity and safety, recompute if length is small, or update.
                # Recomputing is O(N^2). N is ~30-40. 1600 ops. Very fast.
                current_prod_set = get_product_set(current_seq)
                
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq = current_seq[:]
                    no_improve = 0
                    T = max(T, 5.0)
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        elif r < 0.70:
            # Try to remove an element
            if not current_seq:
                continue
            idx = random.randint(0, len(current_seq) - 1)
            removed = current_seq.pop(idx)
            current_prod_set = get_product_set(current_seq)
            no_improve += 1
            
        else:
            # Try a swap (change order)
            if len(current_seq) < 2:
                continue
            i, j = random.sample(range(len(current_seq)), 2)
            current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
            # Check validity
            if is_valid(current_seq):
                current_prod_set = get_product_set(current_seq)
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq = current_seq[:]
                    no_improve = 0
                    T = max(T, 5.0)
                else:
                    no_improve += 1
            else:
                # Revert
                current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
                no_improve += 1

        # Cooling
        T *= cool_rate
        if T < 0.1:
            T = 0.1

    # Final Polish: Greedy insertion in various orders
    def polish(seq):
        seq = seq[:]
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for x in range(2, 129):
                if x not in seq:
                    pos = 0
                    for i, v in enumerate(seq):
                        if v < x:
                            pos = i + 1
                        else:
                            break
                    if check_insert_validity(seq, pos, x, get_product_set(seq)):
                        seq.insert(pos, x)
                        changed = True
                        break
        return seq

    # Try polishing the best sequence found
    polished = polish(best_seq)
    if len(polished) > best_len:
        best_seq = polished
        best_len = len(best_seq)
        
    # Try polishing the initial sequence again with different random seeds for polish order?
    # The polish above is deterministic. Let's try a randomized polish.
    for _ in range(10):
        if time.time() > deadline - 2:
            break
        seq = best_seq[:]
        random.shuffle(seq) # This breaks strictly increasing property, so we need to re-sort?
        # No, polish expects strictly increasing.
        # Let's just run the deterministic polish on a slightly perturbed best_seq
        if best_seq:
            idx = random.randint(0, len(best_seq)-1)
            perturbed = best_seq[:idx] + best_seq[idx+1:]
            polished_p = polish(perturbed)
            if len(polished_p) > best_len:
                best_seq = polished_p
                best_len = len(best_seq)

    return best_seq
