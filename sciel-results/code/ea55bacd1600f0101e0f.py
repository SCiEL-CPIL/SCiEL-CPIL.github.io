import time
import random

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin before 60s/90s limit
    
    def get_block_products(sequence):
        """
        Returns the set of all non-empty consecutive block products.
        """
        n = len(sequence)
        products = set()
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= sequence[j]
                products.add(p)
        return products
    
    def is_valid(sequence):
        """
        Checks if the sequence has distinct products for all non-empty consecutive blocks.
        Returns True if valid, False otherwise.
        """
        n = len(sequence)
        if n == 0:
            return True
        products = set()
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= sequence[j]
                if p in products:
                    return False
                products.add(p)
        return True
    
    def is_valid_fast(sequence, existing_products=None):
        """
        Faster validity check that can reuse existing products.
        If existing_products is provided, it should be the products from a prefix
        of the sequence up to some point, and we only check the new blocks.
        For simplicity, this just calls is_valid but could be optimized.
        """
        return is_valid(sequence)

    def greedy_construct():
        """
        Builds a sequence greedily by trying to append each number from 2 to 128.
        """
        values = []
        products = set()
        
        for x in range(2, 129):
            # Compute new products if we append x
            new_prods = []
            p = x
            new_prods.append(p)
            for a in reversed(values):
                p *= a
                new_prods.append(p)
            
            # Check if any new product collides with existing
            collision = False
            for prod in new_prods:
                if prod in products:
                    collision = True
                    break
            
            if not collision:
                values.append(x)
                products.update(new_prods)
        
        return values

    # Initial sequence
    seq = greedy_construct()
    
    # Local Search
    # We will try to improve the sequence using various moves.
    
    def get_unused(seq):
        used = set(seq)
        return [x for x in range(2, 129) if x not in used]

    def try_insert(seq):
        """Try to insert any unused number at any position."""
        unused = get_unused(seq)
        n = len(seq)
        if n == 0:
            # Insert first element
            for x in unused:
                if is_valid([x]):
                    return [x]
            return None
        
        # Randomize order of positions and candidates to escape local minima
        positions = list(range(n + 1))
        random.shuffle(positions)
        unused_copy = unused[:]
        random.shuffle(unused_copy)
        
        for pos in positions:
            for x in unused_copy:
                # Check bounds
                if pos > 0 and seq[pos-1] >= x:
                    continue
                if pos < n and seq[pos] <= x:
                    continue
                
                new_seq = seq[:pos] + [x] + seq[pos:]
                if is_valid(new_seq):
                    return new_seq
        return None

    def try_1out_1in(seq):
        """Try to replace an element with a different unused number."""
        unused = get_unused(seq)
        n = len(seq)
        if n == 0:
            return None
        
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            left_bound = seq[i-1] if i > 0 else 1
            right_bound = seq[i+1] if i < n-1 else 129
            
            candidates = [x for x in unused if left_bound < x < right_bound]
            random.shuffle(candidates)
            
            for x in candidates:
                if x == seq[i]:
                    continue
                new_seq = seq[:i] + [x] + seq[i+1:]
                if is_valid(new_seq):
                    return new_seq
        return None

    def try_2out_2in(seq):
        """Replace two consecutive elements with two new ones."""
        n = len(seq)
        if n < 2:
            return None
        
        unused = get_unused(seq)
        indices = list(range(n - 1))  # Pairs (i, i+1)
        random.shuffle(indices)
        
        for i in indices:
            left_bound = seq[i-1] if i > 0 else 1
            right_bound = seq[i+2] if i+1 < n-1 else 129
            
            # We need two numbers x, y such that left_bound < x < y < right_bound
            candidates = [x for x in unused if left_bound < x < right_bound]
            if len(candidates) < 2:
                continue
            
            candidates_copy = candidates[:]
            random.shuffle(candidates_copy)
            
            # Try pairs
            for idx_x in range(len(candidates_copy)):
                x = candidates_copy[idx_x]
                for idx_y in range(len(candidates_copy)):
                    if idx_x == idx_y:
                        continue
                    y = candidates_copy[idx_y]
                    
                    # Ensure order x < y
                    if x > y:
                        x, y = y, x
                    
                    if not (left_bound < x < y < right_bound):
                        continue
                        
                    new_seq = seq[:i] + [x, y] + seq[i+2:]
                    if is_valid(new_seq):
                        return new_seq
        return None

    def try_1out_2in(seq):
        """Remove one element, insert two new ones at the same position."""
        n = len(seq)
        if n == 0:
            return None
        
        unused = get_unused(seq)
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            left_bound = seq[i-1] if i > 0 else 1
            right_bound = seq[i+1] if i < n-1 else 129
            
            # We need two numbers x, y such that left_bound < x < y < right_bound
            candidates = [x for x in unused if left_bound < x < right_bound]
            if len(candidates) < 2:
                continue
            
            candidates_copy = candidates[:]
            random.shuffle(candidates_copy)
            
            # Try pairs - limit to avoid O(K^2) explosion
            max_pairs = 50
            pair_count = 0
            for idx_x in range(len(candidates_copy)):
                if pair_count >= max_pairs:
                    break
                x = candidates_copy[idx_x]
                for idx_y in range(len(candidates_copy)):
                    if pair_count >= max_pairs:
                        break
                    if idx_x == idx_y:
                        continue
                    y = candidates_copy[idx_y]
                    pair_count += 1
                    
                    # Ensure order x < y
                    if x > y:
                        x, y = y, x
                    
                    if not (left_bound < x < y < right_bound):
                        continue
                        
                    new_seq = seq[:i] + [x, y] + seq[i+1:]
                    if is_valid(new_seq):
                        return new_seq
        return None

    best_seq = seq[:]
    no_improve_count = 0
    
    while time.time() - start_time < timeout:
        improved = False
        
        # 1. Try to increase length (1-out/2-in) - highest priority for length
        new_seq = try_1out_2in(seq)
        if new_seq is not None:
            seq = new_seq
            if len(seq) > len(best_seq):
                best_seq = seq[:]
            improved = True
            no_improve_count = 0
            continue
        
        # 2. Try to insert (1-in) - can increase length if possible
        new_seq = try_insert(seq)
        if new_seq is not None:
            seq = new_seq
            if len(seq) > len(best_seq):
                best_seq = seq[:]
            improved = True
            no_improve_count = 0
            continue
        
        # 3. Try 2-out/2-in to rearrange and enable future moves
        new_seq = try_2out_2in(seq)
        if new_seq is not None:
            seq = new_seq
            if len(seq) > len(best_seq):
                best_seq = seq[:]
            improved = True
            no_improve_count = 0
            continue
        
        # 4. Try 1-out/1-in to shuffle and enable future moves
        new_seq = try_1out_1in(seq)
        if new_seq is not None:
            seq = new_seq
            improved = True
            no_improve_count = 0
            continue
        
        # 5. If nothing worked, we might be stuck
        no_improve_count += 1
        if no_improve_count > 50:
            break

    # Final check to ensure best_seq is valid
    if not is_valid(best_seq):
        # Fallback to greedy
        best_seq = greedy_construct()
    
    return best_seq
