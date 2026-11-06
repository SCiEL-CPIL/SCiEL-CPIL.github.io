import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    def is_valid_fast(seq):
        """Check if all consecutive block products are distinct."""
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

    # 1. Fast Greedy Initialization to get a strong lower bound
    # This helps prune the beam search early.
    def fast_greedy():
        values = []
        products = set()
        for x in range(2, 129):
            tails = []
            prod = x
            valid = True
            for a in reversed(values):
                prod *= a
                if prod in products:
                    valid = False
                    break
                tails.append(prod)
            
            if not valid:
                continue
            
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values

    best_seq = fast_greedy()
    best_len = len(best_seq)
    
    # 2. Beam Search
    # State: (seq_tuple, frozenset_of_products, tuple_of_suffix_products)
    # Using tuples for immutability and hashing speed.
    
    # Dynamic Beam Width: Start wide, narrow down.
    # At x=2, we have few choices. At x=128, we are refining.
    # Let's define a function for beam width based on x.
    def get_beam_width(x):
        # x ranges from 2 to 128.
        # Early on, allow more diversity.
        if x < 20:
            return 1000
        elif x < 50:
            return 500
        elif x < 90:
            return 200
        else:
            return 50

    # Initial state: empty sequence
    # prods: empty set
    # sufs: empty tuple (suffix products of the sequence so far, ending with 1? 
    # Actually, let's define sufs as the products of suffixes of the current sequence.
    # If seq is [], sufs is () or (1,)? 
    # Let's trace:
    # seq = [a]. Suffixes: [a]. Products: {a}.
    # If we add x, new suffixes are [x, x*a].
    # So if seq is empty, we can say "suffixes" is effectively [1] for the multiplication step?
    # Let's use a list of suffix products.
    # If seq is [s1, ..., sk], suffixes are [sk, sk*sk-1, ..., sk*...*s1].
    # If we add x, new suffixes are [x, x*sk, x*sk*sk-1, ...].
    # So we need the previous suffixes to generate new ones.
    # Let's store `sufs` as a tuple of suffix products.
    # For empty seq, sufs = ().
    # When adding x to seq with sufs S:
    # new_sufs = (x,) + tuple(x * s for s in S)
    # Wait, if S is empty, new_sufs = (x,). Correct.
    # If S is (sk,), new_sufs = (x, x*sk). Correct.
    
    beam = [ ((), frozenset(), ()) ]
    
    for x in range(2, 129):
        if time.time() > deadline:
            break
            
        new_beam = []
        K = get_beam_width(x)
        
        for seq, prods, sufs in beam:
            seq_len = len(seq)
            
            # Pruning: If even adding all remaining numbers can't beat best_len, skip
            # Remaining numbers: 128 - x + 1 (including x)
            # Actually, we are processing x now. Remaining after x is 128 - x.
            if seq_len + (128 - x + 1) <= best_len:
                continue
                
            # Option 1: Exclude x
            # State remains (seq, prods, sufs)
            # We add this to new_beam
            new_beam.append((seq, prods, sufs))
            
            # Option 2: Include x
            # Calculate new suffixes
            # new_sufs = (x,) + tuple(x * s for s in sufs)
            # We need to check if any element in new_sufs is in prods.
            # Also, elements in new_sufs are distinct from each other because seq is increasing and >= 2.
            # Proof: s_i > s_{i+1} ... no, suffixes are products of longer prefixes.
            # Let seq = [a, b, c]. Sufs = [c, bc, abc].
            # Add d. New sufs = [d, dc, dcb, dcba].
            # Are they distinct? d < dc < dcb < dcba (since a,b,c >= 2). Yes.
            
            valid = True
            new_sufs_list = [x]
            for s in sufs:
                p = x * s
                new_sufs_list.append(p)
                if p in prods:
                    valid = False
                    break
            
            if valid:
                # Check if x itself is in prods (covered by new_sufs_list[0])
                # Check if any new_sufs in prods (covered by loop)
                # But we also need to check if x is in prods? 
                # new_sufs_list[0] is x. If x in prods, loop would catch it if sufs was empty? 
                # No, if sufs is empty, loop doesn't run. We must check x separately or handle empty sufs.
                # Let's just check the whole new_sufs_list against prods.
                for p in new_sufs_list:
                    if p in prods:
                        valid = False
                        break
                
                if valid:
                    new_prods = prods.union(new_sufs_list)
                    new_seq = seq + (x,)
                    new_sufs = tuple(new_sufs_list)
                    
                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_seq = list(new_seq)
                    
                    new_beam.append((new_seq, new_prods, new_sufs))
        
        if not new_beam:
            break
            
        # Sort and Prune
        # Heuristic: 
        # 1. Length descending
        # 2. Last element ascending (smaller last element is better for future additions)
        # 3. Sum of elements ascending (smaller sum might be "safer"?)
        
        def sort_key(state):
            seq = state[0]
            if not seq:
                last = 0
                s = 0
            else:
                last = seq[-1]
                s = sum(seq)
            return (-len(seq), last, s)
        
        new_beam.sort(key=sort_key)
        beam = new_beam[:K]
        
        # Early stopping check
        # If the best possible length from the current beam is less than best_len, we can stop?
        # max_len_in_beam = max(len(s[0]) for s in beam)
        # if max_len_in_beam + (128 - x) < best_len:
        #     break
        # This is handled in the pruning step for the next iteration, but we can check here.
        
    # 3. Local Search / Polish on the best sequence found
    # Try to insert any numbers from [2, 128] that are not in best_seq
    # This is similar to the current program's greedy insertion but starting from the beam result.
    
    best_seq_set = set(best_seq)
    candidates = [x for x in range(2, 129) if x not in best_seq_set]
    
    # Sort candidates by "safety" - fewer distinct prime factors first?
    # Or just try all in increasing order.
    # Let's try increasing order first, as it maintains the "greedy" feel.
    
    current_seq = list(best_seq)
    for c in candidates:
        if time.time() > deadline:
            break
            
        # Binary search for insertion point
        lo, hi = 0, len(current_seq)
        while lo < hi:
            mid = (lo + hi) // 2
            if current_seq[mid] < c:
                lo = mid + 1
            else:
                hi = mid
        insert_pos = lo
        
        if insert_pos < len(current_seq) and current_seq[insert_pos] == c:
            continue
            
        new_seq = current_seq[:insert_pos] + [c] + current_seq[insert_pos:]
        
        if is_valid_fast(new_seq):
            current_seq = new_seq
    
    if len(current_seq) > best_len:
        best_len = len(current_seq)
        best_seq = current_seq

    return sorted(best_seq)
