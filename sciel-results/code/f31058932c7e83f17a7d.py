import time
import sys
import random
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Primes up to 127
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Precompute factor vectors for 2..128
    def get_vec(n):
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
            if temp == 1:
                break
        return tuple(v)
        
    vecs = {i: get_vec(i) for i in range(2, 129)}
    zero_vec = (0,) * num_primes

    # --- Phase 1: Beam Search to find diverse good seeds ---
    # State: (seq, prefixes, sums)
    beam = [([], [zero_vec], set())]
    
    best_seqs = [] # List of (len, seq)
    
    for x in range(2, 129):
        if time.time() > start_time + 5.0: # Run beam search for 5s
            break
            
        # Dynamic beam width
        if x < 10:
            beam_width = 500
        elif x < 30:
            beam_width = 200
        elif x < 60:
            beam_width = 100
        else:
            beam_width = 50
            
        new_beam = []
        v_x = vecs[x]
        
        for state in beam:
            seq, prefixes, sums = state
            last_prefix = prefixes[-1]
            
            # Option 1: Skip x
            new_beam.append(state)
            
            # Option 2: Take x
            new_prefix_val = [last_prefix[i] + v_x[i] for i in range(num_primes)]
            new_prefix = tuple(new_prefix_val)
            
            new_sums = set()
            valid = True
            for p in prefixes:
                diff_val = [new_prefix[i] - p[i] for i in range(num_primes)]
                diff_tup = tuple(diff_val)
                if diff_tup in sums:
                    valid = False
                    break
                new_sums.add(diff_tup)
            
            if valid:
                new_seq = seq + [x]
                new_prefixes = prefixes + [new_prefix]
                new_sums_set = sums.union(new_sums)
                
                new_beam.append((new_seq, new_prefixes, new_sums_set))
        
        if not new_beam:
            break
            
        new_beam.sort(key=lambda s: len(s[0]), reverse=True)
        beam = new_beam[:beam_width]
        
        # Collect top unique sequences
        # We want diversity, so maybe take top 10 distinct sequences by length
        if beam:
             current_len = len(beam[0][0])
             if not best_seqs or current_len > best_seqs[0][0]:
                 # Replace top if strictly better, or add if same length and different seq
                 # Simple: just keep the top 5 distinct sequences from the beam
                 pass
                 
    # Extract top 5 distinct sequences from the final beam
    seen_seqs = set()
    top_seqs = []
    for state in beam:
        seq_tuple = tuple(state[0])
        if seq_tuple not in seen_seqs:
            seen_seqs.add(seq_tuple)
            top_seqs.append(state[0])
        if len(top_seqs) >= 5:
            break
            
    if not top_seqs:
        return []
        
    # Sort by length desc
    top_seqs.sort(key=len, reverse=True)
    
    best_global_seq = top_seqs[0]
    best_global_len = len(best_global_seq)
    
    # --- Phase 2: Local Search (Insertion) ---
    # We will try to improve each of the top_seqs
    # To manage time, we'll iterate through them.
    
    def check_and_insert(seq, u, pos):
        """
        Checks if u can be inserted at pos in seq.
        Returns True if valid, False otherwise.
        If valid, does NOT modify seq (pure function).
        """
        # 1. Check bounds
        if pos < 0 or pos > len(seq):
            return False
        if pos == 0:
            if seq and u >= seq[0]: return False
        elif pos == len(seq):
            if seq and u <= seq[-1]: return False
        else:
            if not (seq[pos-1] < u < seq[pos]):
                return False
                
        # 2. Collect existing products
        # We don't have the set of products here. We need to pass it or recompute.
        # Recomputing is O(N^2). N is small (~40). 1600 ops. Fast enough.
        existing_products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                existing_products.add(prod)
                
        # 3. Generate new products involving u at pos
        new_products = set()
        
        # Products starting at u (u, u*seq[pos], u*seq[pos]*seq[pos+1]...)
        prod = u
        new_products.add(prod)
        if prod in existing_products:
            return False
        for j in range(pos, n):
            prod *= seq[j]
            new_products.add(prod)
            if prod in existing_products:
                return False
            if len(new_products) > 1:
                 # Check for internal collisions among new products?
                 # Since we are adding to a set, we can check if len(set) < len(list)
                 # But we are adding one by one.
                 pass
                 
        # Products ending at u (...*seq[pos-2]*seq[pos-1]*u)
        prod = u
        # prod is already in new_products
        for j in range(pos-1, -1, -1):
            prod *= seq[j]
            if prod in existing_products:
                return False
            if prod in new_products:
                 # Collision between a product ending at u and one starting at u?
                 # Or two products ending at u?
                 # Two products ending at u: seq[j..pos-1]*u. Distinct j => distinct length => distinct product (since all > 1).
                 # Product ending at u vs starting at u:
                 # seq[j..pos-1]*u == u*seq[pos..k] => seq[j..pos-1] == seq[pos..k]
                 # This is possible.
                 return False
            new_products.add(prod)
            
        return True

    def get_all_products(seq):
        prods = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                prods.add(prod)
        return prods

    def local_search(initial_seq):
        seq = list(initial_seq)
        products = get_all_products(seq)
        used = set(seq)
        unused = [x for x in range(2, 129) if x not in used]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle unused to avoid bias towards small numbers?
            # Or iterate in order?
            # Inspiration 1 shuffled. Let's shuffle.
            random.shuffle(unused)
            
            for u in unused:
                if time.time() > deadline:
                    break
                    
                pos = bisect.bisect_left(seq, u)
                
                # Quick bounds check
                if pos < len(seq) and seq[pos] == u:
                    continue # Should not happen as u is unused
                if pos > 0 and seq[pos-1] >= u:
                    continue
                if pos < len(seq) and seq[pos] <= u:
                    continue
                    
                # Detailed check
                # We need to check if inserting u creates duplicate products.
                # 1. u not in products
                if u in products:
                    continue
                    
                # 2. New products:
                #    Suffixes of left part * u
                #    Prefixes of right part * u
                #    u itself
                
                # Let's compute the new products and check against `products`
                # and against each other.
                
                new_prods = set()
                new_prods.add(u)
                
                # Right side: u * seq[pos], u * seq[pos] * seq[pos+1] ...
                prod = u
                for j in range(pos, len(seq)):
                    prod *= seq[j]
                    if prod in products:
                        break
                    if prod in new_prods:
                        break
                    new_prods.add(prod)
                else:
                    # Left side: seq[pos-1]*u, seq[pos-2]*seq[pos-1]*u ...
                    prod = u
                    for j in range(pos-1, -1, -1):
                        prod *= seq[j]
                        if prod in products:
                            break
                        if prod in new_prods:
                            break
                        new_prods.add(prod)
                    else:
                        # All good!
                        # Perform insertion
                        seq.insert(pos, u)
                        products.update(new_prods)
                        used.add(u)
                        unused.remove(u)
                        improved = True
                        break # Restart loop with new state
            
        return seq

    # Run local search on the top seeds
    # We allocate time based on seed quality
    for i, seed_seq in enumerate(top_seqs):
        if time.time() > deadline - 5:
            break
            
        # Run local search
        improved_seq = local_search(seed_seq)
        
        if len(improved_seq) > best_global_len:
            best_global_len = len(improved_seq)
            best_global_seq = improved_seq
            
    return best_global_seq

# Test locally if needed
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
