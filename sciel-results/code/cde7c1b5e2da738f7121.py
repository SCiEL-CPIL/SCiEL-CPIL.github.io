import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit

    # Primes up to 127
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)

    # Precompute exponent vectors for all candidates 2..128
    def get_vec(n):
        v = np.zeros(num_primes, dtype=np.int32)
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        return v

    vecs = {i: get_vec(i) for i in range(2, 129)}
    zero_vec = np.zeros(num_primes, dtype=np.int32)

    def vec_to_tuple(v):
        return tuple(v)

    def is_valid_insertion(seq, n, pvecs, pset):
        """
        Check if n can be inserted into seq (which is sorted).
        Returns (new_seq, new_pvecs, new_pset) if valid, else None.
        """
        # Find insertion index
        # seq is strictly increasing.
        # We need to find pos such that seq[pos-1] < n < seq[pos]
        # Since we are checking against the *current* seq which is a subset of 2..128,
        # and n is not in seq, we can use bisect.
        import bisect
        pos = bisect.bisect_left(seq, n)
        
        # Construct new prefix vectors
        # pvecs[0] = 0
        # pvecs[i] = sum(v_{seq[0..i-1]})
        # If we insert n at pos (0-indexed in the new sequence, meaning it becomes the (pos+1)-th element),
        # The prefix sums before pos remain the same.
        # The prefix sums from pos onwards are shifted by v_n.
        
        v_n = vecs[n]
        new_pvecs = pvecs[:pos+1] # P_0 .. P_pos (where P_pos is sum of first pos elements)
        # Wait, pvecs has length len(seq)+1.
        # pvecs[0] is sum of 0 elements.
        # pvecs[k] is sum of first k elements.
        # If we insert at index `pos` in the list `seq` (0-indexed), it becomes the (pos+1)-th element.
        # So the new prefix sum at index pos+1 is pvecs[pos] + v_n.
        # The new prefix sum at index pos+2 is pvecs[pos+1] + v_n.
        # ...
        
        # Let's just rebuild or update carefully.
        # New pvecs length is len(seq) + 2.
        new_pvecs = [None] * (len(seq) + 2)
        new_pvecs[0] = zero_vec
        for i in range(pos):
            new_pvecs[i+1] = pvecs[i+1]
        
        # Insert n
        new_pvecs[pos+1] = pvecs[pos] + v_n
        
        for i in range(pos, len(seq)):
            # The old pvecs[i+1] becomes new_pvecs[i+2]
            # new_pvecs[i+2] = new_pvecs[i+1] + v_{seq[i]}
            # Which is (pvecs[pos] + v_n) + sum(v_{seq[pos..i-1]})
            # = pvecs[i] + v_n
            new_pvecs[i+2] = pvecs[i+1] + v_n

        # Check all new block products
        # New blocks are those involving the new element n.
        # These are blocks [n..j] for j >= pos+1 and [i..n] for i <= pos.
        # In terms of prefix differences:
        # Block starting at n (index pos+1) and ending at j (index k+1 in new pvecs? No, index j in seq is index j+1 in pvecs)
        # Let's use indices in new_pvecs.
        # n is at index pos+1 in new_pvecs (as the end of the prefix).
        # Actually, n is the element at index `pos` in the new sequence list.
        # So its prefix sum is new_pvecs[pos+1].
        # Blocks ending at n: new_pvecs[pos+1] - new_pvecs[i] for i in 0..pos
        # Blocks starting at n: new_pvecs[j] - new_pvecs[pos+1] for j in pos+2..len(new_pvecs)-1
        
        new_prods = []
        
        # Blocks ending at n
        for i in range(pos + 1):
            diff = new_pvecs[pos+1] - new_pvecs[i]
            diff_tup = vec_to_tuple(diff)
            if diff_tup in pset:
                return None
            new_prods.append(diff_tup)
            
        # Blocks starting at n (and not ending at n, i.e., length > 1)
        for j in range(pos + 2, len(new_pvecs)):
            diff = new_pvecs[j] - new_pvecs[pos+1]
            diff_tup = vec_to_tuple(diff)
            if diff_tup in pset:
                return None
            new_prods.append(diff_tup)
            
        # Note: The block [n] itself is new_pvecs[pos+1] - new_pvecs[pos].
        # This is included in "Blocks ending at n" with i=pos.
        
        # Check for collisions among the NEW products themselves?
        # The problem states "Products of every nonempty consecutive block must be distinct".
        # This implies ALL products in the final set must be distinct.
        # The existing pset contains all products from the original sequence.
        # The new_prods are the products involving n.
        # We checked them against pset.
        # We must also check if any two new_prods are equal to each other.
        # Since they all involve n, can two different blocks involving n have the same product?
        # Block A: n * ...
        # Block B: ... * n
        # If A and B are distinct blocks, their products must be distinct.
        # Let's just put them in a set and check size.
        prod_set_check = set(new_prods)
        if len(prod_set_check) != len(new_prods):
            return None
            
        new_pset = pset.union(new_prods)
        new_seq = seq[:pos] + [n] + seq[pos:]
        
        return (new_seq, new_pvecs, new_pset)

    def beam_search(beam_width, use_random_tiebreak=False):
        beam = [([], [zero_vec], set())]
        
        for n in range(2, 129):
            if time.time() > deadline - 5.0:
                break

            v_n = vecs[n]
            new_beam = []

            for seq, pvecs, pset in beam:
                # Option 1: Exclude n
                new_beam.append((seq, pvecs, pset))

                # Option 2: Include n (append, since we are iterating in sorted order)
                last_pvec = pvecs[-1]
                new_pvec = last_pvec + v_n
                
                valid = True
                new_prods = []
                for i in range(len(pvecs)):
                    diff = new_pvec - pvecs[i]
                    diff_tup = vec_to_tuple(diff)
                    if diff_tup in pset:
                        valid = False
                        break
                    new_prods.append(diff_tup)
                
                if valid:
                    new_seq = seq + [n]
                    new_pvecs = pvecs + [new_pvec]
                    new_pset = pset.union(new_prods)
                    new_beam.append((new_seq, new_pvecs, new_pset))

            if not new_beam:
                break

            # Deduplicate
            seen_seqs = set()
            unique_new_beam = []
            for state in new_beam:
                key = tuple(state[0])
                if key not in seen_seqs:
                    seen_seqs.add(key)
                    unique_new_beam.append(state)
            
            # Sort
            if use_random_tiebreak:
                # Shuffle to diversify
                random.shuffle(unique_new_beam)
            unique_new_beam.sort(key=lambda x: len(x[0]), reverse=True)
            
            beam = unique_new_beam[:beam_width]

        return beam[0][0] if beam else []

    def polish(seq):
        if not seq:
            return seq
        
        # Reconstruct pvecs and pset
        pvecs = [zero_vec]
        cur = zero_vec
        for val in seq:
            cur = cur + vecs[val]
            pvecs.append(cur)
        pset = set()
        for i in range(len(pvecs)):
            for j in range(i+1, len(pvecs)):
                diff = pvecs[j] - pvecs[i]
                pset.add(vec_to_tuple(diff))
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Try to insert any missing number
            # To be efficient, try numbers that are "close" to existing ones or just all missing ones
            # We'll iterate through all 2..128 and try to insert
            # To avoid O(N^2) checks on large N, we can just try them. N=127 is small.
            for n in range(2, 129):
                if n in seq:
                    continue
                res = is_valid_insertion(seq, n, pvecs, pset)
                if res:
                    seq, pvecs, pset = res
                    improved = True
                    # Restart the loop to try inserting more, as inserting one might enable/disable others
                    # Or just continue? If we insert n, the pvecs change.
                    # It's safer to restart the scan or at least update the state.
                    # Since we update pvecs and pset, we can just continue the for loop?
                    # No, inserting n changes the indices for subsequent checks in the same for loop?
                    # The for loop iterates `n` from 2 to 128.
                    # If we insert `n`, `seq` changes.
                    # The next `n` in the loop is `n+1`.
                    # This is fine.
                    # However, inserting a small number might allow inserting a larger number that was previously blocked?
                    # Or inserting a large number might block a smaller one?
                    # Since we iterate in increasing order of `n`, we might miss opportunities where inserting a larger number first allows a smaller one?
                    # But the sequence must be sorted.
                    # Let's just restart the whole polish while loop after any insertion.
                    break
        
        return seq

    best_seq = []
    
    # Run beam search with increasing beam widths or multiple restarts
    # Start with a moderate beam width
    beam_widths = [50, 100, 200, 500]
    
    for bw in beam_widths:
        if time.time() > deadline - 10.0:
            break
        
        # Run a few restarts for larger beam widths to find diverse solutions
        restarts = 1 if bw < 100 else 3
        for _ in range(restarts):
            if time.time() > deadline - 5.0:
                break
            seq = beam_search(bw, use_random_tiebreak=(bw > 100))
            if len(seq) > len(best_seq):
                best_seq = seq
    
    # Polish the best found
    if best_seq:
        best_seq = polish(best_seq)

    return best_seq
