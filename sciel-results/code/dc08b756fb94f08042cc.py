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
    # Use numpy for fast vector addition
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

    # We will use prefix sums of exponent vectors.
    # If sequence is a_1, ..., a_k, prefix sums P_0=0, P_i = sum_{j=1}^i v_{a_j}.
    # Product of a_i...a_j is P_j - P_{i-1}.
    # We need all P_j - P_i (for 0 <= i < j <= k) to be distinct.
    # This is equivalent to all differences between prefix sums being distinct.
    # Or, we can just track the set of all block products directly.
    # Since k is at most ~50-60, the number of blocks is O(k^2) ~ 3000.
    # Storing 3000 vectors of length 31 in a set is feasible.
    # To speed up hashing, we can convert to tuples or use a custom hash.
    # Tuples of ints are fast to hash in Python.

    def vec_to_tuple(v):
        return tuple(v)

    # Beam Search
    # State: (seq, prefix_vecs_list, prod_set)
    # prefix_vecs_list: list of numpy arrays (or tuples) representing P_0, P_1, ..., P_k
    # prod_set: set of tuples representing all block products

    beam_width = 100
    beam = [([], [zero_vec], set())]

    for n in range(2, 129):
        if time.time() > deadline - 5.0:
            break

        v_n = vecs[n]
        new_beam = []

        for seq, pvecs, pset in beam:
            # Option 1: Exclude n
            # We keep the state. To avoid bloating the beam with identical states,
            # we rely on deduplication later.
            new_beam.append((seq, pvecs, pset))

            # Option 2: Include n
            last_pvec = pvecs[-1]
            new_pvec = last_pvec + v_n  # numpy addition
            
            # Check validity: all new blocks ending at n must not be in pset
            # New blocks are new_pvec - pvecs[i] for i in 0..k
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

        # Sort by length descending, then lexicographic (smaller is better? or arbitrary?)
        # Let's sort by length desc, then by the last element asc (prefer smaller numbers earlier?)
        # Actually, any tie-breaking is fine as long as we keep diverse states.
        # Sorting by tuple(seq) might be expensive. Let's just sort by length.
        # To ensure diversity, we can shuffle before taking top K?
        # Or we can deduplicate first.
        
        # Deduplicate states with same sequence
        seen_seqs = set()
        unique_new_beam = []
        for state in new_beam:
            key = tuple(state[0])
            if key not in seen_seqs:
                seen_seqs.add(key)
                unique_new_beam.append(state)
        
        # Sort by length descending
        unique_new_beam.sort(key=lambda x: len(x[0]), reverse=True)
        
        beam = unique_new_beam[:beam_width]

        # Check if we have a perfect solution (unlikely, but good for early exit)
        if beam and len(beam[0][0]) == 127:
            return beam[0][0]

    # Local Search / Polish
    # Take the best sequence from the beam and try to improve it.
    best_seq = beam[0][0] if beam else []
    best_len = len(best_seq)
    
    # Reconstruct pset for the best sequence to allow further additions
    if best_seq:
        pvecs = [zero_vec]
        cur = zero_vec
        for val in best_seq:
            cur = cur + vecs[val]
            pvecs.append(cur)
        pset = set()
        for i in range(len(pvecs)):
            for j in range(i+1, len(pvecs)):
                diff = pvecs[j] - pvecs[i]
                pset.add(vec_to_tuple(diff))
        
        # Try to add more elements at the end
        last_val = best_seq[-1]
        for n in range(last_val + 1, 129):
            if time.time() > deadline - 2.0:
                break
            v_n = vecs[n]
            new_pvec = pvecs[-1] + v_n
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
                best_seq.append(n)
                pvecs.append(new_pvec)
                pset.update(new_prods)
                best_len = len(best_seq)
            else:
                # If adding n fails, maybe we can swap?
                # For simplicity, we just stop or try next n.
                # A more advanced local search would try swapping elements.
                pass

    return best_seq
