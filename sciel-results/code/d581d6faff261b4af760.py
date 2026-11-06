import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Primes up to 127
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Precompute factor vectors for 2..128
    # Vector is a tuple of exponents for each prime
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

    # State: (seq, prefixes, sums)
    # seq: list of ints
    # prefixes: list of tuples (prefix sum vectors)
    # sums: set of tuples (all subarray sum vectors)
    
    # Initial state
    # prefixes = [0]
    # sums = {}
    beam = [([], [zero_vec], set())]
    
    best_seq = []
    best_len = 0
    
    # Dynamic beam width
    # We want to explore broadly at the start, then focus.
    # Total numbers to consider: 127 (2 to 128)
    
    for x in range(2, 129):
        if time.time() > deadline:
            break
            
        # Determine beam width
        # Early: Wide. Late: Narrow.
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
            # We keep the state as is.
            # To avoid duplicating states in new_beam, we could just carry them over,
            # but since we sort and truncate, we need to add them.
            # However, adding the same state multiple times is wasteful.
            # But different states in `beam` are different.
            # So we add the current state to new_beam.
            new_beam.append(state)
            
            # Option 2: Take x
            # Calculate new prefix sum
            # new_prefix = last_prefix + v_x
            # Since tuples are immutable, we create a new tuple
            new_prefix_val = [last_prefix[i] + v_x[i] for i in range(num_primes)]
            new_prefix = tuple(new_prefix_val)
            
            # Calculate new subarray sums
            # New sums are new_prefix - p for all p in prefixes
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
                # Create new state
                new_seq = seq + [x]
                new_prefixes = prefixes + [new_prefix]
                new_sums_set = sums.union(new_sums)
                
                new_beam.append((new_seq, new_prefixes, new_sums_set))
                
                # Update best
                if len(new_seq) > best_len:
                    best_len = len(new_seq)
                    best_seq = new_seq
        
        if not new_beam:
            break
            
        # Sort new_beam
        # Primary key: Length (descending)
        # Secondary key: "Flexibility" or "Potential"
        # A simple heuristic: Prefer states with smaller last element? 
        # Or prefer states that have used fewer "small" numbers?
        # Actually, since we process x in increasing order, the last element is determined by the path.
        # Let's just sort by length. If lengths are equal, we can keep the first ones or randomize.
        # To maintain diversity, we could sort by length, then by the last element of the sequence (smaller is better? 
        # No, smaller last element means we picked smaller numbers, which might have caused more collisions? 
        # Actually, primes are better. 
        # Let's just sort by length. Python's sort is stable, so it will keep the order of insertion for ties.
        # To improve diversity, we could shuffle the beam before sorting? No, that breaks the "best first" nature.
        # Let's try sorting by length, then by the sum of the last few elements?
        # For now, simple length sort.
        
        new_beam.sort(key=lambda s: len(s[0]), reverse=True)
        
        # Keep top K
        beam = new_beam[:beam_width]
        
        # Early exit if we can't beat best_len
        # Max possible length = best_len + (128 - x)
        # If even taking all remaining doesn't beat best_len, we can stop?
        # No, because we might have a state in beam with length < best_len that can grow?
        # Wait, if max possible length from ANY state in beam is <= best_len, we can stop.
        # Max possible length for a state with current length L is L + (128 - x).
        # The state with max L in beam is beam[0].
        if beam and len(beam[0][0]) + (128 - x) <= best_len:
            break

    return best_seq

# Test locally if needed
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
