import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    # 1. Generate seed superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap
            overlap = 0
            max_check = min(len(new), len(block))
            for o in range(max_check, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Extract permutations from the seed word
    # The seed word contains each permutation of 1..n exactly once
    # We need to extract them in the order they appear
    seed_perms = []
    seen_perms = set()
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i + n])
        if p not in seen_perms:
            seen_perms.add(p)
            seed_perms.append(p)
    
    # Verify we have n! permutations
    # if len(seed_perms) != factorial(n):
    #    # Fallback to simple construction if something went wrong
    #    pass
    
    N = len(seed_perms)
    
    def get_overlap(a, b):
        # Max k such that a[n-k:] == b[:k]
        # k can be 0 to n-1
        # a and b are tuples of length n
        for k in range(n - 1, -1, -1):
            if k == 0:
                return 0
            if a[n - k:] == b[:k]:
                return k
        return 0

    # Calculate current length
    current_length = n
    for i in range(N - 1):
        current_length += n - get_overlap(seed_perms[i], seed_perms[i + 1])
    
    best_perms = seed_perms
    best_length = current_length
    
    # Local Search: 2-Opt
    # We will try to improve by reversing segments
    # To speed up, we can precompute overlaps for adjacent pairs?
    # But 2-opt changes 4 connections.
    
    # Given time constraints, we might not be able to do full 2-opt passes.
    # Let's do a limited 2-opt: try reversing segments of small length first?
    # Or just random swaps?
    # The gap is small (8). A few good moves might suffice.
    
    # Let's try a simple hill climbing with random swaps and 2-opt moves
    # We will run until deadline
    
    perms = best_perms[:]
    length = best_length
    
    # Precompute overlap matrix? No, too memory intensive/slow to build.
    # We will calculate on the fly.
    
    # To make it faster, we can cache overlaps of pairs that are frequently checked?
    # Or just rely on the fact that n=7 is small.
    
    # Let's try a few passes of 2-opt with early termination
    # Or just random 2-opt moves
    
    while time.time() < deadline:
        improved = False
        # Try a random 2-opt move
        # Choose i, j randomly
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        
        # Calculate cost before
        # Connections affected: (i-1, i), (i, i+1), (j-1, j), (j, j+1)
        # If i=0, no (i-1, i). If j=N-1, no (j, j+1).
        
        def calc_cost_change(perms, i, j):
            # Returns (cost_before, cost_after) for the affected edges
            c_before = 0
            c_after = 0
            
            # Edge (i-1, i) -> (i-1, j)
            if i > 0:
                c_before += n - get_overlap(perms[i-1], perms[i])
                c_after += n - get_overlap(perms[i-1], perms[j])
            
            # Edge (i, i+1) -> (j, i+1)  (Note: i+1 becomes the element after j in the reversed segment? No.)
            # Reversal of [i..j]:
            # Old: ... A, P[i], P[i+1], ..., P[j], B ...
            # New: ... A, P[j], P[j-1], ..., P[i], B ...
            
            # Edge (i-1, i) becomes (i-1, j)
            # Edge (i, i+1) becomes (j, i+1) ?? No.
            # The sequence is A, P[i], P[i+1], ..., P[j], B
            # After reversal: A, P[j], P[j-1], ..., P[i], B
            # So edge (i-1, i) is replaced by (i-1, j).
            # Edge (i, i+1) is replaced by (j, i+1) ?
            # Wait. The element at index i is P[i]. The element at index i+1 is P[i+1].
            # In the new sequence, the element at index i is P[j]. The element at index i+1 is P[j-1] (if j>i).
            # So the edge between index i and i+1 is (P[j], P[j-1]).
            # This is NOT (j, i+1).
            
            # Let's trace carefully.
            # Old segment: P[i], P[i+1], ..., P[j]
            # New segment: P[j], P[j-1], ..., P[i]
            
            # Edge entering the segment: (P[i-1], P[i]) -> (P[i-1], P[j])
            # Edge leaving the segment: (P[j], P[j+1]) -> (P[i], P[j+1])
            
            # Internal edges:
            # Old: (P[k], P[k+1]) for k=i to j-1
            # New: (P[k], P[k-1]) for k=j down to i+1
            # So the set of internal edges is the same, just reversed direction?
            # No, overlap is not symmetric. overlap(A,B) != overlap(B,A).
            # So we must recalculate all internal edges.
            
            # This makes 2-opt expensive: O(j-i) cost calculation.
            # If j-i is large, it's slow.
            
            # However, we only need to check if the TOTAL cost decreases.
            # Total cost change = (New Internal Cost - Old Internal Cost) + (New Boundary Cost - Old Boundary Cost)
            
            # Old Internal Cost: sum(n - overlap(P[k], P[k+1])) for k in i..j-1
            # New Internal Cost: sum(n - overlap(P[k], P[k-1])) for k in i+1..j
            # Note: overlap(P[k], P[k-1]) is the overlap of the reversed pair.
            
            # We can calculate this sum.
            
            c_int_before = 0
            c_int_after = 0
            for k in range(i, j):
                c_int_before += n - get_overlap(perms[k], perms[k+1])
                c_int_after += n - get_overlap(perms[k+1], perms[k])
            
            # Boundary costs
            if i > 0:
                c_before += n - get_overlap(perms[i-1], perms[i])
                c_after += n - get_overlap(perms[i-1], perms[j])
            else:
                # No left boundary
                pass
            
            if j < N - 1:
                c_before += n - get_overlap(perms[j], perms[j+1])
                c_after += n - get_overlap(perms[i], perms[j+1])
            else:
                # No right boundary
                pass
            
            return c_int_before + c_before, c_int_after + c_after

        cost_before, cost_after = calc_cost_change(perms, i, j)
        
        if cost_after < cost_before:
            # Accept move
            perms[i:j+1] = reversed(perms[i:j+1])
            length += cost_after - cost_before
            improved = True
            
            if length < best_length:
                best_length = length
                best_perms = perms[:]
                
            # If we found an improvement, we can try to continue or restart
            # To avoid getting stuck, we might want to randomize more
            # But let's just keep going
            
    # Reconstruct word from best_perms
    final_word = list(best_perms[0])
    for i in range(1, len(best_perms)):
        p_prev = best_perms[i-1]
        p_curr = best_perms[i]
        ov = get_overlap(p_prev, p_curr)
        final_word.extend(p_curr[ov:])
        
    return "".join(map(str, final_word))
