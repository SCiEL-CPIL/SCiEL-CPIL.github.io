"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import sys

def construct(n):
    """
    Constructs the shortest superpermutation for n symbols.
    Uses a heuristic search (2-opt/neighborhood) on the permutation order to minimize length.
    """
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin for safety
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # Generate all permutations of 1..n
    perms = list(itertools.permutations(range(1, n + 1)))
    
    # Calculate cost of a sequence of permutations
    # The word is built by overlapping permutations.
    # Cost is sum of (n - overlap) for each transition?
    # Actually, length = n + sum(n - overlap(p_i, p_{i+1}))
    # Overlap(p, q) is max k such that p[-k:] == q[:k]
    
    def calc_overlap(p, q):
        # p, q are tuples of length n
        # Max overlap is n-1 (since they are permutations, they can't be identical if k=n)
        # Wait, if p == q, overlap is n? No, if p == q, we don't need to append anything?
        # But we must include every permutation as a window.
        # If we have ...p... and next is p, the window p is already there.
        # But the construction usually assumes distinct permutations in the sequence?
        # No, the sequence of permutations defines the order in which we "visit" them.
        # If we visit p, then q, the overlap is the max k such that suffix of p == prefix of q.
        # If p == q, overlap is n? Then we add 0?
        # Let's check: if we have ...p and we want to add p.
        # The word ends in p. The next window is p.
        # We don't need to add any symbols.
        # So overlap n is valid.
        
        max_k = n
        for k in range(max_k, 0, -1):
            if p[-k:] == q[:k]:
                return k
        return 0
    
    # Initialize sequence with a random permutation of perms
    # Or use the standard recursive order?
    # Standard order is good, but we can shuffle and 2-opt.
    
    # Start with a random shuffle to explore, or standard?
    # Standard is deterministic. Let's try to improve standard.
    # But standard is already 5913.
    # Let's try a random start and 2-opt.
    
    # To ensure we don't timeout, we need to be fast.
    # Number of perms is 5040.
    # Calc overlap is O(n^2) or O(n).
    # 2-opt on 5040 items.
    
    # Let's use a faster overlap calculation.
    # Precompute overlaps? 5040^2 * 7 is too big.
    # Just calculate on the fly.
    
    # Heuristic: 
    # 1. Start with standard construction order (derived from recursive definition).
    # 2. Apply 2-opt improvements.
    
    # How to get standard order?
    # The standard construction for n=3 is 123121321.
    # Perms: 123, 231, 312, 121 (wait, 121 is not a perm).
    # The windows are 123, 231, 312, 121? No.
    # 123121321
    # 123
    # 231
    # 312
    # 121 (invalid)
    # 213
    # 132
    # 321
    
    # The standard recursive construction generates a specific order.
    # Let's generate it.
    
    def standard_order(n):
        if n == 1:
            return [(1,)]
        prev = standard_order(n - 1)
        result = []
        for p in prev:
            # p is a tuple of 1..n-1
            # We need to insert n.
            # The recursive step:
            # For each p, we append n, then p again?
            # No, the word construction is:
            # W_n = W_{n-1} + n + W_{n-1} ...?
            
            # Actually, the standard superpermutation S_n is built from S_{n-1}.
            # S_n = S_{n-1} followed by for each permutation p of 1..n-1:
            #       append n, then append p (except the first p which is already there?)
            
            # Let's just use the property that the order of permutations in the standard
            # superpermutation is the order they appear in the word.
            
            # Let's just use a random shuffle for now, it's simpler and 2-opt might fix it.
            pass
        return result
    
    # Let's just use a random permutation of the set of perms.
    seq = perms[:]
    random.shuffle(seq)
    
    def calc_total_cost(seq):
        cost = n # First perm adds n symbols
        for i in range(len(seq) - 1):
            cost += n - calc_overlap(seq[i], seq[i+1])
        return cost
    
    current_cost = calc_total_cost(seq)
    
    # 2-opt local search
    # We try to reverse a segment of the sequence.
    # Or swap adjacent?
    # 2-opt is better.
    
    # To speed up, we can precompute overlaps between neighbors?
    # No, just calculate delta.
    
    # Delta for reversing seq[i:j] (inclusive)
    # Old: ... A, B, C, D ...
    # New: ... A, D, C, B ...
    # Change in cost:
    # Remove: overlap(A,B) + overlap(B,C) + overlap(C,D)
    # Add: overlap(A,D) + overlap(D,C) + overlap(C,B)
    
    # Note: overlap is not symmetric. overlap(A,B) != overlap(B,A).
    
    # Let's implement a simple local search with swaps and reversals.
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # Try all pairs (i, j)
        # This is O(N^2) per pass. N=5040. 25M pairs.
        # Too slow for Python if done naively.
        
        # We need a smarter heuristic or a faster language feature.
        # Maybe just try random moves?
        
        # Let's try random 2-opt moves.
        for _ in range(1000): # Fixed number of random moves per pass?
            if time.time() > deadline - 10:
                break
            i = random.randint(0, len(seq) - 2)
            j = random.randint(i + 1, len(seq) - 1)
            
            # Calculate delta for reversing seq[i:j]
            # Let L = i-1, R = j+1
            # If i == 0, no left neighbor.
            # If j == N-1, no right neighbor.
            
            # Cost components affected:
            # 1. Connection between i-1 and i
            # 2. Connection between j and j+1
            # 3. Internal connections in [i, j]
            
            # This is complex.
            # Let's just recalculate cost for the affected range?
            # Or just do a full recalculation if it's fast enough?
            # 5040 * 7 = 35k ops.
            # 1000 moves * 35k = 35M ops.
            # Python can do ~10M-50M ops/sec.
            # So 1000 moves takes ~1-3 seconds.
            # We have 110 seconds.
            # So we can do ~30-100 passes of 1000 moves.
            
            # Let's try it.
            
            # Store old seq
            old_seq = seq[:]
            seq[i:j+1] = reversed(seq[i:j+1])
            
            new_cost = calc_total_cost(seq)
            
            if new_cost < current_cost:
                current_cost = new_cost
                improved = True
            else:
                seq = old_seq
            
        # If no improvement in a pass, stop?
        # Or keep going?
        # Let's keep going until time runs out or no improvement in a full random pass.
        
    # Build the word from the best sequence
    word = list(seq[0])
    for i in range(1, len(seq)):
        p = seq[i]
        prev = seq[i-1]
        overlap = calc_overlap(prev, p)
        word.extend(p[overlap:])
        
    return "".join(map(str, word))

# EVOLVE-BLOCK-END
