<<<<<<< SEARCH
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
=======
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    # Precompute pairwise conflict: conflict[i][j] = True if products a[i]*a[j] == a[k]*a[l]
    # for some k,l in [2,128] (including k=i, l=j etc.)
    # Actually, the real constraint is about consecutive block products.
    # A simpler useful precomputation: for each pair (i,j), compute sig(i)+sig(j)
    # and store in a set so we can quickly check if that sum appears as any single sig.
    # But the actual constraint is: no two consecutive blocks have the same product.
    # Two blocks [i..j] and [k..l] conflict if sum(sigs[i..j]) == sum(sigs[k..l]).
    # This is hard to precompute for all pairs. Instead, let's precompute all
    # "pair sums" sig[i]+sig[j] for i<=j, and also all single sigs, and check
    # if any pair sum equals any other pair sum or single sig.
    # Actually the key insight: if sig[a]+sig[b] == sig[c]+sig[d] for distinct pairs,
    # then having both {a,b} and {c,d} as consecutive blocks would conflict.
    # But the blocks must be consecutive in the sequence, not arbitrary subsets.
    
    # A more practical speedup: precompute sig[i]+sig[j] for all i,j in [2,128]
    # and store in a dict mapping sum_sig -> list of (i,j) pairs.
    # Then when checking if adding x creates a conflict, we can check if
    # any existing consecutive pair in the sequence has the same sum as a new pair.
    # This is still complex. Let's focus on a different angle:
    # Precompute for each number x, which numbers y have the property that
    # sig[x] + sig[y] equals sig[z] for some z (i.e., x*y is a perfect power of primes
    # that matches some single number's signature). This helps detect when
    # a 2-element block product equals a 1-element block product.
    pair_to_single = {}  # (i,j) -> True if sig[i]+sig[j] == sig[k] for some k
    for i in range(2, 129):
        for j in range(i, 129):
            s = add_sig(sigs_fixed[i], sigs_fixed[j])
            # Check if s equals any sig[k]
            for k in range(2, 129):
                if s == sigs_fixed[k]:
                    pair_to_single[(i, j)] = True
                    break
>>>>>>> REPLACE

