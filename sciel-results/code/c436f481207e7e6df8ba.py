"""
Large sets with cardinality-separated subset sums.

Strategy:
- Greedily build a set of distinct integers in 1..256.
- At each step, try adding each candidate (in increasing order) and check
  the cross-cardinality separation constraint using exact DP.
- The constraint: for any two subsets S, T with |S| != |T|, sum(S) != sum(T).
  Equivalently, the set of achievable sums for cardinality k must be disjoint
  from the set of achievable sums for cardinality k' for all k != k'.
- We maintain, for each cardinality k, the set of achievable sums.
- When we add a new element x, the new sums at cardinality k are old_sums[k-1] + x.
  We must ensure no collision with any other cardinality's sums.
- To maximize size, we use a greedy approach but with a twist: we try candidates
  in a smart order (smaller values first tend to create more collisions, so we
  might want larger values; but larger values also increase sum range).
  Actually, let's try: sort candidates by value, and for each, check if it fits.
  If not, skip it. This is a simple greedy.
  
  Better: use a priority or try multiple orderings. But given 60s limit, let's
  do a careful greedy with backtracking on a small window if needed.

  Actually, the key insight: the constraint is that the sum-sets for different
  cardinalities must be disjoint. This is a strong constraint. Let's think about
  what sets satisfy this.

  If we choose powers of 2, then every subset has a unique sum, so all sum-sets
  are disjoint (each sum corresponds to exactly one subset, hence one cardinality).
  Powers of 2 in 1..256: 1,2,4,8,16,32,64,128 -> 8 elements. That's small.

  What about a geometric progression with ratio > 2? E.g., 1, 3, 9, 27, ... 
  These are "superincreasing" sequences. For a superincreasing sequence, every
  subset sum is unique, so the constraint is satisfied. How many can we fit in 1..256?
  1, 3, 9, 27, 81, 243 -> 6 elements. Worse than powers of 2.

  Powers of 2 give 8. Can we do better?

  The constraint allows repeated sums within the SAME cardinality. So we don't need
  unique subset sums overall, just that sums at different cardinalities don't overlap.

  This is a weaker constraint than unique subset sums. So we might be able to fit
  more elements.

  Let's think: if we have elements a1 < a2 < ... < an, and we want the sum-sets
  for different cardinalities to be disjoint.

  One approach: make the elements grow fast enough that the minimum sum at cardinality
  k+1 exceeds the maximum sum at cardinality k. This is the "separated" condition.
  min_sum(k+1) > max_sum(k) for all k.
  
  min_sum(k) = sum of k smallest elements.
  max_sum(k) = sum of k largest elements.
  
  If we ensure sum of k smallest < sum of (k+1) smallest... wait, that's not right.
  
  Actually, for the sum-sets to be disjoint, a sufficient condition is:
  For each k, max possible sum with k elements < min possible sum with k+1 elements.
  max_sum(k) = sum of k largest elements.
  min_sum(k+1) = sum of (k+1) smallest elements.
  We need: sum of k largest < sum of (k+1) smallest for all k from 0 to n-1.
  (For k=0: max_sum(0)=0, min_sum(1)=smallest element, always true.)
  
  This is a strong sufficient condition. It guarantees disjointness. But it's not necessary.
  
  With this condition, we can try to maximize n. Let's try to construct such a set.
  
  Let the elements be a1 < a2 < ... < an.
  Condition: for each k >= 1, a_{n-k+1} + ... + a_n < a_1 + ... + a_{k+1}.
  
  For k=1: a_n < a_1 + a_2
  For k=2: a_{n-1} + a_n < a_1 + a_2 + a_3
  ...
  For k=n-1: a_2 + ... + a_n < a_1 + ... + a_n  (always true since a1>0)
  
  The tightest constraints are for small k.
  
  Let's try to maximize n with elements in 1..256.
  
  This is getting complex. Let me just implement a greedy algorithm that checks the
  actual sum-set disjointness condition (not just the sufficient condition) and see
  how many elements we can fit.

  Algorithm:
  1. Start with empty set.
  2. For each candidate x in 1..256 (try in some order):
     a. Tentatively add x.
     b. Compute the new sum-sets for each cardinality.
     c. Check that all sum-sets are pairwise disjoint.
     d. If valid, keep x; else, skip.
  3. To improve, try different orders of candidates.

  But computing sum-sets naively is exponential. However, the sums are bounded:
  max sum = 256 * n, and n <= 60, so max sum <= 15360. We can use bitsets.

  For each cardinality k, maintain a bitset of achievable sums.
  When adding x, new_sums[k] = old_sums[k] | (old_sums[k-1] << x).
  Then check that for all i != j, new_sums[i] & new_sums[j] == 0.

  This is O(n * max_sum / word_size) per candidate, which is manageable.

  Let's implement this.
"""

import time
from collections import defaultdict

def construct():
    deadline = time.time() + 80  # 80s safety margin within 90s limit
    
    N = 256
    MAX_N = 60
    
    # Greedy: try candidates in increasing order
    # For each candidate, check if adding it maintains the constraint.
    
    # We'll maintain sum_sums[k] as a Python int used as a bitset.
    # Bit i is set if sum i is achievable with exactly k elements.
    
    def check_and_add(current_set, sum_sums, x, max_sum):
        """Try adding x. Return (new_sum_sums, True) if valid, else (None, False)."""
        n = len(current_set)
        # New cardinality ranges from 0 to n+1
        # old sum_sums has keys 0..n
        # new sum_sums will have keys 0..n+1
        
        # Build new sum_sums
        new_sums = {}
        for k in range(n + 1):
            # Old sums at cardinality k remain
            # Plus: sums at cardinality k-1 shifted by x
            if k == 0:
                new_sums[k] = 1  # empty set, sum 0
            else:
                old_k = sum_sums.get(k, 0)
                old_k_minus_1 = sum_sums.get(k - 1, 0)
                new_sums[k] = old_k | ((old_k_minus_1 << x) & ((1 << (max_sum + 1)) - 1))
        
        # Check disjointness
        keys = list(new_sums.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if new_sums[keys[i]] & new_sums[keys[j]]:
                    return None, False
        
        return new_sums, True
    
    # Start with empty set
    current_set = []
    sum_sums = {0: 1}  # cardinality 0: sum 0
    max_sum = 0
    
    # Try candidates in increasing order
    for x in range(1, N + 1):
        if time.time() > deadline:
            break
        if len(current_set) >= MAX_N:
            break
        
        new_sums, valid = check_and_add(current_set, sum_sums, x, max_sum + x)
        if valid:
            current_set.append(x)
            sum_sums = new_sums
            max_sum += x
    
    # Now try to improve by trying different orderings
    # Try decreasing order
    current_set2 = []
    sum_sums2 = {0: 1}
    max_sum2 = 0
    for x in range(N, 0, -1):
        if time.time() > deadline:
            break
        if len(current_set2) >= MAX_N:
            break
        new_sums, valid = check_and_add(current_set2, sum_sums2, x, max_sum2 + x)
        if valid:
            current_set2.append(x)
            sum_sums2 = new_sums
            max_sum2 += x
    
    # Try a mixed approach: try to fit more by being smarter
    # Use a greedy that at each step picks the candidate that allows the most future additions
    # This is expensive, so only do it if we have time.
    
    best_set = current_set if len(current_set) >= len(current_set2) else current_set2
    
    # Try to improve best_set by attempting to insert additional elements
    # We can try to replace some elements or add new ones
    
    # For now, return the best we found
    return list(best_set)
