import numpy as np

N = 17

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network for N=17 wires using Batcher's Odd-Even Mergesort.
    This is an algorithmic construction distinct from the seed (Odd-Even Transposition).
    Batcher's network is generally more efficient (fewer comparators) than Odd-Even Transposition for N > 4.
    """
    comparators = []
    
    def batcher_sort(n, offset):
        if n <= 1:
            return
        half = n // 2
        batcher_sort(half, offset)
        batcher_sort(half, offset + half)
        batcher_merge(half, half, offset, offset + half)
        
    def batcher_merge(n, m, offset1, offset2):
        # Merge two sorted lists of size n and m
        # They are assumed to be sorted.
        # n and m are the sizes.
        # We use the recursive definition of Batcher's merge.
        if n == 0 or m == 0:
            return
        if n == 1 and m == 1:
            comparators.append((offset1, offset2))
            return
            
        # Split the two lists into odd and even indexed elements
        # List 1: offsets offset1, offset1+1, ...
        # List 2: offsets offset2, offset2+1, ...
        
        # The recursive step of Batcher's merge:
        # Merge the odd-indexed elements of both lists
        # Merge the even-indexed elements of both lists
        # Then compare-exchange between the two merged lists at specific positions
        
        # Standard Batcher Merge implementation:
        # 1. Recursively merge the odd elements of list1 and list2
        # 2. Recursively merge the even elements of list1 and list2
        # 3. Compare-exchange (offset1 + i, offset2 + i) for i in range(min(n, m))? 
        #    Actually, the structure is:
        #    M(n, m) = M(ceil(n/2), ceil(m/2)) + M(floor(n/2), floor(m/2)) + min(n, m) comparators?
        #    No, the comparators are added in the "compare-exchange" phase.
        
        # Let's use the standard recursive structure:
        # The merge of two sorted sequences A (len n) and B (len m)
        # is done by:
        # 1. Merging A_odd and B_odd
        # 2. Merging A_even and B_even
        # 3. Comparing A_odd[i] with B_odd[i] ? No.
        
        # Correct Batcher Merge Logic:
        # To merge two sorted lists X (len n) and Y (len m):
        # 1. Split X into X_odd, X_even
        # 2. Split Y into Y_odd, Y_even
        # 3. Recursively merge X_odd and Y_odd
        # 4. Recursively merge X_even and Y_even
        # 5. For i in range(min(n, m)): compare-exchange (X_odd[i], Y_odd[i])? 
        #    No, the final step compares elements at distance 1?
        
        # Let's look up the exact recurrence for Batcher's Odd-Even Merge.
        # From Knuth:
        # M(n, m) where n <= m.
        # 1. M(ceil(n/2), ceil(m/2))
        # 2. M(floor(n/2), floor(m/2))
        # 3. Compare-exchange (i, i + n/2) ? No.
        
        # Actually, it's easier to implement the "Odd-Even Merge" as:
        # def merge(n, m, o1, o2):
        #   if n == 0 or m == 0: return
        #   if n == 1 and m == 1: add (o1, o2)
        #   # Split
        #   n1 = (n + 1) // 2  # odd part size
        #   n2 = n // 2        # even part size
        #   m1 = (m + 1) // 2
        #   m2 = m // 2
        #   
        #   # Merge odd parts
        #   merge(n1, m1, o1, o2)
        #   # Merge even parts
        #   merge(n2, m2, o1+1, o2+1)
        #   
        #   # Compare-exchange
        #   for i in range(min(n, m)):
        #       # The positions are tricky because the recursive merges have "interleaved" the data?
        #       # No, Batcher's merge keeps the data in the same wires, just sorts the sub-sequences.
        #       # The compare-exchange step connects the two halves.
        #       # Specifically, compare (o1 + i) and (o2 + i) is NOT correct.
        #       # The standard description:
        #       # After merging odds and evens, we compare x_i with y_i for i=0..min(n,m)-1?
        #       # Let's trace n=2, m=2.
        #       # A=[a0, a1], B=[b0, b1]
        #       # Odds: a0, b0. Evens: a1, b1.
        #       # Merge Odds (size 1,1): Compare (a0, b0). Result: min(a0,b0) at a0, max at b0.
        #       # Merge Evens (size 1,1): Compare (a1, b1). Result: min(a1,b1) at a1, max at b1.
        #       # Now we have:
        #       # Wire o1: min(a0,b0)
        #       # Wire o1+1: min(a1,b1)
        #       # Wire o2: max(a0,b0)
        #       # Wire o2+1: max(a1,b1)
        #       # We need to sort the whole thing.
        #       # The final step compares (o1+1, o2)? i.e., (min(a1,b1), max(a0,b0)).
        #       # Yes, for n=m=2, the final comparator is (o1+1, o2).
        #       # General rule: compare (o1 + i + 1, o2 + i) for i in range(min(n//2, m//2))?
        #       # Or (o1 + i, o2 + i) with offsets?
        
        # Let's use a known correct implementation of Batcher's Merge.
        # Reference: "Sorting Networks" by Knuth or standard online implementations.
        
        # Recursive step:
        # 1. Merge odd-indexed elements of A and B.
        # 2. Merge even-indexed elements of A and B.
        # 3. Compare-exchange (o1 + i, o2 + i) for i in 0..min(n,m)-1 is WRONG.
        #    The compare-exchange is between the two resulting sorted sequences?
        #    No, the wires are fixed.
        
        # Correct Batcher Merge (from literature):
        # To merge two sorted sequences of length n and m (n<=m) located at offsets o1, o2:
        # 1. Recursively merge the odd-indexed elements (indices 0, 2, 4...) of both.
        #    Size: n_odd = ceil(n/2), m_odd = ceil(m/2).
        #    Offsets: o1, o2.
        # 2. Recursively merge the even-indexed elements (indices 1, 3, 5...) of both.
        #    Size: n_even = floor(n/2), m_even = floor(m/2).
        #    Offsets: o1+1, o2+1.
        # 3. Compare-exchange (o1 + i, o2 + i) for i in range(min(n, m))?
        #    No. The final step compares elements at distance 1?
        #    Let's check n=2, m=2 again.
        #    After steps 1 & 2:
        #    o1: min(a0, b0)
        #    o1+1: min(a1, b1)
        #    o2: max(a0, b0)
        #    o2+1: max(a1, b1)
        #    We need to ensure o1+1 <= o2.
        #    So we compare (o1+1, o2).
        #    This is index i=0 in the "middle" comparison?
        #    The number of comparators in step 3 is min(n, m) ? No, it's min(n_odd, m_odd) ? No.
        #    For n=2, m=2, we have 1 comparator in step 3.
        #    min(n, m) = 2. So it's not min(n, m).
        #    It is min(n_even, m_even) + min(n_odd, m_odd) - 1?
        #    n_even=1, m_even=1. n_odd=1, m_odd=1.
        #    1 + 1 - 1 = 1. Matches.
        
        # Let's try n=3, m=3.
        # Odds: size 2 (idx 0, 2). Evens: size 1 (idx 1).
        # Merge Odds (2,2): Cost M(2,2) = 3 comparators?
        #   M(2,2) -> M(1,1)+M(1,1)+1 = 1+1+1=3.
        # Merge Evens (1,1): Cost M(1,1) = 1 comparator.
        # Step 3: Compare (o1+1, o2) and (o1+2, o2+1)?
        #   The comparators are (o1 + i, o2 + i) for i in 0..min(n,m)-1?
        #   If we use (o1+1, o2) and (o1+2, o2+1), that's 2 comparators.
        #   Total M(3,3) = 3 + 1 + 2 = 6?
        #   Known M(3,3) for Batcher is 6?
        #   Let's check.
        
        # Implementation of Step 3:
        # The comparators are (o1 + i, o2 + i) for i in range(min(n, m))?
        # Wait, if n=2, m=2, range(2) gives i=0,1.
        # i=0: (o1, o2). But o1 already has min(a0,b0) and o2 has max(a0,b0). They are already ordered relative to each other? No, o1 < o2 is guaranteed?
        # In step 1, we sorted the odds. So o1 (odd idx 0 of A) and o2 (odd idx 0 of B) are compared.
        # So o1 <= o2.
        # So comparing (o1, o2) is redundant?
        # Ah, the recursive merge handles the internal sorting.
        # The "Compare-Exchange" step in Batcher's merge is specifically:
        # for i in range(min(n, m)):
        #     compare-exchange (o1 + i, o2 + i)
        # BUT, we must skip the ones that are already handled?
        # No, the standard algorithm says:
        # "Compare-exchange (i, i + n) for i = 0 to min(n, m) - 1" is for a different variant?
        
        # Let's stick to the most robust definition:
        # Batcher's Odd-Even Merge of two sorted lists of length n and m:
        # 1. Recursively merge the odd-indexed elements.
        # 2. Recursively merge the even-indexed elements.
        # 3. Compare-exchange (o1 + i, o2 + i) for i in 0..min(n, m)-1.
        #    HOWEVER, if n != m, the offsets shift?
        #    Actually, the lists are stored in contiguous memory.
        #    List A: o1, o1+1, ..., o1+n-1
        #    List B: o2, o2+1, ..., o2+m-1
        #    Odds of A: o1, o1+2...
        #    Odds of B: o2, o2+2...
        #    After merging odds, the result is in o1..o1+n_odd-1 and o2..o2+m_odd-1?
        #    No, the recursive merge keeps the data in the same wires.
        #    So after step 1, the odd-indexed wires of A and B are sorted relative to each other?
        #    No, the recursive merge sorts the *subsequence* of odds from A and the *subsequence* of odds from B.
        #    The result is that the odd-indexed wires of A contain the smallest odds, and odd-indexed wires of B contain the largest odds?
        #    Yes.
        #    Similarly for evens.
        #    So now:
        #    A_odd (in A's odd wires) < B_odd (in B's odd wires)
        #    A_even (in A's even wires) < B_even (in B's even wires)
        #    We need to merge these two pairs of sorted sequences into one big sorted sequence.
        #    The final step compares A_odd[i] with B_odd[i]? No.
        #    The final step compares A_even[i] with B_odd[i]?
        #    Specifically, compare (o1 + 2*i + 1, o2 + 2*i) ?
        #    Let's check n=2, m=2.
        #    i=0: Compare (o1+1, o2).
        #    This matches our earlier deduction.
        #    So the comparators are (o1 + 2*i + 1, o2 + 2*i) for i in range(min(n_even, m_odd))?
        #    And also (o1 + 2*i, o2 + 2*i + 1)?
        #    For n=2, m=2:
        #    n_even=1, m_odd=1. i=0. (o1+1, o2).
        #    n_odd=1, m_even=1. i=0. (o1+0, o2+1)?
        #    If we add (o1, o2+1), is it needed?
        #    A_odd[0] <= B_odd[0] (from step 1)
        #    A_even[0] <= B_even[0] (from step 2)
        #    We know A_odd[0] <= A_even[0] (A was sorted)
        #    We know B_odd[0] <= B_even[0] (B was sorted)
        #    We compared A_even[0] with B_odd[0].
        #    Do we need to compare A_odd[0] with B_even[0]?
        #    A_odd[0] <= A_even[0].
        #    If A_even[0] <= B_odd[0], then A_odd[0] < B_odd[0] < B_even[0].
        #    If A_even[0] > B_odd[0], then B_odd[0] < A_even[0].
        #    We need to ensure the whole sequence is sorted.
        #    The sequence is:
        #    A_odd[0], A_even[0], B_odd[0], B_even[0] (logical order after merge?)
        #    No, the wires are:
        #    o1 (A_odd), o1+1 (A_even), o2 (B_odd), o2+1 (B_even).
        #    We want o1 <= o1+1 <= o2 <= o2+1.
        #    We have o1 <= o1+1 (A sorted).
        #    We have o2 <= o2+1 (B sorted).
        #    We compared o1+1 and o2.
        #    If o1+1 <= o2, then o1 <= o1+1 <= o2 <= o2+1. Sorted.
        #    If o1+1 > o2, then swap. New values: o2 at o1+1, o1+1 at o2.
        #    Now we have o1, o2, o1+1, o2+1.
        #    Is o1 <= o2?
        #    o1 is A_odd[0]. o2 is B_odd[0] (before swap? No, o2 holds max(A_odd, B_odd) from step 1? No.
        #    Step 1 merged A_odd and B_odd.
        #    Result: o1 holds min(A_odd, B_odd), o2 holds max(A_odd, B_odd).
        #    So o1 <= o2 is ALWAYS true after Step 1.
        #    So if we swap o1+1 and o2, we have:
        #    o1 (min AO, BO), o2 (max AO, BO), o1+1 (min AE, BE), o2+1 (max AE, BE).
        #    Wait, o1+1 and o2 were swapped.
        #    So o1+1 now holds min(AE, BE) if AE<BE? No.
        #    Step 2 merged AE and BE.
        #    o1+1 holds min(AE, BE), o2+1 holds max(AE, BE).
        #    Step 3 compares o1+1 and o2.
        #    If o1+1 > o2, swap.
        #    New o1+1 = o2 (which is max(AO, BO)).
        #    New o2 = o1+1 (which is min(AE, BE)).
        #    We need to check if o1 <= new o1+1?
        #    o1 = min(AO, BO). new o1+1 = max(AO, BO).
        #    min <= max. Yes.
        #    We need to check if new o2 <= o2+1?
        #    new o2 = min(AE, BE). o2+1 = max(AE, BE).
        #    min <= max. Yes.
        #    So 1 comparator is sufficient for 2,2.
        
        # So the rule is:
        # Compare (o1 + 2*i + 1, o2 + 2*i) for i in range(min(n_even, m_odd))?
        # And also (o1 + 2*i, o2 + 2*i + 1)?
        # For n=2, m=2, only the first type was needed.
        # Let's check n=3, m=3.
        # Odds: size 2. Evens: size 1.
        # Step 1: Merge Odds (2,2). 3 comps.
        # Step 2: Merge Evens (1,1). 1 comp.
        # Step 3:
        #   Compare (o1+1, o2) [i=0, type 1]
        #   Compare (o1+3, o2+1)? No, indices out of bounds.
        #   Compare (o1+0, o2+1) [i=0, type 2]?
        #   Let's trace.
        #   Wires:
        #   o1 (AO0), o1+1 (AE0), o1+2 (AO1)
        #   o2 (BO0), o2+1 (BE0), o2+2 (BO1)
        #   After Step 1 (Merge AO, BO):
        #   AO0, AO1 sorted relative to BO0, BO1.
        #   o1: min(AO0, BO0)
        #   o1+2: max(AO0, BO0)
        #   o2: min(AO1, BO1)
        #   o2+2: max(AO1, BO1)
        #   Wait, the recursive merge puts the result in the same wires.
        #   So o1, o1+2 contain the smallest 2 of {AO0, AO1, BO0, BO1}? No.
        #   The recursive merge sorts the *subsequence* of odds.
        #   So o1, o1+2 are the sorted odds of A? No.
        #   Batcher's merge interleaves? No, it keeps them in their respective halves.
        #   "Merge two sorted lists" implies the output is in the union of the two lists' locations.
        #   So after Step 1, the odds of A and B are merged.
        #   The smallest odds are in A's odd wires?
        #   Yes, typically the first list gets the smaller half.
        #   So o1, o1+2 hold the 2 smallest odds.
        #   o2, o2+2 hold the 2 largest odds.
        #   Similarly for evens:
        #   o1+1 holds the smallest even.
        #   o2+1 holds the largest even.
        #   
        #   Now we have:
        #   o1: Small Odd 1
        #   o1+1: Small Even 1
        #   o1+2: Small Odd 2
        #   o2: Large Odd 1
        #   o2+1: Large Even 1
        #   o2+2: Large Odd 2
        #   
        #   We need to sort the whole thing.
        #   We know o1 <= o1+2 (sorted odds in A)
        #   We know o1+1 is the smallest even.
        #   We know o2 <= o2+2 (sorted odds in B)
        #   
        #   Comparators needed:
        #   1. (o1+1, o2): Compare Small Even 1 with Large Odd 1.
        #   2. (o1+2, o2+1): Compare Small Odd 2 with Large Even 1.
        #   
        #   So for n=3, m=3, we need 2 comparators in Step 3.
        #   Total M(3,3) = 3 (odds) + 1 (evens) + 2 = 6.
        
        # Generalizing Step 3:
        # Compare (o1 + 2*i + 1, o2 + 2*i) for i in range(min(n_even, m_odd))
        # Compare (o1 + 2*i, o2 + 2*i + 1) for i in range(min(n_odd, m_even))
        
        n_odd = (n + 1) // 2
        n_even = n // 2
        m_odd = (m + 1) // 2
        m_even = m // 2
        
        # 1. Merge Odds
        batcher_merge(n_odd, m_odd, offset1, offset2)
        # 2. Merge Evens
        batcher_merge(n_even, m_even, offset1 + 1, offset2 + 1)
        
        # 3. Compare-Exchange
        # Type 1: A_even[i] vs B_odd[i]
        for i in range(min(n_even, m_odd)):
            comparators.append((offset1 + 2*i + 1, offset2 + 2*i))
            
        # Type 2: A_odd[i] vs B_even[i]
        for i in range(min(n_odd, m_even)):
            comparators.append((offset1 + 2*i, offset2 + 2*i + 1))
            
    batcher_sort(N, 0)
    return comparators
# EVOLVE-BLOCK-END
