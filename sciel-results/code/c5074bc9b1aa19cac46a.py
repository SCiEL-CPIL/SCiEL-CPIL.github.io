def batcher_odd_merge(n):
    """
    Generates the list of comparators for Batcher's Odd-Merge Sort network for 'n' wires.
    n must be a power of 2.
    """
    comparators = []
    
    def sort_network(wires, start_idx):
        """
        Recursively sorts the wires given by the list 'wires'.
        'wires' is a list of global wire indices.
        """
        m = len(wires)
        if m <= 1:
            return
        
        # Split into odd and even indices within the local list
        # Odd indices: 1, 3, 5...
        # Even indices: 0, 2, 4...
        odd_wires = [wires[i] for i in range(1, m, 2)]
        even_wires = [wires[i] for i in range(0, m, 2)]
        
        # Recursively sort the odd part
        sort_network(odd_wires, start_idx)
        # Recursively sort the even part
        sort_network(even_wires, start_idx)
        
        # Merge the two sorted lists
        # The two lists are now sorted.
        # We need to merge them.
        # The merge step for two sorted lists of size m/2:
        # For i in 0..m/2-1:
        #   compare even_wires[i] with odd_wires[i]
        #   This produces a min and max.
        #   Let min_i = min(even_wires[i], odd_wires[i])
        #   Let max_i = max(even_wires[i], odd_wires[i])
        #   The mins are sorted? No.
        #   The standard Odd-Merge merge step is:
        #   1. Compare corresponding elements: (even[i], odd[i]) for i=0..m/2-1.
        #   2. This creates two sequences: mins and maxs.
        #   3. The mins are not necessarily sorted, nor are the maxs.
        #   4. However, min_i <= max_i.
        #   5. Also, min_i <= min_{i+1}? No.
        #   6. We need to sort the mins and maxs?
        #   Actually, the property of Odd-Merge is that after step 1, 
        #   the sequence of mins is sorted and the sequence of maxs is sorted?
        #   Let's check N=4.
        #   Inputs: A=[a0, a1], B=[b0, b1] (sorted).
        #   Compare (a0, b0) -> min0, max0.
        #   Compare (a1, b1) -> min1, max1.
        #   We know a0 <= a1 and b0 <= b1.
        #   Is min0 <= min1?
        #   min0 = min(a0, b0). min1 = min(a1, b1).
        #   Since a0 <= a1 and b0 <= b1, yes min0 <= min1.
        #   Is max0 <= max1?
        #   max0 = max(a0, b0). max1 = max(a1, b1).
        #   Yes, max0 <= max1.
        #   So mins are sorted, maxs are sorted.
        #   But we need the combined sequence to be sorted.
        #   The combined sequence is [min0, min1, max0, max1]? No.
        #   The wires are physically located.
        #   In the network, the output of comparator (i, j) goes to wire i (min) and wire j (max).
        #   So after comparing (even[i], odd[i]):
        #      Wire even[i] holds min(even[i], odd[i]).
        #      Wire odd[i] holds max(even[i], odd[i]).
        #   Let's call the set of even indices E and odd indices O.
        #   After the cross-comparisons, the wires in E contain the mins, and wires in O contain the maxs.
        #   Since mins are sorted and maxs are sorted, we just need to merge the two sorted lists:
        #   List M (in wires E) and List X (in wires O).
        #   Wait, the positions in the final sorted array are 0, 1, ..., m-1.
        #   The mins should occupy the first half? Not necessarily.
        #   Actually, in Odd-Merge, the merge step is:
        #   1. Compare (E[i], O[i]) for i=0..m/2-1.
        #   2. Then, we need to merge the two resulting sorted sequences.
        #   But wait, if mins are sorted and maxs are sorted, and every min <= every max?
        #   No, min_i <= max_i, but min_{i+1} can be > max_i?
        #   Example: A=[1, 100], B=[50, 60].
        #   min0 = 1, max0 = 50.
        #   min1 = 60, max1 = 100.
        #   Mins: [1, 60]. Maxs: [50, 100].
        #   Combined sorted: [1, 50, 60, 100].
        #   Current wires: E holds [1, 60], O holds [50, 100].
        #   We need to merge these two sorted lists.
        #   How do we merge two sorted lists in a sorting network?
        #   We use a standard merge network (like Bitonic merge or another Odd-Merge merge).
        #   However, the structure of Odd-Merge is that the merge of two sorted lists of size m/2 is done by:
        #   1. Comparing (E[i], O[i]).
        #   2. Then applying a merge network to the resulting "interleaved" wires?
        #   Actually, the recursive definition of Odd-Merge Sort is:
        #   Sort(N):
        #     Sort odd indices (size N/2)
        #     Sort even indices (size N/2)
        #     Merge(N/2, N/2)
        #   
        #   Merge(M, N) where M<=N:
        #     If M=1:
        #        Compare (even[0], odd[0]) ... wait, indices are tricky.
        #     The standard merge step for two sorted lists A and B of size m:
        #     For i=0 to m-1: compare A[i] with B[i].
        #     Then, we have two sorted lists: Mins and Maxs.
        #     We need to merge Mins and Maxs.
        #     But Mins and Maxs are located in specific wires.
        #     The wires holding Mins are the "lower" outputs of the comparators.
        #     The wires holding Maxs are the "upper" outputs.
        #     In the Odd-Merge construction, the next step is to recursively merge the two halves?
        #     No, the merge itself is not recursive in the same way.
        #     Let's look at the comparator count formula: $C(N) = 2 C(N/2) + N/2 \log_2(N/2) + N/2$?
        #     For N=16, C(16) = 60.
        #     C(8) = 19. C(4) = 5. C(2) = 1.
        #     C(16) = 2*C(8) + Merge(8,8).
        #     60 = 2*19 + Merge(8,8) => Merge(8,8) = 22.
        #     Merge(m, m) comparators:
        #     Step 1: m comparators (A[i] vs B[i]).
        #     Step 2: Merge the two resulting sorted lists of size m/2?
        #     No, the resulting lists are size m.
        #     Wait, after comparing A[i] and B[i], we have m mins and m maxs.
        #     Total 2m elements.
        #     The mins are sorted, the maxs are sorted.
        #     We need to merge these two sorted lists of size m.
        #     This requires a merge network for two lists of size m.
        #     But we are already inside a merge!
        #     This suggests the merge step is:
        #     1. Compare corresponding elements.
        #     2. Recursively merge the two resulting sorted sequences.
        #     But the sequences are interleaved in the wires.
        #     Actually, the "Mins" are in the wires that were the first inputs to the comparators, and "Maxs" in the second.
        #     If we define the merge of two sorted lists A and B (size m) as:
        #       1. For i=0..m-1: Compare A[i], B[i].
        #       2. Let A' be the mins, B' be the maxs.
        #       3. Merge A' and B' (which are sorted) using the same merge procedure?
        #       No, that would be infinite recursion.
        #       
        #     Correct Odd-Merge Merge:
        #     To merge two sorted lists A and B of size m:
        #     1. For i=0..m-1: Compare A[i] with B[i].
        #     2. This produces two sorted lists: M (mins) and X (maxs).
        #     3. The wires are now arranged such that M and X are in separate "halves" of the wire indices?
        #     No, they are in the original wire positions.
        #     4. We then apply a standard merge to the wires containing M and X?
        #     Actually, the key insight of Odd-Merge is that after step 1, the sequence of mins is sorted and the sequence of maxs is sorted.
        #     Furthermore, the mins are in the wires $0, 2, 4...$ and maxs in $1, 3, 5...$ (if we started with A in evens and B in odds).
        #     So we have two sorted lists in even and odd wires.
        #     To merge them into a single sorted list, we just need to sort the entire set of 2m wires?
        #     No, we can use the fact that they are interleaved sorted lists.
        #     This is equivalent to merging two sorted lists.
        #     The cost of merging two sorted lists of size m is $M(m)$.
        #     $M(m) = m + M(m/2) + M(m/2)$?
        #     Let's check counts.
        #     C(2) = 1.
        #     C(4) = 2*C(2) + M(2).
        #     M(2) should be 1?
        #     If M(2)=1, C(4) = 2(1) + 1 = 3? But C(4) is 5.
        #     So M(2) must be 3?
        #     Let's trace N=4.
        #     Sort(4):
        #       Sort(2) on odds (1,3): 1 comp.
        #       Sort(2) on evens (0,2): 1 comp.
        #       Merge(2,2):
        #         Compare (0,1), (2,3). (2 comps).
        #         Now we have mins in 0,2 and maxs in 1,3.
        #         Mins are sorted, Maxs are sorted.
        #         We need to merge wires [0,2] (sorted) and [1,3] (sorted).
        #         This is a merge of two sorted lists of size 2.
        #         How many comps to merge two sorted lists of size 2?
        #         List A: a0<=a1. List B: b0<=b1.
        #         We want to sort a0,a1,b0,b1.
        #         Standard merge for size 2:
        #         Compare (a1, b0).
        #         Then compare (a0, b0) and (a1, b1)?
        #         Actually, for N=4, the total is 5.
        #         We used 1+1+2 = 4 so far. We need 1 more.
        #         The remaining comparator is (1,2)?
        #         Let's check.
        #         After (0,1) and (2,3):
        #         Wire 0: min(a0,b0). Wire 1: max(a0,b0).
        #         Wire 2: min(a1,b1). Wire 3: max(a1,b1).
        #         We know min(a0,b0) <= min(a1,b1) and max(a0,b0) <= max(a1,b1).
        #         We need to sort the 4 wires.
        #         The only "crossing" pair is Wire 1 and Wire 2.
        #         Compare (1,2).
        #         So Merge(2,2) consists of:
        #         1. Compare (0,1), (2,3).
        #         2. Compare (1,2).
        #         Total 3 comparators.
        #         So C(4) = 1 + 1 + 3 = 5. Correct.
        #
        #     So the Merge(m,m) step is:
        #     1. For i=0..m-1: Compare (E[i], O[i]). (m comparators)
        #     2. Recursively merge the two resulting sorted lists?
        #        No, the resulting lists are in E and O wires.
        #        The next step is to sort the wires E and O?
        #        No, we just need to merge the two sorted sequences.
        #        The sequence in E is sorted. The sequence in O is sorted.
        #        We need to merge them.
        #        This is equivalent to applying a merge network to the 2m wires, but exploiting the sortedness.
        #        Actually, the structure is:
        #        Merge(m,m) = m comparators (cross) + Merge(m/2, m/2) on the "mins" + Merge(m/2, m/2) on the "maxs"?
        #        No, that would be $m + 2 M(m/2)$.
        #        Let's check $M(2)$. $M(2) = 2 + 2 M(1)$. $M(1)=0$. So $M(2)=2$?
        #        But we found $M(2)=3$ (including the cross comps? No, the cross comps are part of the merge).
        #        Let's redefine $M(m)$ as the number of comparators to merge two sorted lists of size m.
        #        $M(1) = 1$ (Compare the two elements).
        #        $M(2) = 3$ (Compare (0,1), (2,3), then (1,2)).
        #        $M(4) = ?$
        #        $C(8) = 2 C(4) + M(4) = 2(5) + M(4) = 10 + M(4)$.
        #        We know $C(8) = 19$. So $M(4) = 9$.
        #        Formula: $M(m) = m + 2 M(m/2)$?
        #        $M(4) = 4 + 2 M(2) = 4 + 2(3) = 10$. Close, but 9.
        #        Maybe $M(m) = m + 2 M(m/2) - 1$?
        #        $M(2) = 2 + 2(1) - 1 = 3$. Correct.
        #        $M(4) = 4 + 2(3) - 1 = 9$. Correct.
        #        $M(8) = 8 + 2(9) - 1 = 25$.
        #        $C(16) = 2 C(8) + M(8) = 2(19) + 25 = 38 + 25 = 63$.
        #        But reference is 60.
        #        So my recursive structure is slightly off or there is optimization.
        #
        #        Let's just hardcode the 60-comparator network for N=16 if I can find it, or use a known library.
        #        Since I can't use external libs, I will implement the recursive generator carefully.
        #        The discrepancy might be in the base case or the merge structure.
        #        
        #        Alternative: Use Bitonic Sort (80 comps). It's simpler and guaranteed correct.
        #        Score 120/80 = 1.5.
        #        If I can get 60, score is 2.0.
        #        Let's try to find the 60-comp network online in my training data.
        #        The "Batcher's Odd-Merge" for 16 is indeed 60.
        #        Why did my count give 63?
        #        Maybe $M(1)=0$? No, merging two single elements takes 1 comp.
        #        Maybe the cross-comparisons are optimized?
        #        
        #        Let's write the code for Bitonic Sort first as a fallback, but try to implement Odd-Merge correctly.
        #        Actually, I can just list the 60 comparators if I know them.
        #        I don't have them memorized.
        #        
        #        Let's stick to Bitonic Sort (80 comps) to ensure correctness and a good score (1.5).
        #        Or, I can try to optimize Bitonic? No, Bitonic is fixed.
        #        
        #        Wait, I can implement the recursive Odd-Merge and count. If it gives 60, great. If 63, still better than 120.
        #        
        #        Let's implement a generic recursive Odd-Merge.

def construct():
    N = 16
    comparators = []
    
    def sort_network(wires):
        m = len(wires)
        if m <= 1:
            return
        
        # Split into even and odd indices in the local list
        evens = [wires[i] for i in range(0, m, 2)]
        odds = [wires[i] for i in range(1, m, 2)]
        
        # Recursively sort
        sort_network(evens)
        sort_network(odds)
        
        # Merge
        # 1. Compare corresponding elements
        for i in range(m // 2):
            comparators.append((evens[i], odds[i]))
            
        # 2. Merge the two resulting sorted lists
        # The mins are in evens, maxs in odds.
        # We need to merge the sequence of evens and the sequence of odds.
        # This is equivalent to sorting the combined list of 2m wires, 
        # but knowing that evens are sorted and odds are sorted.
        # We can use a standard merge network on the 2m wires?
        # No, we can recursively apply the merge logic?
        # The structure of the merge of two sorted lists A and B of size m is:
        # Compare (A[i], B[i]) for i=0..m-1.
        # Then merge the mins and maxs.
        # The mins are in A, maxs in B.
        # So we need to merge the sorted list A and the sorted list B?
        # No, A now contains mins, B contains maxs.
        # We need to merge A and B.
        # This is the same problem as before!
        # So Merge(m) = m + Merge(m/2) + Merge(m/2)?
        # Let's assume the wires are rearranged?
        # In the network, the wires are fixed.
        # The "Mins" are in wires evens. The "Maxs" are in wires odds.
        # We need to sort the 2m wires.
        # We can just call a merge function on the list of wires [evens + odds]?
        # No, that would be O(m log m).
        # 
        # Actually, the standard Odd-Merge merge step is:
        # 1. Compare (A[i], B[i]).
        # 2. Apply Merge(m/2) to the first half and Merge(m/2) to the second half?
        # No.
        # 
        # Let's look at the N=4 case again.
        # Merge(2):
        # Compare (0,1), (2,3).
        # Then Compare (1,2).
        # This is not recursive Merge(1) + Merge(1).
        # Merge(1) would be Compare(?, ?).
        # 
        # I will use Bitonic Sort. It is robust and 80 is a good improvement.
        pass

# Bitonic Sort Implementation
def bitonic_sort_network(n):
    comps = []
    
    def bitonic_merge(wires, k, direction):
        # k is the size of the bitonic sequence
        if k <= 1:
            return
        # Compare pairs
        for i in range(k // 2):
            a = wires[i]
            b = wires[i + k // 2]
            if direction == 0: # ascending
                if a > b:
                    a, b = b, a
            # always add comparator
            comps.append((min(a,b), max(a,b)))
            
        # Recursively sort halves
        bitonic_merge(wires[:k//2], k//2, 0)
        bitonic_merge(wires[k//2:], k//2, direction)

    def bitonic_sort(wires, k, direction):
        if k <= 1:
            return
        if k > 2:
            bitonic_sort(wires[:k//2], k//2, 0)
            bitonic_sort(wires[k//2:], k//2, 1)
        bitonic_merge(wires, k, direction)

    wires = list(range(n))
    bitonic_sort(wires, n, 0)
    return comps

def construct():
    return bitonic_sort_network(16)

# Let's verify the count for Bitonic N=16.
# N=2: 1
# N=4: 2 (sort halves) + 2 (merge) + 2 (sort halves in merge)?
# Bitonic count formula: N/2 * log2(N) * (log2(N)+1) / 2?
# N=16: 16/2 * 4 * 5 / 2 = 8 * 10 = 80.
# Yes, 80 comparators.

# Can we do better?
# Batcher's Odd-Merge is 60.
# I will try to implement the recursive Odd-Merge correctly.
# The key is the merge step.
# Merge(m, m):
#   1. For i=0..m-1: Compare (A[i], B[i]).
#   2. The result is two sorted lists: Mins (in A) and Maxs (in B).
#   3. We need to merge Mins and Maxs.
#   4. Note that Mins[i] <= Maxs[i] is NOT necessarily true for all i?
#      Actually, Min_i <= Max_i is true.
#      Also Min_i <= Min_{i+1} and Max_i <= Max_{i+1}.
#      We need to merge two sorted lists.
#      We can use a standard merge network?
#      But the wires are interleaved.
#      The wires in A are even indices, wires in B are odd indices.
#      We can just apply a sorting network to the 2m wires?
#      No, that would be too expensive.
#      
#      Actually, there is a simpler way:
#      The merge of two sorted lists of size m can be done with m + Merge(m/2) + Merge(m/2) comparators IF we can rearrange wires.
#      But we can't rearrange.
#      
#      I'll stick with Bitonic (80). It's safe.

