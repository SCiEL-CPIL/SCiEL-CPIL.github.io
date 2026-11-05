    def _batcher_merge_sort(n):
        if n <= 1:
            return []
        n1 = (n + 1) // 2
        n2 = n // 2
        
        net = []
        # Sort left
        net.extend(_batcher_merge_sort(n1))
        # Sort right
        right_net = _batcher_merge_sort(n2)
        net.extend([(i + n1, j + n1) for i, j in right_net])
        
        # Merge
        # The merge network for Batcher's sort is:
        # 1. Merge the odd-indexed elements of the result.
        #    These come from even-indexed elements of Left and odd-indexed elements of Right.
        #    Left even indices: 0, 2, ..., n1-1 (if n1 odd)
        #    Right odd indices: n1+1, n1+3, ...
        #    This is hard to map directly without a helper.
        
        # Let's use a simpler known network: **Acker's** is too hard.
        # How about **Odd-Even Merge Sort**?
        
        # I will use a pre-computed small network if possible? No, must generate.
        
        # Let's fall back to **OET** but with a **better deletion strategy**.
        # The current deletion is O(N^2 * 2^N).
        # We can optimize `_is_valid` by caching? No.
        
        # What if we just run the current code but with a **tighter time limit** and **returning the best found**?
        # The current code returns the network if time runs out.
        
        # Maybe the score 1.0 is because it's too slow to remove many?
        # 136 removals * 0.1s = 13.6s per pass.
        # If it removes 50, that's 5 passes = 68s. It times out.
        # So it only does 1-2 passes.
        
        # To speed up:
        # 1. Vectorize the removal check?
        #    We can check if a comparator is redundant by checking if the network *without* it sorts all inputs.
        #    This is what we do.
        
        # 2. **Parallelize**? No.
        
        # 3. **Reduce verification cost**:
        #    Instead of checking all 131k inputs, can we check a subset?
        #    No, must be correct.
        
        # 4. **Better Base Network**:
        #    If I can generate a network with ~100 comparators, the deletion will be faster.
        #    Let's implement **Bitonic Sort for 16** and **Merge 1**.
        
        # Bitonic Sort 16: 80 comparators.
        # Merge 1 into 16:
        # We can add the 17th wire and run a "merge" network of size 1 and 16.
        # Batcher's merge of 1 and 16:
        # It's a binary search.
        # Comparators:
        # (7, 16)
        # (3, 16) or (11, 16) depending on result?
        # No, the network must be fixed.
        # The merge network for 1 and N is:
        # Compare (N/2, N)
        # Then recursively merge the winner with the appropriate half.
        # This is a tree of comparators.
        # Depth log2(N).
        # Number of comparators: log2(N) + log2(N/2) + ... = log2(N!) ? No.
        # It's just a binary search tree, which has N leaves? No.
        # To insert 1 element into N sorted elements, you need log2(N) comparators?
        # No, you need a network that sorts 1+N elements where N are sorted.
        # This is equivalent to a "merge" network.
        # Batcher's merge of 1 and N has size O(log N).
        # Specifically, it compares the single element with the middle, then the winner with the middle of the appropriate half, etc.
        # This creates a chain of comparators.
        # (7, 16)
        # (3, 16) AND (11, 16)? No, the network is fixed.
        # We need to compare (3, 16) and (11, 16) in parallel?
        # If (7,16) puts min at 7, max at 16.
        # Then we need to merge 7 into 0..6 and 16 into 8..15?
        # No, the sorted list is 0..15.
        # If 16 < 7, then 16 is in the left half.
        # If 16 > 7, then 16 is in the right half.
        # The network must handle both cases.
        # So we need comparators for both branches.
        # This doubles the work.
        # So it's not just log2(N).
        # It's a full merge network.
        # Size of merge(1, N) is roughly N/2?
        # Actually, merging 1 and N is equivalent to sorting N+1 elements where N are sorted.
        # This is the same as a "selection" network.
        # It has size O(N).
        # So 80 + 8 = 88?
        
        # Let's just use **Batcher's Odd-Merge Sort** for 17.
        # I will implement it now.
    