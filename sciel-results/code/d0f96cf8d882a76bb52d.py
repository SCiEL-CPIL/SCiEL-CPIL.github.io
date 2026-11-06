import numpy as np
import time
import random

N = 17
MAX_COMPARATORS = 4096

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin

    # Pre-compute all 2^N boolean inputs
    # inputs shape: (2^N, N), dtype uint8
    inputs = np.zeros((2**N, N), dtype=np.uint8)
    for i in range(N):
        inputs[:, i] = (np.arange(2**N) >> (N - 1 - i)) & 1

    def simulate(network):
        """Simulate the network on all inputs, return final state."""
        state = inputs.copy()
        for i, j in network:
            si = state[:, i]
            sj = state[:, j]
            # min to i, max to j
            state[:, i] = np.minimum(si, sj)
            state[:, j] = np.maximum(si, sj)
        return state

    def is_sorted(state):
        """Check if all rows are sorted (non-decreasing)."""
        return np.all(state[:, 1:] >= state[:, :-1])

    def is_valid(network):
        """Check if the network sorts all 2^N inputs."""
        state = simulate(network)
        return is_sorted(state)

    # Generate initial network: Bitonic(16) on wires 0-15 + Insertion of wire 16
    def generate_bitonic(n):
        """Generate Bitonic sort network for n wires (n must be power of 2)."""
        if n == 1:
            return []
        # Recursive step: Sort first half, Sort second half (reversed), Merge
        # Standard Bitonic sort:
        # 1. Recursively sort first n/2
        # 2. Recursively sort second n/2 (in reverse order to make bitonic)
        # 3. Merge the two halves
        
        # Actually, standard Bitonic sort construction:
        # bitonic(n):
        #   if n == 1: return
        #   bitonic(n/2) on [0, n/2)
        #   bitonic(n/2) on [n/2, n) with reverse direction?
        #   merge(n/2, n/2)
        
        # Simpler: Use the known structure.
        # For n=16, we can hardcode or generate.
        # Let's generate recursively.
        
        def bitonic_sort(wires, direction):
            # wires: list of wire indices
            # direction: 1 for ascending, -1 for descending
            n = len(wires)
            if n == 1:
                return []
            
            mid = n // 2
            left_wires = wires[:mid]
            right_wires = wires[mid:]
            
            # Sort left half ascending
            net_left = bitonic_sort(left_wires, 1)
            # Sort right half descending (to form bitonic sequence)
            net_right = bitonic_sort(right_wires, -1)
            
            # Merge the two halves
            # Merge requires comparing (i, j) where i in left, j in right
            # Specifically, for k in 0..mid-1: compare (wires[k], wires[n-1-k])
            # Wait, the merge step of Bitonic sort:
            # Given two sorted sequences of length n/2, one asc, one desc.
            # We perform log(n) stages of comparisons.
            
            net_merge = []
            # Stage 1: compare (0, n-1), (1, n-2), ...
            # Stage 2: compare (0, n-2), (1, n-3), ... (excluding middle)
            # ...
            
            # Standard merge for Bitonic:
            # for k in range(1, mid+1):
            #   for i in range(0, mid - k):
            #     compare (wires[i], wires[n-1-i]) ? No.
            
            # Let's use the explicit merge structure.
            # Merge two sorted lists of size m (asc and desc).
            # The result is sorted asc.
            # Comparators:
            # Level 1: (0, 2m-1), (1, 2m-2), ... (m-1, m)  -> m comparators
            # Level 2: (0, 2m-2), (1, 2m-3), ... (m-2, m+1) -> m-1 comparators? No.
            
            # Actually, the merge of two bitonic sequences of size n/2:
            # It consists of log(n) stages.
            # Stage 1: Compare (i, n-1-i) for i=0..n/2-1.
            # Stage 2: Compare (i, n-2-i) for i=0..n/4-1?
            
            # Let's just use a known generator or hardcode for 16.
            # Hardcoding 49 comparators is tedious.
            # Let's implement the recursive Bitonic sort correctly.
            
            # Correction: The recursive step for Bitonic Sort is:
            # 1. Sort first half ascending.
            # 2. Sort second half descending.
            # 3. Merge the two halves.
            
            # The merge of two sorted halves (size n/2) into one sorted half (size n):
            # This is done by comparing (i, n-1-i) for i=0 to n/2-1?
            # No, that's one stage.
            # The merge requires log(n) stages.
            
            # Let's implement merge_bitonic(wires, n):
            # Returns list of comparators to merge two sorted halves of wires.
            
            def merge_bitonic(wires, n):
                # wires is the full set of n wires
                # First half wires[0:n/2] is sorted asc
                # Second half wires[n/2:n] is sorted desc
                # We want to sort wires[0:n] asc.
                
                comparators = []
                # Stage 1: compare (0, n-1), (1, n-2), ..., (n/2-1, n/2)
                for i in range(n // 2):
                    comparators.append((wires[i], wires[n - 1 - i]))
                
                # Recursively merge the two new halves?
                # After stage 1, the sequence is bitonic.
                # The first half (0 to n/2-1) is sorted asc?
                # The second half (n/2 to n-1) is sorted desc?
                # Yes, property of Bitonic merge.
                # So we recursively merge the first half and the second half.
                
                if n > 2:
                    left_merge = merge_bitonic(wires[:n//2], n//2)
                    right_merge = merge_bitonic(wires[n//2:], n//2)
                    # Note: The right half is sorted desc, but merge_bitonic expects
                    # left asc, right desc.
                    # Does the right half remain "left asc, right desc" relative to its own indices?
                    # Yes, the recursive structure holds.
                    
                    comparators.extend(left_merge)
                    comparators.extend(right_merge)
                
                return comparators

            # Now, the full Bitonic sort for n wires:
            # 1. Recursively sort first n/2 asc.
            # 2. Recursively sort second n/2 desc.
            # 3. Merge.
            
            net = []
            net.extend(bitonic_sort(left_wires, 1))
            
            # For descending sort, we can just swap the output of ascending sort?
            # Or call bitonic_sort with a flag.
            # Let's modify bitonic_sort to accept direction.
            
            # Re-defining bitonic_sort with direction
            def bitonic_sort_dir(wires, direction):
                n = len(wires)
                if n == 1:
                    return []
                
                mid = n // 2
                left_wires = wires[:mid]
                right_wires = wires[mid:]
                
                # Always sort first half asc, second half desc for the merge step
                # But if the overall direction is desc, we swap the roles?
                # Standard Bitonic Sort:
                # To sort N elements:
                # 1. Sort first N/2 ascending.
                # 2. Sort second N/2 descending.
                # 3. Merge.
                
                # This produces an ascending sort.
                # To produce a descending sort, we can sort ascending and then swap all pairs?
                # Or, sort first N/2 descending, second N/2 ascending, and merge?
                
                # Let's just generate Ascending Bitonic Sort.
                # For the "descending" part in the recursion, we need a network that sorts descending.
                # A descending sort network is the same as ascending, but with inputs/outputs swapped?
                # No, we can just take the ascending network and reverse the order of comparators?
                # No.
                
                # Simplest: Generate Ascending Bitonic for all.
                # For the second half, we want it to be descending.
                # We can generate an Ascending network for the second half, and then
                # add a "reverse" step? No.
                
                # Actually, if we generate an Ascending network for the second half,
                # it will be Ascending. We need Descending.
                # We can just swap the indices in the comparators of the second half?
                # No.
                
                # Let's use the property:
                # Bitonic Sort (Asc) on N:
                #   Bitonic Sort (Asc) on N/2 (left)
                #   Bitonic Sort (Desc) on N/2 (right)
                #   Merge (N)
                
                # Bitonic Sort (Desc) on N:
                #   Bitonic Sort (Desc) on N/2 (left)
                #   Bitonic Sort (Asc) on N/2 (right)
                #   Merge (N) [Same merge structure? No, merge structure depends on order]
                
                # The merge structure for Ascending result:
                # Left Asc, Right Desc -> Asc.
                
                # So we need a function that generates a network to sort a given set of wires
                # in a specific direction.
                
                def gen_sort(wires, direction):
                    n = len(wires)
                    if n == 1:
                        return []
                    
                    mid = n // 2
                    left = wires[:mid]
                    right = wires[mid:]
                    
                    if direction == 1: # Ascending
                        # Left Asc, Right Desc
                        net_left = gen_sort(left, 1)
                        net_right = gen_sort(right, -1)
                    else: # Descending
                        # Left Desc, Right Asc
                        net_left = gen_sort(left, -1)
                        net_right = gen_sort(right, 1)
                    
                    # Merge
                    # The merge comparators are the same regardless of final direction?
                    # No, the merge step assumes Left Asc, Right Desc to produce Asc.
                    # If we want Desc, we assume Left Desc, Right Asc to produce Desc.
                    # The comparators are (i, n-1-i) etc.
                    
                    net_merge = []
                    # Stage 1
                    for i in range(n // 2):
                        net_merge.append((wires[i], wires[n - 1 - i]))
                    
                    if n > 2:
                        # Recursively merge
                        # The left half of the merged array is now sorted in the target direction?
                        # If target is Asc: Left half is Asc, Right half is Desc.
                        # So we recursively merge Left (Asc) and Right (Desc).
                        # This is the same structure.
                        
                        # For Asc target:
                        # Left half (wires[0:mid]) needs to be sorted Asc.
                        # Right half (wires[mid:n]) needs to be sorted Desc.
                        # Wait, after the first stage of merge, the left half is Asc and right half is Desc.
                        # So we can just call gen_sort on the halves with the target direction?
                        # No, the merge function `merge_bitonic` I wrote earlier assumed
                        # Left Asc, Right Desc -> Asc.
                        
                        # Let's just use the `merge_bitonic` logic which is:
                        # Compare (i, n-1-i)
                        # Then recursively merge left half and right half.
                        # This works for producing an Ascending sort from Bitonic input.
                        
                        # If we are producing a Descending sort, the input is Bitonic (Left Desc, Right Asc).
                        # The comparators (i, n-1-i) will produce Left Desc, Right Asc?
                        # Yes.
                        
                        # So the merge structure is identical.
                        
                        net_merge.extend(gen_sort(left, direction)) # This is wrong. The merge is part of the sort.
                        
                        # Actually, the recursion in `gen_sort` handles the sorting of the halves.
                        # The `net_merge` handles the final merge.
                        # But `gen_sort` calls `gen_sort` on halves, which includes their own merges.
                        # So we just need to add the top-level merge comparators.
                        
                        # The recursive call `gen_sort(left, direction)` will sort the left half.
                        # But we already sorted the left half in `net_left`.
                        # This is double counting.
                        
                        # Correct structure:
                        # gen_sort(wires, dir):
                        #   if n==1 return []
                        #   net_left = gen_sort(left, dir1)
                        #   net_right = gen_sort(right, dir2)
                        #   net_merge = merge_comparators(wires, n)
                        #   return net_left + net_right + net_merge
                        
                        # This is correct.
                        
                        pass # Handled below

                    return net_left + net_right + net_merge

                # Re-implementing cleanly
                def gen_sort(wires, direction):
                    n = len(wires)
                    if n == 1:
                        return []
                    
                    mid = n // 2
                    left = wires[:mid]
                    right = wires[mid:]
                    
                    if direction == 1:
                        net_left = gen_sort(left, 1)
                        net_right = gen_sort(right, -1)
                    else:
                        net_left = gen_sort(left, -1)
                        net_right = gen_sort(right, 1)
                    
                    # Merge comparators
                    net_merge = []
                    for i in range(n // 2):
                        net_merge.append((wires[i], wires[n - 1 - i]))
                    
                    # Recursive merge?
                    # No, the standard Bitonic sort definition:
                    # Sort(N) = Sort(N/2) + Sort(N/2) + Merge(N)
                    # Where Merge(N) is a specific network.
                    # My `net_merge` above is only the first stage of the merge?
                    # No, the merge of two sorted sequences of size N/2 requires log(N) stages.
                    # The structure `for i in range(n//2): compare(i, n-1-i)` is just one stage.
                    
                    # The full merge network for size N is:
                    # Stage 1: compare (0, N-1), (1, N-2), ...
                    # Stage 2: compare (0, N-2), (1, N-3), ... (excluding middle)
                    # ...
                    
                    # This is equivalent to:
                    # Merge(N) = Compare(N) + Merge(N/2) on left + Merge(N/2) on right
                    # Where Compare(N) is the first stage.
                    
                    # So:
                    if n > 2:
                        # The left half of the merged result needs to be sorted.
                        # The right half needs to be sorted.
                        # The recursive calls in gen_sort already did that?
                        # No, gen_sort(left) sorts the left half *before* the merge.
                        # After the first stage of merge, the left half is bitonic?
                        # No, after the first stage of merge, the left half is sorted and the right half is sorted?
                        # Yes, that's the property of Bitonic merge.
                        # So we don't need to recursively merge the halves again in the same call?
                        # Wait.
                        
                        # Standard Bitonic Sort:
                        # 1. Sort first N/2.
                        # 2. Sort second N/2.
                        # 3. Merge the two.
                        
                        # The Merge step itself is a recursive process.
                        # Merge(N) = Stage1(N) + Merge(N/2) on left + Merge(N/2) on right.
                        
                        # So `gen_sort` should be:
                        # 1. Recursively sort halves.
                        # 2. Execute Merge(N).
                        
                        # And Merge(N) is:
                        # Stage 1 comparators.
                        # Then Merge(N/2) on left half.
                        # Then Merge(N/2) on right half.
                        
                        # So I need a separate `merge` function.
                        
                        pass

                    return net_left + net_right + net_merge

                # Let's just use a known simple Bitonic generator.
                # https://en.wikipedia.org/wiki/Bitonic_sorter
                # The diagram shows the structure.
                
                # For N=16, we can just hardcode the 49 comparators.
                # This is safer and faster.
                # I will generate them programmatically once.
                
                return [] # Placeholder

            # Let's implement the correct Bitonic generator.
            
            def bitonic_sort_network(n):
                # Returns list of (i, j) for n wires (0 to n-1)
                # n must be power of 2
                if n == 1:
                    return []
                
                def merge(wires, n):
                    # wires: list of indices
                    # n: length
                    # Returns comparators to merge two sorted halves (Left Asc, Right Desc) into Asc.
                    if n == 1:
                        return []
                    
                    comparators = []
                    # Stage 1
                    for i in range(n // 2):
                        comparators.append((wires[i], wires[n - 1 - i]))
                    
                    # Recursive merge
                    if n > 2:
                        left_merge = merge(wires[:n//2], n//2)
                        right_merge = merge(wires[n//2:], n//2)
                        comparators.extend(left_merge)
                        comparators.extend(right_merge)
                    
                    return comparators

                def sort_asc(wires, n):
                    if n == 1:
                        return []
                    
                    mid = n // 2
                    left = wires[:mid]
                    right = wires[mid:]
                    
                    net_left = sort_asc(left, mid)
                    net_right = sort_desc(right, mid)
                    
                    net_merge = merge(wires, n)
                    
                    return net_left + net_right + net_merge

                def sort_desc(wires, n):
                    if n == 1:
                        return []
                    
                    mid = n // 2
                    left = wires[:mid]
                    right = wires[mid:]
                    
                    net_left = sort_desc(left, mid)
                    net_right = sort_asc(right, mid)
                    
                    # Merge for Descending?
                    # If Left is Desc and Right is Asc, the merge comparators are the same?
                    # Yes, (i, n-1-i).
                    # And the recursive merges are the same.
                    net_merge = merge(wires, n)
                    
                    return net_left + net_right + net_merge

                wires = list(range(n))
                return sort_asc(wires, n)

            return bitonic_sort_network(n)

    # Generate Bitonic(16)
    net_16 = generate_bitonic(16)
    
    # Add wire 16
    # Insertion: Compare 16 with 15, 14, ..., 0
    # This assumes 0-15 are sorted.
    # (15, 16), (14, 16), ..., (0, 16)
    # Note: The order of insertion matters.
    # If we compare (15, 16), the max goes to 16.
    # Then (14, 16), max goes to 16.
    # This bubbles the max of {0..16} to 16.
    # The min goes to the lowest index.
    # This is a valid insertion sort step.
    
    net_17 = net_16[:]
    for i in range(15, -1, -1):
        net_17.append((i, 16))
        
    # Verify
    if not is_valid(net_17):
        # Fallback to OETN if Bitonic+Insertion fails (shouldn't)
        def generate_oetn():
            net = []
            for phase in range(N):
                for i in range(phase % 2, N - 1, 2):
                    net.append((i, i + 1))
            return net
        net_17 = generate_oetn()

    best_network = net_17[:]
    best_len = len(net_17)

    # Optimization Loop
    # Strategy: Greedy Removal
    # We try to remove each comparator. If the network remains valid, we remove it.
    # We repeat this until no more comparators can be removed.
    
    # To speed up, we can check multiple removals in parallel?
    # No, removals are sequential because the network changes.
    # But we can check many candidates quickly.
    
    current_network = best_network[:]
    
    # Phase 1: Greedy Removal
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for idx in range(len(current_network)):
            if time.time() >= deadline - 5.0:
                break
            
            candidate = current_network[:idx] + current_network[idx+1:]
            if is_valid(candidate):
                current_network = candidate
                improved = True
                # Update best if smaller
                if len(current_network) < best_len:
                    best_network = current_network[:]
                    best_len = len(current_network)
                break # Restart loop as indices changed

    # Phase 2: Try swapping adjacent comparators to unlock more removals
    # This is a local search.
    # We try swapping (i, i+1). If valid, we accept.
    # We do this for a fixed number of iterations or until time limit.
    
    # To make this effective, we should only accept swaps that allow subsequent removals?
    # Or just accept any valid swap?
    # Accepting any valid swap might lead to a local minimum with same size.
    # We want to find a configuration that allows removal.
    
    # Let's do a simple random restart local search.
    # But given time constraints, Greedy Removal is the most reliable.
    
    # Let's try "Pair Removal"
    # Try removing two comparators at once.
    
    if time.time() < deadline - 2.0:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(len(current_network)):
                if time.time() >= deadline - 2.0:
                    break
                for j in range(i + 1, len(current_network)):
                    if time.time() >= deadline - 2.0:
                        break
                    candidate = [net[k] for k in range(len(current_network)) if k != i and k != j]
                    if is_valid(candidate):
                        current_network = candidate
                        improved = True
                        if len(current_network) < best_len:
                            best_network = current_network[:]
                            best_len = len(current_network)
                        break
                if improved:
                    break

    # Final verification
    if not is_valid(best_network):
        return generate_oetn()

    return best_network

# Helper for fallback
def generate_oetn():
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net
