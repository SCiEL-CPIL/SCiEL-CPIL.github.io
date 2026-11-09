import numpy as np
import time

N = 16
_DEADLINE = 85.0  # Leave 5s safety margin for the 90s kill

def _elapsed():
    return time.time() - _START

_START = time.time()

def _generate_bitonic_network(n):
    """
    Generate a bitonic sorting network for n wires (n must be a power of 2).
    Returns a list of (i, j) comparators where i < j and min goes to i, max to j.
    This network sorts in ascending order.
    """
    if n <= 1:
        return []
    
    comps = []
    
    def bitonic_merge(start, length, direction):
        # direction: 1 for ascending, -1 for descending
        if length <= 1:
            return
        half = length // 2
        # Recursive merge of the two halves
        bitonic_merge(start, half, direction)
        bitonic_merge(start + half, half, direction)
        # Compare elements across the two halves
        for i in range(half):
            i1 = start + i
            i2 = start + half + i
            if direction == 1:
                # Ascending: min to i1, max to i2
                if i1 < i2:
                    comps.append((i1, i2))
                else:
                    comps.append((i2, i1))
            else:
                # Descending: max to i1, min to i2
                # Since comparator (a, b) puts min at a, max at b (with a < b),
                # to get max at i1 and min at i2 where i1 < i2, we cannot do it directly.
                # Wait, the task says "0 <= i < j < N" and "min to wire i and max to j".
                # So we can ONLY put min at the lower index.
                # This means we can only create ascending sequences?
                # No, we can create descending sequences if we swap the logic?
                # Actually, if we want descending order at the end, we can just reverse the input?
                # No, the network is fixed.
                # Let's re-read the task. "Construct a fixed compare-exchange network that sorts all 16-input sequences".
                # Usually "sorts" implies ascending.
                # The bitonic sort algorithm uses both ascending and descending merges.
                # If we can only do ascending merges, can we still build a bitonic network?
                # A bitonic sequence is ascending then descending.
                # To merge a bitonic sequence into a sorted sequence, we need to be able to sort both halves.
                # If we cannot sort descending, we cannot use standard bitonic sort.
                # HOWEVER, there is a trick: we can simulate a descending sort by swapping the inputs and outputs?
                # No, the wires are physical.
                # Let's check if we can just use Batcher's Odd-Merge which only uses ascending comparators.
                # Batcher's Odd-Merge is a standard sorting network that uses only ascending comparators.
                # It is more efficient than bitonic in terms of comparator count for some sizes?
                # Actually, for N=16, Batcher uses 63, Bitonic uses 56?
                # Let's verify Bitonic count.
                # B(1)=0, B(2)=1, B(4)=6, B(8)=19, B(16)=56.
                # If Bitonic is 56, it's better than Batcher's 63.
                # But can we implement Bitonic with only ascending comparators?
                # The merge step for a bitonic sequence (asc, desc) into asc:
                # The comparators are between (start+i, start+half+i).
                # If the second half is descending, the comparator (i, i+half) puts min at i.
                # This effectively merges the asc and desc parts.
                # The recursive calls need to sort the halves appropriately.
                # The first half (asc) is sorted ascending.
                # The second half (desc) is sorted descending.
                # To sort a half descending, we need a network that sorts descending.
                # If we can't sort descending, we're stuck.
                # BUT, we can sort ascending and then reverse the wires? No, wires are fixed.
                # Alternative: Use a network that sorts ascending, but apply it to the reversed indices?
                # No, the indices are fixed.
                # Let's look at Batcher's Odd-Merge. It is defined using only ascending comparators.
                # So I should implement Batcher's Odd-Merge.
                pass

    # Since Bitonic is tricky with the direction constraint, let's implement Batcher's Odd-Merge.
    # It is known to be efficient and uses only ascending comparators.
    def batcher(start, end):
        length = end - start
        if length <= 1:
            return
        mid = start + length // 2
        batcher(start, mid)
        batcher(mid, end)
        # Merge
        _batcher_merge(start, mid, length // 2)

    def _batcher_merge(s1, s2, k):
        # Merges two sorted sequences of length k starting at s1 and s2.
        # The sequences are interleaved in the output? No, they are in place.
        # The standard Batcher merge operates on the indices.
        # It's easier to implement the recursive merge on the list of indices.
        # Let's use a helper that takes a list of indices.
        pass

    # This is getting complicated. Let's use a known hardcoded network or a simpler generation.
    # Actually, I can use the `sortingnetworks` library if available? No, only stdlib/np/scipy.
    # Let's try to generate the network using a recursive function that returns the list of comparators.
    
    def gen(n):
        if n <= 1:
            return []
        # Bitonic sort generation
        # We will generate the comparators for a bitonic sort.
        # We assume we can handle direction by swapping the comparison logic in the merge?
        # No, the comparator is fixed.
        # Let's assume the task allows us to build a network that sorts ascending.
        # The standard bitonic network for N=16 has 56 comparators.
        # I will hardcode the 56 comparators for N=16 bitonic sort.
        # This is the safest way to ensure correctness and efficiency.
        
        # Source: Standard bitonic sort network for 16 inputs.
        # I will generate them programmatically to avoid hardcoding errors.
        # But I need to handle the direction.
        # Let's assume we can sort ascending.
        # The bitonic merge of two ascending sequences is not a bitonic merge.
        # Bitonic merge requires one asc and one desc.
        # So we MUST be able to sort descending.
        # If we can't, we can't do bitonic.
        # Can we sort descending with ascending comparators?
        # If we sort ascending, we get [min, ..., max].
        # To get [max, ..., min], we need to reverse the order.
        # We can't reverse wires.
        # So we cannot sort descending.
        # Therefore, we cannot use standard Bitonic Sort.
        # We MUST use Batcher's Odd-Merge, which works with ascending comparators only.
        
        # Let's implement Batcher's Odd-Merge correctly.
        # Reference: https://en.wikipedia.org/wiki/Sorting_network#Batcher's_odd%E2%80%93even_mergesort
        
        def odd_even_merge(indices1, indices2, k):
            # indices1 and indices2 are lists of wire indices for the two sorted halves.
            # They are interleaved in the output? No, the output is a single sorted list of 2k wires.
            # The wires are fixed. The "indices" are the wires.
            # The merge compares specific pairs of wires.
            # The recursive structure is:
            # 1. Merge the odd-indexed elements of the two halves.
            # 2. Merge the even-indexed elements of the two halves.
            # 3. Compare the elements at positions i and i+1 in the merged result?
            # This is hard to map to fixed wires without a permutation.
            # Batcher's network is a fixed network.
            # I will hardcode the 63 comparators for N=16 Batcher.
            # This is the most robust solution.
            
            # I will look up the comparators for N=16 Batcher.
            # Since I can't look them up, I will use the Transposition network (120) and prune.
            # Or I will use a simpler network like Shear Sort? No.
            
            # Let's try to generate the network using a recursive function that builds the list of comparators.
            # I found a Python implementation of Batcher's network online in my training data.
            # It uses a recursive function that takes the start and end indices.
            
            return []

    # I will use the Odd-Even Transposition network (120 comparators) and prune it.
    # This is the most reliable method given the constraints.
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    return network

def _verify_network(comps, n=N):
    """
    Verify the network using the zero-one principle.
    Returns True if the network sorts all 2^n inputs.
    """
    for mask in range(1 << n):
        arr = np.array([(mask >> i) & 1 for i in range(n)], dtype=np.int8)
        for (i, j) in comps:
            a = arr[i]
            b = arr[j]
            if a > b:
                arr[i] = b
                arr[j] = a
        # Check sorted
        for k in range(n - 1):
            if arr[k] > arr[k + 1]:
                return False
    return True

def _prune_greedy(comps, n=N, max_time=60.0):
    """
    Greedily remove redundant comparators from the network.
    Returns the pruned list.
    """
    comps = list(comps)
    deadline = _elapsed() + max_time
    # Repeat passes until no more removals or time is up
    while _elapsed() < deadline:
        removed_any = False
        for idx in range(len(comps)):
            if _elapsed() >= deadline:
                break
            # Try removing comps[idx]
            trial = comps[:idx] + comps[idx+1:]
            if _verify_network(trial, n):
                comps = trial
                removed_any = True
                break  # Restart from beginning since indices shifted
        if not removed_any:
            break
    return comps

def construct():
    # Generate Odd-Even Transposition network for 16 wires
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Verify it's correct
    if not _verify_network(network, N):
        # This should not happen, but fallback to empty or error?
        # If verification fails, we have a bug.
        pass
    
    # Prune redundant comparators
    remaining_time = _DEADLINE - _elapsed()
    if remaining_time > 5:
        network = _prune_greedy(network, N, max_time=max(1, remaining_time - 5))
    
    return network
