import time
import numpy as np

N = 16

# EVOLVE-BLOCK-START
import time

def _verify_network_fast(network, n=16):
    """Verify network sorts all 2^n inputs using zero-one principle with NumPy."""
    if not network:
        return n <= 1
        
    num_inputs = 1 << n
    inputs = np.arange(num_inputs, dtype=np.uint32)
    
    masks = (1 << np.arange(n, dtype=np.uint32))
    vals = ((inputs[:, None] & masks[None, :]) > 0).astype(np.uint8)
    
    for i, j in network:
        if i < 0 or i >= n or j < 0 or j >= n or i >= j:
            return False
        vi = vals[:, i]
        vj = vals[:, j]
        
        vi_min = vi & vj
        vj_max = vi | vj
        
        vals[:, i] = vi_min
        vals[:, j] = vj_max
        
    diffs = vals[:, 1:] - vals[:, :-1]
    return np.all(diffs >= 0)

def _bitonic_merge(n, d=0):
    """
    Returns a list of comparators to merge a bitonic sequence of length n into 
    sorted order. d=0 for ascending, d=1 for descending.
    """
    if n <= 1:
        return []
    
    half = n // 2
    comparators = []
    
    # Step 1: Compare (i, i + half)
    # This splits the bitonic sequence into two bitonic sequences of length half.
    # The first half will contain the smaller elements (if d=0) or larger (if d=1)?
    # Actually, the standard bitonic merge for ascending:
    # Compare (i, i+half). The smaller values go to the first half, larger to the second.
    # Both halves are bitonic.
    for i in range(half):
        comparators.append((i, i + half))
    
    # Step 2: Recursively merge the two halves.
    # If d=0 (ascending), we want both halves to be ascending.
    # If d=1 (descending), we want both halves to be descending.
    # Wait, the recursive call should maintain the direction of the final sort.
    # In Bitonic Sort, we usually just merge into ascending.
    # Let's stick to merging into Ascending for the top-level call.
    # The recursive merges should also be Ascending if the parent is Ascending?
    # Let's check the structure.
    # Bitonic Sort(2k, Asc):
    #   Sort(k, Asc)
    #   Sort(k, Desc)
    #   Merge(2k, Asc)
    # Merge(2k, Asc):
    #   Compare(i, i+k)
    #   Merge(k, Asc) on first half
    #   Merge(k, Asc) on second half
    
    # So if d=0, we recurse with d=0.
    # If d=1, we recurse with d=1?
    # Let's trace Merge(4, Desc).
    # Input is bitonic. We want Desc output.
    # Compare (0,2), (1,3).
    # Smaller values go to 0,1? No, compare puts min in lower index.
    # So 0 gets min(0,2), 2 gets max(0,2).
    # 1 gets min(1,3), 3 gets max(1,3).
    # If input was e.g. 3, 1, 2, 0 (Bitonic: 3>1, 2>0? No, 3,1,2,0 is not bitonic in the standard sense of one peak).
    # Standard Bitonic: increases then decreases. e.g. 1, 3, 2, 0.
    # Compare (0,2): 1,2 -> min 1 to 0, max 2 to 2. State: 1, 3, 2, 0.
    # Compare (1,3): 3,0 -> min 0 to 1, max 3 to 3. State: 1, 0, 2, 3.
    # First half: 1, 0. Second half: 2, 3.
    # If we want Descending final sort:
    # We need First half Desc, Second half Desc.
    # First half (1,0) is already Desc.
    # Second half (2,3) is Asc. Needs to be Desc.
    # So we need Merge(2, Desc) on second half?
    # Merge(2, Desc) on (2,3): Compare (2,3). Min 2 to 2, Max 3 to 3.
    # Result: 1, 0, 2, 3. Still not Desc overall (1>0 but 2<3).
    # Wait, if I want the whole sequence Descending, I should have split it such that the larger half is first?
    # No, the comparator always puts min in lower index.
    # So the first half will ALWAYS contain the smaller values.
    # Therefore, if the first half has smaller values, and we want the final sequence Descending,
    # the smaller values must be at the END.
    # This implies that a simple "Bitonic Merge to Descending" using standard comparators (min->low index)
    # cannot produce a Descending sequence on wires 0..n-1 if the smaller values end up in 0..n/2-1.
    
    # Correction:
    # To sort Descending on wires 0..n-1, we need the largest values at index 0.
    # But the comparators push min to lower indices.
    # So index 0 will tend to get small values.
    # Thus, it is impossible to sort Descending on wires 0..n-1 using only standard comparators (min->low) 
    # IF we require the result to be $v_0 \ge v_1 \dots$.
    # Wait, is it impossible?
    # If I have 2 wires, (0,1). Min to 0, Max to 1. Result: Ascending.
    # Can I get Descending? No.
    # So how does Bitonic Sort work?
    # It sorts Ascending.
    # The "Descending" phase in Bitonic Sort is used to create a Bitonic sequence.
    # A Bitonic sequence is one that goes up then down.
    # To create a Bitonic sequence on wires 0..n-1, we sort the first half Asc and second half Desc?
    # If we sort the second half Desc on wires n/2..n-1, we need the largest value at n/2.
    # But comparators push min to lower indices.
    # So on wires n/2..n-1, index n/2 is the lowest. It will get the min.
    # So the second half will be Ascending.
    # So we can't easily make a "Down" half using standard comparators on the right side?
    
    # Let's re-read Bitonic Sort definition.
    # Bitonic Sort(2k):
    # 1. Bitonic Sort(k) on first k (Asc)
    # 2. Bitonic Sort(k) on last k (Desc) -> This produces a Descending sequence on the last k wires?
    #    If so, the last k wires have $v_{n/2} \ge v_{n/2+1} \dots$.
    #    But as established, standard comparators make the lowest index get the min.
    #    So $v_{n/2}$ would be the min of that subarray.
    #    So the second half would be Ascending.
    #    So the whole sequence would be Asc + Asc = Asc? No, that's not Bitonic.
    
    # Ah, the trick is to REVERSE the wires for the "Descending" sort.
    # If we want the sequence on wires $n/2 \dots n-1$ to be Descending ($v_{n/2} \ge v_{n/2+1} \dots$),
    # we can apply an Ascending Sort Network to the REVERSED wires.
    # Reversed wires: $n-1, n-2, \dots, n/2$.
    # Let $w_i = n-1-i$ for local index $i$.
    # If we apply Asc Sort to these local indices, the local index 0 (global $n-1$) gets the min.
    # Local index $k-1$ (global $n/2$) gets the max.
    # So global $n/2$ has the max, global $n-1$ has the min.
    # So global sequence is $v_{n/2} \ge v_{n/2+1} \dots \ge v_{n-1}$.
    # This IS Descending.
    
    # So, `Sort(k, Desc)` on global wires $start \dots start+k-1$ is implemented by:
    # 1. Generate `Sort(k, Asc)` network (list of pairs $(a,b)$ with $0 \le a < b < k$).
    # 2. Map each pair $(a,b)$ to global indices:
    #    Global index for local $x$ is $start + (k-1-x)$.
    #    So $a \to start + k - 1 - a$
    #    $b \to start + k - 1 - b$
    #    Since $a < b$, $start + k - 1 - a > start + k - 1 - b$.
    #    Let $u = start + k - 1 - b$ and $v = start + k - 1 - a$.
    #    Then $u < v$.
    #    The comparator is $(u, v)$.
    #    This puts min in $u$ (which is the higher global index) and max in $v$ (lower global index).
    #    Wait.
    #    Local $a$ maps to Global $G_a = start + k - 1 - a$.
    #    Local $b$ maps to Global $G_b = start + k - 1 - b$.
    #    Comparator compares Local $a$ and Local $b$.
    #    In the Asc Sort network, min goes to Local $a$, max to Local $b$.
    #    So min goes to Global $G_a$, max to Global $G_b$.
    #    Since $a < b$, $G_a > G_b$.
    #    So min goes to Higher Global Index, max to Lower Global Index.
    #    This means the Lower Global Index gets the Max.
    #    So the sequence is Descending. Correct.
    
    comparators.extend(_bitonic_merge(half, d))
    
    # The second half is at offset half.
    second_half_comparators = _bitonic_merge(half, d)
    for i, j in second_half_comparators:
        comparators.append((i + half, j + half))
        
    return comparators

def _bitonic_sort(n, d=0):
    """
    Returns a list of comparators to sort n wires.
    d=0 for Ascending, d=1 for Descending.
    """
    if n <= 1:
        return []
    
    half = n // 2
    comparators = []
    
    # Sort first half
    comparators.extend(_bitonic_sort(half, d))
    
    # Sort second half in opposite direction
    # If d=0 (Asc), second half should be Desc.
    # If d=1 (Desc), second half should be Asc.
    second_half_dir = 1 - d
    
    # Generate the network for the second half
    # We need to map the local indices of the second half sort to global indices.
    # If second_half_dir == 0 (Asc):
    #   Standard mapping: local i -> global i + half.
    # If second_half_dir == 1 (Desc):
    #   Reversed mapping: local i -> global (n - 1 - i).
    #   Wait, the second half occupies global indices half .. n-1.
    #   Local index 0 corresponds to global half? Or global n-1?
    #   For Desc sort on wires half..n-1, we want global half to be Max.
    #   As derived before, we apply Asc Sort to reversed wires.
    #   Reversed wires for the block half..n-1 are: n-1, n-2, ..., half.
    #   Local index 0 of the sort network corresponds to the "first" wire in the sort logic.
    #   If we use the standard Asc Sort network, local 0 gets Min.
    #   We want global half to get Max.
    #   So local 0 (Min) should map to global n-1.
    #   Local half-1 (Max) should map to global half.
    #   So mapping: local i -> global (n - 1 - i).
    
    second_half_net = _bitonic_sort(half, second_half_dir)
    
    if second_half_dir == 0:
        # Ascending on second half: local i -> global i + half
        for i, j in second_half_net:
            comparators.append((i + half, j + half))
    else:
        # Descending on second half: local i -> global n - 1 - i
        for i, j in second_half_net:
            # i, j are local indices in 0..half-1
            # Global indices:
            gi = n - 1 - i
            gj = n - 1 - j
            # Since i < j, gi > gj.
            # Comparator must have first < second.
            comparators.append((gj, gi))

    # Merge the bitonic sequence
    comparators.extend(_bitonic_merge(n, d))
    
    return comparators

def construct():
    """Return a sorting network for 16 wires."""
    # Generate Batcher's Bitonic Sort network
    net = _bitonic_sort(N, 0)
    
    # Verify
    if not _verify_network_fast(net, N):
        # Fallback to Odd-Even Transposition if Bitonic fails
        net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    return net
# EVOLVE-BLOCK-END
