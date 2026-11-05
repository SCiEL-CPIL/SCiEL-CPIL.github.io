import numpy as np
import time
import random

N = 18
SIZE = 1 << N  # 262144

def construct():
    """
    Constructs a sorting network using constructive forward search.
    1. Start with all 2^N inputs unsorted.
    2. Repeatedly find the comparator (i, j) that sorts the most remaining unsorted inputs.
    3. Apply it, update the unsorted set, and add to the network.
    4. Stop when all inputs are sorted.
    5. Perform a final greedy deletion pass to remove any redundant comparators.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for the 90s limit
    
    # Initialize states for all 2^N inputs
    # states[j] contains the j-th bit of all 2^N numbers
    k_range = np.arange(SIZE, dtype=np.uint64)
    states = np.empty((N, SIZE), dtype=np.uint8)
    for j in range(N):
        states[j] = ((k_range >> j) & 1).astype(np.uint8)
    
    # unsorted_mask[k] is True if input k is currently unsorted
    # An input is unsorted if there exists j such that states[j][k] < states[j+1][k]
    # i.e., states[j][k] == 0 and states[j+1][k] == 1
    unsorted_mask = np.zeros(SIZE, dtype=bool)
    for j in range(N - 1):
        # states[j] is 0, states[j+1] is 1
        unsorted_mask |= (states[j] == 0) & (states[j+1] == 1)
    
    network = []
    
    # Constructive phase
    while np.any(unsorted_mask):
        if time.time() > deadline - 10:  # Leave time for final polish
            break
            
        best_i, best_j = -1, -1
        best_count = 0
        
        # Find the comparator that fixes the most unsorted inputs
        for i in range(N - 1):
            # Optimization: only consider comparators that could potentially fix unsorted inputs
            # If no unsorted input has a 0 at i and 1 at j, this comparator does nothing to unsorted set
            # But checking this is also expensive. Let's just iterate all pairs.
            si = states[i]
            sj = states[j]
            
            # The comparator swaps values where si=1, sj=0.
            # After swap:
            # new_si = si & sj (0 where si=1, sj=0)
            # new_sj = si | sj (1 where si=1, sj=0)
            
            # An unsorted input k is fixed by (i, j) if:
            # 1. It was unsorted.
            # 2. The inversion at (i, j) is removed: si[k]=1, sj[k]=0.
            # 3. No new inversions are created.
            #    - New inversion at (i, j)? No, new_si=0, new_sj=1 is sorted.
            #    - New inversion at (i-1, i)? If i>0, need states[i-1][k] <= new_si[k].
            #      If states[i-1][k]=1 and new_si[k]=0, new inversion.
            #    - New inversion at (j, j+1)? If j<N-1, need new_sj[k] <= states[j+1][k].
            #      If new_sj[k]=1 and states[j+1][k]=0, new inversion.
            
            # Let's compute the mask of inputs fixed by (i, j) among unsorted ones.
            # Fixed candidates: unsorted & (si == 1) & (sj == 0)
            cand = unsorted_mask & (si == 1) & (sj == 0)
            
            if not np.any(cand):
                continue
                
            # Check for new inversions
            # Inversion at (i-1, i): states[i-1] == 1 and new_si == 0
            # new_si == 0 where si==1, sj==0.
            # So new inversion if states[i-1] == 1 and si==1 and sj==0.
            if i > 0:
                new_inv_left = cand & (states[i-1] == 1)
                cand = cand & ~new_inv_left
                
            # Inversion at (j, j+1): new_sj == 1 and states[j+1] == 0
            # new_sj == 1 where si==1, sj==0.
            # So new inversion if si==1 and sj==0 and states[j+1] == 0.
            if j < N - 1:
                new_inv_right = cand & (states[j+1] == 0)
                cand = cand & ~new_inv_right
                
            count = np.count_nonzero(cand)
            if count > best_count:
                best_count = count
                best_i = i
                best_j = j
                
                # Early break if we found a comparator that fixes ALL unsorted inputs?
                # Unlikely in one step, but good to check.
                if count == np.count_nonzero(unsorted_mask):
                    break
                    
        if best_i == -1:
            # No comparator can fix any unsorted input? This shouldn't happen for a valid sorting network problem.
            # It implies the current state is not sortable by any single comparator, which is impossible
            # if the network is correct. But we are building it.
            # Actually, if we are stuck, it means our greedy choice led to a dead end?
            # No, sorting networks are always constructible.
            # Wait, if best_count == 0, it means no comparator reduces the unsorted set.
            # This can happen if we are in a local minimum of the "unsorted count" metric?
            # No, the "unsorted count" metric is monotonic decreasing if we pick the right comparator.
            # Is it possible that no single comparator reduces the count?
            # Yes, if the unsorted inputs require a sequence of comparators to fix.
            # Example: 0 1 0 1. Comparator (0,1) fixes first, creates second? No.
            # If we are stuck, we might need to accept a comparator that doesn't reduce the count
            # but enables future reductions.
            # However, for now, let's break and handle in deletion phase or restart.
            # Actually, let's just break to avoid infinite loop.
            break
            
        # Apply the best comparator
        i, j = best_i, best_j
        network.append((i, j))
        
        # Update states
        si = states[i]
        sj = states[j]
        states[i] = si & sj
        states[j] = si | sj
        
        # Update unsorted_mask
        # Recalculate is expensive (N-1 passes).
        # Can we update incrementally?
        # Only inputs affected by (i, j) changed.
        # Affected: si=1, sj=0 (swapped) and si=0, sj=1 (swapped).
        # Actually, si & sj and si | sj only change where si != sj.
        # So only inputs where si != sj are affected.
        # Let's just recalculate for simplicity, it's fast enough with NumPy.
        # N=18, SIZE=2^18. 17 passes of bitwise ops on 2^18 arrays.
        # 17 * 2^18 operations. Very fast.
        new_unsorted = np.zeros(SIZE, dtype=bool)
        for k in range(N - 1):
            new_unsorted |= (states[k] == 0) & (states[k+1] == 1)
        unsorted_mask = new_unsorted
        
    # Final Greedy Deletion Pass
    # Try to remove comparators from the end to the start.
    # We need to re-verify the whole network after each removal.
    # This is expensive, so we only do this if we have time.
    
    def verify_network(net):
        # Apply net to initial states
        temp_states = np.empty((N, SIZE), dtype=np.uint8)
        for j in range(N):
            temp_states[j] = ((k_range >> j) & 1).astype(np.uint8)
            
        for i, j in net:
            temp_states[i] = temp_states[i] & temp_states[j]
            temp_states[j] = temp_states[i] | temp_states[j]
            
        # Check sorted
        for j in range(N - 1):
            if np.any(temp_states[j] < temp_states[j+1]):
                return False
        return True

    # Optimization: Only try deletion if network is not too long and we have time.
    # Actually, the constructive network might already be quite short.
    # Let's try a few deletions from the end.
    
    current_net = network[:]
    
    # Greedy removal from end
    for idx in range(len(current_net) - 1, -1, -1):
        if time.time() > deadline - 2:
            break
        i, j = current_net[idx]
        
        # Try removing
        test_net = current_net[:idx] + current_net[idx+1:]
        
        # Verify
        if verify_network(test_net):
            current_net = test_net
        else:
            # Keep it
            pass
            
    return current_net
