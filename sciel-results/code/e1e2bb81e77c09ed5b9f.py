"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict

def _build_ap_data():
    """
    Pre-compute data structures for fast incremental discrepancy updates.
    For each index i, we store:
    1. The list of (start, step) pairs for all APs containing i.
    2. The list of (start, step) pairs for all APs containing i-1 (to update the "next" element in the AP).
    """
    n = 256
    # ap_starts[i] = list of (start, step) for APs containing index i
    # ap_prev[i] = list of (start, step) for APs containing index i-1
    # Note: An AP is defined by (start, step). It contains indices start, start+step, ...
    # If we flip index i, we affect the sum of all APs that include i.
    # Also, for any AP that included i-1, the "current" element in our tracking 
    # (if we track prefix sums) would effectively shift, but here we track 
    # the sum of the *entire* AP? No, the discrepancy is the max over *prefixes* of APs.
    
    # Wait, the discrepancy is the max absolute *signed sum* on ANY arithmetic progression.
    # This means for a specific AP (start, step), we look at:
    # S_1 = a[start]
    # S_2 = a[start] + a[start+step]
    # S_k = sum_{j=0}^{k-1} a[start + j*step]
    # The discrepancy contribution of this AP is max_k |S_k|.
    
    # This is equivalent to: for each (start, step), we have a sequence of values.
    # We need to track the maximum absolute prefix sum for each such sequence.
    
    # To make this incremental:
    # When we flip a[i], it affects:
    # 1. All APs where i is an element.
    # 2. Specifically, for an AP (s, d), if i is in it, let pos = (i-s)/d.
    #    The prefix sums S_k for k > pos are all changed by +/- 2*a[i] (which is +/-2).
    #    The prefix sums S_k for k <= pos are unchanged.
    #    The max absolute prefix sum for this AP might change.
    
    # Data structure:
    # For each (s, d), we maintain:
    # - The array of values in that AP: vals[s,d]
    # - The prefix sums: pref[s,d]
    # - The current max absolute prefix sum: max_abs[s,d]
    # - To update max_abs efficiently, we can use a segment tree or just recompute 
    #   if the AP is short. Since n=256, APs are short.
    #   However, there are ~ n^2/2 = 32768 APs.
    #   Recomputing max_abs for an AP of length L takes O(L).
    #   Total cost of one flip: sum of lengths of all APs containing i.
    #   How many APs contain i? 
    #   For each step d (1..256), i is in at most 1 AP for that d (the one starting at i - k*d >= 0).
    #   Actually, for a fixed d, i is in the AP starting at s = i - k*d.
    #   There is only one such AP for a given d? No.
    #   The AP is defined by start and step.
    #   If step is d, the APs are 0, d, 2d...; 1, 1+d, ...; etc.
    #   Index i is in exactly one AP for each step d (the one starting at i % d).
    #   So index i is in exactly 256 APs (one for each step d=1..256).
    #   The length of the AP for step d containing i is roughly n/d.
    #   Total work per flip: sum_{d=1}^{256} (n/d) ~ n * H_n ~ 256 * 6 ~ 1500 operations.
    #   This is very fast! We can do ~10,000-50,000 flips per second.
    
    pass

def construct():
    n = 256
    deadline = time.time() + 55  # Safety margin
    
    # Pre-compute AP information for each index
    # For each index i, list of (start, step) pairs such that i is in the AP (start, step)
    # And for each such AP, we need to know the position of i in that AP.
    
    # Let's structure the state:
    # signs: numpy array of length n
    # ap_data: dict keyed by (start, step) -> 
    #           {
    #               'vals': list of signs in this AP
    #               'pref': numpy array of prefix sums
    #               'max_abs': current max absolute prefix sum
    #           }
    # This is too much memory and overhead to create dicts for 32k APs.
    # Let's use arrays.
    # Number of APs: sum_{d=1}^n d = n(n+1)/2 = 32896.
    # We can flatten (start, step) into a single index.
    # idx = start * (n+1) + step ? No, start < step.
    # Let's just iterate d from 1 to n, and r from 0 to d-1 (where r < n).
    # Total APs is small enough to store in a list of dicts or parallel arrays.
    
    # Let's use a simpler approach:
    # For each (d, r), we have an AP.
    # Let's precompute the indices for each (d, r).
    
    ap_indices = {}
    ap_id_map = {} # (d, r) -> id
    current_id = 0
    ap_lengths = []
    
    for d in range(1, n + 1):
        for r in range(d):
            if r >= n:
                continue
            indices = list(range(r, n, d))
            if not indices:
                continue
            ap_indices[current_id] = indices
            ap_id_map[(d, r)] = current_id
            ap_lengths.append(len(indices))
            current_id += 1
            
    num_aps = current_id
    
    # For each index i, which APs contain it?
    # And what is its position in that AP?
    # index_to_aps[i] = list of (ap_id, pos_in_ap)
    index_to_aps = [[] for _ in range(n)]
    for ap_id, indices in ap_indices.items():
        for pos, idx in enumerate(indices):
            index_to_aps[idx].append((ap_id, pos))
            
    # State
    signs = np.random.choice([-1, 1], size=n)
    
    # For each AP, maintain:
    # ap_pref[ap_id]: numpy array of prefix sums
    # ap_max[ap_id]: max absolute value of ap_pref[ap_id]
    
    ap_pref = [None] * num_aps
    ap_max = [0] * num_aps
    
    def init_ap_state():
        for ap_id in range(num_aps):
            indices = ap_indices[ap_id]
            vals = signs[indices]
            pref = np.cumsum(vals)
            ap_pref[ap_id] = pref
            ap_max[ap_id] = np.max(np.abs(pref))
            
    init_ap_state()
    
    # Global max discrepancy
    current_max = np.max(ap_max)
    best_signs = signs.copy()
    best_max = current_max
    
    rng = random.Random(42)
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        if T < 0.01:
            break
            
        i = rng.randint(0, n - 1)
        old_val = signs[i]
        new_val = -old_val
        delta = 2 * new_val # Change in value is +/- 2
        
        # Calculate change in max discrepancy
        # This is tricky. We need to know if the *global* max changes.
        # Let's just update the affected APs and track the new global max.
        # Since we have ap_max array, we can update it.
        
        old_global_max = current_max
        
        # To efficiently update, we need to know the new max for each affected AP.
        # And we need to know if the *old* max of an AP was the global max.
        # If so, and it decreases, the global max might decrease.
        # If any AP increases above the old global max, global max increases.
        
        # Let's just recalculate the affected APs' maxes.
        # And keep track of the new global max.
        
        # Optimization: Only recalculate if necessary?
        # No, just do it. It's fast enough.
        
        # We need to be careful: if the old global max was from one of these APs,
        # and it drops, the new global max is max(new_affected_maxes, max_of_unaffected_aps).
        # We don't have max_of_unaffected_aps easily.
        # However, we can maintain a count of how many APs have max == current_max?
        # Or simpler: just take the max of all ap_max after update?
        # That's O(num_aps) = 32k. Too slow for 50k flips/sec.
        
        # Better: Maintain a max-heap or a counter of max values?
        # Or: Notice that current_max is an integer.
        # We can keep a list of ap_ids that currently have max == current_max.
        # If we update one, and it drops, we remove it from the list.
        # If the list becomes empty, we need to find the new max.
        # Finding the new max is O(num_aps).
        # How often does the global max drop? Rarely.
        # How often does it rise? Sometimes.
        
        # Let's try a simpler heuristic:
        # 1. Update affected APs.
        # 2. Check if any affected AP has new_max > old_global_max. If so, new_global_max = max(new_maxes).
        # 3. Else, if the old_global_max was achieved ONLY by affected APs, and they all dropped, we need to rescan.
        #    To avoid frequent rescans, we can just rescan if the number of "candidates" for global max is small.
        
        # Let's stick to the robust but slightly slower method:
        # Update affected APs.
        # Compute new_global_max = max(max(affected_new_maxes), max(unaffected_maxes)).
        # We can store ap_max in a numpy array and use np.max?
        # np.max(ap_max) is C-speed. 32k elements. ~10-20us.
        # 50k flips/sec * 20us = 1s/sec. Feasible.
        
        affected_aps = index_to_aps[i]
        
        for ap_id, pos in affected_aps:
            # Update prefix sums
            # pref[pos:] += delta
            ap_pref[ap_id][pos:] += delta
            # Update max
            # np.max(np.abs(ap_pref[ap_id])) is O(L)
            # We can optimize this if we track the argmax?
            # For now, just do it.
            new_ap_max = np.max(np.abs(ap_pref[ap_id]))
            ap_max[ap_id] = new_ap_max
            
        # Update global max
        # To speed up, we can check if the max of affected APs is > current_max.
        # If yes, new_max is that.
        # If no, we need to check if the old max is still present.
        # Let's just do np.max(np.array(ap_max)) for simplicity and correctness first.
        # If it's too slow, we optimize.
        
        # Actually, np.max on a python list of floats/ints is slow.
        # Let's keep ap_max in a numpy array.
        
        # Let's refactor to use numpy for ap_max.
        
        break # Restart with numpy ap_max
        
    # Refactored version
    ap_max_np = np.array(ap_max, dtype=np.int64)
    
    while time.time() < deadline:
        if T < 0.01:
            break
            
        i = rng.randint(0, n - 1)
        old_val = signs[i]
        new_val = -old_val
        delta = 2 * new_val
        
        affected_aps = index_to_aps[i]
        
        old_global_max = current_max
        
        # Update affected APs
        # We need to update ap_max_np in place.
        # But ap_max_np is a view? No, it's a copy.
        # Let's just use a list for ap_max and convert to np.array only for max?
        # Or keep ap_max as a numpy array and update it.
        
        # Let's use a numpy array for ap_max.
        # We need to update specific indices.
        
        # Wait, ap_pref is a list of numpy arrays.
        # ap_max_np is a numpy array.
        
        # Update loop
        for ap_id, pos in affected_aps:
            ap_pref[ap_id][pos:] += delta
            # Recompute max for this AP
            # Optimization: If the max was at the end, and we added to the end, 
            # the max might just be abs(old_end + delta).
            # But it could be in the middle.
            # Just recompute.
            new_ap_max = np.max(np.abs(ap_pref[ap_id]))
            ap_max_np[ap_id] = new_ap_max
            
        # Update global max
        # Check if any affected AP exceeded old_global_max
        # max_affected = max(ap_max_np[ap_id for ap_id, _ in affected_aps])
        # This is a python loop.
        
        # Simpler: new_global_max = np.max(ap_max_np)
        # This is O(N_aps) in C.
        new_global_max = np.max(ap_max_np)
        
        # Acceptance criterion
        delta_max = new_global_max - old_global_max
        
        if delta_max <= 0:
            # Improvement or neutral
            signs[i] = new_val
            current_max = new_global_max
            if new_global_max < best_max:
                best_max = new_global_max
                best_signs = signs.copy()
        else:
            # Worse
            # Accept with probability exp(-delta_max / T)
            # Since delta_max is integer, and T is small, this is rare.
            if rng.random() < np.exp(-delta_max / T):
                signs[i] = new_val
                current_max = new_global_max
                
        T *= cooling_rate
        
    return best_signs.tolist()

# EVOLVE-BLOCK-END
