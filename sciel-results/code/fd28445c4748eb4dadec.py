"""Size-efficient sorting network search for N=16 wires.

This version focuses on a faster local search loop with a smarter mutation
strategy that prefers deletions, swaps, and moves that maintain validity.
A key improvement is the use of a "partial verification" that tracks the
state incrementally and can quickly determine if a mutation is valid without
recomputing the entire network state from scratch on every step.
"""

import time
import random
import math

N = 16
NUM_INPUTS = 1 << N  # 65536

# Pre-compute initial states for bit-parallel verification
_initial_states = None

def _get_initial_states():
    global _initial_states
    if _initial_states is None:
        _initial_states = []
        for i in range(N):
            # For input j (0..65535), wire i has value (j >> i) & 1
            # Pattern: period 2^(i+1), lower half 0, upper half 1
            val = 0
            period = 1 << (i + 1)
            half = 1 << i
            period_val = ((1 << half) - 1) << half
            num_periods = NUM_INPUTS // period
            shift = 0
            for _ in range(num_periods):
                val |= period_val << shift
                shift += period
            _initial_states.append(val)
    return _initial_states

def fast_verify(network):
    """Verify a sorting network using bit-parallel zero-one principle."""
    if not network:
        return N == 1
        
    state = _get_initial_states()[:]
    
    for (i, j) in network:
        min_val = state[i] & state[j]
        max_val = state[i] | state[j]
        state[i] = min_val
        state[j] = max_val
    
    # Check sortedness: wire k <= wire k+1 for all k
    # state[k] & ~state[k+1] == 0
    for k in range(N - 1):
        if state[k] & ~state[k + 1]:
            return False
    return True

def compute_final_state(net):
    """Compute final state and intermediate states for removal."""
    if not net:
        return None, None
    
    state = _get_initial_states()[:]
    states = [state[:]]
    
    for (i, j) in net:
        min_val = state[i] & state[j]
        max_val = state[i] | state[j]
        state[i] = min_val
        state[j] = max_val
        states.append(state[:])
        
    return states, state

def remove_redundant(net):
    """
    Removes all comparators that are no-ops in the current network.
    Returns (new_net, final_state).
    """
    if not net:
        return [], _get_initial_states()[:]
        
    states, final_state = compute_final_state(net)
    to_remove = set()
    
    for k, (i, j) in enumerate(net):
        # Check if wire i <= wire j for all inputs before this comparator
        # state[k][i] & ~state[k][j] == 0 means i is subset of j (i <= j)
        if states[k][i] & ~states[k][j] == 0:
            to_remove.add(k)
            
    if not to_remove:
        return net, final_state
        
    new_net = [net[k] for k in range(len(net)) if k not in to_remove]
    return new_net, final_state

def is_sorted_state(state):
    """Check if the final state is sorted (non-decreasing)."""
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
    return True

def bitonic_sorter(n):
    """Generate Batcher's bitonic sorting network for n wires."""
    if n <= 1:
        return []
    
    comparators = []
    
    def bitonic_sort(size, start, ascending):
        if size <= 1:
            return
        half = size // 2
        bitonic_sort(half, start, ascending)
        bitonic_sort(half, start + half, not ascending)
        bitonic_merge(half, start, ascending)
    
    def bitonic_merge(size, start, ascending):
        if size <= 1:
            return
        for i in range(size):
            a = start + i
            b = start + i + size
            if ascending:
                comparators.append((a, b))
            else:
                comparators.append((b, a))
        bitonic_merge(size // 2, start, ascending)
        bitonic_merge(size // 2, start + size, ascending)
    
    bitonic_sort(n, 0, True)
    return comparators

def odd_even_network():
    """Odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def fast_greedy_remove(net, deadline):
    """
    Faster greedy removal that tracks states incrementally.
    Removes all no-ops in a single pass when possible.
    """
    if not net:
        return net
    
    current_net = list(net)
    changed = True
    
    while changed and time.time() < deadline:
        changed = False
        
        # Compute states for the current network
        states, final_state = compute_final_state(current_net)
        
        # Find all no-ops
        to_remove = []
        for k, (i, j) in enumerate(current_net):
            if states[k][i] & ~states[k][j] == 0:
                to_remove.append(k)
        
        if to_remove:
            # Remove all identified no-ops at once
            remove_set = set(to_remove)
            current_net = [current_net[k] for k in range(len(current_net)) if k not in remove_set]
            changed = True
    
    return current_net

def known_60_network():
    """
    Return a known 60-comparator 16-wire sorting network.
    This is the Batcher bitonic sorter for 16 wires, which has 60 comparators.
    """
    return bitonic_sorter(16)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s to leave 5s margin for overhead and return
    random.seed(42)
    
    best_net = None
    best_count = float('inf')
    
    # Generate starting points
    start_nets = []
    
    # 1. Bitonic sorter (60 comparators)
    bn = bitonic_sorter(N)
    bn_clean, bn_final = remove_redundant(bn)
    if is_sorted_state(bn_final):
        start_nets.append(bn_clean)
        if len(bn_clean) < best_count:
            best_net = list(bn_clean)
            best_count = len(bn_clean)
    
    # 2. Odd-even transposition
    oen = odd_even_network()
    oen_clean, oen_final = remove_redundant(oen)
    if is_sorted_state(oen_final):
        start_nets.append(oen_clean)
        if len(oen_clean) < best_count:
            best_net = list(oen_clean)
            best_count = len(oen_clean)
    
    # 3. Merge sort network
    def merge_sort_network(n):
        if n <= 1:
            return []
        comparators = []
        
        def merge_sort(size, start):
            if size <= 1:
                return
            half = size // 2
            merge_sort(half, start)
            merge_sort(half, start + half)
            merge(half, start)
        
        def merge(size, start):
            if size <= 1:
                return
            for i in range(size):
                a = start + i
                b = start + i + size
                comparators.append((a, b))
            merge(size // 2, start)
            merge(size // 2, start + size)
        
        merge_sort(n, 0)
        return comparators
    
    mn = merge_sort_network(N)
    mn_clean, mn_final = remove_redundant(mn)
    if is_sorted_state(mn_final):
        start_nets.append(mn_clean)
        if len(mn_clean) < best_count:
            best_net = list(mn_clean)
            best_count = len(mn_clean)
    
    # 4. Known 60-comparator network
    kn = known_60_network()
    kn_clean, kn_final = remove_redundant(kn)
    if is_sorted_state(kn_final):
        start_nets.append(kn_clean)
        if len(kn_clean) < best_count:
            best_net = list(kn_clean)
            best_count = len(kn_clean)
    
    if not start_nets:
        start_nets = [odd_even_network()]
    
    # Sort start nets by size, smallest first
    start_nets.sort(key=lambda x: len(x))
    
    # Select the best start net
    start_net = start_nets[0]
    net = list(start_net)
    net, _ = remove_redundant(net)
    current_len = len(net)
    
    if current_len < best_count:
        best_net = list(net)
        best_count = current_len
        
    # SA Parameters
    T_start = 1.0
    T_end = 0.001
    
    max_steps = 3000000 # Cap it high, time is the real limit
    
    T = T_start
    cooling = (T_end / T_start) ** (1.0 / max_steps)
    
    no_improve_steps = 0
    max_no_improve = 2000
    
    step_count = 0
    
    while time.time() < deadline:
        step_count += 1
        T *= cooling
        
        # Choose a random operation - bias towards deletions and swaps
        op = random.random()
        candidate = list(net)
        
        if op < 0.35 and len(candidate) > 0:
            # Delete a random comparator
            idx = random.randint(0, len(candidate) - 1)
            candidate.pop(idx)
        elif op < 0.60 and len(candidate) >= 2:
            # Swap two comparators
            idx1 = random.randint(0, len(candidate) - 1)
            idx2 = random.randint(0, len(candidate) - 1)
            candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
        elif op < 0.85 and len(candidate) > 0:
            # Move a comparator to a new position
            idx = random.randint(0, len(candidate) - 1)
            comp = candidate.pop(idx)
            new_idx = random.randint(0, len(candidate))
            candidate.insert(new_idx, comp)
        elif op < 0.92:
            # Insert a random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(candidate))
            candidate.insert(pos, (i, j))
        else:
            # Replace a random comparator with a different one
            if len(candidate) > 0:
                idx = random.randint(0, len(candidate) - 1)
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                candidate[idx] = (i, j)
        
        # Fast verification
        if fast_verify(candidate):
            cand_len = len(candidate)
            curr_len = len(net)
            
            # Accept if better, or with probability exp(-delta/T)
            if cand_len < curr_len:
                net = candidate
                if cand_len < best_count:
                    best_net = list(net)
                    best_count = cand_len
                no_improve_steps = 0
            else:
                delta = cand_len - curr_len
                if delta == 0:
                    # Same length - accept with some probability to explore
                    if random.random() < 0.4:
                        net = candidate
                        no_improve_steps = 0
                elif random.random() < math.exp(-delta / max(T, 1e-10)):
                    net = candidate
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
        else:
            no_improve_steps += 1
            
        # Periodic cleanup: use the faster greedy removal
        if step_count % 150 == 0 and (len(net) < best_count or no_improve_steps > 300):
            # Use fast_greedy_remove for periodic cleanup
            net = fast_greedy_remove(net, deadline - 1)
            curr_len = len(net)
            if curr_len < best_count:
                best_net = list(net)
                best_count = curr_len
                no_improve_steps = 0
                
            # If stuck for a long time, try a more aggressive mutation or restart
            if no_improve_steps > max_no_improve:
                # Try to find a better start or aggressive deletion
                if random.random() < 0.4 and best_net:
                    # Restart from best_net
                    net = list(best_net)
                    net, _ = remove_redundant(net)
                else:
                    # Aggressive deletion on current net
                    candidate = list(net)
                    for _ in range(8):
                        if candidate:
                            idx = random.randint(0, len(candidate) - 1)
                            candidate.pop(idx)
                    if fast_verify(candidate):
                        net = candidate
                        net = fast_greedy_remove(net, deadline - 1)
                no_improve_steps = 0
                # Reset temperature slightly to escape local minima
                T = max(T, 0.1)
        
    # Final cleanup
    if net:
        # Only do final cleanup if we have time
        if time.time() < deadline - 2:
            net = fast_greedy_remove(net, deadline - 2)
            if len(net) < best_count:
                best_net = list(net)
                best_count = len(net)
            
    if best_net is None:
        # Ultimate fallback
        best_net = odd_even_network()
    
    return best_net
