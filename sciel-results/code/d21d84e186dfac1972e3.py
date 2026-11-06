"""Size-efficient sorting network search for N=16 wires.

Structural change: use comparator dependency analysis to identify essential
comparators, then apply targeted removal and local search.

Key insight: For each comparator (i,j) at position k, compute:
- state_before: the state just before this comparator
- state_after: the state just after this comparator
A comparator is "redundant" if removing it still yields a valid sorting network.
We prioritize removal candidates by how "weak" their effect is (smaller state change
means more likely redundant).

This replaces generic simulated annealing with a principled dependency-based search.
"""

import time
import random
import math
import numpy as np

N = 16
NUM_INPUTS = 1 << N  # 65536
NUM_WORDS = NUM_INPUTS // 64  # 1024 uint64 words per wire

_initial_states_np = None

def _get_initial_states_np():
    global _initial_states_np
    if _initial_states_np is None:
        _initial_states_np = np.zeros((N, NUM_WORDS), dtype=np.uint64)
        for i in range(N):
            for w in range(NUM_WORDS):
                word_start = w * 64
                word_val = 0
                for bit in range(64):
                    pos = word_start + bit
                    if (pos >> i) & 1:
                        word_val |= (1 << bit)
                _initial_states_np[i, w] = word_val
    return _initial_states_np

def fast_verify(network):
    """Verify a sorting network using numpy bit-parallel zero-one principle."""
    state = _get_initial_states_np().copy()
    for (i, j) in network:
        min_val = state[i] & state[j]
        max_val = state[i] | state[j]
        state[i] = min_val
        state[j] = max_val
    for i in range(N - 1):
        if np.any(state[i] & ~state[i + 1]):
            return False
    return True

def fast_verify_from_state(state, network):
    """Verify remaining network from a given state."""
    s = state.copy()
    for (i, j) in network:
        min_val = s[i] & s[j]
        max_val = s[i] | s[j]
        s[i] = min_val
        s[j] = max_val
    for i in range(N - 1):
        if np.any(s[i] & ~s[i + 1]):
            return False
    return True

def compute_all_states(network):
    """
    Compute the state after each comparator in the network.
    Returns a list of states: states[k] = state after first k comparators.
    """
    states = [_get_initial_states_np().copy()]
    current = states[0]
    for (i, j) in network:
        new_state = current.copy()
        min_val = new_state[i] & new_state[j]
        max_val = new_state[i] | new_state[j]
        new_state[i] = min_val
        new_state[j] = max_val
        states.append(new_state)
        current = new_state
    return states

def measure_comparator_impact(state_before, state_after, i, j):
    """
    Measure how much a comparator changes the state.
    Returns a score: 0 if no change, higher if larger change.
    """
    # Count bits that changed in wires i and j
    changed_i = np.count_nonzero(state_before[i] ^ state_after[i])
    changed_j = np.count_nonzero(state_before[j] ^ state_after[j])
    return changed_i + changed_j

def dependency_based_removal(net, deadline):
    """
    Remove redundant comparators using dependency analysis.
    Priority: comparators with smallest impact are tried first.
    """
    net = list(net)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        n_comps = len(net)
        if n_comps == 0:
            break
        
        # Compute all states
        states = compute_all_states(net)
        
        # Score each comparator by its impact (smaller = more likely removable)
        impacts = []
        for k in range(n_comps):
            i, j = net[k]
            impact = measure_comparator_impact(states[k], states[k+1], i, j)
            impacts.append((impact, k))
        
        # Sort by impact (smallest first) - try removing low-impact ones first
        impacts.sort()
        
        for impact, idx in impacts:
            if time.time() > deadline:
                break
            if idx >= len(net):
                continue
            
            # Try removing comparator at idx
            candidate = net[:idx] + net[idx+1:]
            if fast_verify(candidate):
                net = candidate
                changed = True
                break
    
    return net

def local_search_removal(net, deadline):
    """
    Local search that tries removing pairs of comparators.
    Sometimes two comparators together are redundant even though neither alone is.
    """
    net = list(net)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        n_comps = len(net)
        if n_comps < 2:
            break
        
        # Try removing pairs of adjacent comparators
        for idx in range(n_comps - 1):
            if time.time() > deadline:
                break
            candidate = net[:idx] + net[idx+2:]
            if fast_verify(candidate):
                net = candidate
                changed = True
                break
        
        if not changed:
            # Try removing non-adjacent pairs
            for idx1 in range(n_comps):
                if time.time() > deadline:
                    break
                for idx2 in range(idx1 + 1, n_comps):
                    if time.time() > deadline:
                        break
                    # Remove both idx1 and idx2 (adjust index)
                    if idx2 == idx1 + 1:
                        continue  # already tried adjacent
                    candidate = net[:idx1] + net[idx1+1:idx2] + net[idx2+1:]
                    if fast_verify(candidate):
                        net = candidate
                        changed = True
                        break
                if changed:
                    break
    
    return net

def swap_local_search(net, deadline):
    """
    Try swapping comparator order to enable further removals.
    """
    net = list(net)
    best_net = list(net)
    best_count = len(net)
    
    if len(net) < 2:
        return net
    
    for _ in range(200):  # Limited number of swap attempts
        if time.time() > deadline:
            break
        
        # Try swapping two random comparators
        idx1 = random.randint(0, len(net) - 1)
        idx2 = random.randint(0, len(net) - 1)
        if idx1 == idx2:
            continue
        
        candidate = list(net)
        candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
        
        if not fast_verify(candidate):
            continue
        
        # If valid, try to remove more from this candidate
        reduced = dependency_based_removal(candidate, deadline)
        if len(reduced) < best_count:
            best_net = reduced
            best_count = len(reduced)
            net = list(reduced)
    
    return best_net

def bitonic_sorter(n):
    """
    Generate Batcher's bitonic sorting network for n wires (n must be power of 2).
    """
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

def construct():
    deadline = time.time() + 82  # 82s to leave 8s margin for overhead
    
    # Start from bitonic sorter (64 comparators for n=16)
    net = bitonic_sorter(N)
    best_net = list(net)
    best_count = len(net)
    
    # Verify initial network
    if not fast_verify(best_net):
        best_net = odd_even_network()
        best_count = len(best_net)
        net = list(best_net)
    
    # Phase 1: Dependency-based removal (principled, not random)
    net = dependency_based_removal(net, deadline)
    if len(net) < best_count:
        best_net = list(net)
        best_count = len(net)
    
    # Phase 2: Pair removal (sometimes two comparators are jointly redundant)
    net = local_search_removal(net, deadline)
    if len(net) < best_count:
        best_net = list(net)
        best_count = len(net)
    
    # Phase 3: Swap-based local search to enable further removals
    net = swap_local_search(net, deadline)
    if len(net) < best_count:
        best_net = net
        best_count = len(net)
    
    # Phase 4: Another round of dependency-based removal
    net = dependency_based_removal(best_net, deadline)
    if len(net) < best_count:
        best_net = net
        best_count = len(net)
    
    # Phase 5: Try from odd-even network as well
    if time.time() < deadline:
        net2 = odd_even_network()
        net2 = dependency_based_removal(net2, deadline)
        net2 = local_search_removal(net2, deadline)
        if len(net2) < best_count:
            best_net = net2
            best_count = len(net2)
    
    # Final verification
    if not fast_verify(best_net):
        # Fallback to a known valid network
        best_net = bitonic_sorter(N)
    
    return best_net
