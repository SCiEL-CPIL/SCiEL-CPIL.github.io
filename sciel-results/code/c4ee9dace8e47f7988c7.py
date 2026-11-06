"""
Size-efficient sorting networks on 17 wires – Coarse-to-fine greedy deletion approach.
Systematically identifies and removes redundant comparators using bitmask verification.
"""

import time
import random

N = 17
TOTAL_INPUTS = 1 << N  # 131072

# Precompute initial state bitmasks for verification
# state[i] is a bitmask where bit k is 1 if input k has a 1 at wire i
# Input k is an integer 0 to 2^N - 1
INITIAL_STATE = [0] * N
for i in range(N):
    mask = 0
    bit_val = 1 << i
    # We can construct this more efficiently using patterns, but for N=17 it's fast enough
    # or we can use the iterative shift method
    # Pattern for wire i: blocks of 2^i zeros, 2^i ones
    # Let's use the efficient construction from the current program
    block = 1 << i
    val = 0
    for m in range(block, TOTAL_INPUTS, 2 * block):
        bits_to_set = (1 << block) - 1
        val |= bits_to_set << m
    INITIAL_STATE[i] = val

def _is_valid(network):
    """
    Check if network sorts all 0/1 inputs using zero-one principle.
    Uses bitmask simulation for speed.
    """
    if not network:
        return N <= 1
    
    # Copy initial state
    state = list(INITIAL_STATE)
    n = N
    
    # Apply comparators
    # Optimization: local variable binding
    st = state
    for i, j in network:
        si = st[i]
        sj = st[j]
        st[i] = si & sj
        st[j] = si | sj
        
    # Check if state is sorted: S[0] subseteq S[1] subseteq ... subseteq S[N-1]
    # This is equivalent to checking S[i] & ~S[i+1] == 0 for all i
    for i in range(n - 1):
        if st[i] & ~st[i + 1]:
            return False
            
    return True

def _construct_odd_even_transposition():
    """Standard odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _try_redundant_removal(network, deadline):
    """
    Try to remove redundant comparators from the network.
    Returns a smaller valid network if found, otherwise returns the original.
    """
    current = list(network)
    current_len = len(current)
    
    # Try removing each comparator one at a time
    for idx in range(len(current)):
        if time.time() > deadline:
            break
        
        # Create candidate without comparator at idx
        candidate = current[:idx] + current[idx+1:]
        
        # Check if candidate is still valid
        if _is_valid(candidate):
            # Successfully removed a redundant comparator
            current = candidate
            current_len -= 1
    
    return current

def construct():
    """
    Construct a sorting network with as few comparators as possible.
    Uses coarse-to-fine greedy deletion to systematically remove redundant comparators.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    # Start with Odd-Even Transposition
    current_network = _construct_odd_even_transposition()
    best_network = list(current_network)
    best_len = len(current_network)
    
    # Verify start
    if not _is_valid(current_network):
        # Should not happen, but just in case
        return best_network

    # Phase 1: Aggressive redundant removal
    # Repeatedly try to remove comparators until no more can be removed
    phase1_deadline = start_time + 40.0
    while time.time() < phase1_deadline:
        prev_len = len(current_network)
        current_network = _try_redundant_removal(current_network, phase1_deadline)
        
        if len(current_network) < best_len:
            best_network = list(current_network)
            best_len = len(current_network)
        
        # If no progress, break
        if len(current_network) == prev_len:
            break
    
    # Phase 2: Local search with targeted moves
    # Try swapping adjacent comparators, then re-check for redundancies
    phase2_deadline = deadline
    while time.time() < phase2_deadline:
        improved = False
        
        # Try swapping adjacent comparators
        for idx in range(len(current_network) - 1):
            if time.time() > phase2_deadline:
                break
            
            # Create candidate with swap
            candidate = list(current_network)
            candidate[idx], candidate[idx + 1] = candidate[idx + 1], candidate[idx]
            
            # Check if candidate is valid
            if _is_valid(candidate):
                # Try to remove redundancies from this swapped version
                optimized = _try_redundant_removal(candidate, phase2_deadline)
                
                if len(optimized) < len(current_network):
                    current_network = optimized
                    improved = True
                    if len(current_network) < best_len:
                        best_network = list(current_network)
                        best_len = len(current_network)
                    break
        
        # If no improvement after trying all swaps, break
        if not improved:
            break
    
    return best_network

# Keep the required function name and signature
# The task asks to keep function names and signatures of the seed program.
# The seed program had `construct()` as the main entry point.
# It also had helper functions, but `construct` is the interface.
