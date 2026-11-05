"""Size-efficient sorting networks on 18 wires."""
import numpy as np
import time

N = 18

# EVOLVE-BLOCK-START

def _apply_network(net, x):
    """Apply network to array x (in-place). x is length N."""
    for i, j in net:
        if x[i] > x[j]:
            x[i], x[j] = x[j], x[i]

def _verify_network(net, skip_idx=-1):
    """Verify network sorts all 2^N boolean inputs. Returns True if valid.
    If skip_idx >= 0, the comparator at that index is ignored."""
    total = 1 << N
    idx = np.arange(total, dtype=np.uint32)
    state = np.zeros((N, total), dtype=np.uint8)
    for k in range(N):
        state[k] = (idx >> k) & 1
    
    for idx_c, (i, j) in enumerate(net):
        if idx_c == skip_idx:
            continue
        new_i = state[i] & state[j]
        new_j = state[i] | state[j]
        state[i] = new_i
        state[j] = new_j
    
    for k in range(N - 1):
        if np.any(state[k] > state[k+1]):
            return False
    return True

def _depth(net):
    """Compute the parallel depth of the network."""
    if not net:
        return 0
    # depth[i] = the depth at which wire i last had a comparator
    depth = [0] * N
    for i, j in net:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth)

def _odd_even_transposition():
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    """Construct a sorting network with as few comparators as possible."""
    start_time = time.time()
    deadline = start_time + 75  # Safety margin: 90s limit, 75s deadline
    
    # Start with the seed odd-even transposition network
    best_net = _odd_even_transposition()
    best_len = len(best_net)
    
    # Try to optimize by removing redundant comparators
    # Strategy: iteratively try removing each comparator and check validity
    # If valid, keep it removed; otherwise restore.
    
    net = list(best_net)
    
    # Helper to run a pass of removal
    def run_pass():
        nonlocal net, best_len, best_net
        i = 0
        while i < len(net) and time.time() < deadline:
            if _verify_network(net, skip_idx=i):
                net.pop(i)
                best_len = len(net)
                best_net = list(net)
                # Don't increment i
            else:
                i += 1

    # Run multiple passes
    for _ in range(10):
        if time.time() > deadline:
            break
        run_pass()
    
    return best_net

# EVOLVE-BLOCK-END
