import time
import numpy as np

N = 17
NUM_BOOLEANS = 1 << N  # 131072

# Precompute masks for speed in _check_single
_MASKS = [1 << i for i in range(N)]

def _check_single(network, x):
    """Check if a single Boolean input x is sorted by the network."""
    val = x
    for (i, j) in network:
        bi = val & _MASKS[i]
        bj = val & _MASKS[j]
        if bi and not bj:
            # Swap: clear i, set j
            val = (val & ~_MASKS[i]) | (val | _MASKS[j])
            # Or: val ^= (_MASKS[i] | _MASKS[j])
            val ^= (_MASKS[i] | _MASKS[j])
    # Check sorted
    for i in range(N - 1):
        if (val & _MASKS[i]) and not (val & _MASKS[i + 1]):
            return False
    return True

def _check_full(network):
    """Full verification."""
    if not network:
        return N == 1
    bits = np.arange(1, NUM_BOOLEANS, dtype=np.uint32)[:, None]
    for (i, j) in network:
        mask_i = 1 << i
        mask_j = 1 << j
        bi = bits & mask_i
        bj = bits & mask_j
        swap_mask = (bi != 0) & (bj == 0)
        bits = np.where(swap_mask, bits ^ (mask_i | mask_j), bits)
    
    for i in range(N - 1):
        bi = bits & (1 << i)
        bj = bits & (1 << (i + 1))
        if np.any((bi != 0) & (bj == 0)):
            return False
    return True

def _find_bad(network):
    """Find a bad input."""
    for x in range(1, NUM_BOOLEANS):
        if not _check_single(network, x):
            return x
    return None

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0

    def odd_even():
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]

    net = odd_even()
    best_net = list(net)
    
    if not _check_full(net):
        return best_net

    bad_x = None
    order = None
    idx = 0
    stuck_count = 0
    max_stuck = 100 # Try this many swaps/removals before declaring stuck

    while time.time() < deadline:
        if order is None or idx >= len(order):
            order = list(range(len(net)))
            np.random.shuffle(order)
            idx = 0
            stuck_count = 0

        pos = order[idx]
        idx += 1
        
        # 1. Try Removal
        candidate = net[:pos] + net[pos + 1:]
        
        if bad_x is not None:
            if not _check_single(candidate, bad_x):
                continue

        if _check_full(candidate):
            net = candidate
            best_net = list(net)
            bad_x = None
            order = None
            idx = 0
            stuck_count = 0
        else:
            stuck_count += 1
            # Update bad_x
            if bad_x is None or not _check_single(candidate, bad_x):
                bx = _find_bad(candidate)
                if bx is not None:
                    bad_x = bx
        
        # 2. If stuck on removals, try Swaps
        if stuck_count >= max_stuck:
            # Try a few random swaps
            swaps_tried = 0
            while time.time() < deadline and swaps_tried < 20:
                i = np.random.randint(0, len(net))
                j = np.random.randint(0, len(net))
                if i == j:
                    continue
                # Swap i and j
                candidate = list(net)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                
                if bad_x is not None:
                    if not _check_single(candidate, bad_x):
                        swaps_tried += 1
                        continue

                if _check_full(candidate):
                    net = candidate
                    best_net = list(net)
                    bad_x = None
                    stuck_count = 0
                    order = None
                    idx = 0
                    break
                swaps_tried += 1

    return best_net
# EVOLVE-BLOCK-END
