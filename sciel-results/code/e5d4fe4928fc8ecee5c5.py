"""Size-efficient sorting network search for N=16 wires.

Strategy:
1. Use a fast bit-parallel zero-one principle verifier.
2. Start from a known-good baseline (odd-even transposition, 120 comparators).
3. Aggressive local search:
   - Greedy removal of redundant comparators.
   - "Swap" moves: replace one comparator with another at a different position.
   - "Insert then remove": insert a random comparator, then do a full greedy removal pass.
   - Multiple random restarts with different seeds to escape local minima.
4. Time-budgeted: return the best network found before the deadline.

The key insight from the inspirations is that a structured initial network (like bitonic)
might be closer to optimal than odd-even, but odd-even with aggressive optimization
has been shown to work. We'll combine:
- Fast verification (bit-parallel, pre-computed masks)
- Aggressive local search with multiple move types
- Multiple restarts to escape local minima
"""

import time
import random
import sys

N = 16
TOTAL_INPUTS = 1 << N  # 65536

# Pre-compute the initial state masks for all 16 wires
# mask[i] is a 65536-bit integer where bit j is 1 iff input j has a 1 on wire i
def _build_masks():
    masks = []
    for i in range(N):
        # Bit i of input j is 1 iff (j >> i) & 1 == 1
        # Pattern: 2^i zeros, then 2^i ones, repeating
        period = 1 << (i + 1)
        half = 1 << i
        # Build one period: lower half zeros, upper half ones
        period_val = ((1 << half) - 1) << half
        num_periods = TOTAL_INPUTS // period
        val = 0
        shift = 0
        for _ in range(num_periods):
            val |= period_val << shift
            shift += period
        masks.append(val)
    return masks

_MASKS = _build_masks()

def fast_verify(network):
    """Verify a sorting network using bit-parallel zero-one principle."""
    state = _MASKS[:]
    for (i, j) in network:
        min_val = state[i] & state[j]
        max_val = state[i] | state[j]
        state[i] = min_val
        state[j] = max_val
    
    # Check sorted: for all i, state[i] & ~state[i+1] == 0
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
    return True

def odd_even_network():
    """Odd-even transposition sorting network: 120 comparators for 16 wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def bitonic_sort_network(n, offset=0, inc=True):
    """Generate a bitonic sorting network for n wires starting at offset.
    inc=True means sort in increasing order, inc=False means decreasing."""
    if n <= 1:
        return []
    
    def _bs(n, offset, inc):
        if n <= 1:
            return []
        if n == 2:
            if inc:
                return [(offset, offset + 1)]
            else:
                # For decreasing, we want max at offset, min at offset+1
                # But comparator (i,j) always puts min at i, max at j
                # So for decreasing on wires (offset, offset+1), we need
                # wire offset to have the larger value.
                # We can't directly do this with a single comparator in the same direction.
                # Actually, in bitonic sort, the "decreasing" sort is implemented
                # by sorting in increasing order but on reversed wire indices,
                # or by using comparators in the opposite direction.
                # Since our comparator (i,j) always does min->i, max->j,
                # to get decreasing order on wires [offset, offset+1],
                # we need the larger value at offset and smaller at offset+1.
                # This means we need a comparator that puts max at offset, min at offset+1.
                # But our interface only allows (i,j) with i<j, and min->i, max->j.
                # So we cannot directly express a "reverse" comparator.
                # 
                # Workaround: In bitonic sort, the "decreasing" half is actually
                # sorted in increasing order, but the merge step accounts for the direction.
                # Let me reconsider the bitonic sort algorithm.
                #
                # Actually, the standard bitonic sort network uses comparators that
                # can go in either direction. The direction is determined by the
                # desired output order. For a comparator between wires a and b
                # (a < b), if we want the smaller value at a, we use (a,b).
                # If we want the larger value at a, we use (b,a) -- but our interface
                # requires i < j, so we can't do (b,a) if b > a.
                #
                # Hmm, this is a problem. Let me re-read the task:
                # "Each compare-exchange writes min to wire i and max to j."
                # And "0 <= i < j < N".
                # So the comparator is always min->lower wire, max->higher wire.
                # This means all comparators are "ascending" -- they enforce
                # output[i] <= output[j].
                #
                # But a sorting network for increasing order only needs ascending comparators!
                # The bitonic sort algorithm, when properly implemented, only uses
                # comparators that enforce the desired final order.
                #
                # Let me just use the standard bitonic sort construction that works
                # with ascending comparators. The key is that the "decreasing" sort
                # in bitonic sort is actually just a sort in the opposite direction,
                # which can be achieved by reversing the wire indices.
                #
                # For simplicity, let me just use odd-even as the baseline and focus
                # on the local search, which is the main contribution.
                return [(offset, offset + 1)]  # Fallback: just use ascending
        half = n // 2
        res = []
        # Sort first half in desired direction
        res += _bs(half, offset, inc)
        # Sort second half in opposite direction
        res += _bs(half, offset + half, not inc)
        # Merge: compare (offset+i, offset+i+half) for i in 0..half-1
        # Direction depends on inc
        for i in range(half):
            if inc:
                res.append((offset + i, offset + i + half))
            else:
                # For decreasing merge, we want the larger value at the lower wire
                # But our comparator always puts min at lower wire
                # So we need to swap the indices? No, we can't.
                # Actually, for a decreasing bitonic merge, the comparators should
                # enforce output[offset+i] >= output[offset+i+half], which means
                # we need max at offset+i and min at offset+i+half.
                # With our interface, we can only do min->lower, max->higher.
                # So we need to use (offset+i+half, offset+i) but that violates i<j.
                # 
                # This is getting complicated. Let me just skip bitonic and use
                # odd-even as the baseline, since the local search is what matters.
                res.append((offset + i, offset + i + half))
        return res
    
    return _bs(n, offset, inc)

def greedy_remove(net, deadline):
    """Try to remove each comparator; return new net if any removal succeeds."""
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline:
                return net
            candidate = net[:idx] + net[idx+1:]
            if fast_verify(candidate):
                net = candidate
                changed = True
                break
    return net

def local_search(net, deadline):
    """Perform local search with multiple move types."""
    best_net = list(net)
    best_count = len(net)
    
    # First, do greedy removal
    net = greedy_remove(net, deadline)
    best_net = list(net)
    best_count = len(net)
    
    # Now do random local search
    random.seed(42)
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        move_type = random.choice(['remove', 'swap', 'insert_remove'])
        
        if move_type == 'remove':
            # Try removing a random comparator
            if len(net) <= 1:
                continue
            idx = random.randint(0, len(net) - 1)
            candidate = net[:idx] + net[idx+1:]
            if fast_verify(candidate):
                net = candidate
                # After removal, try further removals
                net = greedy_remove(net, deadline)
        
        elif move_type == 'swap':
            # Try replacing one comparator with another
            if len(net) <= 1:
                continue
            idx = random.randint(0, len(net) - 1)
            # Generate a random new comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            candidate = net[:idx] + [(i, j)] + net[idx+1:]
            if fast_verify(candidate):
                net = candidate
                # After swap, try further removals
                net = greedy_remove(net, deadline)
        
        elif move_type == 'insert_remove':
            # Insert a random comparator, then do greedy removal
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(net))
            candidate = net[:pos] + [(i, j)] + net[pos:]
            # Try to remove comparators from the candidate
            candidate = greedy_remove(candidate, deadline)
            if len(candidate) < len(net):
                net = candidate
        
        if len(net) < best_count:
            best_net = list(net)
            best_count = len(net)
    
    return best_net

def construct():
    deadline = time.time() + 85  # 85s to leave 5s margin
    
    # Start from odd-even transposition
    net = odd_even_network()
    best_net = list(net)
    best_count = len(net)
    
    # Do local search from the initial network
    result = local_search(net, deadline)
    if len(result) < best_count:
        best_net = result
        best_count = len(result)
    
    # Try a few more restarts with different initial perturbations
    random.seed(123)
    for restart in range(3):
        if time.time() > deadline - 5:
            break
        
        # Start from the best network found so far, perturb it
        net = list(best_net)
        # Remove a few random comparators
        for _ in range(min(5, len(net))):
            if len(net) > 1:
                idx = random.randint(0, len(net) - 1)
                net = net[:idx] + net[idx+1:]
        
        # Do local search from the perturbed network
        result = local_search(net, deadline)
        if len(result) < best_count:
            best_net = result
            best_count = len(result)
    
    return best_net
