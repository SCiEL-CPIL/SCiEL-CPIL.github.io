"""Size-efficient sorting network search for N=16 wires.

Strategy:
1. Use bit-parallel zero-one verification for fast local search.
2. Start from a good baseline (odd-even transposition, 120 comparators).
3. Aggressive greedy removal to get down quickly.
4. Then do a more sophisticated local search:
   - Try swapping adjacent comparators.
   - Try inserting a comparator and then removing others (kick-start).
   - Try moving a comparator to a different position.
5. Keep the best network found within the time budget.
"""

import time
import random

N = 16
TOTAL_INPUTS = 1 << N  # 65536
ALL_ONES = (1 << TOTAL_INPUTS) - 1

# Pre-compute initial states for all 16 wires
_initial_states = None

def _get_initial_states():
    global _initial_states
    if _initial_states is None:
        _initial_states = []
        for i in range(N):
            val = 0
            period = 1 << (i + 1)
            half = 1 << i
            period_val = ((1 << half) - 1) << half
            num_periods = TOTAL_INPUTS // period
            shift = 0
            for _ in range(num_periods):
                val |= period_val << shift
                shift += period
            _initial_states.append(val)
    return _initial_states

def fast_verify(network):
    """Verify a sorting network using bit-parallel zero-one principle."""
    state = _get_initial_states()[:]
    
    for (i, j) in network:
        min_val = state[i] & state[j]
        max_val = state[i] | state[j]
        state[i] = min_val
        state[j] = max_val
    
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
    return True

def odd_even_network():
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def batcher_network():
    """Generate Batcher's bitonic sorting network for N=16."""
    # Bitonic sort for n=16 (2^4)
    # Stages: for k in 1..log2(n):
    #   for each block of size 2^k:
    #     merge two halves of size 2^(k-1)
    # The standard bitonic sorter has n/2 * log2(n) * (log2(n)+1) / 2 comparators?
    # Actually let me just build it recursively.
    
    def bitonic_sort(k, dir):
        """Return list of comparators for a bitonic sort of 2^k elements starting at position 0, direction dir (1=asc, -1=desc)."""
        if k == 1:
            return []
        
        # First, sort the two halves in opposite directions
        left = bitonic_sort(k-1, -dir)
        right = bitonic_sort(k-1, dir)
        
        # Then merge
        merge = []
        for i in range(2**(k-1)):
            a = i
            b = i + 2**(k-1)
            if dir == 1:
                merge.append((a, b))
            else:
                merge.append((b, a))
        
        # Actually, the bitonic merge is more complex. Let me use the standard construction.
        # For a bitonic merge of two sorted sequences (one asc, one desc) of length m:
        # For each i in 0..m-1: compare (i, i+m)
        # Then recursively merge the two halves.
        
        # Let me redo this properly.
        return left + right + merge
    
    # Actually, let me just use a known simple construction.
    # For n=16, the bitonic sorter has 4 stages.
    # Stage 1: compare (0,1),(2,3),(4,5),...,(14,15) [8 comparators]
    # Stage 2: compare (0,2),(1,3),(4,6),(5,7),...,(12,14),(13,15) [8 comparators]
    #          then compare (0,4),(1,5),(2,6),(3,7),... [8 comparators]
    # Stage 3: similar pattern for 8-element blocks
    # Stage 4: merge 8+8
    
    # This is getting complicated. Let me just use odd-even and rely on local search.
    return odd_even_network()

def construct():
    deadline = time.time() + 85  # 85s to leave 5s margin
    random.seed(42)
    
    # Start from odd-even transposition
    net = odd_even_network()
    best_net = list(net)
    best_count = len(net)
    
    # Phase 1: Greedy removal until stuck
    def try_remove_all(net):
        """Try to remove each comparator; repeat until no more removals possible."""
        changed = True
        while changed:
            changed = False
            for idx in range(len(net) - 1, -1, -1):
                candidate = net[:idx] + net[idx+1:]
                if fast_verify(candidate):
                    net = candidate
                    changed = True
                    break
        return net
    
    net = try_remove_all(net)
    best_net = list(net)
    best_count = len(net)
    
    # Phase 2: Local search with various moves
    # Moves:
    # 1. Remove a comparator (if possible)
    # 2. Insert a comparator at a random position, then try to remove others
    # 3. Swap two adjacent comparators
    # 4. Move a comparator to a different position
    
    def try_remove_one(net, start_idx=0):
        """Try to remove a single comparator from net, starting search from start_idx."""
        for idx in range(start_idx, len(net)):
            candidate = net[:idx] + net[idx+1:]
            if fast_verify(candidate):
                return candidate
        return net
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        move_type = random.randint(0, 3)
        
        if move_type == 0:
            # Try to remove a comparator
            candidate = try_remove_one(net)
            if len(candidate) < len(net):
                net = candidate
                best_net = list(net)
                best_count = len(net)
        
        elif move_type == 1:
            # Insert a random comparator, then try to remove others
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(net))
            candidate = net[:pos] + [(i, j)] + net[pos:]
            
            # Try to remove comparators from the candidate
            for _ in range(3):  # Try a few removals
                candidate = try_remove_one(candidate)
                if len(candidate) <= len(net):
                    break
            
            if len(candidate) < len(net):
                net = candidate
                best_net = list(net)
                best_count = len(net)
        
        elif move_type == 2:
            # Swap two adjacent comparators
            if len(net) < 2:
                continue
            idx = random.randint(0, len(net) - 2)
            candidate = net[:]
            candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
            if fast_verify(candidate):
                net = candidate
                # No improvement in count, but might enable future removals
        
        elif move_type == 3:
            # Move a comparator to a different position
            if len(net) < 2:
                continue
            src = random.randint(0, len(net) - 1)
            dst = random.randint(0, len(net) - 1)
            if src == dst:
                continue
            candidate = net[:]
            comp = candidate.pop(src)
            candidate.insert(dst, comp)
            if fast_verify(candidate):
                # Try to remove some comparators
                for _ in range(2):
                    candidate = try_remove_one(candidate)
                    if len(candidate) < len(net):
                        break
                if len(candidate) < len(net):
                    net = candidate
                    best_net = list(net)
                    best_count = len(net)
                else:
                    net = candidate
    
    return best_net
