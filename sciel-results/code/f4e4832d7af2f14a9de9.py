import time
import random

N = 16
NUM_INPUTS = 1 << N

# Pre-compute initial states for bit-parallel verification
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
            num_periods = NUM_INPUTS // period
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
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def bitonic_sorter():
    """Batcher's bitonic sorting network for N=16 wires."""
    comps = []
    
    def bitonic_merge(start, size, direction):
        """Merge a bitonic sequence of given size starting at start."""
        if size <= 1:
            return
        half = size // 2
        for i in range(half):
            if direction:
                comps.append((start + i, start + i + half))
            else:
                comps.append((start + i + half, start + i))
        bitonic_merge(start, half, direction)
        bitonic_merge(start + half, half, direction)
    
    def bitonic_sort(start, size, direction):
        """Sort a sequence of given size starting at start."""
        if size <= 1:
            return
        half = size // 2
        bitonic_sort(start, half, True)
        bitonic_sort(start + half, half, False)
        bitonic_merge(start, size, direction)
    
    bitonic_sort(0, N, True)
    return comps

def construct():
    deadline = time.time() + 85
    start_time = time.time()
    
    # Start from bitonic sorter (80 comparators for N=16)
    net = bitonic_sorter()
    best_net = list(net)
    best_count = len(net)
    
    def greedy_remove(net, time_limit=None):
        """Aggressively try to remove each comparator; return minimized net."""
        changed = True
        while changed:
            changed = False
            for idx in range(len(net) - 1, -1, -1):
                if time_limit and time.time() > time_limit:
                    return net
                candidate = net[:idx] + net[idx+1:]
                if fast_verify(candidate):
                    net = candidate
                    changed = True
                    break
        return net
    
    # Phase 1: Greedy removal from bitonic
    net = greedy_remove(net, deadline)
    best_net = list(net)
    best_count = len(net)
    
    random.seed(42)
    
    # Phase 2: Local search with swap, move, and insert operations
    def try_swap(net, idx1, idx2):
        """Swap two comparators and check if still valid."""
        if idx1 == idx2:
            return None
        candidate = list(net)
        candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
        if fast_verify(candidate):
            return candidate
        return None
    
    def try_move(net, from_idx, to_idx):
        """Move a comparator from one position to another."""
        if from_idx == to_idx:
            return None
        comp = net[from_idx]
        candidate = net[:from_idx] + net[from_idx+1:]
        if to_idx > from_idx:
            to_idx -= 1
        candidate = candidate[:to_idx] + [comp] + candidate[to_idx:]
        if fast_verify(candidate):
            return candidate
        return None
    
    def try_insert_remove(net, i, j, pos):
        """Insert a comparator and then greedily remove."""
        candidate = net[:pos] + [(i, j)] + net[pos:]
        candidate = greedy_remove(candidate, deadline)
        return candidate
    
    # Local search loop
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        op_type = random.random()
        
        if op_type < 0.4:
            # Swap two comparators
            if len(net) < 2:
                continue
            idx1 = random.randint(0, len(net) - 1)
            idx2 = random.randint(0, len(net) - 1)
            candidate = try_swap(net, idx1, idx2)
            if candidate is not None:
                candidate = greedy_remove(candidate, deadline)
                if len(candidate) < len(net):
                    net = candidate
                    best_net = list(net)
                    best_count = len(net)
        
        elif op_type < 0.7:
            # Move a comparator
            if len(net) < 2:
                continue
            from_idx = random.randint(0, len(net) - 1)
            to_idx = random.randint(0, len(net) - 1)
            candidate = try_move(net, from_idx, to_idx)
            if candidate is not None:
                candidate = greedy_remove(candidate, deadline)
                if len(candidate) < len(net):
                    net = candidate
                    best_net = list(net)
                    best_count = len(net)
        
        else:
            # Insert a random comparator, then greedily remove
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(net))
            candidate = try_insert_remove(net, i, j, pos)
            if len(candidate) < len(net):
                net = candidate
                best_net = list(net)
                best_count = len(net)
    
    return best_net
