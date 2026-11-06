"""Size-efficient sorting network search for N=16 wires.

Strategy:
1. Start from the known optimal 60-comparator AKS-style / hand-crafted 16-wire network if we can find one quickly,
   otherwise start from a smaller known-good network (e.g., odd-even 120) and aggressively optimize.
2. Use local search: try removing each comparator and verifying; try inserting a comparator at a random position
   and see if it enables further removals.
3. Use the zero-one principle for verification: a network sorts all inputs iff it sorts all 2^16 = 65536 boolean inputs.
   We can speed this up with bit-parallel simulation: represent each wire's value across many inputs as a bitmask.

Given the 90s wall-clock limit, we need to be very careful with verification cost.
Verifying one network against all 65536 boolean inputs naively is 65536 * C comparators, which for C~60 is ~4M ops,
which is fine in Python if we use bit-parallelism.

Bit-parallel verification:
- Represent the state of all 16 wires for all 65536 inputs as a list of 16 integers, each 65536 bits wide (i.e., Python ints).
- Initially, wire i has bit j set if input j has a 1 in position i. Input j is a 16-bit number, so bit i of j indicates wire i's value.
- For a comparator (i, j): 
    min_val = state[i] & state[j]
    max_val = state[i] | state[j]
    state[i] = min_val
    state[j] = max_val
- At the end, check that wire 0 is all 0s (state[0] == 0) and wire 15 is all 1s (state[15] == (1 << 65536) - 1), and more generally that the wires are sorted: state[i] should be a prefix of 0s followed by 1s, but actually for a sorting network, the output must be sorted for every input. With bit-parallel, we can check that for each wire i, state[i] is monotone: actually, the zero-one principle says we just need to check that the output is sorted for each boolean input. In bit-parallel form, the output on wire i for input j is bit j of state[i]. The output is sorted for input j iff for all i, bit j of state[i] <= bit j of state[i+1]. This is equivalent to: state[i] & ~state[i+1] == 0 for all i (i.e., wherever wire i has a 1, wire i+1 also has a 1). So we check: for all i in 0..14, (state[i] & ~state[i+1]) == 0.

This is very fast: 16 AND/NOT operations on 65536-bit integers, plus 60 comparator updates (each with AND/OR). This should take well under 1 second.

So we can do many local search steps within the time budget.

Plan:
- Define a fast verify(network) function using bit-parallelism.
- Start from a known 60-comparator network if we can hardcode it. Since I don't have it memorized, I'll start from a smaller baseline. Actually, the problem states "A verified 60-comparator construction is the reference". I should try to find a 60-comparator network. But I don't have one hardcoded. Let me start from odd-even (120) and try to reduce it.

Actually, let me think about known results. The optimal number of comparators for a 16-wire sorting network is known to be 60 (or is it 61? The problem says 60 is the reference). Let me just do a local search from the odd-even network and see how far we can get.

Given the time constraint, I'll:
1. Implement fast bit-parallel verification.
2. Start from odd-even transposition (120 comparators).
3. Try to remove each comparator one by one (greedy removal).
4. Then do a more sophisticated local search: try swapping, inserting, etc.

But 120 -> 60 is a big gap. Greedy removal might get stuck. Let me also try starting from a different initial network, like a batcher's bitonic sorter, which for 16 wires uses 16*log2(16)*log2(16)/4 = 16*4*4/4 = 64 comparators? Let me compute: batcher's bitonic sorter for n wires uses n/2 * log2(n) * (log2(n)+1) / 2? Actually, the number of comparators in a bitonic sorting network for n=2^k is n/2 * k * (k+1) / 2? No, let me recall: a bitonic sorter for 2^k elements has k stages, and the total number of comparators is (n/2) * k * (k+1) / 2? For n=16, k=4: (16/2) * 4 * 5 / 2 = 8 * 10 = 80. Hmm, that's more than 60.

Actually, I recall that the optimal sorting network for 16 wires is indeed 60 comparators, and it was found by exhaustive search or clever construction. Since I can't hardcode it, I'll do a local search.

Let me implement:
- fast_verify(network): returns True if the network sorts all 65536 boolean inputs.
- local_search(network): try to remove each comparator; if removal keeps it valid, remove it. Repeat until no more removals possible.
- Then try to add a comparator at a random position and see if it enables further removals (this is a form of simulated annealing / random restart).

Given the 90s limit, I'll:
1. Implement fast bit-parallel verification.
2. Start from odd-even transposition (120 comparators).
3. Try to remove each comparator one by one (greedy removal).
4. Then do a more sophisticated local search: try swapping, inserting, etc.

But 120 -> 60 is a big gap. Greedy removal might get stuck. Let me also try starting from a different initial network, like a batcher's bitonic sorter, which for 16 wires uses 16*log2(16)*log2(16)/4 = 16*4*4/4 = 64 comparators? Let me compute: batcher's bitonic sorter for n wires uses n/2 * log2(n) * (log2(n)+1) / 2? Actually, the number of comparators in a bitonic sorting network for n=2^k is n/2 * k * (k+1) / 2? No, let me recall: a bitonic sorter for 2^k elements has k stages, and the total number of comparators is (n/2) * k * (k+1) / 2? For n=16, k=4: (16/2) * 4 * 5 / 2 = 8 * 10 = 80. Hmm, that's more than 60.

Actually, I recall that the optimal sorting network for 16 wires is indeed 60 comparators, and it was found by exhaustive search or clever construction. Since I can't hardcode it, I'll do a local search.

Let me implement:
- fast_verify(network): returns True if the network sorts all 65536 boolean inputs.
- local_search(network): try to remove each comparator; if removal keeps it valid, remove it. Repeat until no more removals possible.
- Then try to add a comparator at a random position and see if it enables further removals (this is a form of simulated annealing / random restart).

Given the 90s limit, I'll do a time-budgeted search.
"""

import time
import random

N = 16
ALL_ONES = (1 << (1 << N)) - 1  # 65536 ones

# Pre-compute initial states for speed
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
            num_periods = (1 << N) // period
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

def construct():
    deadline = time.time() + 85  # 85s to leave 5s margin
    start_time = time.time()
    
    # Start from odd-even transposition
    net = odd_even_network()
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
    
    # Phase 1: Greedy removal from odd-even
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
    for iteration in range(2000):
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
                # Try to remove after swap
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
