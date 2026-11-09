import time
import random

N = 17
TOTAL_MASKS = 1 << N  # 131072
DEADLINE_FRACTION = 0.95

def _build_seed():
    """Odd-even transposition network for N wires."""
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net

def _verify_network(net):
    """
    Returns True if the network sorts all 131,072 Boolean inputs.
    Uses Python big integers for speed.
    Each wire is represented by a Python int of TOTAL_MASKS bits.
    Bit k of wire_i represents the value of input k on wire i.
    """
    # Initialize wires
    # For mask m (0 to 2^N - 1), the input vector is the bits of m.
    # Wire i gets bit i of m.
    # So wire_i is an integer where bit k is 1 if (k >> i) & 1 == 1.
    # We can construct these integers efficiently.
    
    wires = [0] * N
    # For each bit position i, the pattern of 1s in the input masks is periodic.
    # Bit i of mask m is 1 for ranges of length 2^i, then 0 for 2^i, etc.
    # Instead of looping 131k times, we can construct the integer directly or use a loop if N is small enough.
    # Actually, a simple loop over masks is 131k iterations, which is fast in Python (~10ms).
    # But we can do better.
    # Let's just use the loop for initialization once, it's negligible compared to search.
    
    # Actually, we can construct the initial wires using bit manipulation tricks.
    # For bit i, the pattern is 0...0 1...1 0...0 1...1 ...
    # The period is 2^(i+1). The first half is 0, second half is 1.
    # This is equivalent to summing 1 << k for all k where (k >> i) & 1.
    
    # Let's just initialize with a loop for simplicity and correctness, it's done once.
    # Or even better, precompute the initial wires once globally? No, they are constant.
    
    # Precompute initial wires
    # wire[i] = sum(1 << k for k in range(TOTAL_MASKS) if (k >> i) & 1)
    
    # This loop is 17 * 131072 = ~2.2M operations. In Python this might take ~0.2s.
    # We can do it faster.
    # The integer for bit i is a pattern of 0s and 1s.
    # For i=0: 101010... (LSB is 1) -> 0x5555...
    # For i=1: 00110011... (LSB is 0, next 1) -> 0x3333...
    
    # Let's just use the loop. 0.2s is acceptable for a 60s budget.
    # But we can optimize:
    # The pattern for bit i repeats every 2^(i+1).
    # The block of 1s is 2^i long.
    # We can build the integer by repeating a pattern.
    
    # Actually, the simplest fast way:
    # wire[i] = 0
    # for k in range(TOTAL_MASKS):
    #     if (k >> i) & 1: wire[i] |= (1 << k)
    
    # Let's try to generate them via bit operations.
    # The value of bit i in mask k is (k >> i) & 1.
    # This is the i-th bit of the counter k.
    # The sequence of i-th bits for k=0..2^N-1 is a specific pattern.
    
    # Let's just precompute them once in a helper or at module level?
    # No, let's compute them inside the function but cache them?
    # The function is called many times. The initial wires are ALWAYS the same.
    # So we should compute them once.
    
    # I will compute them in construct() and pass them or store them in a global.
    # For this function signature, I'll compute them. To avoid recomputation, 
    # I'll use a global cache.
    
    global _INITIAL_WIRES
    if not hasattr(_verify_network, '_INITIAL_WIRES'):
        _verify_network._INITIAL_WIRES = [0] * N
        for i in range(N):
            w = 0
            # Pattern: 0 for 2^i, 1 for 2^i, repeat.
            # We can build this by setting bits.
            # For k in range(TOTAL_MASKS):
            #     if (k >> i) & 1: w |= 1 << k
            # This is O(N * 2^N). 17 * 131072 = 2.2M. 
            # Let's optimize this initialization.
            # The integer is a sum of blocks.
            # Block size 2^(i+1). First 2^i are 0, next 2^i are 1.
            # The value of the block of 1s is (2^(2^i) - 1) << (2^i).
            # Wait, no. The bits are at positions k.
            # The bits set are k such that (k >> i) & 1 == 1.
            # These k are in ranges [j*2^(i+1) + 2^i, j*2^(i+1) + 2^(i+1) - 1].
            # The sum of 1<<k for k in [A, B] is (2^(B+1) - 2^A).
            # So for each block j, the contribution is 2^(j*2^(i+1) + 2^(i+1)) - 2^(j*2^(i+1) + 2^i).
            # This is still a loop over 2^(N-i-1) blocks.
            
            # Given N=17, the total number of operations to build all wires is small enough 
            # to just do the naive loop once. 2.2M bit sets is fast (<50ms).
            for k in range(TOTAL_MASKS):
                if (k >> i) & 1:
                    w |= (1 << k)
            _verify_network._INITIAL_WIRES[i] = w
    
    wires = _verify_network._INITIAL_WIRES[:]
    
    # Apply network
    for i, j in net:
        wi = wires[i]
        wj = wires[j]
        # min goes to i, max goes to j
        # min(wi, wj) is bitwise AND? No.
        # For each bit k:
        # if wi_k == 1 and wj_k == 0: swap.
        # if wi_k == 0 and wj_k == 1: no swap (min is 0, max is 1, already correct? 
        # Wait. min(0,1)=0 to i, max(0,1)=1 to j. So no change.
        # if wi_k == 1 and wj_k == 1: min=1, max=1. No change.
        # if wi_k == 0 and wj_k == 0: min=0, max=0. No change.
        
        # So swap happens only if wi_k=1 and wj_k=0.
        # Let S = wi & ~wj. This has 1s where swap is needed.
        # New wi = wi ^ S (turn off the 1s that moved to j)
        # New wj = wj ^ S (turn on the 1s that moved from i)
        
        S = wi & ~wj
        wires[i] = wi ^ S
        wires[j] = wj ^ S
        
    # Check if sorted
    # Sorted mask for k ones is 0...0 1...1.
    # The value is (2^k - 1) << (N-k).
    # We need to check if the output wires match the expected sorted wires for each mask.
    # Expected sorted wires:
    # For a mask with popcount k, the wires should be:
    # wires[0..N-k-1] = 0
    # wires[N-k..N-1] = 1
    
    # We can construct the "expected" wires for all masks simultaneously?
    # Or we can check the condition:
    # For each wire i, the bit k should be 1 if and only if popcount(mask_k) > N-1-i.
    # i.e., popcount(mask_k) >= N-i.
    
    # Let's precompute the "expected" wires for sorted output.
    # expected_wires[i] should have bit k set if popcount(k) >= N-i.
    
    if not hasattr(_verify_network, '_EXPECTED_WIRES'):
        _verify_network._EXPECTED_WIRES = [0] * N
        for i in range(N):
            w = 0
            # Bit k is 1 if popcount(k) >= N-i
            # Let threshold = N-i
            # for k in range(TOTAL_MASKS):
            #     if k.bit_count() >= threshold: w |= 1 << k
            # This is another 2.2M operations.
            # We can do this once.
            for k in range(TOTAL_MASKS):
                if k.bit_count() >= (N - i):
                    w |= (1 << k)
            _verify_network._EXPECTED_WIRES[i] = w
    
    expected = _verify_network._EXPECTED_WIRES
    
    for i in range(N):
        if wires[i] != expected[i]:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 60 * DEADLINE_FRACTION
    
    # Start with seed
    current_net = _build_seed()
    
    # Verify seed (should be True)
    if not _verify_network(current_net):
        # Fallback if seed is broken (should not happen)
        return current_net
        
    best_net = current_net[:]
    
    # All possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Greedy Removal Strategy
    # Try to remove each comparator. If the network is still valid, keep it removed.
    # Repeat until no more removals are possible or time runs out.
    
    # To make it faster, we can randomize the order of removal attempts.
    # Or we can iterate through the list multiple times.
    
    # Optimization: 
    # Instead of re-verifying the whole network from scratch every time, 
    # we could maintain the state, but since removal changes the global flow, 
    # we must re-verify.
    # However, verification is ~0.1ms. 
    # 136 comparators * 131072 masks * bitwise ops.
    # Actually, the verification cost is O(L * N * (1/word_size)) roughly.
    # With big ints, it's very fast.
    
    # Let's estimate: 
    # 136 comparators.
    # Each comparator: ~5 big int ops.
    # 17 wires.
    # Total ops ~ 136 * 5 * 17 ~ 11k big int ops.
    # Big int ops on 131k bits (~16KB) are fast.
    # Probably < 0.5ms per verification.
    
    # We can do ~1000 verifications in 0.5s.
    # In 55s, we can do ~10,000 verifications.
    # Each verification corresponds to trying to remove one comparator.
    # So we can try ~10,000 removals.
    
    # Strategy:
    # 1. Initialize with seed.
    # 2. Loop:
    #    a. Pick a random index in current_net.
    #    b. Remove it.
    #    c. Verify.
    #    d. If valid, keep removed.
    #    e. If invalid, put it back.
    #    f. If no removals succeeded in a full pass, try to move/swap?
    #       Or just stop if greedy removal stagnates.
    #       To escape local minima, we can occasionally add a comparator? 
    #       But we want to minimize size. 
    #       If we are stuck, we might need to rearrange.
    #       But greedy removal is very powerful for sorting networks.
    
    # Let's add a "shuffle" mechanism if stuck.
    # If we can't remove anything, we can try swapping two comparators?
    # Swapping doesn't change size, so it doesn't help minimize size directly, 
    # but it might allow further removals.
    
    stuck_count = 0
    max_stuck = 1000 # Try 1000 random removal attempts before declaring stuck
    
    while time.time() < deadline:
        if len(current_net) == 0:
            break
            
        # Try to remove a random comparator
        idx = random.randint(0, len(current_net) - 1)
        removed_comp = current_net[idx]
        new_net = current_net[:idx] + current_net[idx+1:]
        
        if _verify_network(new_net):
            # Success
            current_net = new_net
            best_net = current_net[:]
            stuck_count = 0
        else:
            # Failure
            stuck_count += 1
            
            # If stuck for a while, try a different move?
            # For now, just keep trying random removals.
            # If we really get stuck, we might need to add a comparator to rearrange?
            # But adding increases size. 
            # Maybe we can do a "local search" on the order?
            # Let's stick to removal first. It's the most direct path to score improvement.
            
            # If stuck_count is high, we might be in a local minimum where no single removal works.
            # We could try removing 2 comparators at once? 
            # Or swapping?
            # Let's try swapping two comparators if stuck_count > 500.
            if stuck_count > 500 and len(current_net) >= 2:
                i1 = random.randint(0, len(current_net) - 1)
                i2 = random.randint(0, len(current_net) - 1)
                if i1 != i2:
                    current_net[i1], current_net[i2] = current_net[i2], current_net[i1]
                    # We don't verify after swap because size didn't change, 
                    # but it might enable future removals.
                    # We don't update best_net because size is same.
                    # Reset stuck_count? No, keep it high to keep trying swaps/removals.
                    # Actually, swapping is cheap. Let's just do it.
                    stuck_count = 0 # Reset to try removals again

    return best_net

# Pre-compute initial and expected wires to avoid overhead in first call?
# The function _verify_network handles caching.
# Let's trigger the cache initialization now.
_verify_network([])

if __name__ == "__main__":
    # Quick test
    net = construct()
    print(f"Found network with {len(net)} comparators")
    print("Valid:", _verify_network(net))
