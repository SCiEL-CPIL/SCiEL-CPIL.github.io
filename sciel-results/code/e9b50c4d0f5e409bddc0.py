import numpy as np
import time
import random
import sys

# Set a seed for reproducibility if needed, but we can use random
random.seed(42)

N = 18

def construct():
    """
    Constructs a sorting network using a multi-phase approach:
    1. Generate a compact seed network using Batcher's Merge Sort.
    2. Aggressive Greedy Removal (backward pass, multiple iterations).
    3. Local Search (Simulated Annealing / Hill Climbing) to escape local minima.
       Moves include: remove one, add one, swap positions, replace pair.
    
    Verification is done bit-parallel using NumPy for speed.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # --- Helpers ---

    def build_merge_sort_network(n):
        """Recursive construction of Batcher's merge sort network."""
        if n <= 1:
            return []
        m = n // 2
        # Sort two halves
        left = build_merge_sort_network(m)
        # Offset the left network by m
        left_shifted = [(i + m, j + m) for i, j in left]
        right = build_merge_sort_network(n - m)
        
        # Merge stage
        # This creates a network that merges two sorted lists of size m
        # using the standard Batcher merge pattern.
        # The pattern is: 
        # (0, m), (1, m+1), ... for the first m elements against the second m?
        # No, the standard merge network for size 2m is defined recursively.
        
        # Let's use a simpler, known efficient construction for small N.
        # Actually, let's implement the recursive merge directly.
        
        def merge_network(m):
            """Network to merge two sorted lists of length m into one sorted list of length 2m."""
            if m == 1:
                return [(0, 1)]
            k = m // 2
            # The structure is complex. Let's use a different seed:
            # Shell sort or Odd-Even Transposition is safer and known.
            # But we want something smaller.
            # Let's stick with Odd-Even Transposition but try to optimize the seed generation.
            pass
        
        # Actually, let's just use the Odd-Even Transposition seed but generated cleanly.
        # It's guaranteed correct.
        net = []
        for phase in range(n):
            for i in range(phase % 2, n - 1, 2):
                net.append((i, i + 1))
        return net

    # Initialize Seed
    # Let's try to get a better seed.
    # A known good network for N=18?
    # Let's just start with Odd-Even Transposition. It's 153 comparators.
    # Or we can try to use a "pre-sorted" assumption? No.
    # Let's use the Odd-Even Transposition.
    
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))

    # Setup Verification
    size = 1 << N
    # Use uint8 arrays for states
    k_range = np.arange(size, dtype=np.uint64)
    states = np.empty((N, size), dtype=np.uint8)
    for j in range(N):
        states[j] = ((k_range >> j) & 1).astype(np.uint8)

    def is_sorted():
        # Check if wire i <= wire i+1 for all i
        # Equivalent to: no 0 followed by 1 in any wire pair (vertical)
        # Wait, "wire i <= wire i+1" means for every input k: states[i][k] <= states[i+1][k]
        # Violation: states[i][k] == 1 and states[i+1][k] == 0
        # So check if there is any k where states[i][k] > states[i+1][k]
        # This is fast with numpy.
        return not np.any(states[:-1] > states[1:])

    def apply_comparator(net_states, i, j):
        # Apply (i, j) to net_states in place
        si = net_states[i]
        sj = net_states[j]
        net_states[i] = si & sj
        net_states[j] = si | sj

    def revert_comparator(net_states, i, j, si_copy, sj_copy):
        net_states[i] = si_copy
        net_states[j] = sj_copy

    def evaluate_network(net):
        """
        Returns a copy of the final state after applying the network.
        If the network sorts, is_sorted() on the result will be True.
        We return the result array to check.
        """
        # Make a copy of the initial states
        s = states.copy()
        for i, j in net:
            apply_comparator(s, i, j)
        return s

    # --- Phase 1: Greedy Removal ---
    # Start with the seed network.
    # We maintain the state of the network after applying all comparators up to index `idx`.
    # Actually, it's easier to re-evaluate if we do complex moves, but for simple removal from the end,
    # we can maintain the state incrementally if we traverse backwards carefully.
    # However, removal in the middle shifts indices.
    
    # Strategy:
    # 1. Try to remove each comparator from back to front.
    # 2. If valid, remove.
    # 3. Repeat until no change or time limit.
    
    current_net = network[:]
    
    # Optimization: We can compute the full state once, then "undo" from the back?
    # No, undoing a min/max is not trivial without storing previous states.
    # So we just re-simulate? That's O(N_comp * N_comp) if we re-simulate from scratch each time.
    # 153 comparators. Re-simulating 153 times is 153*153 = 23k ops. Negligible.
    # Even with NumPy overhead, it's fast.
    
    def try_remove(idx):
        """
        Tries to remove the comparator at index idx from current_net.
        Returns True if successful.
        Modifies current_net and global 'current_state' if we were tracking it.
        Since we aren't tracking state incrementally in this simple loop, we just re-verify.
        """
        # Save
        saved = current_net[idx]
        # Temporarily remove
        del current_net[idx]
        
        # Verify
        # We can optimize verification by checking if the removed comparator is actually doing anything
        # in the current context, but full verification is safer.
        # Full verification:
        s = states.copy()
        for i, j in current_net:
            apply_comparator(s, i, j)
        valid = is_sorted() # Check on s
        
        if valid:
            return True
        else:
            # Restore
            current_net.insert(idx, saved)
            return False

    # Greedy Loop
    # We will do several passes.
    # To be faster, we can try to remove multiple? No, sequential is standard.
    
    # Let's implement a faster "backward greedy" where we maintain the state.
    # State = output of net[0:idx]
    # If we remove net[idx], we need to know if net[0:idx] + net[idx+1:] sorts.
    # This requires re-running the tail.
    
    # Let's just use the simple "re-evaluate from scratch" if it's fast enough.
    # 153 comps. Evaluation is ~153 * (2 * 262144 ops). 
    # NumPy ops on 262144 uint8s are very fast (~ms).
    # So ~153ms per full evaluation? Maybe less.
    # 10 passes * 153 comps * 153ms = 2.3 seconds.
    # This is very fast. We can afford many more passes or complex searches.
    
    # Let's refine the verification to be as fast as possible.
    # We can break early? No, need to process all comparators.
    
    # Let's run the greedy removal loop.
    last_improvement_time = start_time
    pass_count = 0
    while time.time() < deadline - 10.0: # Leave 10s for phase 2
        pass_count += 1
        removed_any = False
        # Try removing from the end
        for idx in range(len(current_net) - 1, -1, -1):
            if try_remove(idx):
                removed_any = True
                last_improvement_time = time.time()
                # If we removed one, the list shrinks, loop continues to next idx (which is now previous)
                # Since we iterate backwards, `idx` decreases.
                # But `try_remove` pops. 
                # If we pop at idx, the next iteration checks idx-1 (which is now at the end? No).
                # Example: [A, B, C]. idx=2 (C). Remove C. List [A, B]. Next idx=1 (B). Correct.
                pass
        
        if not removed_any:
            break

    # --- Phase 2: Local Search (Simulated Annealing) ---
    # Now we try to escape local minima.
    # Moves:
    # 1. Add a random comparator (i, j) at a random position.
    # 2. Remove a random comparator.
    # 3. Swap two adjacent comparators.
    # 4. Move a comparator to a different position.
    
    # Objective: Minimize length.
    # Score = len(net)
    
    current_net = current_net[:]
    best_net = current_net[:]
    best_score = len(best_net)
    
    # Temperature parameters
    T = 1.0
    cooling = 0.99
    
    # Time budget for SA: e.g., 50% of remaining time
    sa_deadline = time.time() + (deadline - time.time()) * 0.5
    
    while time.time() < sa_deadline:
        T *= cooling
        
        # Pick a move
        move_type = random.randint(0, 3)
        
        # Save original
        orig_net = current_net[:]
        new_net = orig_net[:]
        
        # Apply move
        if move_type == 0: # Remove
            if len(new_net) > 0:
                idx = random.randint(0, len(new_net) - 1)
                del new_net[idx]
            else:
                continue
        elif move_type == 1: # Add
            i = random.randint(0, N - 2)
            j = i + random.randint(1, N - 1 - i)
            pos = random.randint(0, len(new_net))
            new_net.insert(pos, (i, j))
        elif move_type == 2: # Swap Adjacent
            if len(new_net) > 1:
                idx = random.randint(0, len(new_net) - 2)
                new_net[idx], new_net[idx+1] = new_net[idx+1], new_net[idx]
            else:
                continue
        else: # Move
            if len(new_net) > 1:
                idx_from = random.randint(0, len(new_net) - 1)
                idx_to = random.randint(0, len(new_net) - 1)
                if idx_from != idx_to:
                    comp = new_net.pop(idx_from)
                    new_net.insert(idx_to, comp)
                else:
                    continue
            else:
                continue
        
        # Evaluate
        # Cost is len(new_net). 
        # If len is larger, only accept if probability allows (to escape).
        # If len is smaller, always accept.
        
        s = states.copy()
        for i, j in new_net:
            apply_comparator(s, i, j)
        
        valid = is_sorted()
        
        if valid:
            new_score = len(new_net)
            delta = new_score - len(current_net)
            
            if delta < 0:
                # Better
                current_net = new_net
                if new_score < best_score:
                    best_score = new_score
                    best_net = current_net[:]
                    last_improvement_time = time.time()
            else:
                # Worse or equal
                # Accept with probability exp(-delta / T)
                if delta == 0:
                    # Keep it to explore
                    if random.random() < 0.5:
                        current_net = new_net
                else:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        current_net = new_net
        
    return best_net

# To ensure the function works even if something goes wrong, wrap in try/except
# and return the seed if all else fails.

def construct_safe():
    try:
        return construct()
    except Exception as e:
        print(f"Error in construct: {e}", file=sys.stderr)
        # Fallback: Odd-Even Transposition
        net = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                net.append((i, i + 1))
        return net

# The task requires `construct` to be the entry point.
# I will rename the internal function to `_construct_impl` and make `construct` call it with safety.

def _construct_impl():
    start_time = time.time()
    deadline = start_time + 85.0

    # --- Helpers ---
    
    def is_sorted(states):
        return not np.any(states[:-1] > states[1:])

    def apply_comparator(net_states, i, j):
        si = net_states[i]
        sj = net_states[j]
        net_states[i] = si & sj
        net_states[j] = si | sj

    def try_remove(net, idx):
        saved = net[idx]
        del net[idx]
        s = initial_states.copy()
        for i, j in net:
            apply_comparator(s, i, j)
        valid = is_sorted(s)
        if valid:
            return True
        else:
            net.insert(idx, saved)
            return False

    def try_move(net, idx_from, idx_to):
        comp = net[idx_from]
        del net[idx_from]
        # Adjust index if from < to
        if idx_from < idx_to:
            idx_to -= 1
        net.insert(idx_to, comp)
        s = initial_states.copy()
        for i, j in net:
            apply_comparator(s, i, j)
        valid = is_sorted(s)
        if valid:
            return True
        else:
            # Revert: this is hard if we don't save the list.
            # Better to implement moves with explicit list copying in the SA loop instead of in-place modifications that require complex reverts.
            pass
        # This approach is messy. Let's stick to list copying in SA.

    # Initialization
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    
    size = 1 << N
    k_range = np.arange(size, dtype=np.uint64)
    initial_states = np.empty((N, size), dtype=np.uint8)
    for j in range(N):
        initial_states[j] = ((k_range >> j) & 1).astype(np.uint8)

    current_net = network[:]
    
    # Phase 1: Greedy Removal
    # We use a helper that returns True if removal is valid.
    def greedy_remove():
        improved = True
        while improved and time.time() < deadline - 15.0:
            improved = False
            # Copy for trial
            for idx in range(len(current_net) - 1, -1, -1):
                # Try remove
                comp = current_net[idx]
                del current_net[idx]
                
                s = initial_states.copy()
                for i, j in current_net:
                    apply_comparator(s, i, j)
                
                if is_sorted(s):
                    improved = True
                else:
                    current_net.insert(idx, comp)
        
    greedy_remove()
    
    # Phase 2: Simulated Annealing
    best_net = current_net[:]
    best_score = len(best_net)
    
    T = 1.0
    cooling = 0.995
    sa_deadline = time.time() + (deadline - time.time()) * 0.8 # Use 80% of remaining for SA
    
    while time.time() < sa_deadline:
        T *= cooling
        
        move_type = random.randint(0, 3)
        
        new_net = current_net[:]
        
        # Apply move
        if move_type == 0: # Remove
            if len(new_net) == 0: continue
            idx = random.randint(0, len(new_net) - 1)
            del new_net[idx]
        elif move_type == 1: # Add
            i = random.randint(0, N - 2)
            j = i + random.randint(1, N - 1 - i)
            pos = random.randint(0, len(new_net))
            new_net.insert(pos, (i, j))
        elif move_type == 2: # Swap Adjacent
            if len(new_net) < 2: continue
            idx = random.randint(0, len(new_net) - 2)
            new_net[idx], new_net[idx+1] = new_net[idx+1], new_net[idx]
        else: # Move
            if len(new_net) < 2: continue
            idx_from = random.randint(0, len(new_net) - 1)
            idx_to = random.randint(0, len(new_net) - 1)
            if idx_from == idx_to: continue
            comp = new_net.pop(idx_from)
            new_net.insert(idx_to, comp)
        
        # Evaluate
        s = initial_states.copy()
        for i, j in new_net:
            apply_comparator(s, i, j)
        valid = is_sorted(s)
        
        if valid:
            new_score = len(new_net)
            old_score = len(current_net)
            delta = new_score - old_score
            
            if delta < 0:
                current_net = new_net
                if new_score < best_score:
                    best_score = new_score
                    best_net = current_net[:]
            else:
                if delta == 0:
                    if random.random() < 0.3: # Slight bias to stay
                        current_net = new_net
                else:
                    # T should be scaled to the problem size?
                    # If T is too high, we accept too many bad moves.
                    # If T is too low, we get stuck.
                    # Delta is integer.
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        current_net = new_net

    return best_net

def construct():
    try:
        return _construct_impl()
    except Exception as e:
        print(f"Construct failed: {e}", file=sys.stderr)
        net = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                net.append((i, i + 1))
        return net

if __name__ == "__main__":
    res = construct()
    print(f"Comparators: {len(res)}")
