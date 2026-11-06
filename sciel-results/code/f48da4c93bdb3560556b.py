"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 18

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s kill

    # 1. Generate all 2^N boolean inputs
    # We use the periodic tiling method for speed
    inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for col in range(N):
        period = 1 << (col + 1)
        half = 1 << col
        block = np.zeros(period, dtype=np.uint8)
        block[half:] = 1
        repeats = (1 << N) // period
        inputs[:, col] = np.tile(block, repeats)

    # 2. Fast Verification Setup
    # We will verify against all 2^N inputs for correctness, but use a faster internal loop structure.
    # For local search, we can sometimes use a subset for fast rejection, but given the small N=18,
    # full verification is ~37MB memory and fast enough with NumPy.
    
    state_buffer = np.empty_like(inputs)

    def verify(net):
        """Verify against all 2^N inputs. Returns True if sorted."""
        np.copyto(state_buffer, inputs)
        for i, j in net:
            mi = state_buffer[:, i] & state_buffer[:, j]
            ma = state_buffer[:, i] | state_buffer[:, j]
            state_buffer[:, i] = mi
            state_buffer[:, j] = ma
        # Check sortedness: no '10' pattern
        return np.all(state_buffer[:, :-1] <= state_buffer[:, 1:])

    def _odd_even_network():
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Start with odd-even transposition
    best_network = _odd_even_network()
    best_count = len(best_network)
    
    if not verify(best_network):
        return best_network

    # Pre-compute a list of all possible comparators for random selection
    all_comparators = [(i, j) for i in range(N) for j in range(i + 1, N)]
    num_comparators = len(all_comparators)

    def search(start_net, seed, time_budget):
        """
        Hybrid Local Search:
        1. Greedy Removal (Targeted)
        2. Simulated Annealing with Random Swap/Move/Remove
        """
        rng = np.random.RandomState(seed)
        current = list(start_net)
        current_count = len(current)
        best_local = list(current)
        best_local_count = current_count
        
        T = 1.0
        T_min = 0.05
        cooling_rate = 0.999
        t0 = time.time()
        last_improve = t0
        
        # Phase 1: Greedy Removal
        # Try to remove comparators. To make this efficient, we scan the network.
        # If removing a comparator breaks sorting, we skip it.
        # We iterate until no more removals are possible or time runs out.
        while time.time() < t0 + time_budget:
            progress = False
            # Iterate in reverse to handle index shifts naturally if we were modifying in place,
            # but we create a new list each time for simplicity in verification.
            for idx in range(len(current) - 1, -1, -1):
                if time.time() > t0 + time_budget:
                    break
                candidate = current[:idx] + current[idx+1:]
                if verify(candidate):
                    current = candidate
                    current_count -= 1
                    best_local = list(current)
                    best_local_count = current_count
                    last_improve = time.time()
                    progress = True
                    break # Restart scan from beginning/end as structure changed significantly
            
            if not progress:
                break # No more greedy removals possible

        # Phase 2: Simulated Annealing
        # Now we try to swap or move comparators to find better configurations that allow more removals.
        while time.time() < t0 + time_budget and T > T_min:
            if not current:
                break
            
            move_type = rng.random()
            
            if move_type < 0.5:
                # Remove a comparator (might fail, but good for exploration if we accept bad states? 
                # No, we only accept valid states in this simple SA. 
                # Actually, standard SA for sorting networks often allows invalid intermediate states 
                # to escape local minima, but verifying validity is expensive. 
                # Let's stick to valid-state-only search for simplicity and speed, 
                # relying on Swap/Move to navigate the valid landscape.)
                
                # Actually, if we only allow valid states, "Remove" is only accepted if valid.
                # This is effectively greedy removal again but random order.
                idx = rng.randint(len(current))
                candidate = list(current)
                candidate.pop(idx)
                
            elif move_type < 0.8:
                # Swap: Change one comparator to a different random one
                idx = rng.randint(len(current))
                # Pick a random comparator from the universe
                new_comp = all_comparators[rng.randint(num_comparators)]
                candidate = list(current)
                candidate[idx] = new_comp
                
            else:
                # Move: Remove one, insert at random position
                idx = rng.randint(len(current))
                comp = current[idx]
                candidate = list(current)
                candidate.pop(idx)
                insert_idx = rng.randint(len(candidate) + 1)
                candidate.insert(insert_idx, comp)

            # Verify candidate
            if not verify(candidate):
                T *= cooling_rate
                continue
                
            cand_count = len(candidate)
            
            # Acceptance criteria
            if cand_count < current_count:
                # Always accept improvement
                current = candidate
                current_count = cand_count
                if current_count < best_local_count:
                    best_local = list(current)
                    best_local_count = current_count
                    last_improve = time.time()
                T = 1.0 # Reset temperature on improvement
            else:
                # Same size or worse. 
                # If same size, we might accept to explore landscape.
                # If worse, we generally reject unless T is high enough.
                if cand_count == current_count:
                    # Accept neutral moves with probability 1.0 or high prob to allow drift
                    if rng.random() < 0.5:
                        current = candidate
                else:
                    # Worse size. Accept with probability exp(-delta/T)
                    # delta is increase in count
                    delta = cand_count - current_count
                    if rng.random() < np.exp(-delta / T):
                        current = candidate
                        current_count = cand_count
            
            T *= cooling_rate
            
            # Stagnation check
            if time.time() - last_improve > 2.0:
                # Restart from best local
                current = list(best_local)
                current_count = best_local_count
                T = 1.0
                last_improve = time.time()
                
        return best_local, best_local_count

    # Multi-restart search
    global_seed = 1
    while time.time() < deadline:
        time_left = deadline - time.time()
        if time_left < 2.0:
            break
            
        # Shorter restarts to allow more exploration
        time_budget = min(4.0, time_left)
        
        # Start from best known global
        start_net = best_network
        
        cand_net, cand_count = search(start_net, global_seed, time_budget)
        
        if cand_count < best_count:
            best_network = cand_net
            best_count = cand_count
            
        global_seed += 1

    # Final safety check
    if not verify(best_network):
        best_network = _odd_even_network()

    return best_network

# EVOLVE-BLOCK-END
