import time
import numpy as np

N = 18
TIME_LIMIT = 80  # Safety margin for 90s limit

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Pre-compute bit-parallel states for fast verification
    # Number of 64-bit blocks needed for 2^18 inputs
    B = 2**N // 64  # 4096
    
    # states is a flat array of size N * B
    # Wire w is stored in indices [w*B, (w+1)*B)
    states = np.zeros(N * B, dtype=np.uint64)
    
    # Fill states
    # k is input index 0 to 2^N - 1
    # Wire w is 1 if (k >> w) & 1
    # We fill block by block
    for k in range(2**N):
        block_idx = k >> 6
        bit_idx = k & 63
        mask = 1 << bit_idx
        for w in range(N):
            if (k >> w) & 1:
                states[w * B + block_idx] |= mask

    def _fast_verify(comparators):
        """
        Verify network using bit-parallel simulation on flat array.
        Returns True if network sorts all inputs.
        """
        current = states.copy()
        
        for i, j in comparators:
            si = i * B
            sj = j * B
            a = current[si:si+B]
            b = current[sj:sj+B]
            # min goes to i, max goes to j
            current[si:si+B] = a & b
            current[sj:sj+B] = a | b
            
        # Check if sorted
        for w in range(N - 1):
            sw = w * B
            sw1 = (w + 1) * B
            if np.any(current[sw:sw+B] & ~current[sw1:sw1+B]):
                return False
        return True

    best_network = _odd_even_transposition()
    best_len = len(best_network)
    
    # Verify seed
    if not _fast_verify(best_network):
        return best_network

    current_network = list(best_network)
    
    def greedy_reduce(net):
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            idx = 0
            while idx < len(net):
                if time.time() > deadline - 5:
                    return net
                trial = net[:idx] + net[idx+1:]
                if _fast_verify(trial):
                    net = trial
                    changed = True
                    # Do not increment idx, check the new element at idx
                else:
                    idx += 1
        return net

    current_network = greedy_reduce(current_network)
    if len(current_network) < best_len:
        best_network = current_network
        best_len = len(current_network)

    # Phase 2: Local Search with Restarts
    np.random.seed(42)
    
    # Helper to create a perturbed network for restart
    def perturb(net, amount=5):
        if len(net) <= amount:
            return list(net)
        # Remove 'amount' random comparators
        indices = np.random.choice(len(net), size=amount, replace=False)
        # Sort indices to remove correctly
        indices = np.sort(indices)
        new_net = []
        remove_set = set(indices)
        for i, comp in enumerate(net):
            if i not in remove_set:
                new_net.append(comp)
        return new_net

    while time.time() < deadline - 10:
        # Try to improve current_network
        if len(current_network) > 1:
            # Random Swap
            idx = np.random.randint(len(current_network) - 1)
            trial = current_network.copy()
            trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
            if _fast_verify(trial):
                current_network = trial
                current_network = greedy_reduce(current_network)
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)
        
        if len(current_network) > 0 and time.time() < deadline - 10:
            # Wire Mutation
            idx = np.random.randint(len(current_network))
            i, j = current_network[idx]
            a, b = np.random.choice(N, 2, replace=False)
            if a > b:
                a, b = b, a
            if a == i and b == j:
                continue
            trial = current_network.copy()
            trial[idx] = (a, b)
            if _fast_verify(trial):
                current_network = trial
                current_network = greedy_reduce(current_network)
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)
        
        # Periodic Restart: Perturb the best network found so far
        if time.time() < deadline - 10 and len(best_network) > 0:
            # Every few seconds, or if stuck, restart from best
            # Simple heuristic: if we haven't improved best in last 5s? 
            # For simplicity, just do a restart from best every 5 seconds.
            if int(time.time() - start_time) % 5 == 0:
                 # This check is imprecise. Better to track last_improve_time
                 pass

    # Better Restart Strategy:
    # We'll restructure the loop slightly to allow explicit restarts.
    
    last_improve_time = start_time
    last_best_len = best_len
    
    while time.time() < deadline - 10:
        # If stuck for 3 seconds, restart from best with perturbation
        if time.time() - last_improve_time > 3.0:
            # Perturb best network
            perturbed = perturb(best_network, amount=5)
            current_network = greedy_reduce(perturbed)
            if len(current_network) < best_len:
                best_network = list(current_network)
                best_len = len(current_network)
                last_improve_time = time.time()
                last_best_len = best_len
            else:
                # If perturbation didn't yield improvement immediately, 
                # just use the perturbed one as current and continue search
                current_network = perturbed
                # Try a few local steps on this new current
                for _ in range(50):
                    if time.time() > deadline - 10: break
                    if len(current_network) > 1:
                        idx = np.random.randint(len(current_network) - 1)
                        trial = current_network.copy()
                        trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
                        if _fast_verify(trial):
                            current_network = trial
                            current_network = greedy_reduce(current_network)
                            if len(current_network) < best_len:
                                best_network = list(current_network)
                                best_len = len(current_network)
                                last_improve_time = time.time()
                                last_best_len = best_len
                                break # Restart search on this new best
                    if len(current_network) > 0:
                        idx = np.random.randint(len(current_network))
                        i, j = current_network[idx]
                        a, b = np.random.choice(N, 2, replace=False)
                        if a > b:
                            a, b = b, a
                        if a == i and b == j:
                            continue
                        trial = current_network.copy()
                        trial[idx] = (a, b)
                        if _fast_verify(trial):
                            current_network = trial
                            current_network = greedy_reduce(current_network)
                            if len(current_network) < best_len:
                                best_network = list(current_network)
                                best_len = len(current_network)
                                last_improve_time = time.time()
                                last_best_len = best_len
                                break

        # Normal local search step
        if len(current_network) > 1:
            idx = np.random.randint(len(current_network) - 1)
            trial = current_network.copy()
            trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
            if _fast_verify(trial):
                current_network = trial
                current_network = greedy_reduce(current_network)
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)
                    last_improve_time = time.time()
                    last_best_len = best_len

        if len(current_network) > 0 and time.time() < deadline - 10:
            idx = np.random.randint(len(current_network))
            i, j = current_network[idx]
            a, b = np.random.choice(N, 2, replace=False)
            if a > b:
                a, b = b, a
            if a == i and b == j:
                continue
            trial = current_network.copy()
            trial[idx] = (a, b)
            if _fast_verify(trial):
                current_network = trial
                current_network = greedy_reduce(current_network)
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)
                    last_improve_time = time.time()
                    last_best_len = best_len

    return best_network
# EVOLVE-BLOCK-END
