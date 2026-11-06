"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 18
TOTAL_INPUTS = 1 << N  # 262144
DEADLINE = 55.0  # seconds, safety margin before 60s limit

# EVOLVE-BLOCK-START
def construct():
    """
    Build a sorting network on N=18 wires using Bit-Parallel verification.
    Uses Python big integers to represent the state of each wire across all 2^18 inputs.
    min(a,b) is bitwise AND, max(a,b) is bitwise OR.
    Sorted check: (wire[i] & ~wire[i+1]) == 0 for all i.
    """
    start_time = time.time()

    # --- Initialize Bit-Parallel Wires ---
    # wire[i] is a Python int where bit k is 1 if input k has 1 on wire i.
    # Input k is the integer k. Wire i corresponds to bit i of k.
    # So wire[i] has bits set at positions k where (k >> i) & 1 == 1.

    def make_mask(i):
        L = 1 << (i + 1)
        num_blocks = TOTAL_INPUTS // L
        ones_val = (1 << (1 << i)) - 1
        block = ones_val << (1 << i)
        G = ((1 << TOTAL_INPUTS) - 1) // ((1 << L) - 1)
        mask = block * G
        return mask

    # Pre-compute masks
    masks = [make_mask(i) for i in range(N)]

    def verify_network(network):
        """Verify network sorts all inputs. Returns True if correct."""
        wires = masks[:]
        for i, j in network:
            new_wi = wires[i] & wires[j]
            new_wj = wires[i] | wires[j]
            wires[i] = new_wi
            wires[j] = new_wj
        for i in range(N - 1):
            if (wires[i] & ~wires[i+1]) != 0:
                return False
        return True

    def find_violation(network):
        """Find the first adjacent inversion pair, or None if sorted."""
        wires = masks[:]
        for i, j in network:
            new_wi = wires[i] & wires[j]
            new_wj = wires[i] | wires[j]
            wires[i] = new_wi
            wires[j] = new_wj
        for i in range(N - 1):
            if (wires[i] & ~wires[i+1]) != 0:
                return (i, i + 1)
        return None

    # --- Start with odd-even transposition network ---
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))

    assert verify_network(network), "Initial network should be correct"

    best_network = network[:]
    best_size = len(network)

    # --- Phase 1: Greedy removal ---
    removed = True
    while removed and (time.time() - start_time) < DEADLINE - 15:
        removed = False
        for idx in range(len(network)):
            if (time.time() - start_time) > DEADLINE - 10:
                break
            candidate = network[:]
            candidate.pop(idx)
            if verify_network(candidate):
                network = candidate
                if len(network) < best_size:
                    best_network = network[:]
                    best_size = len(network)
                removed = True
                break  # Restart since indices changed

    # --- Phase 2: Error-guided local search ---
    current = best_network[:]
    # Pre-compute all valid (i,j) pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    for iteration in range(20000):
        if (time.time() - start_time) > DEADLINE - 5:
            break
        if len(current) <= 70:
            break
            
        # Random removal
        idx = random.randint(0, len(current) - 1)
        candidate = current[:]
        candidate.pop(idx)
        
        # If it's still valid, accept
        if verify_network(candidate):
            current = candidate
            if len(current) < best_size:
                best_network = current[:]
                best_size = len(current)
            continue
            
        # If broken, find the violation and try to repair
        violation = find_violation(candidate)
        repaired = False
        if violation:
            vi, vj = violation
            # Try inserting comparators involving the violated wires
            repair_candidates = []
            for w in range(N):
                if w != vi and w != vj:
                    # (w, vi) if w < vi, else (vi, w)
                    if w < vi:
                        repair_candidates.append((w, vi))
                    else:
                        repair_candidates.append((vi, w))
                if w != vj:
                    if w < vj:
                        repair_candidates.append((w, vj))
                    else:
                        repair_candidates.append((vj, w))
            # Also try (vi, vj) itself
            repair_candidates.append((vi, vj))
            
            random.shuffle(repair_candidates)
            
            for (i, j) in repair_candidates[:30]:
                if (time.time() - start_time) > DEADLINE - 5:
                    break
                # Try inserting at a random position
                pos = random.randint(0, len(candidate))
                temp = candidate[:]
                temp.insert(pos, (i, j))
                
                if verify_network(temp):
                    current = temp
                    repaired = True
                    if len(current) < best_size:
                        best_network = current[:]
                        best_size = len(current)
                    break
        
        if not repaired:
            # If we can't repair, keep current
            pass

    # --- Phase 3: Swap-based local search ---
    current = best_network[:]
    for iteration in range(10000):
        if (time.time() - start_time) > DEADLINE - 5:
            break
        if len(current) <= 70:
            break
            
        # Try swapping two comparators
        if len(current) < 2:
            break
        i1 = random.randint(0, len(current) - 1)
        i2 = random.randint(0, len(current) - 1)
        if i1 == i2:
            continue
            
        candidate = current[:]
        candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
        
        if verify_network(candidate):
            current = candidate
            if len(current) < best_size:
                best_network = current[:]
                best_size = len(current)

    # --- Phase 4: Final Greedy Removal ---
    final_net = best_network[:]
    removed = True
    while removed and (time.time() - start_time) < DEADLINE - 2:
        removed = False
        for idx in range(len(final_net)):
            if (time.time() - start_time) > DEADLINE - 1:
                break
            candidate = final_net[:]
            candidate.pop(idx)
            if verify_network(candidate):
                final_net = candidate
                removed = True
                break

    if len(final_net) < best_size:
        best_network = final_net[:]
        best_size = len(final_net)

    return best_network
# EVOLVE-BLOCK-END
