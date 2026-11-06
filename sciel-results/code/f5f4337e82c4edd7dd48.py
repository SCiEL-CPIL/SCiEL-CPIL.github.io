"""Size-efficient sorting networks on 17 wires – Simulated Annealing Search."""
import time
import random
import numpy as np

N = 17
MAX_COMP = 4096

# ---------- fast verification helpers ----------
def _make_valid_mask():
    """Boolean array of length 2**N where entry k is True iff k is a sorted 0/1 state."""
    m = np.zeros(1 << N, dtype=bool)
    for k in range(N + 1):
        if k == 0:
            val = 0
        else:
            val = (1 << N) - (1 << (N - k))
        m[val] = True
    return m

_VALID = _make_valid_mask()

def verify(network):
    """Return True if network sorts all 2**N boolean inputs (zero-one principle)."""
    if not network:
        return False
    
    states = np.arange(1 << N, dtype=np.uint32)
    for i, j in network:
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        mask = (bi == 1) & (bj == 0)
        states[mask] ^= (1 << i) | (1 << j)
    return bool(np.all(_VALID[states]))

def compute_depth(network):
    """Greedy layering depth (reported separately; does not affect score)."""
    if not network:
        return 0
    layers = []
    for comp in network:
        placed = False
        for layer in layers:
            wires = set()
            for c in layer:
                wires.add(c[0]); wires.add(c[1])
            if comp[0] not in wires and comp[1] not in wires:
                layer.append(comp)
                placed = True
                break
        if not placed:
            layers.append([comp])
    return len(layers)

def seed_network():
    """Odd-even transposition network (136 comparators for N=17)."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def random_comparator():
    """Generate a random comparator (i, j) with i < j."""
    i, j = rng.sample(range(N), 2)
    if i > j:
        i, j = j, i
    return (i, j)

def simulated_annealing(network, deadline, rng, verbose=False):
    """
    Simulated Annealing to minimize comparator count.
    """
    current = list(network)
    best = list(current)
    best_cost = len(current)
    
    # Energy function: count if valid, else large penalty
    def energy(net):
        if verify(net):
            return len(net)
        else:
            return 10000 + len(net)
    
    current_energy = energy(current)
    best_energy = current_energy
    
    # Annealing parameters
    T = 50.0
    T_min = 0.01
    cooling_rate = 0.995
    
    start_time = time.time()
    
    while T > T_min and time.time() < deadline:
        # Generate a neighbor
        move_type = rng.random()
        new_net = list(current)
        
        if move_type < 0.3 and len(new_net) > 0:
            # Delete
            idx = rng.randint(0, len(new_net) - 1)
            new_net.pop(idx)
        elif move_type < 0.5:
            # Insert
            pos = rng.randint(0, len(new_net))
            new_net.insert(pos, random_comparator())
        elif move_type < 0.8 and len(new_net) >= 2:
            # Swap
            i, j = rng.sample(range(len(new_net)), 2)
            new_net[i], new_net[j] = new_net[j], new_net[i]
        else:
            # Replace
            if new_net:
                idx = rng.randint(0, len(new_net) - 1)
                new_net[idx] = random_comparator()
        
        new_energy = energy(new_net)
        delta_E = new_energy - current_energy
        
        # Acceptance criterion
        if delta_E < 0 or (T > 0 and rng.random() < np.exp(-delta_E / T)):
            current = new_net
            current_energy = new_energy
            
            if current_energy < best_energy:
                best = list(current)
                best_energy = current_energy
                if verbose:
                    print(f"  SA improved to {best_energy}", flush=True)
        
        T *= cooling_rate
        
        # Periodic greedy removal to tighten the network
        if time.time() < deadline and rng.random() < 0.05:
            # Try to remove any redundant comparator
            improved = True
            while improved and time.time() < deadline:
                improved = False
                indices = list(range(len(current)))
                rng.shuffle(indices)
                for idx in indices:
                    if time.time() >= deadline:
                        break
                    test_net = current[:idx] + current[idx+1:]
                    if verify(test_net):
                        current = test_net
                        current_energy = len(current)
                        if current_energy < best_energy:
                            best = list(current)
                            best_energy = current_energy
                            if verbose:
                                print(f"  Greedy improved to {best_energy}", flush=True)
                        improved = True
                        break
                        
    return best

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin before 60s hard limit
    global rng
    rng = random.Random(42)

    best_net = seed_network()
    best_count = len(best_net)
    print(f"Seed network: {best_count} comparators", flush=True)

    if not verify(best_net):
        best_net = []
        best_count = 0

    # Run Simulated Annealing
    print("Running Simulated Annealing...", flush=True)
    best_net = simulated_annealing(best_net, deadline, rng, verbose=True)
    best_count = len(best_net)
    print(f"SA done: {best_count}", flush=True)

    # Final verification
    if not verify(best_net):
        best_net = seed_network()
        best_count = len(best_net)

    depth = compute_depth(best_net)
    print(f"Final: {best_count} comparators, depth {depth}", flush=True)
    return best_net
