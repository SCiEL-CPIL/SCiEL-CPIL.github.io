"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
SEED_COMPARATORS = 153  # odd-even transposition seed size

# EVOLVE-BLOCK-START
def _apply_network(net, vec):
    """Apply a list of (i,j) comparators to a vector (numpy array)."""
    for i, j in net:
        if vec[i] > vec[j]:
            vec[i], vec[j] = vec[j], vec[i]
    return vec

def _is_sorted_network(net, n_vecs=None, seed_vecs=None):
    """Check if network sorts all given vectors. Returns True if sorted."""
    if seed_vecs is None:
        # Use a set of random vectors
        rng = np.random.default_rng(42)
        seed_vecs = [rng.integers(0, 1000, size=N) for _ in range(n_vecs or 200)]
    for v in seed_vecs:
        vv = v.copy()
        _apply_network(net, vv)
        if not np.all(np.diff(vv) >= 0):
            return False
    return True

def _depth(net):
    """Compute parallel depth (longest path)."""
    if not net:
        return 0
    depth = np.zeros(N, dtype=int)
    for i, j in net:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return int(np.max(depth))

def _random_vector_check(net, n_tests=500):
    """Fast random check."""
    rng = np.random.default_rng(12345)
    for _ in range(n_tests):
        v = rng.integers(0, 10000, size=N)
        vv = v.copy()
        _apply_network(net, vv)
        if not np.all(np.diff(vv) >= 0):
            return False, v
    return True, None

def _seed_network():
    """Odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _build_batcher(n):
    """Build a Batcher odd-even merge sort network for n wires (n must be power of 2)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    half = n // 2
    # Recursively sort two halves
    left = _build_batcher(half)
    right = [(i + half, j + half) for i, j in _build_batcher(half)]
    # Odd-even merge
    merge = []
    for phase in range(n):
        parity = phase % 2
        for i in range(parity, n - 1, 2):
            merge.append((i, i + 1))
    return left + right + merge

def _merge_two(nets_a, nets_b, n):
    """Merge two sorting networks for n wires using odd-even merge pattern."""
    # This is a simplified approach; proper merge requires bitonic-like structure
    # For now, just concatenate
    return nets_a + nets_b

def _random_comparator(n, rng):
    i = rng.integers(0, n - 1)
    j = rng.integers(i + 1, n)
    return (int(i), int(j))

def _local_search(net, max_iters=2000, time_limit=40.0, rng_seed=0):
    """Hill-climbing: try to remove or swap comparators."""
    rng = np.random.default_rng(rng_seed)
    net = list(net)
    best_net = list(net)
    best_len = len(net)
    start_time = time.time()
    
    # Pre-generate test vectors
    test_rng = np.random.default_rng(777)
    test_vecs = [test_rng.integers(0, 10000, size=N) for _ in range(300)]
    # Also add some structured vectors
    test_vecs.append(np.arange(N, 0, -1))  # reverse
    test_vecs.append(np.arange(N))  # sorted
    test_vecs.append(np.array([N-1-i for i in range(N)]))  # reverse again
    # Random permutations
    for _ in range(50):
        perm = np.random.default_rng(42 + _).permutation(N)
        test_vecs.append(perm)
    
    def check(net_list):
        for v in test_vecs:
            vv = v.copy()
            _apply_network(net_list, vv)
            if not np.all(np.diff(vv) >= 0):
                return False
        return True
    
    # First verify current net
    if not check(net):
        return best_net
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if len(net) <= 68:  # lower bound
            break
            
        # Try random modifications
        op = rng.integers(0, 3)
        if op == 0 and len(net) > 2:
            # Remove a random comparator
            idx = rng.integers(0, len(net))
            new_net = net[:idx] + net[idx+1:]
            if check(new_net):
                net = new_net
                if len(net) < best_len:
                    best_len = len(net)
                    best_net = list(net)
        elif op == 1:
            # Swap two adjacent comparators
            if len(net) < 2:
                continue
            idx = rng.integers(0, len(net) - 1)
            new_net = list(net)
            new_net[idx], new_net[idx+1] = new_net[idx+1], new_net[idx]
            if check(new_net):
                net = new_net
        elif op == 2:
            # Replace a random comparator with a different one
            if not net:
                continue
            idx = rng.integers(0, len(net))
            new_i, new_j = _random_comparator(N, rng)
            new_net = list(net)
            new_net[idx] = (new_i, new_j)
            if check(new_net):
                net = new_net
                if len(net) < best_len:
                    best_len = len(net)
                    best_net = list(net)
    
    return best_net

def _greedy_add(net, n_add=10, time_limit=10.0):
    """Try to add useful comparators (not directly helpful for minimization, but for restructuring)."""
    pass  # Not used in minimization

def construct():
    """Construct a sorting network with as few comparators as possible."""
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin before 90s kill
    
    # Start with seed network
    seed_net = _seed_network()
    best_net = list(seed_net)
    best_len = len(seed_net)
    
    # Try Batcher-based construction as an alternative starting point
    # Batcher for 16 wires, then extend to 18
    # Actually, let's try a different approach: start from a known small network structure
    
    # Phase 1: Local search from seed
    t1 = time.time()
    net1 = _local_search(seed_net, max_iters=5000, time_limit=30.0, rng_seed=0)
    if len(net1) < best_len:
        best_net = net1
        best_len = len(net1)
    
    # Phase 2: Try another random restart
    if time.time() < deadline - 20.0:
        # Create a perturbed version of seed
        rng = np.random.default_rng(123)
        perturbed = list(seed_net)
        # Remove some random comparators and see if still valid
        for _ in range(20):
            if len(perturbed) <= 80:
                break
            idx = rng.integers(0, len(perturbed))
            trial = perturbed[:idx] + perturbed[idx+1:]
            if _random_vector_check(trial, 200)[0]:
                perturbed = trial
        net2 = _local_search(perturbed, max_iters=3000, time_limit=20.0, rng_seed=1)
        if len(net2) < best_len:
            best_net = net2
            best_len = len(net2)
    
    # Phase 3: Final local search from best
    if time.time() < deadline - 5.0:
        remaining_time = max(2.0, deadline - 5.0 - time.time())
        net3 = _local_search(best_net, max_iters=5000, time_limit=remaining_time, rng_seed=99)
        if len(net3) < best_len:
            best_net = net3
            best_len = len(net3)
    
    # Verify the best network one more time with more tests
    if time.time() < deadline - 1.0:
        ok, bad_vec = _random_vector_check(best_net, 1000)
        if not ok:
            # Fall back to seed
            best_net = seed_net
    
    return best_net
# EVOLVE-BLOCK-END
