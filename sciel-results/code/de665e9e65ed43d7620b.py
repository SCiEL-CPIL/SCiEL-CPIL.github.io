import itertools
import time
import random
import sys

def construct():
    start_time = time.time()
    # Evaluator kills at 150s. We aim to return by 140s to be safe.
    deadline = start_time + 140.0

    def _time_left():
        return deadline - time.time()

    # --- Bitmask Utilities ---
    
    def count_violations_sampled(adj, n, sample_sizes=None, max_samples=500):
        """
        Quickly check for violations by sampling subsets.
        Returns (count, violating_subset_or_None).
        """
        if sample_sizes is None:
            # Prioritize smaller sizes as they are more common violations
            sample_sizes = list(range(6, min(n, 10) + 1))
            
        found_count = 0
        example_violation = None
        
        for k in sample_sizes:
            if k > n:
                continue
            # Sample a few subsets of this size
            # For small n, we might check all, but for n=22, C(22,6) is 74613, which is fine.
            # For n=22, C(22,7) is 170544. 
            # Let's just check all subsets for k=6,7 if n <= 20, else sample.
            if n <= 20:
                subsets = list(itertools.combinations(range(n), k))
            else:
                # Sample up to max_samples subsets
                subsets = []
                for _ in range(max_samples):
                    subsets.append(tuple(sorted(random.sample(range(n), k))))
            
            for subset in subsets:
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                # Check regularity
                # Get degrees
                first_deg = None
                is_reg = True
                for v in subset:
                    deg = (adj[v] & mask).bit_count()
                    if first_deg is None:
                        first_deg = deg
                    elif deg != first_deg:
                        is_reg = False
                        break
                
                if is_reg:
                    found_count += 1
                    if example_violation is None:
                        example_violation = subset
                    # If we just need to know if it exists, we could return early,
                    # but counting helps gauge "closeness" to valid.
                    # However, for speed, let's limit the count.
                    if found_count > 10:
                        return found_count, example_violation

        return found_count, example_violation

    def is_valid_fast(adj, n):
        """Check subsets of size 6 and 7 exhaustively (if feasible) or heavily sampled."""
        # For n=22, C(22,6)=74k, C(22,7)=170k. Total ~245k checks.
        # Each check is ~6-7 bit_counts. ~1.5M bit_counts.
        # Python bit_count is fast. This should take < 1s.
        # So we can do exhaustive check for k=6,7.
        for k in [6, 7]:
            if k > n:
                continue
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                first_deg = None
                is_reg = True
                for v in subset:
                    deg = (adj[v] & mask).bit_count()
                    if first_deg is None:
                        first_deg = deg
                    elif deg != first_deg:
                        is_reg = False
                        break
                if is_reg:
                    return False
        return True

    def is_valid_full(adj, n):
        """Check all subsets of size >= 6. Only called if fast check passes."""
        for k in range(6, n + 1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                first_deg = None
                is_reg = True
                for v in subset:
                    deg = (adj[v] & mask).bit_count()
                    if first_deg is None:
                        first_deg = deg
                    elif deg != first_deg:
                        is_reg = False
                        break
                if is_reg:
                    return False
        return True

    def get_edges_from_adj(adj, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    def generate_paley_like(n):
        """
        Generate a Paley-like graph on n vertices.
        Uses a prime p >= n.
        """
        # Find a prime p >= n
        p = n
        while True:
            is_prime = True
            for i in range(2, int(p**0.5) + 1):
                if p % i == 0:
                    is_prime = False
                    break
            if is_prime:
                break
            p += 1
        
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                diff = u - v
                # Legendre symbol check: x^2 == 1 mod p
                # For Paley graph, edge if (u-v) is a quadratic residue mod p
                # Simpler: check if (u-v)^2 mod p is a square? 
                # Actually, standard Paley: edge if (u-v) is a QR mod p.
                # Let's just use a fixed random seed for "structured" random if Paley is tricky to implement correctly for all n quickly.
                # Let's stick to random for robustness, but use a specific seed for reproducibility in search.
                pass
        return adj

    def local_search(n, max_iterations=2000, timeout=10.0):
        """
        Perform local search on a random graph to eliminate violations.
        """
        start_ls = time.time()
        rng = random.Random(int(start_ls * 1000) % 10000)
        
        # Initialize random graph
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        best_adj = adj[:]
        best_score = float('inf')
        
        for i in range(max_iterations):
            if time.time() - start_ls > timeout:
                break
            if _time_left() < 10:
                break

            # Check if current graph is valid (fast check)
            if is_valid_fast(adj, n):
                # Do full check to be sure
                if is_valid_full(adj, n):
                    return adj
                
            # Find a violation to guide the move
            # Use sampling to find a violation quickly
            count, viol_subset = count_violations_sampled(adj, n, sample_sizes=[6, 7, 8], max_samples=100)
            
            if count == 0:
                # No violation found in sample, try fast check again
                if is_valid_fast(adj, n):
                    if is_valid_full(adj, n):
                        return adj
                # If fast check fails but sample didn't find it, sample more or do exhaustive
                # For now, assume it's close. Pick random edge to flip.
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u == v:
                    continue
                u, v = min(u, v), max(u, v)
            else:
                # We have a violation. Try to flip an edge within or connected to this subset
                # Strategy: Flip an edge (u, v) where u, v are in the subset.
                # This changes degrees of u and v within the subset, potentially breaking regularity.
                if len(viol_subset) >= 2:
                    u, v = rng.sample(list(viol_subset), 2)
                    u, v = min(u, v), max(u, v)
                else:
                    u = rng.randint(0, n - 1)
                    v = rng.randint(0, n - 1)
                    if u == v:
                        continue
                    u, v = min(u, v), max(u, v)
            
            # Flip edge (u, v)
            if adj[u] & (1 << v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            # Simulated Annealing / Hill Climbing logic:
            # If we want to minimize violations, we should accept moves that reduce count.
            # But count is expensive to compute fully.
            # Let's just do random walk with occasional restarts if we don't find valid quickly.
            
            # Periodically, if we are stuck, restart
            if i % 100 == 0:
                # Check if we made progress? Hard to track without full count.
                # Just keep going.
                pass

        return None

    # --- Main Search Logic ---

    # Try n=22
    if _time_left() > 30:
        for attempt in range(5):
            if _time_left() < 10:
                break
            adj = local_search(22, max_iterations=1000, timeout=_time_left() - 10)
            if adj:
                edges = get_edges_from_adj(adj, 22)
                return {'n': 22, 'edges': edges}

    # Try n=21
    if _time_left() > 20:
        for attempt in range(10):
            if _time_left() < 10:
                break
            adj = local_search(21, max_iterations=1500, timeout=_time_left() - 10)
            if adj:
                edges = get_edges_from_adj(adj, 21)
                return {'n': 21, 'edges': edges}

    # Try n=20 (High probability of success)
    if _time_left() > 10:
        # Random restarts for n=20
        # C(20,6) = 38760. C(20,7)=77520. Total ~116k checks. Very fast.
        # A random graph on 20 vertices is very likely to have no regular induced subgraph of size >= 6.
        # Let's verify:
        # Probability that a specific 6-set is regular?
        # It's rare.
        # So we can just try random graphs.
        
        for seed in range(1000):
            if _time_left() < 5:
                break
            rng = random.Random(seed)
            adj = [0] * 20
            for u in range(20):
                for v in range(u + 1, 20):
                    if rng.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            if is_valid_fast(adj, 20):
                if is_valid_full(adj, 20):
                    edges = get_edges_from_adj(adj, 20)
                    return {'n': 20, 'edges': edges}
            
            # If fast check fails, we could try to fix it with local search, 
            # but random restarts are probably faster for n=20.
            # However, if we have time, let's try a quick local search on the best candidate?
            # No, just restart.

    # Fallback n=19
    if _time_left() > 5:
        for seed in range(100):
            if _time_left() < 2:
                break
            rng = random.Random(seed + 10000)
            adj = [0] * 19
            for u in range(19):
                for v in range(u + 1, 19):
                    if rng.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            if is_valid_fast(adj, 19):
                if is_valid_full(adj, 19):
                    edges = get_edges_from_adj(adj, 19)
                    return {'n': 19, 'edges': edges}

    # Fallback: Small valid graph
    # n=5 is always valid (vacuously)
    return {'n': 5, 'edges': []}

if __name__ == "__main__":
    # Self-test
    res = construct()
    print(f"Constructed graph with n={res['n']}, edges={len(res['edges'])}")
