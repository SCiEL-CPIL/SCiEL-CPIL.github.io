import time
import random
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    For k=3, uses cyclic Cayley graphs Z_N with generators {a, -a, b}.
    Searches for the smallest N and generators that satisfy the girth constraint.
    """
    if k != 3:
        return _luw_construct(k, g)
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def _luw_construct(k, g):
        """LUW construction fallback."""
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    aa, bb = rule[c]
                    y[c] = add(x[c], mul(y[aa], x[bb])) if side == 0 else sub(x[c], mul(x[aa], y[bb]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def _autocorr_cycle_check(N, a, b, max_len):
        """
        Algebraic pre-filter: use FFT to detect any closed walk of length <= max_len
        that is not a trivial back-and-forth. If found, girth < max_len+1.
        Returns True if no short cycle detected (candidate passes).
        """
        import numpy as np
        gens = [a % N, (N - a) % N, b % N]
        # Build generator indicator
        G = np.zeros(N, dtype=np.float64)
        for s in gens:
            G[s] = 1.0
        G[0] = 0.0  # no self-loops
        
        # Compute G^l via FFT for l=3..max_len
        Gf = np.fft.rfft(G)
        # Power in frequency domain
        Gf_pow = Gf.copy()
        for l in range(3, max_len + 1):
            Gf_pow = Gf_pow * Gf
            # autocorrelation: sum over x of G^l[x] * G[-x]
            # This equals sum over closed walks of length l
            # But we need to exclude trivial backtracks. For l < girth,
            # the only closed walks of length l are backtracks, so:
            # number of non-trivial closed walks = total - backtrack_count
            # For simplicity, just check if total closed walks > expected backtrack count
            # Expected backtrack count for l steps: 3 * 2^(l-1) per starting vertex * N / ... 
            # Actually simpler: if autocorrelation > N * 3 * 2^(l-2) (approx), there's a cycle
            # Let's just compute and check
            pass
        
        # More direct: compute G^l[0] for each l
        # G^l[0] = number of closed walks of length l from any vertex (summed)
        # = N * (number of closed walks of length l from vertex 0)
        # For a 3-regular graph with girth >= g, closed walks of length l < g
        # are only backtracks: count = 3 * 2^(l-1) per vertex
        # So total = N * 3 * 2^(l-1)
        
        Gf_l = Gf.copy()
        for l in range(3, max_len + 1):
            Gf_l = Gf_l * Gf
            # Inverse FFT to get G^l in spatial domain
            Gl = np.fft.irfft(Gf_l, N)
            # Gl[0] = sum of all entries? No, Gl[x] = sum_y G^l[y] * ... 
            # Actually Gl[x] = (G^l)[x] for the convolution
            # The number of closed walks of length l from vertex 0 is Gl[0]
            # Wait, let me think again.
            # If G is the adjacency matrix (as a vector), then G^l in the group ring
            # gives the number of walks of length l.
            # (G^l)[0] = number of closed walks of length l from 0.
            # But G here is a vector, not a matrix. The convolution G * G gives
            # (G*G)[x] = sum_y G[y] * G[x-y], which is the number of 2-step walks
            # from 0 to x. So (G^l)[0] is the number of l-step closed walks from 0.
            
            closed_walks = Gl[0]
            # Expected backtrack count for 3-regular graph: 3 * 2^(l-1)
            expected_backtracks = 3 * (2 ** (l - 1))
            # Allow small numerical tolerance
            if closed_walks > expected_backtracks + N * 0.01:
                return False
        return True

    def generate_circulant(N, a, b):
        """
        Generates edges for Z_N with generators a, -a, b.
        Returns (n, edges) or None if invalid.
        """
        a = a % N
        b = b % N
        
        # Check for simple graph conditions
        if a == 0 or b == 0:
            return None
        if a == (N - a) % N:  # 2a = 0
            return None
        if b == (N - b) % N:  # 2b = 0
            return None
        if a == b or a == (N - b) % N or (N - a) % N == b or (N - a) % N == (N - b) % N:
            return None
        
        edges = set()
        for i in range(N):
            for j in [(i + a) % N, (i - a) % N, (i + b) % N]:
                if i == j:
                    return None
                u, v = min(i, j), max(i, j)
                edges.add((u, v))
        
        # Verify degree
        deg = [0] * N
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        for d in deg:
            if d != 3:
                return None
        
        return N, list(edges)

    # Search strategy: try N from Moore bound upward
    # Use structured generator selection for better success rate
    
    random.seed(42)
    
    # Golden ratio conjugate for good distribution
    phi = (1 + 5**0.5) / 2
    
    for N in range(512, 1300):
        if time.time() > deadline - 15:
            break
            
        # Try structured choices first, then random
        candidates = []
        
        # Golden ratio based
        a_golden = int(N / phi) % N
        b_golden = int(N / (phi * phi)) % N
        if a_golden != 0 and b_golden != 0:
            candidates.append((a_golden, b_golden))
            candidates.append((b_golden, a_golden))
        
        # Try a few offsets
        for offset in range(-10, 11):
            a_try = (a_golden + offset) % N
            b_try = (b_golden + offset) % N
            if a_try != 0 and b_try != 0:
                candidates.append((a_try, b_try))
        
        # Add more structured candidates
        for frac in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            a_f = int(N * frac) % N
            b_f = int(N * (1.0 - frac)) % N
            if a_f != 0 and b_f != 0 and a_f != b_f:
                candidates.append((a_f, b_f))
        
        # Add random candidates
        for _ in range(300):
            a_r = random.randint(1, N - 1)
            b_r = random.randint(1, N - 1)
            candidates.append((a_r, b_r))
        
        # Deduplicate
        candidates = list(set(candidates))
        
        for a, b in candidates:
            if time.time() > deadline - 15:
                break
            
            res = generate_circulant(N, a, b)
            if res is None:
                continue
                
            n, edges = res
            
            # Fast algebraic pre-filter
            if not _autocorr_cycle_check(N, a, b, g - 1):
                continue
            
            # Check girth
            if check_girth_fast(N, a, b, g):
                return n, edges
    
    # Fallback to LUW
    return _luw_construct(k, g)
