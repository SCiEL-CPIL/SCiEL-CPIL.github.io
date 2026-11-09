# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def ring_routing() -> np.ndarray:
    # Problem data
    N = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    weights = np.array([d[2] for d in demands], dtype=float)
    sources = np.array([d[0] for d in demands], dtype=int)
    targets = np.array([d[1] for d in demands], dtype=int)

    # For each demand, the set of edges used in each direction
    # CW from s to t uses edges s, s+1, ..., t-1 (mod N)
    # CCW uses all other edges
    cw_edges = []
    ccw_edges = []
    for i in range(D):
        s, t = sources[i], targets[i]
        cw = set()
        e = s
        while e != t:
            cw.add(e)
            e = (e + 1) % N
        all_edges = set(range(N))
        ccw = all_edges - cw
        cw_edges.append(cw)
        ccw_edges.append(ccw)

    # Precompute: for each edge, which demands use it in CW and in CCW
    # edge_cw_users[e] = list of demand indices that use edge e in CW
    # edge_ccw_users[e] = list of demand indices that use edge e in CCW
    edge_cw_users = [[] for _ in range(N)]
    edge_ccw_users = [[] for _ in range(N)]
    for i in range(D):
        for e in cw_edges[i]:
            edge_cw_users[e].append(i)
        for e in ccw_edges[i]:
            edge_ccw_users[e].append(i)

    def compute_loads(assignment):
        """Compute edge loads given a boolean assignment (True = CCW)."""
        loads = np.zeros(N, dtype=float)
        for i in range(D):
            if assignment[i]:  # CCW
                for e in ccw_edges[i]:
                    loads[e] += weights[i]
            else:  # CW
                for e in cw_edges[i]:
                    loads[e] += weights[i]
        return loads

    def max_load(assignment):
        return compute_loads(assignment).max()

    def check_feasible(L):
        """
        Check if there exists an assignment such that every edge load <= L.
        Uses 2-SAT. Variable i: True means demand i goes CCW.
        For each edge e, the load is sum of weights of demands using e in their chosen direction.
        We need: for every edge e, load_e <= L.
        
        This is not directly 2-SAT unless we handle it carefully.
        Actually, the constraint "sum of weights on edge e <= L" is a knapsack-like constraint,
        not a simple 2-SAT clause. So 2-SAT alone doesn't work for arbitrary L.
        
        Alternative: use local search / simulated annealing.
        """
        pass  # Not using 2-SAT directly; using local search instead.

    # Strategy: Local search with multiple restarts
    # Start from all-CW, then try flipping demands to reduce max load.
    
    def local_search(initial, max_iters=500000):
        """
        Hill-climbing with random restarts within this function.
        initial: np.array of booleans (True = CCW)
        Returns best assignment found.
        """
        best_assign = initial.copy()
        best_ml = max_load(best_assign)
        current = initial.copy()
        current_ml = best_ml
        no_improve = 0
        max_no_improve = 5000
        
        for it in range(max_iters):
            # Try flipping one random demand
            i = np.random.randint(D)
            current[i] = not current[i]
            new_ml = max_load(current)
            if new_ml < current_ml:
                current_ml = new_ml
                no_improve = 0
                if new_ml < best_ml:
                    best_ml = new_ml
                    best_assign = current.copy()
            else:
                current[i] = not current[i]  # revert
                no_improve += 1
                if no_improve > max_no_improve:
                    break
        
        return best_assign, best_ml

    def local_search_2phase(initial, max_iters=2000000):
        """
        More aggressive local search: try all single flips, pick best improvement.
        """
        current = initial.copy()
        current_ml = max_load(current)
        best_assign = current.copy()
        best_ml = current_ml
        no_improve = 0
        max_no_improve = 100
        
        for it in range(max_iters):
            # Find the best single flip
            best_flip = -1
            best_new_ml = current_ml
            for i in range(D):
                current[i] = not current[i]
                ml = max_load(current)
                current[i] = not current[i]  # revert
                if ml < best_new_ml:
                    best_new_ml = ml
                    best_flip = i
            
            if best_flip >= 0 and best_new_ml < current_ml:
                current[best_flip] = not current[best_flip]
                current_ml = best_new_ml
                no_improve = 0
                if current_ml < best_ml:
                    best_ml = current_ml
                    best_assign = current.copy()
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
        
        return best_assign, best_ml

    # Run multiple strategies
    deadline = time.time() + 600  # 600s safety margin
    best_overall = None
    best_overall_ml = float('inf')

    # Strategy 1: All CW
    start = time.time()
    assign_cw = np.zeros(D, dtype=bool)
    a, ml = local_search_2phase(assign_cw, max_iters=1000000)
    if ml < best_overall_ml:
        best_overall_ml = ml
        best_overall = a
    if time.time() > deadline - 60:
        return (best_overall.astype(float) * 1.0) if best_overall is not None else np.ones(D)

    # Strategy 2: All CCW
    start = time.time()
    assign_ccw = np.ones(D, dtype=bool)
    a, ml = local_search_2phase(assign_ccw, max_iters=1000000)
    if ml < best_overall_ml:
        best_overall_ml = ml
        best_overall = a
    if time.time() > deadline - 60:
        return (best_overall.astype(float) * 1.0) if best_overall is not None else np.ones(D)

    # Strategy 3: Random restarts
    np.random.seed(42)
    for restart in range(200):
        if time.time() > deadline - 60:
            break
        init = np.random.randint(0, 2, D).astype(bool)
        a, ml = local_search_2phase(init, max_iters=500000)
        if ml < best_overall_ml:
            best_overall_ml = ml
            best_overall = a
        # Also try single-flip polish
        if time.time() > deadline - 60:
            break

    # Final polish: exhaustive single-flip check on best
    if best_overall is not None and time.time() < deadline - 30:
        current = best_overall.copy()
        current_ml = best_overall_ml
        improved = True
        while improved:
            improved = False
            for i in range(D):
                current[i] = not current[i]
                ml = max_load(current)
                current[i] = not current[i]
                if ml < current_ml:
                    current[i] = not current[i]
                    current_ml = ml
                    improved = True
                    break
            if not improved:
                # Try all pairs? Too expensive. Just do single flips.
                pass
        if current_ml < best_overall_ml:
            best_overall_ml = current_ml
            best_overall = current

    # Convert to output format: >= 0.5 means CCW
    result = best_overall.astype(float) * 1.0
    return result

# EVOLVE-BLOCK-END
