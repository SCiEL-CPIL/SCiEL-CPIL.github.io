<<<<<<< SEARCH
import time
import random
import math
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Returns (n, edges).
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    if k == 3:
        # Primary: Voltage lift of Petersen with improved search (inspiration 1)
        # Fallback: Greedy configuration model (strategy directive)
        # Final fallback: LUW / Petersen lift from current program
        result = _voltage_lift_petersen_improved(k, g, deadline, start_time)
        if result is not None:
            return result
        
        result = _greedy_configuration_model(k, g, deadline, start_time)
        if result is not None:
            return result
            
        # Last resort: original Petersen lift from current program
        result = _construct_petersen_lift(g, deadline, start_time)
        if result is not None:
            return result
            
        return _construct_luw(k, g, d=11)
    else:
        return _construct_luw(k, g, d=7)

def _voltage_lift_petersen_improved(k, g, deadline, start_time):
    """
    Voltage lifting of the Petersen graph with improved search.
    Uses numpy for constraint checking and simulated annealing for voltage assignment.
    """
    if k != 3 or g != 16:
        return None

    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]

    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    adj = [[] for _ in range(n_base)]
    for u, v in pet_edges:
        adj[u].append(v)
        adj[v].append(u)

    # Generate all non-backtracking closed walks of length [5, g-1]
    max_L = g - 1
    walk_vectors = set()

    for start_v in range(n_base):
        stack = [(start_v, -1, 0, [0] * 15)]
        while stack:
            current, prev, length, counts = stack.pop()

            if length >= max_L:
                continue

            for next_v in adj[current]:
                if next_v == prev:
                    continue
                e_idx = edge_index[(current, next_v)]
                u, v = pet_edges[e_idx]
                d = 1 if current == u else -1

                new_counts = counts.copy()
                new_counts[e_idx] += d
                new_l = length + 1

                if next_v == start_v and new_l >= 5:
                    walk_vectors.add(tuple(new_counts))

                stack.append((next_v, current, new_l, new_counts))

    if not walk_vectors:
        return None

    # Filter out zero vectors and duplicates
    unique_constraints = set()
    for c in walk_vectors:
        if all(x == 0 for x in c):
            continue
        first_non_zero = -1
        for i in range(15):
            if c[i] != 0:
                first_non_zero = i
                break
        if first_non_zero != -1:
            if c[first_non_zero] < 0:
                c = tuple(-x for x in c)
            unique_constraints.add(c)

    C = np.array(list(unique_constraints), dtype=np.int64)
    if C.size == 0:
        return None

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def solve_voltage(m, max_iters=10000):
        # Initialize with random values
        v = np.random.randint(1, m, size=15).astype(np.int64)

        def cost(v):
            residues = np.mod(C @ v, m)
            return np.sum(residues == 0)

        best_v = v.copy()
        best_cost = cost(v)

        if best_cost == 0:
            return v, 0

        # Simulated annealing with adaptive temperature
        T = 1.0
        cooling = 0.9999
        min_T = 0.01

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            # Pick a random variable to change
            j = random.randint(0, 14)
            old_val = v[j]
            v[j] = random.randint(1, m - 1)

            new_cost = cost(v)
            delta = new_cost - best_cost

            if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_v = v.copy()
                if new_cost == 0:
                    return v, 0
            else:
                v[j] = old_val

            T *= cooling

        return best_v, best_cost

    # Try primes in increasing order starting from 53
    # Moore bound for (3,16) is 512. n = 10*m. So m >= 52.
    # Smallest prime >= 52 is 53.
    m = 53
    while m < 2000 and time.time() < deadline - 20:
        if is_prime(m):
            # Try multiple restarts with different strategies
            for restart in range(30):
                if time.time() > deadline - 15:
                    break

                v, cost = solve_voltage(m, max_iters=5000)

                if cost == 0:
                    # Construct graph
                    n = 10 * m
                    edges = []
                    for u, v_base in pet_edges:
                        e_idx = edge_index[(u, v_base)]
                        vol = v[e_idx]
                        for layer in range(m):
                            u_lift = u * m + layer
                            v_lift = v_base * m + (layer + vol) % m
                            if u_lift < v_lift:
                                edges.append((u_lift, v_lift))
                            else:
                                edges.append((v_lift, u_lift))

                    # Verify
                    if _verify_girth_fast(n, edges, g):
                        return (n, edges)

        m += 1

    return None
=======
import time
import random
import math
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Returns (n, edges).
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    if k == 3:
        # Primary: Voltage lift of Petersen with linear algebra approach
        # Fallback: Greedy configuration model
        # Final fallback: LUW / Petersen lift
        result = _voltage_lift_petersen_la(k, g, deadline, start_time)
        if result is not None:
            return result
        
        result = _greedy_configuration_model(k, g, deadline, start_time)
        if result is not None:
            return result
            
        result = _construct_petersen_lift(g, deadline, start_time)
        if result is not None:
            return result
            
        return _construct_luw(k, g, d=11)
    else:
        return _construct_luw(k, g, d=7)

def _voltage_lift_petersen_la(k, g, deadline, start_time):
    """
    Voltage lifting using linear algebra over F_p.
    For each prime m, we need to find voltages v_e such that for every
    non-backtracking closed walk W of length < g in the base graph,
    sum_{e in W} sign(e) * v_e != 0 (mod m).
    
    We use a greedy constraint-propagation approach:
    1. Generate all constraints (closed walks)
    2. For each variable, determine which constraints it affects
    3. Assign values greedily, choosing values that satisfy the most constraints
    """
    if k != 3 or g != 16:
        return None

    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]

    n_base = 10
    n_edges = 15
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    adj = [[] for _ in range(n_base)]
    for u, v in pet_edges:
        adj[u].append(v)
        adj[v].append(u)

    # Generate all non-backtracking closed walks of length 3 to g-1
    max_L = g - 1
    walk_vectors = []

    for start_v in range(n_base):
        stack = [(start_v, -1, 0, [0] * n_edges)]
        while stack:
            current, prev, length, counts = stack.pop()

            if length >= max_L:
                continue

            for next_v in adj[current]:
                if next_v == prev:
                    continue
                e_idx = edge_index[(current, next_v)]
                u, v = pet_edges[e_idx]
                d = 1 if current == u else -1

                new_counts = counts.copy()
                new_counts[e_idx] += d
                new_l = length + 1

                if next_v == start_v and new_l >= 3:
                    walk_vectors.append(tuple(new_counts))

                stack.append((next_v, current, new_l, new_counts))

    if not walk_vectors:
        return None

    # Build constraint matrix: each row is a constraint (closed walk)
    # C[i][j] = coefficient of edge j in constraint i
    C_list = []
    for w in walk_vectors:
        if any(x != 0 for x in w):
            C_list.append(w)

    C = np.array(C_list, dtype=np.int64)
    n_constraints = C.shape[0]
    
    if n_constraints == 0:
        return None

    # For each edge, find which constraints it appears in
    edge_to_constraints = [[] for _ in range(n_edges)]
    for i in range(n_constraints):
        for j in range(n_edges):
            if C[i][j] != 0:
                edge_to_constraints[j].append(i)

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def solve_voltage_greedy(m, max_backtracks=100000):
        """
        Greedy constraint propagation with backtracking.
        Assign voltages to edges one at a time.
        """
        v = [-1] * n_edges  # -1 means unassigned
        assigned = [False] * n_edges
        
        # Track which constraints are satisfied/unsatisfied
        constraint_status = [0] * n_constraints  # 0=unknown, 1=satisfied, -1=unsatisfied
        constraint_sum = [0] * n_constraints  # current partial sum
        
        def check_constraint(i, edge_idx, val):
            """Check if constraint i can still be satisfied given current assignments."""
            s = constraint_sum[i] + C[i][edge_idx] * val
            # Check if remaining unassigned edges can make s != 0 mod m
            remaining = 0
            for j in range(n_edges):
                if not assigned[j] and j != edge_idx:
                    remaining += abs(C[i][j])
            # If s == 0 and no remaining edges can change it, constraint is violated
            if s % m == 0 and remaining == 0:
                return False
            return True

        def try_assign(edge_idx, val):
            """Try to assign val to edge_idx, return True if consistent."""
            if assigned[edge_idx]:
                return v[edge_idx] == val
            
            assigned[edge_idx] = True
            v[edge_idx] = val
            
            # Update constraint sums
            for ci in edge_to_constraints[edge_idx]:
                constraint_sum[ci] += C[ci][edge_idx] * val
                # Check if this constraint is now violated
                if constraint_sum[ci] % m == 0:
                    # Check if remaining edges can fix it
                    remaining = 0
                    for j in range(n_edges):
                        if not assigned[j]:
                            remaining += abs(C[ci][j])
                    if remaining == 0:
                        # Violation
                        constraint_sum[ci] -= C[ci][edge_idx] * val
                        assigned[edge_idx] = False
                        v[edge_idx] = -1
                        return False
            
            return True

        def unassign(edge_idx):
            assigned[edge_idx] = False
            v[edge_idx] = -1
            for ci in edge_to_constraints[edge_idx]:
                constraint_sum[ci] -= C[ci][edge_idx] * v[edge_idx]

        # Order edges by number of constraints they affect (most constrained first)
        edge_order = sorted(range(n_edges), key=lambda j: -len(edge_to_constraints[j]))
        
        backtracks = 0
        stack = []  # (edge_idx, val)
        
        # Greedy forward pass
        for edge_idx in edge_order:
            found = False
            for val in range(1, m):
                if try_assign(edge_idx, val):
                    found = True
                    stack.append((edge_idx, val))
                    break
            if not found:
                # Backtrack
                while stack:
                    if backtracks > max_backtracks:
                        return None, -1
                    backtracks += 1
                    prev_edge, prev_val = stack.pop()
                    unassign(prev_edge)
                    # Try next value for prev_edge
                    found = False
                    for val in range(prev_val + 1, m):
                        if try_assign(prev_edge, val):
                            found = True
                            stack.append((prev_edge, val))
                            # Now continue with remaining edges
                            for edge_idx in edge_order:
                                if not assigned[edge_idx]:
                                    found2 = False
                                    for val2 in range(1, m):
                                        if try_assign(edge_idx, val2):
                                            found2 = True
                                            stack.append((edge_idx, val2))
                                            break
                                    if not found2:
                                        # Continue backtracking
                                        break
                                    else:
                                        break
                            if found:
                                break
                    if found:
                        break
                if not stack:
                    return None, -1
        
        # Verify all constraints
        for i in range(n_constraints):
            s = sum(C[i][j] * v[j] for j in range(n_edges))
            if s % m == 0:
                return None, -1
        
        return v, 0

    # Try primes in increasing order
    # Moore bound for (3,16) is 512. n = 10*m. So m >= 52.
    m = 53
    while m < 500 and time.time() < deadline - 20:
        if is_prime(m):
            # Try multiple random initial orderings
            for restart in range(50):
                if time.time() > deadline - 15:
                    break
                
                # Shuffle edge order for diversity
                random_edge_order = list(range(n_edges))
                random.shuffle(random_edge_order)
                
                v, cost = solve_voltage_greedy(m, max_backtracks=50000)
                
                if v is not None and cost == 0:
                    # Construct graph
                    n = 10 * m
                    edges = []
                    for u, v_base in pet_edges:
                        e_idx = edge_index[(u, v_base)]
                        vol = v[e_idx]
                        for layer in range(m):
                            u_lift = u * m + layer
                            v_lift = v_base * m + (layer + vol) % m
                            if u_lift < v_lift:
                                edges.append((u_lift, v_lift))
                            else:
                                edges.append((v_lift, u_lift))

                    # Verify
                    if _verify_girth_fast(n, edges, g):
                        return (n, edges)

        m += 1

    return None
>>>>>>> REPLACE

