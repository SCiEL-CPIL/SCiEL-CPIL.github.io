"""Rewritten program: Grid search over threshold allocation rules with exact IC payment derivation."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import heapq

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction mechanism.
    
    Args:
        distribution_spec: specification of the value distribution
        support: support of the value space
        constraints: additional constraints
    
    Returns:
        dict with 'allocations': (16, 2) array and 'payments': (16,) array
    """
    # Parse distribution
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            if 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=float)
            elif 'dist' in distribution_spec:
                probs = np.array(distribution_spec['dist'], dtype=float)
            elif 'p' in distribution_spec:
                probs = np.array(distribution_spec['p'], dtype=float)
            else:
                probs = None
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = None
    else:
        probs = None

    if probs is None or len(probs) != 16:
        probs = np.ones(16) / 16.0
    else:
        probs = probs / probs.sum()

    # Types: (v1, v2)
    types = [(i, j) for i in range(4) for j in range(4)]
    n = 16
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # Function to compute optimal payments for a given allocation matrix X
    def compute_optimal_payments(X, probs):
        """
        Computes the minimal payments P that satisfy IC and IR for a fixed allocation X.
        Uses the fact that optimal utilities U are the longest paths in the IC graph.
        U(t) = max(0, max_{t'} [U(t') + (v(t) - v(t')) dot X(t')])
        P(t) = v(t) dot X(t) - U(t)
        """
        # Build graph edges: t' -> t with weight w(t', t) = (v(t) - v(t')) dot X(t')
        # We want max U(t) such that U(t) >= U(t') + w(t', t) and U(t) >= 0.
        # This is equivalent to finding the longest path in a graph where nodes are types.
        # Since the graph is dense (16 nodes), we can use Bellman-Ford or Floyd-Warshall.
        # Floyd-Warshall is O(N^3) = 16^3 = 4096, very fast.
        
        # dist[i][j] = max utility difference from i to j?
        # Let's define D[t][t'] as the max "gain" in utility from t' to t.
        # Constraint: U(t) - U(t') >= (v(t) - v(t')) dot X(t')
        # Let C[t][t'] = (v(t) - v(t')) dot X(t')
        # We want max U(t) subject to U(t) >= U(t') + C[t][t'] and U(t) >= 0.
        # This is equivalent to U(t) = max(0, max_{path ending at t} sum(C))
        
        # Initialize U with 0
        U = np.zeros(n)
        
        # Relaxation: U(t) = max(U(t), U(t') + C[t][t'])
        # Iterate until convergence.
        for _ in range(n): # Bellman-Ford style
            changed = False
            for t in range(n):
                for t2 in range(n):
                    if t == t2: continue
                    v_t = np.array(types[t])
                    v_t2 = np.array(types[t2])
                    # C[t][t2] = (v(t) - v(t2)) dot X(t2)
                    c_val = np.dot(v_t - v_t2, X[t2])
                    new_u = U[t2] + c_val
                    if new_u > U[t] + 1e-9:
                        U[t] = new_u
                        changed = True
            if not changed:
                break
        
        # Ensure U(t) >= 0
        U = np.maximum(U, 0)
        
        # Calculate Payments
        # P(t) = v(t) dot X(t) - U(t)
        P = np.zeros(n)
        for t in range(n):
            v_t = np.array(types[t])
            P[t] = np.dot(v_t, X[t]) - U[t]
            
        # Check IR: P(t) <= v(t) dot X(t) => U(t) >= 0 (Satisfied)
        # Check P(t) >= 0?
        # If P(t) < 0, it means U(t) > v(t) dot X(t).
        # Can this happen?
        # U(t) is the max utility. If the agent gets high utility, payment is low.
        # If U(t) > Value, then Payment < 0.
        # But we require P(t) >= 0.
        # So we must ensure U(t) <= v(t) dot X(t).
        # If the derived U(t) violates this, the mechanism is infeasible with P>=0?
        # Actually, IR is U(t) >= 0. P(t) >= 0 is a separate constraint.
        # If P(t) < 0, we can set P(t) = 0, which increases U(t) to v(t) dot X(t).
        # But increasing U(t) might violate IC for others?
        # If we increase U(t), we might need to increase U(t') to maintain IC?
        # The longest path algorithm finds the *minimum* U that satisfies IC and U>=0.
        # If this minimum U leads to P < 0, then any feasible U (which is >= min U) will have even larger U, hence smaller P.
        # So if min U(t) > v(t) dot X(t), then NO feasible payment with P>=0 exists for this X?
        # Wait. If min U(t) > v(t) dot X(t), then P(t) < 0 for the "best" case.
        # We can't make P(t) larger (revenue lower) to satisfy P>=0?
        # We CAN make P(t) larger by making U(t) smaller? No, U(t) is bounded below by IC.
        # So if the IC-lower-bound on U(t) is higher than the value, we cannot satisfy P(t)>=0.
        # In that case, this allocation X is infeasible for P>=0.
        
        if np.any(P < -1e-9):
            return None, -np.inf
        
        # Clip small negatives
        P = np.maximum(P, 0)
        
        # Calculate Revenue
        revenue = np.dot(P, probs)
        return P, revenue

    best_revenue = -np.inf
    best_X = None
    best_P = None
    
    # Candidate Allocations
    # 1. Independent Thresholds
    for tau1 in range(5):
        for tau2 in range(5):
            X = np.zeros((16, 2))
            for i, (v1, v2) in enumerate(types):
                if v1 >= tau1:
                    X[i, 0] = 1.0
                if v2 >= tau2:
                    X[i, 1] = 1.0
            P, rev = compute_optimal_payments(X, probs)
            if P is not None and rev > best_revenue:
                best_revenue = rev
                best_X = X.copy()
                best_P = P.copy()
                
    # 2. Joint Threshold (Sum)
    for tau in range(7): # 0 to 6
        X = np.zeros((16, 2))
        for i, (v1, v2) in enumerate(types):
            if v1 + v2 >= tau:
                X[i, 0] = 1.0
                X[i, 1] = 1.0
        P, rev = compute_optimal_payments(X, probs)
        if P is not None and rev > best_revenue:
            best_revenue = rev
            best_X = X.copy()
            best_P = P.copy()
            
    # 3. No Allocation
    X = np.zeros((16, 2))
    P, rev = compute_optimal_payments(X, probs)
    if P is not None and rev > best_revenue:
        best_revenue = rev
        best_X = X.copy()
        best_P = P.copy()

    # 4. Full Allocation
    X = np.ones((16, 2))
    P, rev = compute_optimal_payments(X, probs)
    if P is not None and rev > best_revenue:
        best_revenue = rev
        best_X = X.copy()
        best_P = P.copy()
        
    # Fallback if all fail (should not happen with X=0)
    if best_X is None:
        best_X = np.zeros((16, 2))
        best_P = np.zeros(16)
        
    return {'allocations': best_X, 'payments': best_P}
# EVOLVE-BLOCK-END
