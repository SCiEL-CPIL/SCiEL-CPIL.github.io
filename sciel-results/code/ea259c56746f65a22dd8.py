"""Exact low-rank tensor decomposition search via projected integer lattice search."""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [3, 4, 7]
DEADLINE = 140.0  # safety margin under 150s wall-clock limit


def _target_tensor(n, m, p):
    """T[i,j,k] = 1 for all (i,j,k)."""
    return np.ones((n, m, p), dtype=object)


def _test_ranks(n, m, p, seed):
    """Generate random rank-1 test tensors (i,j,k) -> a_i * b_j * c_k."""
    rng = random.Random(seed)
    tests = []
    for _ in range(40):
        a = [rng.randint(-3, 3) for _ in range(n)]
        b = [rng.randint(-3, 3) for _ in range(m)]
        c = [rng.randint(-3, 3) for _ in range(p)]
        tests.append((a, b, c))
    return tests


def _build_constraints(n, m, p, tests, r):
    """
    Build linear constraint matrix. Variables: U (n*m*r), V (m*p*r), W (p*n*r).
    For each test (a,b,c) and each (i,j,k):
      a_i b_j c_k = sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
    This is cubic, not linear. So instead we use a different formulation:
    We require that for the specific bilinear structure, the product matches.
    
    Actually the tensor equation T[i,j,k] = sum_l U[i*m+j,l]*V[j*p+k,l]*W[k*n+i,l]
    is cubic in the variables. We can't do linear search directly.
    
    Alternative: use the fact that for rank-1 terms, each term is a product of
    three vectors. We search over the U matrix entries (small integers), then for
    each candidate U, the remaining equations become linear in V and W given W,
    or we can do a two-stage search.
    
    Simpler: enumerate small integer U (n*m rows, r cols), then for each, check
    if there exist integer V, W satisfying the equations. Given U and V, W is
    determined by: for each (i,j,k), W[k*n+i] = (T[i,j,k] - sum_l U[i*m+j,l]*V[j*p+k,l]*W_other)/... 
    This is still coupled.
    
    Let me use a cleaner approach: search over r rank-1 terms directly.
    Each rank-1 term: u (n*m vector), v (m*p vector), w (p*n vector).
    We need: for all (i,j,k): sum_t u_t[i*m+j]*v_t[j*p+k]*w_t[k*n+i] = 1.
    
    Search: try small integer vectors for u, v, w.
    """
    pass


def _search_small_rank(n, m, p, r, deadline, seed):
    """
    Search for r rank-1 terms with small integer factors.
    Each term t has: u_t (n*m,), v_t (m*p,), w_t (p*n,) with entries in {-1,0,1}.
    Check if sum_t u_t[i*m+j]*v_t[j*p+k]*w_t[k*n+i] == 1 for all (i,j,k).
    """
    rng = random.Random(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    target = np.ones((n, m, p), dtype=int)
    
    # Precompute index mappings
    # For a given term (u, v, w), the contribution to (i,j,k) is u[i*m+j]*v[j*p+k]*w[k*n+i]
    
    def evaluate_terms(terms):
        """terms: list of (u, v, w) tuples. Returns residual norm."""
        total = np.zeros((n, m, p), dtype=int)
        for u, v, w in terms:
            for i in range(n):
                for j in range(m):
                    uv = u[i * m + j]
                    if uv == 0:
                        continue
                    for k in range(p):
                        total[i, j, k] += uv * v[j * p + k] * w[k * n + i]
        return np.sum((total - target) ** 2)
    
    # Random search over small integer vectors
    # To make it tractable, restrict to vectors with mostly zeros
    # Strategy: each u has at most 2 nonzeros, etc.
    
    # Actually, let's try a structured approach:
    # For r=1: u[i*m+j]*v[j*p+k]*w[k*n+i] = 1 for all i,j,k
    # This means u[i*m+j]=1, v[j*p+k]=1, w[k*n+i]=1 for all -> the schoolbook with r=1 only works if n=m=p=1.
    # So r=1 is impossible for n=3,m=4,p=7.
    
    # For general r, let's do random search with small entries.
    # Limit: each vector has entries in {-1, 0, 1}, and we try random sparse patterns.
    
    max_iter = 200000
    for it in range(max_iter):
        if time.time() > deadline:
            return None
        terms = []
        for t in range(r):
            # Random sparse vector
            u = np.zeros(nm, dtype=int)
            v = np.zeros(mp, dtype=int)
            w = np.zeros(pn, dtype=int)
            # Pick a few nonzero positions
            for _ in range(rng.randint(1, min(3, nm))):
                pos = rng.randint(0, nm - 1)
                u[pos] = rng.choice([-1, 1])
            for _ in range(rng.randint(1, min(3, mp))):
                pos = rng.randint(0, mp - 1)
                v[pos] = rng.choice([-1, 1])
            for _ in range(rng.randint(1, min(3, pn))):
                pos = rng.randint(0, pn - 1)
                w[pos] = rng.choice([-1, 1])
            terms.append((u, v, w))
        
        if evaluate_terms(terms) == 0:
            return terms
    
    return None


def _try_fixed_structure(n, m, p, r, deadline):
    """
    Try a structured decomposition where the r terms follow a pattern.
    For matrix multiplication, known constructions exist.
    
    One idea: use the fact that the tensor T[i,j,k]=1 can be written as
    sum over a "basis" of the multiplication. For small dimensions,
    we can try rank = n*p or m*p or n*m with structured factors.
    
    Let's try rank = n*p: for each (i,k), have a term.
    Term (i,k): u_{i,k}[j] = 1 for all j, v_{i,k}[j,k'] = delta(k,k')... 
    This doesn't directly work.
    
    Actually, the standard lower bound for 3x4x7 is unknown to be below 84.
    Let's just try to find ANY decomposition with rank < 84.
    """
    pass


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 84
    
    # Build schoolbook as fallback
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    best = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to find lower rank decompositions
    # Start from rank 1 upward, but skip obviously impossible ranks
    # For 3x4x7, the tensor rank is at least max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # So try ranks from 28 upward
    
    min_possible = max(n * m, m * p, p * n)
    
    for r in range(min_possible, best_rank):
        if time.time() > DEADLINE - 10:
            break
        # Try random search for this rank
        result = _search_small_rank(n, m, p, r, DEADLINE - 5, seed=r * 1000)
        if result is not None:
            # Convert to required format
            U_new = [[0] * r for _ in range(n * m)]
            V_new = [[0] * r for _ in range(m * p)]
            W_new = [[0] * r for _ in range(p * n)]
            for t in range(r):
                u, v, w = result[t]
                for idx in range(n * m):
                    U_new[idx][t] = int(u[idx])
                for idx in range(m * p):
                    V_new[idx][t] = int(v[idx])
                for idx in range(p * n):
                    W_new[idx][t] = int(w[idx])
            best = {"U": U_new, "V": V_new, "W": W_new}
            best_rank = r
            break  # Found lowest rank, stop
    
    return best
