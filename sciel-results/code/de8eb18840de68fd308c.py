import time
import random
import numpy as np

_N = 20
_DEADLINE = 120.0
_START = time.time()
_SEED = 12345

def _elapsed():
    return time.time() - _START

def _is_k23_free(A):
    # A: (20,20) int adjacency (0/1), symmetric, zero diag
    # For each pair (i,j), common neighbours = row i & row j ; need popcount <= 2
    # Vectorized: C = A @ A  gives C[i,j] = number of common neighbours of i,j
    C = A @ A
    # off-diagonal entries are common-neighbor counts
    # need all C[i,j] <= 2 for i != j
    return int(np.max(np.where(np.eye(_N, dtype=bool), 0, C))) <= 2

def _edge_count(A):
    return int(np.sum(A) // 2)

def _greedy(A, rng):
    # Try to add a random missing edge if it keeps K2,3-free
    # Work on a copy
    B = A.copy()
    # Precompute pair-common-neighbor counts
    C = B @ B
    for _ in range(200000):
        if _elapsed() > _DEADLINE - 2.0:
            break
        i = rng.randrange(_N)
        j = rng.randrange(_N)
        if i == j or B[i, j]:
            continue
        # adding edge i-j: new common-neighbor count for pairs (i, j) becomes C[i,j]
        # also for pairs (i, x) where x is neighbor of j: common neighbors gain j
        # for pairs (j, x) where x is neighbor of i: common neighbors gain i
        # Check: C[i,j] must be <= 2 after (it's currently C[i,j], adding edge makes common = C[i,j])
        # Actually common neighbors of i and j after adding edge = C[i,j] (unchanged, since i,j not neighbors yet)
        if C[i, j] > 2:
            continue
        # For x in N(j): common(i,x) currently C[i,x]; after adding i-j, j becomes common neighbor of i and x
        # so new common(i,x) = C[i,x] + 1 (if x not already... j is neighbor of x and now neighbor of i)
        # need C[i,x] + 1 <= 2 => C[i,x] <= 1 for all x in N(j)
        # Similarly for x in N(i): C[j,x] <= 1
        # Vectorized check
        ok = True
        # neighbors of j (excluding i)
        njs = np.where(B[j] == 1)[0]
        if len(njs) > 0:
            if np.max(C[i, njs]) > 1:
                ok = False
        if ok:
            nis = np.where(B[i] == 1)[0]
            if len(nis) > 0:
                if np.max(C[j, nis]) > 1:
                    ok = False
        if not ok:
            continue
        # Add edge
        B[i, j] = 1
        B[j, i] = 1
        # Update C: common neighbors increase
        # For all x: C[i,x] += B[j,x]; C[j,x] += B[i,x]; C[x,i] += B[j,x]; C[x,j] += B[i,x]
        # and C[i,j] += 1 (they are now common? no)
        # Actually common(i,x) = sum_k A[i,k]A[x,k]. Adding A[i,j]=1: for x != i, common(i,x) gains A[j,x]
        # Let's just recompute relevant rows
        C[i, :] += B[j, :]
        C[:, i] += B[j, :]
        C[j, :] += B[i, :]
        C[:, j] += B[i, :]
        C[i, j] += 1  # j is now neighbor of i and neighbor of j? no. Recompute: common(i,j)=sum A[i,k]A[j,k], adding A[i,j] doesn't change (A[j,j]=0)
        C[j, i] += 1
        # Wait, that double counts. Let me just recompute C fully occasionally for safety.
    return B

def _local_search(A, rng):
    # Try removing an edge and adding two others, or moving an edge
    B = A.copy()
    C = B @ B
    best = _edge_count(B)
    for it in range(50000):
        if _elapsed() > _DEADLINE - 2.0:
            break
        # Try to add a random missing edge
        i = rng.randrange(_N)
        j = rng.randrange(_N)
        if i == j or B[i, j]:
            continue
        if C[i, j] > 2:
            continue
        njs = np.where(B[j] == 1)[0]
        if len(njs) > 0 and np.max(C[i, njs]) > 1:
            continue
        nis = np.where(B[i] == 1)[0]
        if len(nis) > 0 and np.max(C[j, nis]) > 1:
            continue
        # add
        B[i, j] = 1
        B[j, i] = 1
        C[i, :] += B[j, :]
        C[:, i] += B[j, :]
        C[j, :] += B[i, :]
        C[:, j] += B[i, :]
        C[i, j] += 1
        C[j, i] += 1
        best = max(best, _edge_count(B))
    return B, best

def _try_pair_move(A, rng):
    # Remove edge (a,b), add (a,c) and (b,d) where c,d are non-neighbors
    B = A.copy()
    C = B @ B
    best = _edge_count(B)
    edges = [(i, j) for i in range(_N) for j in range(i + 1, _N) if B[i, j]]
    for _ in range(20000):
        if _elapsed() > _DEADLINE - 2.0:
            break
        if not edges:
            break
        a, b = edges[rng.randrange(len(edges))]
        c = rng.randrange(_N)
        d = rng.randrange(_N)
        if c in (a, b) or d in (a, b) or c == d:
            continue
        if B[a, c] or B[b, d]:
            continue
        # Remove (a,b), add (a,c),(b,d)
        # Check resulting graph is K2,3-free
        B[a, b] = 0
        B[b, a] = 0
        B[a, c] = 1
        B[c, a] = 1
        B[b, d] = 1
        B[d, b] = 1
        if _is_k23_free(B):
            C = B @ B
            edges = [(i, j) for i in range(_N) for j in range(i + 1, _N) if B[i, j]]
            best = max(best, _edge_count(B))
        else:
            # revert
            B[a, b] = 1
            B[b, a] = 1
            B[a, c] = 0
            B[c, a] = 0
            B[b, d] = 0
            B[d, b] = 0
    return B, best

def construct():
    rng = random.Random(_SEED)
    np.random.seed(_SEED)
    best_edges = None
    best_count = -1

    # Start from a random graph (some density) and greedily add
    for trial in range(200):
        if _elapsed() > _DEADLINE - 3.0:
            break
        # Random initial adjacency
        A = np.zeros((_N, _N), dtype=np.int64)
        # Start with a random sparse graph
        for i in range(_N):
            for j in range(i + 1, _N):
                if rng.random() < 0.15:
                    A[i, j] = 1
                    A[j, i] = 1
        # Greedy add
        B = _greedy(A, rng)
        cnt = _edge_count(B)
        if cnt > best_count:
            best_count = cnt
            best_edges = B
        # Local search
        C2, cnt2 = _local_search(B, rng)
        if cnt2 > best_count:
            best_count = cnt2
            best_edges = C2
        # Pair move
        D2, cnt3 = _try_pair_move(C2, rng)
        if cnt3 > best_count:
            best_count = cnt3
            best_edges = D2

    if best_edges is None:
        # Fallback: 5 copies of K4
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges

    # Extract edges
    edges = []
    for i in range(_N):
        for j in range(i + 1, _N):
            if best_edges[i, j]:
                edges.append([i, j])
    return edges
