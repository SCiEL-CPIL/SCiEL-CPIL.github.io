import numpy as np
import itertools
import random
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def construct():
    # 1. Generate all norm-2 integer vectors (squared norm 4)
    vectors = []
    
    # Type A: one +/- 2
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            vectors.append(v)
            
    # Type B: four +/- 1
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, val in zip(combo, signs):
                v[pos] = val
            vectors.append(v)
            
    N = len(vectors)
    
    # 2. Build conflict graph
    vec_arr = np.array(vectors, dtype=np.int8)
    dots = vec_arr @ vec_arr.T
    
    # Conflict if dot > 2
    rows, cols = np.where(dots > 2)
    valid = rows < cols
    rows = rows[valid]
    cols = cols[valid]
    
    adj = [[] for _ in range(N)]
    for r, c in zip(rows, cols):
        adj[r].append(c)
        adj[c].append(r)
        
    degrees = np.array([len(a) for a in adj], dtype=np.int32)
    
    # 3. Spectral Partitioning - compute Fiedler vector
    data = np.ones(len(rows), dtype=np.int32)
    row_sym = np.concatenate([rows, cols])
    col_sym = np.concatenate([cols, rows])
    data_sym = np.concatenate([data, data])
    A_sparse = csr_matrix((data_sym, (row_sym, col_sym)), shape=(N, N))
    
    D_sparse = csr_matrix(np.diag(degrees))
    L = D_sparse - A_sparse
    
    try:
        eigvals, eigvecs = eigsh(L, k=3, which='SM')
        idx = np.argsort(eigvals)
        fiedler_vec = eigvecs[:, idx[1]]
    except Exception:
        fiedler_vec = np.random.randn(N)
    
    def clean_independent_set(partition):
        """Greedy independent set from a candidate partition."""
        sorted_part = sorted(partition, key=lambda x: degrees[x], reverse=True)
        s = set()
        in_s = [False] * N
        for v in sorted_part:
            conflict = False
            for u in adj[v]:
                if in_s[u]:
                    conflict = True
                    break
            if not conflict:
                s.add(v)
                in_s[v] = True
        return s
    
    def kempe_chain_local_search(S, max_iter=50):
        """
        Optimized Kempe-chain based local search.
        """
        S = set(S)
        in_S = [False] * N
        for v in S:
            in_S[v] = True
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # Collect all vertices not in S
            T = [v for v in range(N) if not in_S[v]]
            if not T:
                break
            
            # Find connected components in the induced subgraph on T
            visited_comp = [False] * N
            components = []
            
            for start_v in T:
                if visited_comp[start_v]:
                    continue
                
                comp = []
                queue = [start_v]
                visited_comp[start_v] = True
                
                while queue:
                    u = queue.pop()
                    comp.append(u)
                    for v in adj[u]:
                        if not in_S[v] and not visited_comp[v]:
                            visited_comp[v] = True
                            queue.append(v)
                
                components.append(comp)
            
            # Sort components by size descending, only check top 20 to save time
            components.sort(key=len, reverse=True)
            top_comps = components[:20]
            
            best_gain = 0
            best_comp = None
            best_I = None
            
            for comp in top_comps:
                # Compute neighbors in S
                R = set()
                for v in comp:
                    for u in adj[v]:
                        if in_S[u]:
                            R.add(u)
                
                # Quick upper bound: len(comp) - len(R)
                if len(comp) - len(R) <= best_gain:
                    continue
                
                # Compute max independent set within comp (greedy)
                sorted_comp = sorted(comp, key=lambda x: degrees[x], reverse=True)
                I = set()
                in_I = [False] * len(comp) # Local optimization not really possible without mapping
                # Just use set for I
                for v in sorted_comp:
                    conflict = False
                    for u in adj[v]:
                        if u in I:
                            conflict = True
                            break
                    if not conflict:
                        I.add(v)
                
                gain = len(I) - len(R)
                if gain > best_gain:
                    best_gain = gain
                    best_comp = comp
                    best_I = I
            
            if best_comp is not None and best_gain > 0:
                # Perform the swap
                R = set()
                for v in best_comp:
                    for u in adj[v]:
                        if in_S[u]:
                            R.add(u)
                
                for v in R:
                    in_S[v] = False
                    S.discard(v)
                for v in best_I:
                    in_S[v] = True
                    S.add(v)
                improved = True
        
        return S
    
    def merge_two_sets(S1, S2):
        """
        Merge two independent sets by starting with the larger one
        and trying to add vertices from the other that don't conflict.
        """
        S = set(S1) if len(S1) >= len(S2) else set(S2)
        other = set(S2) if len(S1) >= len(S2) else set(S1)
        
        in_S = [False] * N
        for v in S:
            in_S[v] = True
        
        # Try to add vertices from other that don't conflict
        for v in other:
            if in_S[v]:
                continue
            conflict = False
            for u in adj[v]:
                if in_S[u]:
                    conflict = True
                    break
            if not conflict:
                in_S[v] = True
                S.add(v)
        
        # Run Kempe chain local search
        S = kempe_chain_local_search(S)
        return S
    
    # 4. Generate multiple partitions and find the best
    best_set = set()
    best_size = 0
    start_time = time.time()
    time_limit = 110 # Leave 10s buffer
    
    # Partition strategy 1: Sign split of Fiedler vector
    P1 = [i for i in range(N) if fiedler_vec[i] > 0]
    P2 = [i for i in range(N) if fiedler_vec[i] <= 0]
    
    S1_clean = clean_independent_set(P1)
    S2_clean = clean_independent_set(P2)
    
    S1_final = kempe_chain_local_search(S1_clean)
    S2_final = kempe_chain_local_search(S2_clean)
    
    # Merge the two
    S_merged = merge_two_sets(S1_final, S2_final)
    
    if len(S_merged) > best_size:
        best_set = S_merged
        best_size = len(S_merged)
    
    # Partition strategy 2: Threshold splits at different percentiles
    for pct in [0.3, 0.4, 0.5, 0.6, 0.7]:
        if time.time() - start_time > time_limit:
            break
        threshold = np.percentile(fiedler_vec, pct * 100)
        P1_t = [i for i in range(N) if fiedler_vec[i] > threshold]
        P2_t = [i for i in range(N) if fiedler_vec[i] <= threshold]
        
        if not P1_t or not P2_t:
            continue
        
        S1_t = clean_independent_set(P1_t)
        S2_t = clean_independent_set(P2_t)
        
        S1_t = kempe_chain_local_search(S1_t)
        S2_t = kempe_chain_local_search(S2_t)
        
        S_t = merge_two_sets(S1_t, S2_t)
        
        if len(S_t) > best_size:
            best_set = S_t
            best_size = len(S_t)
    
    # Partition strategy 3: Random restarts with perturbed Fiedler vector
    for restart in range(5):
        if time.time() - start_time > time_limit:
            break
        noise = np.random.randn(N) * 0.1
        perturbed = fiedler_vec + noise
        P1_r = [i for i in range(N) if perturbed[i] > 0]
        P2_r = [i for i in range(N) if perturbed[i] <= 0]
        
        S1_r = clean_independent_set(P1_r)
        S2_r = clean_independent_set(P2_r)
        
        S1_r = kempe_chain_local_search(S1_r)
        S2_r = kempe_chain_local_search(S2_r)
        
        S_r = merge_two_sets(S1_r, S2_r)
        
        if len(S_r) > best_size:
            best_set = S_r
            best_size = len(S_r)
    
    final_vectors = []
    for i in range(N):
        if i in best_set:
            final_vectors.append(vectors[i])
            
    return np.array(final_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
