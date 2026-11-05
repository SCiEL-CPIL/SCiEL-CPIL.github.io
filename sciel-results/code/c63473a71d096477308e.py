import time
from itertools import combinations

DIMENSIONS = [3, 4, 6]
_START_TIME = time.time()
_DEADLINE = 140  # seconds, safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 24
    pn = p * n  # 18
    total_monomials = n * m * p  # 72
    
    def check_deadline():
        return (time.time() - _START_TIME) < _DEADLINE
    
    # Target: 72 monomials, each with coefficient 1
    # Monomial (i,j,k) is at flat index i*m*p + j*p + k
    # U index: i*m + j, V index: j*p + k, W index: k*n + i
    
    # Build the set of all 72 monomial indices
    # A rank-1 tensor with U-mask u (bitmask over 12 bits), V-mask v (bitmask over 24 bits), W-mask w (bitmask over 18 bits)
    # covers monomial (i,j,k) iff u has bit (i*m+j) AND v has bit (j*p+k) AND w has bit (k*n+i)
    
    # Precompute for each monomial (i,j,k): its U-index, V-index, W-index
    monomial_info = []  # list of (u_idx, v_idx, w_idx) for each monomial
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                monomial_info.append((u_idx, v_idx, w_idx))
    
    # Candidate generation: small subsets
    # U indices: 0..11 (12 elements)
    # V indices: 0..23 (24 elements)
    # W indices: 0..17 (18 elements)
    
    # Generate candidate rank-1 tensors as triples of frozensets
    # For efficiency, use bitmasks (Python ints)
    
    # Strategy: generate candidates from single elements and pairs
    # U candidates: single bits and pairs
    U_candidates = []
    for i in range(nm):
        U_candidates.append(1 << i)
    for i in range(nm):
        for j in range(i+1, nm):
            U_candidates.append((1 << i) | (1 << j))
    
    V_candidates = []
    for i in range(mp):
        V_candidates.append(1 << i)
    for i in range(mp):
        for j in range(i+1, mp):
            V_candidates.append((1 << i) | (1 << j))
    
    W_candidates = []
    for i in range(pn):
        W_candidates.append(1 << i)
    for i in range(pn):
        for j in range(i+1, pn):
            W_candidates.append((1 << i) | (1 << j))
    
    # For a candidate (u_mask, v_mask, w_mask), compute which monomials it covers
    # A monomial (i,j,k) is covered iff:
    #   (u_mask >> (i*m+j)) & 1 == 1 AND
    #   (v_mask >> (j*p+k)) & 1 == 1 AND
    #   (w_mask >> (k*n+i)) & 1 == 1
    
    def compute_coverage(u_mask, v_mask, w_mask):
        """Return a bitmask of covered monomials (72 bits)."""
        covered = 0
        for idx, (u_idx, v_idx, w_idx) in enumerate(monomial_info):
            if (u_mask >> u_idx) & 1 and (v_mask >> v_idx) & 1 and (w_mask >> w_idx) & 1:
                covered |= (1 << idx)
        return covered
    
    # Precompute coverage for all candidate triples is too expensive.
    # Instead, generate a smaller candidate pool and compute on-the-fly.
    
    # Smaller candidate pool: focus on structured candidates
    # U: rows of the matrix (groups of m=4 consecutive indices)
    # V: columns (groups of p=6) and rows (groups of m=4)
    # W: rows (groups of n=3) and columns (groups of p=6)
    
    candidate_pool = []
    
    # Add structured candidates
    # U: single rows (4 elements each)
    for i in range(n):
        mask = 0
        for j in range(m):
            mask |= (1 << (i*m + j))
        candidate_pool.append(('U_row', mask))
    
    # U: single elements
    for i in range(nm):
        candidate_pool.append(('U_elem', 1 << i))
    
    # U: pairs within same row
    for i in range(n):
        for j1 in range(m):
            for j2 in range(j1+1, m):
                mask = (1 << (i*m + j1)) | (1 << (i*m + j2))
                candidate_pool.append(('U_pair', mask))
    
    # V: single columns (6 elements each)
    for j in range(m):
        mask = 0
        for k in range(p):
            mask |= (1 << (j*p + k))
        candidate_pool.append(('V_col', mask))
    
    # V: single elements
    for i in range(mp):
        candidate_pool.append(('V_elem', 1 << i))
    
    # V: pairs within same column
    for j in range(m):
        for k1 in range(p):
            for k2 in range(k1+1, p):
                mask = (1 << (j*p + k1)) | (1 << (j*p + k2))
                candidate_pool.append(('V_pair', mask))
    
    # W: single rows (3 elements each)
    for k in range(p):
        mask = 0
        for i in range(n):
            mask |= (1 << (k*n + i))
        candidate_pool.append(('W_row', mask))
    
    # W: single elements
    for i in range(pn):
        candidate_pool.append(('W_elem', 1 << i))
    
    # W: pairs within same row
    for k in range(p):
        for i1 in range(n):
            for i2 in range(i1+1, n):
                mask = (1 << (k*n + i1)) | (1 << (k*n + i2))
                candidate_pool.append(('W_pair', mask))
    
    # For greedy selection, we need to evaluate many (u,v,w) triples.
    # This is O(|U_cand| * |V_cand| * |W_cand|) which is too large.
    # Instead, use a different approach:
    # 
    # Alternate strategy: use the constructive rank-66 decomposition as baseline,
    # then try to find a lower-rank decomposition using a different method.
    #
    # Let's try a randomized local search / simulated annealing approach
    # starting from the rank-66 decomposition and trying to reduce rank.
    #
    # Actually, let's try a simpler approach: generate random rank-r decompositions
    # and check if they're valid, for decreasing r.
    
    # Fallback: constructive rank-66 decomposition
    rank_66 = 66
    U66 = [[0] * rank_66 for _ in range(nm)]
    V66 = [[0] * rank_66 for _ in range(mp)]
    W66 = [[0] * rank_66 for _ in range(pn)]
    current_rank = 0
    
    def add_rank1(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal current_rank
        if current_rank >= rank_66 or not check_deadline():
            return
        for idx, val in zip(u_indices, u_vals):
            U66[idx][current_rank] += val
        for idx, val in zip(v_indices, v_vals):
            V66[idx][current_rank] += val
        for idx, val in zip(w_indices, w_vals):
            W66[idx][current_rank] += val
        current_rank += 1
    
    def add_1_4_2(i_off, j_off, k_off):
        for j in range(4):
            for k in range(2):
                u_idx = (i_off) * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])
    
    def add_2_2_2(i_off, j_off, k_off):
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        
        add_rank1([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        add_rank1([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1])
        add_rank1([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1])
        add_rank1([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1])
        add_rank1([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1])
        add_rank1([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1])
        add_rank1([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1])
    
    def add_2_4_2(i_off, j_off, k_off):
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off + 2, k_off)
    
    for block_k in range(3):
        k_off = block_k * 2
        add_1_4_2(0, 0, k_off)
        add_2_4_2(1, 0, k_off)
    
    # Now try to improve: use GF(2) greedy forward selection
    # Target: all 72 monomials covered exactly once (mod 2)
    # Start with empty residual (all 72 bits set)
    
    # Generate a focused candidate pool for GF(2) search
    # Each candidate is a triple (u_mask, v_mask, w_mask)
    # We'll generate candidates from small structured sets
    
    gf2_candidates = []
    
    # U masks: single elements and pairs
    u_masks = []
    for i in range(nm):
        u_masks.append(1 << i)
    for i in range(nm):
        for j in range(i+1, nm):
            u_masks.append((1 << i) | (1 << j))
    
    # V masks: single elements and pairs
    v_masks = []
    for i in range(mp):
        v_masks.append(1 << i)
    for i in range(mp):
        for j in range(i+1, mp):
            v_masks.append((1 << i) | (1 << j))
    
    # W masks: single elements and pairs
    w_masks = []
    for i in range(pn):
        w_masks.append(1 << i)
    for i in range(pn):
        for j in range(i+1, pn):
            w_masks.append((1 << i) | (1 << j))
    
    # Precompute coverage for each (u_mask, v_mask, w_mask) is too expensive.
    # Instead, compute on-the-fly during greedy selection.
    
    # Greedy forward selection over GF(2):
    # residual = all 72 bits set
    # At each step, find candidate (u,v,w) that maximizes popcount(coverage & residual) - popcount(coverage & ~residual)
    # i.e., maximizes new coverage while minimizing re-coverage
    
    # This is O(|candidates| * 72) per step, which is too slow for large candidate pools.
    # Use a smaller candidate pool.
    
    # Focused candidate pool: use the structure of the problem
    # U: 12 elements, V: 24 elements, W: 18 elements
    # Generate candidates from single elements only (72*24*18 = 31104 candidates)
    # Plus pairs in U (66*24*18 = 28512), pairs in V (12*276*18 = 59544), pairs in W (12*24*153 = 43968)
    # Total ~163k candidates, each requiring 72-bit coverage computation
    
    # Let's precompute coverage for all single-element candidates
    # and use a smarter search
    
    # Actually, let's try a different approach: use the rank-66 decomposition
    # and try to find a lower-rank one via randomized search.
    
    # Try ranks from 65 down to some minimum
    # For each rank r, try to find a valid decomposition
    
    # Given time constraints, let's just return the rank-66 decomposition
    # if we can't find something better quickly.
    
    if not check_deadline():
        return {"U": U66, "V": V66, "W": W66}
    
    # Try a quick greedy GF(2) search with a limited candidate pool
    # Use only single-element candidates for speed
    # This gives 12 * 24 * 18 = 5184 candidates
    
    residual = (1 << 72) - 1  # all 72 monomials uncovered
    selected = []  # list of (u_mask, v_mask, w_mask)
    
    # Precompute coverage for single-element candidates
    # For u_mask = 1<<ui, v_mask = 1<<vi, w_mask = 1<<wi:
    # covers monomial (i,j,k) iff ui == i*m+j AND vi == j*p+k AND wi == k*n+i
    
    # For a given (ui, vi, wi), the monomial it covers is unique if it exists
    # ui = i*m+j, vi = j*p+k, wi = k*n+i
    # From ui: i = ui // m, j = ui % m
    # From vi: j' = vi // p, k = vi % p
    # From wi: k' = wi // n, i' = wi % n
    # Consistent iff i == i', j == j', k == k'
    
    # So single-element candidates cover at most 1 monomial each.
    # This is just the schoolbook approach, which gives rank 72.
    
    # For pairs, we can cover more. Let's try a limited search.
    
    # Given the complexity, let's return the rank-66 decomposition
    return {"U": U66, "V": V66, "W": W66}
