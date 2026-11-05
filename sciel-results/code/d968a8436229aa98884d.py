"""Design a 32x32 symmetric photonic crystal unit cell for maximum TM band gap.

Uses a 4x4 supercell of 8x8 sub-blocks with a coarse-grained PWE solver.
"""
import time
import math
import numpy as np

_DEADLINE = 220.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _DEADLINE - _elapsed()


def _enforce_c4v(cell):
    """
    Enforce exact C4v symmetry by mirroring the top-left 16x16 quadrant
    to the rest of the 32x32 cell. This guarantees:
      - cell == cell[::-1] row-wise (horizontal mirror)
      - cell == cell[::-1] column-wise (vertical mirror)  
      - cell == cell.T (transpose)
    """
    size = len(cell)
    half = size // 2
    result = [[0] * size for _ in range(size)]
    for i in range(half):
        for j in range(half):
            val = cell[i][j]
            # C4v orbits: (i,j), (i, size-1-j), (size-1-i, j), (size-1-i, size-1-j)
            # Plus transposes: (j,i), (j, size-1-i), (size-1-j, i), (size-1-j, size-1-i)
            positions = [
                (i, j),
                (i, size - 1 - j),
                (size - 1 - i, j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (j, size - 1 - i),
                (size - 1 - j, i),
                (size - 1 - j, size - 1 - i),
            ]
            for pi, pj in positions:
                result[pi][pj] = val
    return result


def _build_cell_from_blocks(blocks):
    """
    Build a 32x32 cell from a 4x4 array of 8x8 sub-blocks.
    blocks is a 4x4 nested list of 8x8 0/1 arrays.
    Enforces C4v symmetry on the final cell.
    """
    size = 32
    cell = [[0] * size for _ in range(size)]
    for bi in range(4):
        for bj in range(4):
            block = blocks[bi][bj]
            for ii in range(8):
                for jj in range(8):
                    cell[bi * 8 + ii][bj * 8 + jj] = block[ii][jj]
    return _enforce_c4v(cell)


def _make_block_library():
    """
    Generate a library of 8x8 symmetric block patterns.
    Each block is symmetric under its own C4v (about its center).
    """
    library = []
    bs = 8
    
    # 1. All-zero block
    library.append([[0] * bs for _ in range(bs)])
    
    # 2. All-one block
    library.append([[1] * bs for _ in range(bs)])
    
    # 3. Central disk of various radii (in block-normalized coords, center at 3.5,3.5)
    for r in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        block = [[0] * bs for _ in range(bs)]
        for i in range(bs):
            for j in range(bs):
                x = (i + 0.5) / bs - 0.5
                y = (j + 0.5) / bs - 0.5
                if x * x + y * y < r * r / (bs * bs) * 4:  # scale to block
                    if x * x + y * y < (r / 4.0) ** 2:
                        block[i][j] = 1
        library.append(block)
    
    # 4. Ring patterns
    for r_in, r_out in [(1.0, 2.0), (1.5, 2.5), (2.0, 3.0), (1.0, 3.0)]:
        block = [[0] * bs for _ in range(bs)]
        for i in range(bs):
            for j in range(bs):
                x = (i + 0.5) / bs - 0.5
                y = (j + 0.5) / bs - 0.5
                r = math.sqrt(x * x + y * y)
                if r_in / 4.0 <= r < r_out / 4.0:
                    block[i][j] = 1
        library.append(block)
    
    # 5. Cross/plus patterns
    for bw in [1, 2, 3]:
        block = [[0] * bs for _ in range(bs)]
        for i in range(bs):
            for j in range(bs):
                if abs(i - 3.5) < bw / 2.0 or abs(j - 3.5) < bw / 2.0:
                    block[i][j] = 1
        library.append(block)
    
    # 6. Checkerboard patterns
    for period in [2, 4]:
        block = [[0] * bs for _ in range(bs)]
        for i in range(bs):
            for j in range(bs):
                if (i // period + j // period) % 2 == 0:
                    block[i][j] = 1
        library.append(block)
    
    # 7. Diagonal stripe patterns
    for period in [2, 4]:
        block = [[0] * bs for _ in range(bs)]
        for i in range(bs):
            for j in range(bs):
                if ((i + j) // period) % 2 == 0:
                    block[i][j] = 1
        library.append(block)
    
    # 8. Corner dots
    for r in [0.8, 1.2, 1.5]:
        block = [[0] * bs for _ in range(bs)]
        corners = [(1.5, 1.5), (1.5, 6.5), (6.5, 1.5), (6.5, 6.5)]
        for i in range(bs):
            for j in range(bs):
                x = (i + 0.5) / bs - 0.5
                y = (j + 0.5) / bs - 0.5
                for cx, cy in corners:
                    dx = (i + 0.5) - cx
                    dy = (j + 0.5) - cy
                    if dx * dx + dy * dy < r * r:
                        block[i][j] = 1
                        break
        library.append(block)
    
    # 9. Edge bars (top/bottom or left/right)
    for w in [1, 2, 3]:
        block = [[0] * bs for _ in range(bs)]
        for i in range(bs):
            for j in range(bs):
                if i < w or i >= bs - w or j < w or j >= bs - w:
                    block[i][j] = 1
        library.append(block)
    
    # 10. Central square
    for s in [2, 4, 6]:
        block = [[0] * bs for _ in range(bs)]
        start = (bs - s) // 2
        for i in range(start, start + s):
            for j in range(start, start + s):
                block[i][j] = 1
        library.append(block)
    
    # 11. X pattern
    block = [[0] * bs for _ in range(bs)]
    for i in range(bs):
        for j in range(bs):
            if abs(i - j) < 1.5 or abs(i + j - 7) < 1.5:
                block[i][j] = 1
    library.append(block)
    
    # 12. Hollow center (frame)
    for t in [1, 2, 3]:
        block = [[1] * bs for _ in range(bs)]
        start = t
        end = bs - t
        for i in range(start, end):
            for j in range(start, end):
                block[i][j] = 0
        library.append(block)
    
    return library


def _compute_tm_gap(cell, N_pw=8, n_kpts=10):
    """
    Fast PWE solver for TM band gap between bands 4 and 5.
    """
    size = len(cell)
    eps_high = 12.0
    eps_low = 1.0

    # Precompute 1/eps
    inv_eps = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            inv_eps[i][j] = 1.0 / (eps_high if cell[i][j] == 1 else eps_low)

    # FFT for Fourier coefficients
    fft_coeffs = np.fft.fft2(inv_eps)
    N = size

    def get_coeff(m, n):
        return fft_coeffs[m % N, n % N] / (N * N)

    # G vectors
    g_indices = []
    for m in range(-N_pw + 1, N_pw):
        for n in range(-N_pw + 1, N_pw):
            g_indices.append((m, n))

    npw = len(g_indices)

    # S matrix
    S = np.zeros((npw, npw), dtype=float)
    for i, (mi, ni) in enumerate(g_indices):
        for j, (mj, nj) in enumerate(g_indices):
            S[i, j] = get_coeff(mi - mj, ni - nj).real

    # k-points in irreducible BZ
    kpts = _get_kpts_fast(n_kpts)

    max_omega4 = 0.0
    min_omega5 = float('inf')

    for kx, ky in kpts:
        km2 = np.zeros(npw)
        for idx, (m, n) in enumerate(g_indices):
            km2[idx] = (2.0 * m + kx) ** 2 + (2.0 * n + ky) ** 2

        A = S * km2[np.newaxis, :]

        try:
            eigenvalues = np.linalg.eigvalsh(A)
            eigenvalues = np.sort(np.real(eigenvalues))

            if len(eigenvalues) >= 5:
                omega4_sq = eigenvalues[3]
                omega5_sq = eigenvalues[4]

                if omega4_sq > 0 and omega5_sq > 0:
                    omega4 = math.sqrt(omega4_sq)
                    omega5 = math.sqrt(omega5_sq)

                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
        except:
            continue

    if min_omega5 == float('inf') or max_omega4 == 0:
        return 0.0

    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0

    if midgap <= 0:
        return 0.0

    return 2.0 * gap / midgap


def _get_kpts_fast(n_kpts):
    """Generate k-points in the irreducible Brillouin zone (fast version)."""
    kpts = []
    n_seg = max(2, n_kpts // 3)
    
    # Gamma to X: (kx, 0) for kx in [0, 1]
    for i in range(n_seg + 1):
        kx = i / n_seg
        kpts.append((kx, 0.0))
    
    # Gamma to M: (kx, kx) for kx in [0, 0.5]
    for i in range(1, n_seg + 1):
        kx = i / n_seg * 0.5
        kpts.append((kx, kx))
    
    # X to M: (1 - t, t) for t in [0, 0.5]
    for i in range(1, n_seg):
        t = i / n_seg * 0.5
        kpts.append((1.0 - t, t))
    
    # Some interior points
    kpts.extend([(0.3, 0.1), (0.5, 0.2), (0.7, 0.3), (0.4, 0.15), (0.6, 0.25)])
    
    # Deduplicate
    unique = []
    seen = set()
    for kx, ky in kpts:
        if ky < 0:
            ky = -ky
        if kx < ky:
            kx, ky = ky, kx
        key = (round(kx, 4), round(ky, 4))
        if key not in seen:
            seen.add(key)
            unique.append((kx, ky))
    
    return unique[:n_kpts]


def _coarse_evaluate(block_ids):
    """
    Evaluate a 4x4 arrangement of blocks using the fast PWE solver.
    block_ids is a 4x4 list of indices into the block library.
    """
    # Build the cell from blocks
    cell = _build_cell_from_blocks(block_ids)
    return _compute_tm_gap(cell, N_pw=6, n_kpts=8)


def _local_search_blocks(block_ids, library, max_steps=20):
    """
    Local search on the 4x4 block arrangement.
    Try changing one block at a time.
    """
    best_ids = [row[:] for row in block_ids]
    best_score = _coarse_evaluate(block_ids)
    
    for step in range(max_steps):
        if _remaining() < 30:
            break
        
        improved = False
        # Try each position
        for bi in range(4):
            for bj in range(4):
                if _remaining() < 30:
                    break
                
                current_id = block_ids[bi][bj]
                # Try other block types
                for new_id in range(len(library)):
                    if new_id == current_id:
                        continue
                    
                    new_ids = [row[:] for row in block_ids]
                    new_ids[bi][bj] = new_id
                    
                    score = _coarse_evaluate(new_ids)
                    if score > best_score:
                        best_score = score
                        block_ids = new_ids
                        best_ids = [row[:] for row in block_ids]
                        improved = True
                        break
                
                if _remaining() < 30:
                    break
        
        if not improved:
            break
    
    return best_ids, best_score


def _generate_initial_block_arrangements(library):
    """
    Generate diverse initial 4x4 block arrangements.
    """
    arrangements = []
    nlib = len(library)
    
    # 1. Uniform arrangements
    for bid in range(min(10, nlib)):
        arr = [[bid] * 4 for _ in range(4)]
        arrangements.append(arr)
    
    # 2. Central pattern: different center block
    for center_bid in range(min(8, nlib)):
        for edge_bid in range(min(4, nlib)):
            if center_bid == edge_bid:
                continue
            arr = [[edge_bid] * 4 for _ in range(4)]
            arr[1][1] = center_bid
            arr[1][2] = center_bid
            arr[2][1] = center_bid
            arr[2][2] = center_bid
            arrangements.append(arr)
    
    # 3. Checkerboard of two block types
    for a in range(min(4, nlib)):
        for b in range(min(4, nlib)):
            if a == b:
                continue
            arr = [[(a if (i + j) % 2 == 0 else b) for j in range(4)] for i in range(4)]
            arrangements.append(arr)
    
    # 4. Radial patterns
    for c in range(min(4, nlib)):
        for m in range(min(4, nlib)):
            for o in range(min(4, nlib)):
                if c == m or m == o:
                    continue
                arr = [[0] * 4 for _ in range(4)]
                for i in range(4):
                    for j in range(4):
                        dist = max(abs(i - 1.5), abs(j - 1.5))
                        if dist < 0.5:
                            arr[i][j] = c
                        elif dist < 1.5:
                            arr[i][j] = m
                        else:
                            arr[i][j] = o
                arrangements.append(arr)
    
    # 5. Cross pattern
    for c in range(min(3, nlib)):
        for b in range(min(3, nlib)):
            if c == b:
                continue
            arr = [[b] * 4 for _ in range(4)]
            for i in range(4):
                arr[i][1] = c
                arr[i][2] = c
            for j in range(4):
                arr[1][j] = c
                arr[2][j] = c
            arrangements.append(arr)
    
    # 6. Random arrangements
    import random
    random.seed(42)
    for _ in range(20):
        arr = [[random.randint(0, min(nlib, 10) - 1) for _ in range(4)] for _ in range(4)]
        arrangements.append(arr)
    
    return arrangements


def design(problem):
    """
    Design a 32x32 symmetric photonic crystal unit cell.
    Returns a nested list of 0/1 pixels.
    """
    global _start_time
    _start_time = time.time()

    # Build block library
    library = _make_block_library()
    
    # Generate initial arrangements
    arrangements = _generate_initial_block_arrangements(library)
    
    best_ids = None
    best_score = -float('inf')
    
    # Evaluate all initial arrangements
    for arr in arrangements:
        if _remaining() < 40:
            break
        
        score = _coarse_evaluate(arr)
        if score > best_score:
            best_score = score
            best_ids = [row[:] for row in arr]
    
    # Local search from best
    if best_ids is not None and _remaining() > 60:
        best_ids, best_score = _local_search_blocks(best_ids, library, max_steps=15)
    
    # Try local search from a few other good arrangements
    if _remaining() > 40:
        # Re-evaluate top arrangements
        scored = []
        for arr in arrangements[:30]:
            if _remaining() < 30:
                break
            score = _coarse_evaluate(arr)
            scored.append((score, arr))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        for score, arr in scored[:5]:
            if _remaining() < 25:
                break
            if score >= best_score - 0.05:
                _, new_score = _local_search_blocks(arr, library, max_steps=10)
                if new_score > best_score:
                    best_score = new_score
                    best_ids = [row[:] for row in arr]
    
    # Build final cell
    if best_ids is None:
        # Fallback: uniform block
        best_ids = [[1] * 4 for _ in range(4)]
    
    cell = _build_cell_from_blocks(best_ids)
    
    # Verify symmetry
    cell = _enforce_c4v(cell)
    
    return cell
