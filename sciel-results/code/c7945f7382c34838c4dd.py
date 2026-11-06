import math
import time
import numpy as np

def design(problem):
    """Design a symmetric 32x32-pixel unit cell for maximum TM 4-5 band gap.

    Strategy: start from a 2x2 array of square dielectric rods (the base
    structure known to enhance Bragg scattering for higher-order TM gaps),
    then refine rod size and position with a fast local search driven by an
    exact-coefficient plane-wave gap evaluator.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)

    start_time = time.time()
    deadline = start_time + 105  # safety margin well under the 120s budget

    N = size

    # ------------------------------------------------------------------
    # Fast exact plane-wave TM band-gap evaluator
    # ------------------------------------------------------------------
    # Number of plane-wave truncation per axis (full set = (2*npw+1)^2).
    # The evaluator uses 289 plane waves => npw = 8.
    npw = 8

    # G vectors (in units of 2*pi, i.e. lattice constant = 1)
    mx = np.arange(-npw, npw + 1)
    my = np.arange(-npw, npw + 1)
    GX, GY = np.meshgrid(mx, my, indexing='ij')
    GX = GX.ravel()
    GY = GY.ravel()
    nG = len(GX)

    # Dictionary mapping (mx,my) -> index for fast difference lookup
    G_dict = {}
    for idx in range(nG):
        G_dict[(GX[idx], GY[idx])] = idx

    # Precompute the difference-index matrix: D[i,j] = index of (G_i - G_j)
    # or -1 if that difference is outside the truncation set.
    D = np.full((nG, nG), -1, dtype=int)
    for i in range(nG):
        for j in range(nG):
            key = (GX[i] - GX[j], GY[i] - GY[j])
            if key in G_dict:
                D[i, j] = G_dict[key]

    # Pixel-centre coordinates
    x = (np.arange(N) + 0.5) / N
    y = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Precompute exp terms: phase[i, p] = exp(-i*(GX[i]*x_p + GY[i]*y_p))
    # p indexes pixels row-major.
    phase = -1j * (GX[:, None] * x[None, :] + GY[:, None] * y[None, :])
    # We need a full 2D grid of pixels, so build the phase per pixel.
    # phase2d[i, r, c] = exp(-i*(GX[i]*x_c + GY[i]*y_r))
    phase2d = -1j * (GX[:, None, None] * x[None, None, :] +
                     GY[:, None, None] * y[None, :, None])
    exp2d = np.exp(phase2d)  # (nG, N, N)

    # Wave vectors over the irreducible BZ (k in units where BZ edge = pi).
    # These cover the boundary and interior of the irreducible wedge.
    k_points = [
        (0.0, 0.0),
        (math.pi, 0.0),
        (math.pi / 2, math.pi / 2),
        (math.pi / 2, 0.0),
        (math.pi / 4, math.pi / 4),
        (math.pi / 2, math.pi / 4),
        (3 * math.pi / 8, math.pi / 8),
        (math.pi / 3, math.pi / 6),
        (2 * math.pi / 5, math.pi / 5),
        (math.pi / 2, math.pi / 6),
        (math.pi / 4, 0.0),
        (3 * math.pi / 4, 0.0),
        (math.pi / 6, 0.0),
        (math.pi / 2, math.pi / 8),
        (5 * math.pi / 8, math.pi / 8),
        (math.pi / 5, math.pi / 5),
        (math.pi / 2, 3 * math.pi / 8),
        (math.pi / 12, math.pi / 12),
        (math.pi / 3, 0.0),
        (math.pi / 2, math.pi / 3),
    ]

    def compute_gap(cell_2d):
        """Compute the gap-midgap ratio between TM bands 4 and 5."""
        eps = np.where(np.asarray(cell_2d, dtype=int) == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps

        # Fourier coefficients of 1/eps
        inv_eps_coeffs = np.sum(inv_eps[None, :, :] * exp2d, axis=(1, 2)) / (N * N)

        max_omega4 = 0.0
        min_omega5 = np.inf

        for kx, ky in k_points:
            # (k + G)^2 for each plane wave
            kG_sq = (kx + 2 * math.pi * GX) ** 2 + (ky + 2 * math.pi * GY) ** 2

            # Build the Hermitian matrix H[i,j] = A[i-j] * (k+G_j)^2
            # A[d] = inv_eps_coeffs[D[i,j]]
            A = inv_eps_coeffs[np.maximum(D, 0)]
            # Zero out entries where D was -1 (difference outside truncation)
            A = np.where(D >= 0, A, 0.0)
            H = A * kG_sq[None, :]
            H = (H + H.conj().T) / 2.0

            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)

            if len(eigvals) >= 6:
                o4 = eigvals[3]  # 4th band (0-indexed 3)
                o5 = eigvals[4]  # 5th band (0-indexed 4)
                if o4 > max_omega4:
                    max_omega4 = o4
                if o5 < min_omega5:
                    min_omega5 = o5

        if min_omega5 == np.inf or max_omega4 <= 0:
            return 0.0
        midgap = 0.5 * (min_omega5 + max_omega4)
        if midgap <= 0:
            return 0.0
        return (min_omega5 - max_omega4) / midgap

    # ------------------------------------------------------------------
    # Rod-based parametrisation with D4 symmetry
    # ------------------------------------------------------------------
    # A 2x2 array of square rods, each of side `s`, placed at the centre of
    # each quadrant (16x16).  Rod centres in the full 32x32 cell are at
    # (7.5, 7.5), (23.5, 7.5), (7.5, 23.5), (23.5, 23.5).
    def make_cell(rod_size, offset=0):
        """Build the full 32x32 symmetric cell from a 2x2 rod array."""
        cell = np.zeros((N, N), dtype=int)
        centres = [7.5, 23.5]
        half = rod_size / 2.0
        for cy in centres:
            for cx in centres:
                r0 = int(round(cy - half + offset))
                r1 = int(round(cy + half + offset))
                c0 = int(round(cx - half + offset))
                c1 = int(round(cx + half + offset))
                # clamp
                r0 = max(0, min(N - 1, r0))
                r1 = max(0, min(N - 1, r1))
                c0 = max(0, min(N - 1, c0))
                c1 = max(0, min(N - 1, c1))
                if r1 >= r0 and c1 >= c0:
                    cell[r0:r1 + 1, c0:c1 + 1] = 1
        return cell

    # Initial best
    best_cell = make_cell(8)
    best_gap = compute_gap(best_cell)

    # ------------------------------------------------------------------
    # Local search over rod size and position offset
    # ------------------------------------------------------------------
    rod_sizes = [6, 7, 8, 9, 10, 11, 12]
    offsets = [-1, 0, 1]

    # First pass: scan the grid
    for s in rod_sizes:
        if time.time() > deadline:
            break
        for off in offsets:
            if time.time() > deadline:
                break
            cell = make_cell(s, off)
            gap = compute_gap(cell)
            if gap > best_gap:
                best_gap = gap
                best_cell = cell.copy()

    # Second pass: fine local search around the best (size +/- 1, offset +/- 1)
    # Determine the best size/offset by scanning again with finer steps.
    # We do a small hill-climb: try perturbing size by +/-1 and offset by +/-1.
    # We track best (s, off) by re-evaluating neighbours.
    # Find which (s, off) gave best_cell.
    best_s, best_off = None, None
    for s in rod_sizes:
        for off in offsets:
            if np.array_equal(make_cell(s, off), best_cell):
                best_s, best_off = s, off
                break
        if best_s is not None:
            break
    if best_s is None:
        best_s, best_off = 8, 0

    for _ in range(3):
        improved = False
        for ds in [-1, 1]:
            for do in [-1, 0, 1]:
                if time.time() > deadline:
                    break
                s2 = best_s + ds
                off2 = best_off + do
                if s2 < 4 or s2 > 14:
                    continue
                cell = make_cell(s2, off2)
                gap = compute_gap(cell)
                if gap > best_gap + 1e-12:
                    best_gap = gap
                    best_cell = cell.copy()
                    best_s, best_off = s2, off2
                    improved = True
        if not improved:
            break

    # ------------------------------------------------------------------
    # Final pixel-level refinement: flip individual pixels in the top-left
    # quadrant (D4-symmetric) to fine-tune the gap.
    # ------------------------------------------------------------------
    def quadrant_to_full(q):
        q = np.asarray(q, dtype=int)
        top = np.hstack([q, q[:, ::-1]])
        bottom = top[::-1, :]
        return np.vstack([top, bottom])

    def full_to_quadrant(cell):
        return np.asarray(cell[:16, :16], dtype=int)

    q = full_to_quadrant(best_cell)

    # Hill-climb: try flipping each pixel in the quadrant; keep the best
    # improving flip each round.
    for _ in range(4):
        if time.time() > deadline:
            break
        best_flip = None
        best_flip_gain = 1e-12
        for r in range(16):
            if time.time() > deadline:
                break
            for c in range(16):
                if time.time() > deadline:
                    break
                q2 = q.copy()
                q2[r, c] = 1 - q2[r, c]
                cell = quadrant_to_full(q2)
                gap = compute_gap(cell)
                gain = gap - best_gap
                if gain > best_flip_gain:
                    best_flip_gain = gain
                    best_flip = (r, c)
        if best_flip is not None:
            r, c = best_flip
            q[r, c] = 1 - q[r, c]
            best_cell = quadrant_to_full(q)
            best_gap += best_flip_gain
        else:
            break

    # Return as nested list of 0/1
    return best_cell.tolist()
