"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.linalg import eig

def _make_cell_from_quadrant(quadrant, size=32):
    """
    Expand an 8x8 upper-right quadrant to a full 32x32 cell.
    quadrant: 8x8 array of 0/1.
    The cell is invariant under 4-fold symmetry (D4).
    Pixels (i, j) in [0, 32) map to quadrant (i, j) if i<16, j<16.
    For i>=16, mirror i. For j>=16, mirror j.
    Also handle the center line carefully.
    Since 32 is even, there is no single center pixel.
    Quadrant indices 0..7 map to global indices 0..7.
    Global indices 24..31 map to quadrant indices 7..0 (reversed).
    """
    cell = np.zeros((size, size), dtype=np.float64)
    half = size // 2
    q_size = half // 2
    
    # Fill the upper-right quadrant (0..15, 0..15)
    # The quadrant provided is 8x8, covering x in [0, 0.5), y in [0, 0.5)
    # Wait, 32x32 cell. Half is 16.
    # The "quadrant" in the prompt context usually refers to the fundamental domain of the symmetry group.
    # For D4 symmetry on a 32x32 grid:
    # The fundamental domain is 1/8 of the area? No, D4 has 8 elements.
    # 32x32 / 8 = 128 pixels.
    # The fundamental domain is a triangle.
    # However, the prompt says "invariant under the square symmetry group".
    # The seed program used a circle parameterization.
    # Let's stick to the 8x8 quadrant idea but map it correctly.
    # Actually, a simpler discrete representation for D4 symmetry:
    # Define the cell by its values in the region 0 <= i < 16, 0 <= j < 16, with the constraint that cell[i,j] = cell[15-i, j] ? No.
    # Symmetry about center: cell[i,j] == cell[31-i, 31-j].
    # Also cell[i,j] == cell[j,i] (transpose).
    # And cell[i,j] == cell[31-i, j] ? No, that's reflection across horizontal midline.
    # D4 includes: identity, rot 90, rot 180, rot 270, refl x, refl y, refl x=y, refl x=-y.
    # For a 32x32 grid, the fundamental domain is the set of pixels (i,j) such that:
    # 0 <= i < 16, 0 <= j < 16, and i <= j? No.
    # Let's just use the 8x8 block in the corner (0..7, 0..7) and mirror it?
    # If I define pixels in [0, 8) x [0, 8), and mirror to [24, 32) x [24, 32), etc., I need to handle the middle.
    # Actually, the simplest robust way is to define the 16x16 upper-right quadrant and enforce symmetry within it?
    # No, the seed program's `_make_cell_from_params` built the whole cell.
    # Let's use a 16x16 quadrant (upper right of the 32x32) and mirror it to the other 3 quadrants.
    # Upper-right quadrant: i in [16, 32), j in [16, 32).
    # Center of cell is (15.5, 15.5).
    # Symmetry: cell[i,j] = cell[31-i, 31-j] (180 rot)
    # cell[i,j] = cell[j,i] (transpose)
    # cell[i,j] = cell[31-i, j] (horizontal flip)
    # If I define the top-right 16x16 block, I can generate the rest.
    # But I also need to satisfy cell[i,j] = cell[i, 31-j] (vertical flip) within the top-right?
    # No, the top-right block maps to top-left via vertical flip?
    # Let's define the fundamental domain as the triangle in the top-right quadrant where i <= j?
    # This is complicated.
    
    # Alternative: Use the 8x8 corner (0..7, 0..7) and replicate it to all 4 corners, and fill the center cross?
    # This assumes the design is composed of 4 identical 8x8 blocks in the corners and a symmetric center.
    # This is a restricted design space, but it's a valid structural change.
    # Let's try:
    # cell[0:8, 0:8] = q
    # cell[24:32, 0:8] = q[::-1, :]  # Bottom-left? No, 180 rot of top-right is bottom-left.
    # Top-right is [16:32, 16:32].
    # Let's map the 8x8 quadrant to the Top-Right corner [24:32, 24:32]?
    # No, let's map it to [0:8, 0:8] (Top-Left) and mirror.
    
    # Let's use the 16x16 upper-right quadrant (indices 16..31, 16..31) as the source.
    # Actually, let's just use the 8x8 block at the center of the edge?
    
    # Let's go with the 8x8 block in the top-left corner (0..7, 0..7).
    # We will enforce D4 symmetry by mirroring this block to all 4 corners and filling the center 16x16 area with a symmetric pattern derived from the block?
    # This might be too restrictive.
    
    # Better: Define the 16x16 upper-right quadrant (i in 16..31, j in 16..31).
    # Generate the full 32x32 from this.
    # cell[i,j] for i<16, j<16 (Top-Left) is determined by symmetry.
    # Symmetry: cell[i,j] = cell[31-i, 31-j] (180 rot).
    # cell[i,j] = cell[j,i] (transpose).
    # cell[i,j] = cell[31-i, j] (horiz flip).
    # From these, the Top-Left quadrant is determined by the Top-Right?
    # Top-Right (i>15, j>15).
    # Let's define the source as the 16x16 Top-Right quadrant.
    # Source: src[16..31, 16..31].
    # Top-Left (i<16, j<16):
    # cell[i,j] = cell[31-i, 31-j] = src[31-i, 31-j].
    # Bottom-Right (i>15, j>15): src.
    # Bottom-Left (i>15, j<16):
    # cell[i,j] = cell[j,i] (transpose). Since j<16, i>15, (j,i) is in Top-Left.
    # So cell[i,j] = cell[j,i] = src[31-j, 31-i].
    # Top-Right (i<16, j>15):
    # cell[i,j] = cell[j,i] = src[31-j, 31-i]? No.
    # (i,j) in Top-Right. (j,i) in Bottom-Left.
    # cell[i,j] = cell[j,i] = src[31-i, 31-j]? No.
    # Let's check:
    # cell[i,j] (TR).
    # Flip x: cell[31-i, j]. (31-i) > 15, j > 15. This is in BR.
    # So cell[i,j] = cell[31-i, j] (BR).
    # BR is src. So cell[i,j] = src[31-i, j].
    
    # So if I define src (16x16, indices 16..31), I can build the cell.
    # But I want to search over a smaller space.
    # The 16x16 src has 256 pixels.
    # Can I reduce it?
    # The src itself must satisfy symmetries?
    # The full cell has D4 symmetry.
    # The Top-Right quadrant (16x16) does NOT have internal symmetry.
    # So I have to search 256 bits.
    # This is too large for simple hill climbing in 120s.
    
    # Let's go back to the 8x8 corner idea.
    # Define an 8x8 block in the Top-Left corner (0..7, 0..7).
    # Assume the design is formed by replicating this block to all 4 corners (0..7, 0..7), (24..31, 0..7), (0..7, 24..31), (24..31, 24..31) with appropriate flips.
    # And fill the center 16x16 area (8..23, 8..23) with a symmetric pattern, e.g., all 0 or all 1 or derived from the block.
    # This is a very restricted space (64 bits).
    
    cell = np.zeros((size, size), dtype=np.float64)
    
    # Place the 8x8 block in the four corners
    # Top-Left
    cell[0:8, 0:8] = quadrant
    # Bottom-Right (180 rot of TL)
    cell[24:32, 24:32] = quadrant[::-1, ::-1]
    # Top-Right (Transpose of TL? No, Transpose of TL is TL if symmetric. But D4 includes transpose.)
    # cell[i,j] = cell[j,i].
    # For i<8, j>23: (i,j) is in TR. (j,i) is in BL.
    # This mapping is getting messy.
    
    # Let's use the 16x16 Top-Right quadrant but reduce the search space by assuming it is composed of 4 8x8 blocks?
    # No.
    
    # Let's use the 8x8 quadrant in the Top-Right corner (24..31, 24..31) and mirror it to the other 3 corners.
    # This assumes the design is "corner-based".
    # Many photonic crystals are corner-based (holes at corners).
    
    # TL (0..7, 0..7) = TR flipped 180?
    # TR is (24..31, 24..31).
    # 180 rot of TR is TL.
    # So TL = TR[::-1, ::-1].
    # BL (24..31, 0..7) = TR flipped horizontally?
    # TR (i, j) with i>15, j>15.
    # Horiz flip: (31-i, j). 31-i is <16. j>15. This is in TL? No, TL is i<16, j<16.
    # (31-i, j) is in Top-Right? No.
    # i in 24..31 -> 31-i in 0..7.
    # j in 24..31 -> j in 24..31.
    # So (31-i, j) is in Top-Right? No, i' < 16, j' > 15. This is Top-Right? No, Top-Right is i>15, j>15.
    # This is Top-Left? No, j>15.
    # This is the region i<16, j>15. Let's call it Top-Right-Left? No, it's just the top-right part of the left half.
    # Let's just use the 16x16 Top-Right quadrant as the source and search it.
    # To make it fast, I will use a genetic algorithm or hill climbing with a very fast solver.
    
    # Source: 16x16 array.
    # I will pass the 16x16 array.
    pass # Placeholder, will be replaced by the actual implementation below.
    
def _pw_solver_fast(cell, n_pw=17):
    """
    Fast plane-wave solver for TM bands.
    cell: 32x32 array of 0/1.
    Returns: (max_omega4, min_omega5) over a set of k-points.
    """
    size = cell.shape[0]
    eps_low = 1.0
    eps_high = 12.0
    
    # 1/eps map
    inv_eps = np.where(cell > 0.5, 1.0/eps_high, 1.0/eps_low)
    
    # Fourier coefficients of 1/eps
    # G = 2*pi*m, 2*pi*n
    # We need coefficients for G in [-16, 16] x [-16, 16] (for 17x17 PWs, indices -8..8? No, 17 PWs means indices -8..8? 17 is odd. -8..8 is 17 values.)
    # Wait, 289 plane waves = 17x17.
    # Indices: -8, -7, ..., 0, ..., 8.
    
    indices = np.arange(-n_pw//2, n_pw//2 + 1)
    
    # Compute FFT of inv_eps
    # inv_eps is 32x32.
    # FFT gives coefficients for frequencies 0..31.
    # We need frequencies -16..15.
    # fftshift to center.
    
    fft_inv_eps = np.fft.fft2(inv_eps)
    fft_inv_eps = np.fft.fftshift(fft_inv_eps)
    
    # The FFT output at index (kx, ky) corresponds to frequency (kx - 16, ky - 16) if we consider 0..31.
    # With fftshift, index (kx, ky) corresponds to frequency (kx - 16) if kx<16, else (kx-32).
    # Actually, fftshift moves the zero frequency to the center.
    # For N=32, center is 16.
    # Index 16 corresponds to freq 0.
    # Index 15 corresponds to freq -1.
    # Index 17 corresponds to freq 1.
    # So freq m is at index (m + 16) % 32.
    
    # We need coeffs for m in -8..8.
    # Indices in FFT array: -8+16=8 to 8+16=24.
    
    # Extract the 17x17 block of Fourier coefficients
    # Note: FFT coefficients are scaled by 1/N^2 for the inverse transform, but here we want the Fourier series coefficients.
    # f(x) = sum c_G exp(i G x).
    # c_G = (1/N^2) sum f(x) exp(-i G x).
    # np.fft.fft2 computes sum f(x) exp(-i 2pi k x / N).
    # So c_G = fft2(f)[k] / N^2.
    
    coeff_block = fft_inv_eps[8:25, 8:25] / (size * size)
    
    # K-points
    # Irreducible BZ: 0 <= kx <= ky <= pi.
    # Sample points:
    # Gamma (0,0)
    # M (pi, 0)
    # X (pi/2, pi/2)
    # And some interior points.
    
    k_points = [
        (0.0, 0.0),
        (np.pi, 0.0),
        (np.pi/2, np.pi/2),
        (np.pi/4, np.pi/4),
        (3*np.pi/4, np.pi/4),
        (np.pi/2, 0.0),
        (np.pi/4, 0.0),
        (np.pi/2, np.pi/4)
    ]
    
    max_omega4 = 0.0
    min_omega5 = np.inf
    
    # Precompute G vectors
    Gx = 2 * np.pi * indices
    Gy = 2 * np.pi * indices
    Gx_grid, Gy_grid = np.meshgrid(Gx, Gy, indexing='ij')
    
    for kx, ky in k_points:
        # Build A and B
        # A is diagonal: |k - G|^2
        kx_G = kx - Gx_grid
        ky_G = ky - Gy_grid
        A_diag = kx_G**2 + ky_G**2
        
        # B is coeff_block
        B = coeff_block
        
        # Solve A c = omega^2 B c
        # B is Hermitian positive definite.
        # Use eig(A, B)
        
        try:
            vals, _ = eig(A_diag.reshape(-1, 1) * np.eye(n_pw**2, dtype=np.complex128), B)
            # Wait, A is diagonal.
            # A_mat = np.diag(A_diag.flatten())
            # This is slow.
            # Better: Solve B^{-1} A c = omega^2 c.
            # Or use scipy.linalg.eig with A and B.
            
            A_mat = np.diag(A_diag.flatten().astype(np.complex128))
            
            vals, _ = eig(A_mat, B)
            vals_real = np.real(vals)
            vals_pos = vals_real[vals_real > 1e-6]
            vals_pos.sort()
            
            if len(vals_pos) >= 5:
                omega4 = np.sqrt(vals_pos[3])
                omega5 = np.sqrt(vals_pos[4])
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        except Exception:
            continue
            
    return max_omega4, min_omega5

def _score_cell(cell):
    try:
        max_omega4, min_omega5 = _pw_solver_fast(cell, n_pw=17)
        if min_omega5 > max_omega4:
            gap = min_omega5 - max_omega4
            midgap = (min_omega5 + max_omega4) / 2.0
            if midgap > 0:
                return gap / midgap
            else:
                return 0.0
        else:
            return -1.0
    except Exception:
        return -1.0

def design(problem):
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 100.0
    
    # Initial guess: 16x16 Top-Right quadrant
    # Let's start with a rod in the center of the quadrant?
    # Or a hole?
    # Let's start with all 0 (air) and flip pixels.
    
    # Source quadrant: 16x16
    src = np.zeros((16, 16), dtype=np.float64)
    
    # Helper to build full cell from src
    def build_cell(src):
        cell = np.zeros((32, 32), dtype=np.float64)
        # TR: 16..31, 16..31
        cell[16:32, 16:32] = src
        # TL: 0..15, 0..15
        # cell[i,j] = cell[31-i, 31-j] = src[31-i, 31-j]
        # i in 0..15, j in 0..15
        # 31-i in 16..31, 31-j in 16..31
        cell[0:16, 0:16] = src[::-1, ::-1]
        # BR: 16..31, 0..15
        # cell[i,j] = cell[j,i] (transpose)
        # i in 16..31, j in 0..15
        # j in 0..15, i in 16..31 -> (j,i) is in TL
        # cell[i,j] = cell[j,i] = src[31-j, 31-i]
        # j in 0..15, i in 16..31
        # 31-j in 16..31, 31-i in 0..15
        # So cell[16:32, 0:16] = src[:, ::-1]?
        # Let's check indices.
        # Let i=16, j=0. cell[16,0].
        # cell[16,0] = cell[0,16] (transpose).
        # cell[0,16] is in TR? No, 0<16, 16>15. This is TR? No, TR is i>15, j>15.
        # My quadrant definition:
        # TR: i in 16..31, j in 16..31.
        # TL: i in 0..15, j in 0..15.
        # BL: i in 16..31, j in 0..15.
        # BR: i in 0..15, j in 16..31.
        
        # Symmetries:
        # 1. cell[i,j] = cell[31-i, 31-j] (180 rot)
        # 2. cell[i,j] = cell[j,i] (transpose)
        # 3. cell[i,j] = cell[31-i, j] (horiz flip)
        
        # From 1: TL is 180 rot of TR.
        # TL[i,j] = TR[31-i, 31-j].
        # From 2: BR[i,j] = BL[j,i].
        # From 3: BL[i,j] = BR[31-i, j].
        
        # Let's express everything in terms of TR (src).
        # TL[i,j] = src[31-i, 31-j] for i,j in 0..15.
        # BR[i,j] (i in 0..15, j in 16..31).
        # BR[i,j] = BL[j,i] (transpose).
        # BL[j,i] (j in 16..31, i in 0..15).
        # BL[j,i] = BR[31-j, i] (horiz flip).
        # This is circular.
        
        # Let's use 1 and 2.
        # BR[i,j] = cell[i,j].
        # cell[i,j] = cell[j,i] = TL[j,i] = src[31-j, 31-i].
        # So BR[i,j] = src[31-j, 31-i] for i in 0..15, j in 16..31.
        
        # BL[i,j] (i in 16..31, j in 0..15).
        # cell[i,j] = cell[j,i] = BR[j,i].
        # BR[j,i] (j in 0..15, i in 16..31).
        # BR[j,i] = src[31-i, 31-j].
        # So BL[i,j] = src[31-i, 31-j] for i in 16..31, j in 0..15.
        
        # Let's verify.
        # cell[16,0] (BL).
        # Formula: src[31-16, 31-0] = src[15, 31].
        # src is 16x16, indices 0..15. src[15, 31] is out of bounds.
        # Ah, src indices are 0..15.
        # TR indices are 16..31.
        # src[k,l] corresponds to cell[16+k, 16+l].
        # So cell[i,j] = src[i-16, j-16] for i,j in 16..31.
        
        # TL[i,j] = cell[31-i, 31-j].
        # 31-i in 16..31, 31-j in 16..31.
        # TL[i,j] = src[(31-i)-16, (31-j)-16] = src[15-i, 15-j].
        
        # BR[i,j] (i in 0..15, j in 16..31).
        # BR[i,j] = cell[j,i] = TL[j,i] = src[15-j, 15-i].
        
        # BL[i,j] (i in 16..31, j in 0..15).
        # BL[i,j] = cell[j,i] = BR[j,i] = src[15-i, 15-j].
        
        # So:
        cell[0:16, 0:16] = src[::-1, ::-1]
        cell[0:16, 16:32] = src[:, ::-1] # Wait. BR[i,j] = src[15-j, 15-i].
        # i in 0..15, j in 16..31.
        # Let j' = j-16. j' in 0..15.
        # BR[i, j'+16] = src[15-(j'+16), 15-i] = src[-1-j', 15-i]. Out of bounds.
        
        # Let's re-evaluate BR[i,j].
        # i in 0..15, j in 16..31.
        # cell[i,j] = cell[j,i].
        # j in 16..31, i in 0..15. (j,i) is in BL.
        # So cell[i,j] = BL[j,i].
        # BL[j,i] (j in 16..31, i in 0..15).
        # BL[j,i] = cell[i,j] (transpose). Circular.
        
        # Use 180 rot: cell[j,i] = cell[31-j, 31-i].
        # 31-j in 0..15, 31-i in 16..31.
        # (31-j, 31-i) is in BR.
        # So cell[j,i] = BR[31-j, 31-i].
        # And cell[i,j] = BR[31-j, 31-i].
        
        # So BR is determined by BR?
        
        # Let's use Horiz Flip: cell[i,j] = cell[31-i, j].
        # For BR[i,j] (i<16, j>15):
        # cell[i,j] = cell[31-i, j].
        # 31-i in 16..31, j in 16..31.
        # (31-i, j) is in TR.
        # So BR[i,j] = TR[31-i, j] = src[(31-i)-16, j-16] = src[15-i, j-16].
        
        # For BL[i,j] (i>15, j<16):
        # cell[i,j] = cell[31-i, j].
        # 31-i in 0..15, j in 0..15.
        # (31-i, j) is in TL.
        # So BL[i,j] = TL[31-i, j] = src[15-(31-i), 15-j] = src[i-16, 15-j].
        
        # Let's check consistency with Transpose.
        # BR[i,j] = src[15-i, j-16].
        # BL[j,i] = src[j-16, 15-i].
        # cell[i,j] = cell[j,i] => BR[i,j] = BL[j,i].
        # src[15-i, j-16] vs src[j-16, 15-i].
        # These are transposes of each other in the src array.
        # So src must be symmetric? No, the cell must be symmetric.
        # If I define src arbitrarily, the cell will NOT be symmetric under transpose unless src is symmetric.
        # But the task requires the cell to be symmetric under transpose.
        # So I must constrain src to be symmetric?
        # Or I must generate the cell in a way that enforces symmetry.
        
        # If I define src (TR), and generate TL, BR, BL using the symmetries, will the result be symmetric?
        # Yes, if I use the correct mappings.
        # But the mapping for BR and BL depends on src.
        # If I pick a random src, the resulting cell will satisfy 180 rot and Horiz Flip.
        # Does it satisfy Transpose?
        # cell[i,j] = cell[j,i].
        # We derived BR[i,j] = src[15-i, j-16].
        # And BL[j,i] = src[j-16, 15-i].
        # For cell to be symmetric, we need BR[i,j] = BL[j,i].
        # So src[15-i, j-16] = src[j-16, 15-i].
        # This implies src must be symmetric with respect to its diagonal?
        # Let u = 15-i, v = j-16.
        # src[u,v] = src[v,u].
        # Yes, src must be symmetric.
        
        # So I should search over symmetric 16x16 matrices.
        # This reduces the search space to 136 independent pixels (16*17/2).
        
        # For now, I will enforce symmetry on src during the search.
        
        cell[16:32, 16:32] = src
        cell[0:16, 0:16] = src[::-1, ::-1]
        cell[0:16, 16:32] = src[:, ::-1] # This is BR?
        # BR[i,j] (i in 0..15, j in 16..31).
        # j' = j-16.
        # BR[i, j'] = src[15-i, j'].
        # So cell[0:16, 16:32] = src[:, ::-1]?
        # src[:, ::-1] has indices (u, 15-v).
        # We want (15-i, j').
        # If we take src.T, we get (v, u).
        # If we take src[:, ::-1], we get (u, 15-v).
        # We want row 15-i, col j'.
        # So cell[0:16, 16:32] = src[::-1, :]?
        # Row 15-i is row -1-i.
        # Col j' is col j'.
        # So cell[0:16, 16:32] = src[::-1, :].
        
        cell[16:32, 0:16] = src[:, ::-1] # BL?
        # BL[i,j] (i in 16..31, j in 0..15).
        # i' = i-16.
        # BL[i', j] = src[i', 15-j].
        # So cell[16:32, 0:16] = src[:, ::-1].
        
        return cell

    # Initial src: Symmetric 16x16
    src = np.zeros((16, 16), dtype=np.float64)
    # Make a rod in the center of the quadrant?
    # Center of quadrant is (7.5, 7.5).
    for i in range(16):
        for j in range(16):
            if i >= j: # Symmetric
                dx = i - 7.5
                dy = j - 7.5
                if dx*dx + dy*dy < 4.0**2:
                    src[i,j] = 1.0
                    src[j,i] = 1.0
    
    best_cell = build_cell(src)
    best_score = _score_cell(best_cell)
    
    # Hill climbing
    np.random.seed(42)
    step_size = 1.0
    
    while time.time() < deadline:
        # Pick a random pixel in the upper triangle of src
        i = np.random.randint(16)
        j = np.random.randint(i, 16)
        
        new_src = src.copy()
        new_src[i,j] = 1.0 - new_src[i,j]
        new_src[j,i] = new_src[i,j]
        
        new_cell = build_cell(new_src)
        new_score = _score_cell(new_cell)
        
        if new_score > best_score:
            best_score = new_score
            best_cell = new_cell
            src = new_src
        else:
            # Random restart if stuck?
            pass
            
    # Convert to nested list
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(1 if best_cell[i, j] > 0.5 else 0)
        result.append(row)
    
    return result
# EVOLVE-BLOCK-END
