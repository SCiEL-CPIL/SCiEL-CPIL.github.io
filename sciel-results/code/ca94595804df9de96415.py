"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell for max TM band gap between bands 4 and 5."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin: 120s budget, leave 5s for overhead
    
    size = problem["pixels"]
    if size != 32:
        size = 32
        
    def enforce_d4_symmetry(cell):
        """
        Enforces full D4 symmetry (rotations and reflections) by averaging.
        cell is a 2D list or numpy array.
        """
        c = np.array(cell, dtype=float)
        n = c.shape[0]
        
        # Identity
        sym_ops = [c]
        
        # Rotation 90
        sym_ops.append(np.rot90(c, 1))
        # Rotation 180
        sym_ops.append(np.rot90(c, 2))
        # Rotation 270
        sym_ops.append(np.rot90(c, 3))
        
        # Reflections (transpose, anti-transpose, etc)
        # D4 consists of 4 rotations and 4 reflections.
        # We can generate all 8 by reflecting the rotations.
        # Reflection across y-axis: c[:, ::-1]
        sym_ops.append(c[:, ::-1])
        sym_ops.append(np.rot90(c[:, ::-1], 1))
        sym_ops.append(np.rot90(c[:, ::-1], 2))
        sym_ops.append(np.rot90(c[:, ::-1], 3))
        
        # Average all 8
        avg = np.mean(sym_ops, axis=0)
        # Threshold at 0.5
        return (avg > 0.5).astype(int).tolist()

    def generate_diamond(center_r, inner_r=0.0, outer_r=0.5):
        """Generates a pattern based on Manhattan distance or Euclidean distance rotated."""
        # Let's use a rotated square (diamond) defined by |x| + |y| < r
        # In pixel coordinates centered at 0
        cell = [[0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                # Map to [-0.5, 0.5)
                x = (j + 0.5) / size - 0.5
                y = (i + 0.5) / size - 0.5
                # Diamond: |x| + |y| < r
                if abs(x) + abs(y) < center_r:
                    cell[i][j] = 1
        return cell

    def generate_cross(arm_width):
        cell = [[0]*size for _ in range(size)]
        c = size // 2
        half_w = arm_width // 2
        for i in range(size):
            for j in range(size):
                if abs(i - c + 0.5) <= half_w + 0.5 or abs(j - c + 0.5) <= half_w + 0.5:
                    cell[i][j] = 1
        return cell

    def generate_checkerboard_2x2():
        cell = [[0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if (i // 2) % 2 == (j // 2) % 2:
                    cell[i][j] = 1
        return cell

    candidates = []
    
    # Candidate 1: Small central diamond
    candidates.append(generate_diamond(0.25))
    
    # Candidate 2: Larger central diamond
    candidates.append(generate_diamond(0.35))
    
    # Candidate 3: Cross
    candidates.append(generate_cross(4))
    
    # Candidate 4: Cross with thicker arms
    candidates.append(generate_cross(6))
    
    # Candidate 5: 2x2 Checkerboard (high symmetry, often good for gaps)
    candidates.append(generate_checkerboard_2x2())
    
    # Process candidates: enforce symmetry and check time
    valid_candidates = []
    for c in candidates:
        if time.time() > deadline:
            break
        sym_c = enforce_d4_symmetry(c)
        # Verify symmetry one last time just in case
        # (The enforce function should be perfect, but double check cost is low for 32x32)
        arr = np.array(sym_c)
        if np.array_equal(arr, arr.T) and np.array_equal(arr, np.rot90(arr, 2)):
             valid_candidates.append(sym_c)
             
    if not valid_candidates:
        # Fallback: simple central block
        cell = [[0]*size for _ in range(size)]
        c = size // 2
        for i in range(c-5, c+5):
            for j in range(c-5, c+5):
                cell[i][j] = 1
        return enforce_d4_symmetry(cell)

    # Heuristic: Pick the one with fill factor closest to 0.5
    def fill_factor(cell):
        return sum(sum(row) for row in cell) / (size*size)
        
    scored = [(abs(fill_factor(c) - 0.5), c) for c in valid_candidates]
    scored.sort()
    return scored[0][1]

# EVOLVE-BLOCK-END
