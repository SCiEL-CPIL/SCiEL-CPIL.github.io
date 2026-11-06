import numpy as np

def design(problem):
    size = problem["pixels"]
    
    grid = np.zeros((size, size), dtype=int)
    c = size // 2
    
    # Seed: A cross shape
    # Width 2 for robustness
    grid[c-1:c+2, c-4:c+5] = 1
    grid[c-4:c+5, c-1:c+2] = 1
    
    steps = 6
    for _ in range(steps):
        new_grid = np.zeros_like(grid)
        for i in range(size):
            for j in range(size):
                n = 0
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni = (i + di) % size
                        nj = (j + dj) % size
                        n += grid[ni, nj]
                
                if grid[i, j] == 1:
                    if n == 2 or n == 3:
                        new_grid[i, j] = 1
                else:
                    if n == 3:
                        new_grid[i, j] = 1
        grid = new_grid
        
        # Optional: Enforce symmetry during evolution to keep it "on track"
        # But running on full grid with periodic BC preserves D4 if initial state is D4.
        # Is the initial state D4? Yes, the cross is D4.
        # Does periodic Life preserve D4? Yes, if the initial state is D4 and the rule is local and symmetric.
        # So we don't strictly need to enforce it at every step, but doing it at the end is a good safety check.
        
    # Final Symmetry Enforcement (just in case)
    final_grid = np.zeros_like(grid)
    for i in range(size):
        for j in range(size):
            sym_indices = {
                (i, j),
                (i, size-1-j),
                (size-1-i, j),
                (size-1-i, size-1-j),
                (j, i),
                (j, size-1-i)
            }
            count = sum(1 for r, c_idx in sym_indices if grid[r, c_idx] == 1)
            if count * 2 >= len(sym_indices):
                final_grid[i, j] = 1
                
    return final_grid.tolist()
