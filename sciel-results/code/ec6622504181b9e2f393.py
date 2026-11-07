import numpy as np
import time

class Optimizer:
        def __init__(self, budget, dim):
            self.budget = budget
            self.dim = dim

        def __call__(self, func):
            start_time = time.time()
            deadline = start_time + 280.0 # Safety margin
            
            # Initial Grid: 2^dim cells
            # Bounds: [-5, 5]
            # Cell size: 10 / 2^dim ? No, 2^dim cells means dividing each dim into 2.
            # So 2^5 = 32 cells.
            
            active_cells = []
            # Generate 2^dim initial cells
            for mask in range(2 ** self.dim):
                center = np.zeros(self.dim)
                half_width = 2.5 # 10/2 / 2 = 2.5? 
                # Dim i: if bit i is 0, center is -2.5, if 1, center is 2.5.
                # Wait, range is [-5, 5]. Midpoint is 0.
                # Split into [-5, 0] and [0, 5].
                # Center of [-5, 0] is -2.5.
                # Center of [0, 5] is 2.5.
                for i in range(self.dim):
                    if (mask >> i) & 1:
                        center[i] = 2.5
                    else:
                        center[i] = -2.5
                # Store cell: (center, half_width_vector)
                # Half width is 2.5 for all dims.
                active_cells.append((center, 2.5))
            
            best_x = None
            best_f = float('inf')
            calls = 0
            
            # Evaluate initial grid
            for center, hw in active_cells:
                if time.time() > deadline:
                    break
                f_val = func(center)
                calls += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = center.copy()
            
            # Adaptive Refinement
            # We need to select the best cell from active_cells.
            # We need to store the function value for each cell center.
            # Let's restructure active_cells to store (center, half_width, f_val)
            
            # Re-initialize with values
            active_cells_with_val = []
            # We already evaluated them, but we didn't store them in a list with values properly for selection.
            # Let's just re-evaluate or store during first pass.
            # Better: Store during first pass.
            
            # Let's rewrite the loop to store values.
            active_cells = []
            for mask in range(2 ** self.dim):
                center = np.zeros(self.dim)
                for i in range(self.dim):
                    if (mask >> i) & 1:
                        center[i] = 2.5
                    else:
                        center[i] = -2.5
                f_val = func(center)
                calls += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = center.copy()
                active_cells.append((center, 2.5, f_val))
                
            # If we ran out of budget in initial grid, return
            if calls >= self.budget:
                return best_x, best_f
                
            # Main Loop
            while calls < self.budget:
                if time.time() > deadline:
                    break
                
                # Find best cell
                best_idx = 0
                best_cell_f = active_cells[0][2]
                for i in range(1, len(active_cells)):
                    if active_cells[i][2] < best_cell_f:
                        best_cell_f = active_cells[i][2]
                        best_idx = i
                        
                best_cell = active_cells[best_idx]
                center = best_cell[0]
                hw = best_cell[1]
                
                # Remove best cell from active list
                active_cells.pop(best_idx)
                
                # Subdivide into 2^dim children
                # New half width is hw / 2
                new_hw = hw / 2.0
                
                for mask in range(2 ** self.dim):
                    if calls >= self.budget or time.time() > deadline:
                        break
                    
                    child_center = center.copy()
                    for i in range(self.dim):
                        if (mask >> i) & 1:
                            child_center[i] = center[i] + new_hw
                        else:
                            child_center[i] = center[i] - new_hw
                    
                    # Clip to bounds? The grid logic keeps us within bounds if we start within.
                    # Initial bounds [-5, 5]. Centers +/-2.5, hw 2.5.
                    # Children centers +/- (2.5 +/- 1.25). All within [-5, 5].
                    
                    f_val = func(child_center)
                    calls += 1
                    
                    if f_val < best_f:
                        best_f = f_val
                        best_x = child_center.copy()
                        
                    active_cells.append((child_center, new_hw, f_val))
                
            return best_x, best_f
