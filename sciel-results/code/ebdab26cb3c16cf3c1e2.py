import time

def design(problem):
        size = problem["pixels"]
        # Parameters for a super-ellipse dielectric rod
        # n=4 (squircle), a=0.27 (half-width)
        n_exp = 4.0
        a = 0.27
        
        cell = [[0 for _ in range(size)] for _ in range(size)]
        
        # Iterate over top-left quadrant
        # Due to symmetry, we can just iterate over all and check, 
        # but constructing via quadrant is faster and ensures symmetry.
        # Center is at (size/2, size/2) = (16, 16)
        # Pixel centers are at (i+0.5, j+0.5)
        
        for i in range(size):
            for j in range(size):
                # Coordinates relative to center
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                
                # Super-ellipse equation: |x/a|^n + |y/a|^n < 1
                # Using absolute values handles all quadrants
                val = (abs(x) / a) ** n_exp + (abs(y) / a) ** n_exp
                if val < 1.0:
                    cell[i][j] = 1
                else:
                    cell[i][j] = 0
                    
        return cell
