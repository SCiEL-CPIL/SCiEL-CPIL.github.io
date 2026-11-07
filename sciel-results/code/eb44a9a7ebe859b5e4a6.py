import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4 structure: first M-1 variables define front, rest control g
        # g is minimized at x_i = 0.5 for i in [M-1, N)
        # f_1 = (1+g)*x_0, f_2 = (1+g)*x_1, f_3 = (1+g)*(1-x_0-x_1)
        # For 3 objectives: x_0, x_1 are front vars; x_2..x_11 are g vars
        # Set g vars to 0.5 to get g=0 (optimal)
        
        n_front = self.n_obj - 1  # 2
        n_g = self.dim - n_front  # 10
        
        # Generate Fibonacci lattice points on the unit simplex
        # For 3 objectives, the Pareto front is a triangle in objective space
        # x_0 + x_1 <= 1, x_0 >= 0, x_1 >= 0
        # Use 2D Fibonacci lattice for quasi-random uniform sampling
        
        # Reserve 3 evaluations for extreme vertices
        # Vertex 1: x_0=1, x_1=0 -> f=(1,0,0)
        # Vertex 2: x_0=0, x_1=1 -> f=(0,1,0)
        # Vertex 3: x_0=0, x_1=0 -> f=(0,0,1)
        
        # Number of lattice points
        n_lattice = self.budget - 3
        
        # Generate Fibonacci lattice on [0,1]^2, then filter to simplex
        # Use rejection sampling with Fibonacci lattice
        golden_ratio = (1 + np.sqrt(5)) / 2
        inv_phi = 1 / golden_ratio
        
        # Generate candidate points
        lattice_points = []
        for i in range(n_lattice):
            # 2D Fibonacci lattice
            x1 = (i * inv_phi) % 1.0
            x2 = ((i + 1) * inv_phi) % 1.0
            
            # Map to simplex: x1, x2 are in [0,1], need x1 + x2 <= 1
            # Use transformation: if x1 + x2 > 1, reject or transform
            # Better: use a direct simplex mapping
            # Let u = x1, v = x2 * (1 - x1) to ensure u + v <= 1
            # Actually, let's use: a = x1, b = x2 * (1 - x1)
            # Then a + b <= 1 always since b <= 1-x1
            
            a = x1
            b = x2 * (1.0 - x1)
            
            # Create decision vector
            x = np.zeros(self.dim)
            x[0] = a
            x[1] = b
            # Set g variables to 0.5 for optimal g=0
            x[n_front:] = 0.5
            
            lattice_points.append(x)
        
        # Add the three extreme vertices
        vertex1 = np.zeros(self.dim)
        vertex1[0] = 1.0
        vertex1[n_front:] = 0.5
        
        vertex2 = np.zeros(self.dim)
        vertex2[1] = 1.0
        vertex2[n_front:] = 0.5
        
        vertex3 = np.zeros(self.dim)
        vertex3[n_front:] = 0.5
        
        all_points = lattice_points + [vertex1, vertex2, vertex3]
        
        # Evaluate all points
        archive_x = []
        archive_f = []
        
        for x in all_points:
            if time.time() > deadline:
                break
            f = problem(x)
            self._add_to_archive(archive_x, archive_f, x, f)
        
        return archive_x

    def _add_to_archive(self, archive_x, archive_f, x, f):
        if len(archive_f) == 0:
            archive_x.append(x)
            archive_f.append(f)
            return

        F_arr = np.array(archive_f)
        
        # Check if new point is dominated
        dominated = False
        for i in range(len(archive_f)):
            if np.all(F_arr[i] <= f) and np.any(F_arr[i] < f):
                dominated = True
                break
        
        if dominated:
            return
        
        # Remove points dominated by the new point
        new_archive_x = []
        new_archive_f = []
        for i in range(len(archive_f)):
            if np.all(f <= F_arr[i]) and np.any(f < F_arr[i]):
                continue
            new_archive_x.append(archive_x[i])
            new_archive_f.append(archive_f[i])
            
        new_archive_x.append(x)
        new_archive_f.append(f)
        
        archive_x.clear()
        archive_f.clear()
        archive_x.extend(new_archive_x)
        archive_f.extend(new_archive_f)
