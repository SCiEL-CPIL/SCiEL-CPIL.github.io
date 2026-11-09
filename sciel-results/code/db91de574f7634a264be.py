import numpy as np
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    best_val = 0
    best_centers = None
    best_sides = None
    
    # Define a Node class for the guillotine tree
    class Node:
        def __init__(self, is_leaf, left=None, right=None, dir=None):
            self.is_leaf = is_leaf
            self.left = left
            self.right = right
            self.dir = dir # 0: Horizontal (top/bottom), 1: Vertical (left/right)
            self.param_idx = -1

    def generate_topology(k, mode='random'):
        """
        Generates a random guillotine tree topology with k leaves.
        mode: 'random', 'balanced', 'comb'
        """
        if k == 1:
            return Node(True)
        
        if mode == 'balanced':
            i = k // 2
        elif mode == 'comb':
            # Always split 1 leaf vs k-1 leaves
            if np.random.rand() < 0.5:
                i = 1
            else:
                i = k - 1
        else: # random
            i = np.random.randint(1, k)
            
        dir = np.random.randint(0, 2)
        
        left = generate_topology(i, mode)
        right = generate_topology(k - i, mode)
        
        return Node(False, left, right, dir)

    def assign_params(root):
        """Assigns parameter indices to internal nodes."""
        count = 0
        def rec(node):
            nonlocal count
            if not node.is_leaf:
                node.param_idx = count
                count += 1
                rec(node.left)
                rec(node.right)
        rec(root)
        return count

    def compute_solution(root, x):
        """
        Given the tree and split ratios x, computes centers and sides.
        Returns (centers, sides, valid_flag)
        """
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        leaf_idx = 0
        total_sum = 0.0
        
        def rec(node, w, h, cx, cy):
            nonlocal leaf_idx, total_sum
            if node.is_leaf:
                side = min(w, h)
                if side <= 1e-9:
                    return
                centers[leaf_idx] = [cx, cy]
                sides[leaf_idx] = side
                leaf_idx += 1
                total_sum += side
                return
            
            split = x[node.param_idx]
            # Clip split to valid range to avoid zero/negative dimensions
            split = np.clip(split, 0.01, 0.99)
            
            if node.dir == 0: # Horizontal split (top/bottom)
                h_top = h * split
                h_bot = h * (1 - split)
                # Top child
                cy_top = cy - h/2 + h_top/2
                rec(node.left, w, h_top, cx, cy_top)
                # Bottom child
                cy_bot = cy - h/2 + h_top + h_bot/2
                rec(node.right, w, h_bot, cx, cy_bot)
            else: # Vertical split (left/right)
                w_left = w * split
                w_right = w * (1 - split)
                # Left child
                cx_left = cx - w/2 + w_left/2
                rec(node.left, w_left, h, cx_left, cy)
                # Right child
                cx_right = cx - w/2 + w_left + w_right/2
                rec(node.right, w_right, h, cx_right, cy)

        rec(root, 1.0, 1.0, 0.5, 0.5)
        
        if leaf_idx != n:
            return None, None, False
            
        # Validation
        tol = 1e-7
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol or cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return None, None, False
            if s <= 0:
                return None, None, False
        
        for i in range(n):
            for j in range(i+1, n):
                xi, yi = centers[i]
                xj, yj = centers[j]
                si = sides[i]
                sj = sides[j]
                t = (si + sj) / 2.0 - tol
                if abs(xi - xj) < t and abs(yi - yj) < t:
                    return None, None, False
                    
        return centers, sides, True

    def objective(x, root):
        """
        Objective function for optimization.
        Returns negative sum of sides.
        """
        centers, sides, valid = compute_solution(root, x)
        if not valid:
            return 1e10 # Large penalty
        return -np.sum(sides)

    # List of topologies to try
    topologies = []
    np.random.seed(42)
    
    # Add some structured topologies
    topologies.append(generate_topology(n, 'balanced'))
    topologies.append(generate_topology(n, 'comb'))
    
    # Add random topologies
    for _ in range(30):
        topologies.append(generate_topology(n, 'random'))
        
    # Optimize each topology
    for root in topologies:
        n_params = assign_params(root)
        if n_params == 0:
            continue
            
        # Multiple restarts for each topology
        for restart in range(5):
            # Random initial guess
            x0 = np.random.uniform(0.3, 0.7, n_params)
            
            res = minimize(
                objective, 
                x0, 
                args=(root,),
                method='L-BFGS-B', 
                bounds=[(0.05, 0.95)] * n_params,
                options={'maxiter': 200, 'ftol': 1e-9}
            )
            
            if res.success or res.fun < 1e9:
                centers, sides, valid = compute_solution(root, res.x)
                if valid:
                    val = np.sum(sides)
                    if val > best_val:
                        best_val = val
                        best_centers = centers.copy()
                        best_sides = sides.copy()

    # Fallback if no valid solution found (should not happen)
    if best_centers is None:
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        # Simple 2x5 grid
        for i in range(n):
            col = i % 5
            row = i // 5
            centers[i] = [(col + 0.5) / 5.0, (row + 0.5) / 2.0]
            sides[i] = 1.0 / 5.0
        return centers, sides

    return best_centers, best_sides
