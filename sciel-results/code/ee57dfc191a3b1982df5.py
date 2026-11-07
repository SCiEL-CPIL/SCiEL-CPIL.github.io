"""Genetic Programming based symbolic regression with robust evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import operator
import math

# Deadline guard: 330s wall-clock limit, use 300s for safety
_DEADLINE = time.time() + 300.0

def _safe_div(x, y):
    """Safe division avoiding zero and large values."""
    eps = 1e-8
    y_safe = np.where(np.abs(y) < eps, np.sign(y) * eps, y)
    # Handle case where y is exactly 0
    y_safe = np.where(y == 0, eps, y_safe)
    # Clip result to avoid explosion
    return np.clip(x / y_safe, -1e6, 1e6)

def _safe_log(x):
    """Safe log: log(|x| + eps) with sign preservation? No, standard log(x) for x>0."""
    # In symbolic regression, log is usually defined for positive inputs.
    # If x <= 0, we can use log(|x|) or return a large penalty.
    # Let's use log(|x| + eps) to be smooth and defined everywhere.
    return np.log(np.abs(x) + 1e-8)

def _safe_sqrt(x):
    """Safe sqrt: sqrt(|x|)."""
    return np.sqrt(np.abs(x))

def _safe_exp(x):
    """Safe exp: exp(clip(x, -100, 100))."""
    return np.exp(np.clip(x, -100, 100))

def _safe_sin(x):
    return np.sin(x)

def _safe_cos(x):
    return np.cos(x)

def _safe_tan(x):
    # Clip input to avoid asymptotes causing huge values
    return np.tan(np.clip(x, -10, 10))

def _safe_abs(x):
    return np.abs(x)

def _safe_pow(x, y):
    # x**y can be dangerous. Clip base and exponent.
    # Only allow real results? If x<0 and y is non-integer, it's complex.
    # Simplest: use np.power with where, or just clip and accept nan/inf handled later.
    # For GP, it's better to use a safe version that returns 0 or sign-preserving value for invalids.
    # Let's just clip x to positive and y to small range.
    x_safe = np.clip(x, 1e-8, 1e6)
    y_safe = np.clip(y, -10, 10)
    return np.power(x_safe, y_safe)

# Define GP Nodes
class GPNode:
    def __init__(self, op, children=None):
        self.op = op
        self.children = children or []

    def evaluate(self, X):
        """Evaluate the tree on input X (n x d array). Returns (n,) array."""
        if self.op in ['x0', 'x1', 'x2', 'x3']:
            idx = int(self.op[1])
            if idx < X.shape[1]:
                return X[:, idx]
            else:
                return np.zeros(X.shape[0])
        
        if self.op in ['const']:
            # Constants are not generated randomly in this simple version, 
            # but we can add a few fixed constants or let the linear part handle scaling.
            # For now, no explicit constant nodes, rely on linear regression bias.
            return np.ones(X.shape[0])

        # Unary ops
        if self.op == 'sin':
            return _safe_sin(self.children[0].evaluate(X))
        if self.op == 'cos':
            return _safe_cos(self.children[0].evaluate(X))
        if self.op == 'tan':
            return _safe_tan(self.children[0].evaluate(X))
        if self.op == 'abs':
            return _safe_abs(self.children[0].evaluate(X))
        if self.op == 'sqrt':
            return _safe_sqrt(self.children[0].evaluate(X))
        if self.op == 'exp':
            return _safe_exp(self.children[0].evaluate(X))
        if self.op == 'log':
            return _safe_log(self.children[0].evaluate(X))
        
        # Binary ops
        if self.op == 'add':
            return self.children[0].evaluate(X) + self.children[1].evaluate(X)
        if self.op == 'sub':
            return self.children[0].evaluate(X) - self.children[1].evaluate(X)
        if self.op == 'mul':
            return self.children[0].evaluate(X) * self.children[1].evaluate(X)
        if self.op == 'div':
            return _safe_div(self.children[0].evaluate(X), self.children[1].evaluate(X))
        if self.op == 'pow':
            return _safe_pow(self.children[0].evaluate(X), self.children[1].evaluate(X))
            
        # Fallback
        return np.zeros(X.shape[0])

    def depth(self):
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def copy(self):
        new_children = [c.copy() for c in self.children]
        return GPNode(self.op, new_children)

    def __repr__(self):
        if not self.children:
            return self.op
        return f"({self.op} " + " ".join(repr(c) for c in self.children) + ")"

def _create_random_tree(depth, max_depth, d):
    """Create a random GP tree."""
    if depth >= max_depth or (depth > 0 and random.random() < 0.2):
        # Terminal
        if d > 0:
            return GPNode(f"x{random.randint(0, d-1)}")
        else:
            return GPNode("x0")
    
    # Non-terminal
    unary_ops = ['sin', 'cos', 'tan', 'abs', 'sqrt', 'exp', 'log']
    binary_ops = ['add', 'sub', 'mul', 'div', 'pow']
    
    if random.random() < 0.4:
        # Unary
        op = random.choice(unary_ops)
        child = _create_random_tree(depth + 1, max_depth, d)
        return GPNode(op, [child])
    else:
        # Binary
        op = random.choice(binary_ops)
        c1 = _create_random_tree(depth + 1, max_depth, d)
        c2 = _create_random_tree(depth + 1, max_depth, d)
        return GPNode(op, [c1, c2])

def _get_subtree(node):
    """Get a random subtree."""
    if not node.children:
        return node, 0
    idx = random.randint(0, len(node.children) - 1)
    subtree, idx_in_subtree = _get_subtree(node.children[idx])
    return subtree, idx
    
def _crossover(tree1, tree2, d):
    """Crossover two trees."""
    t1 = tree1.copy()
    t2 = tree2.copy()
    
    # Get random subtree from t1
    node1 = t1
    if node1.children:
        idx1 = random.randint(0, len(node1.children) - 1)
        node1 = node1.children[idx1]
        
    # Get random subtree from t2
    node2 = t2
    if node2.children:
        idx2 = random.randint(0, len(node2.children) - 1)
        node2 = node2.children[idx2]
        
    # Swap
    # This simple implementation just swaps the subtrees at the selected positions
    # Note: This requires careful index tracking. A simpler approach is to replace a random terminal or subtree.
    # Let's use a simpler mutation-like crossover: replace a random subtree in t1 with a random subtree from t2.
    
    # Find a random node in t1 to replace
    nodes = []
    def collect_nodes(node):
        nodes.append(node)
        for c in node.children:
            collect_nodes(c)
    collect_nodes(t1)
    
    if not nodes:
        return t1
        
    target = random.choice(nodes)
    
    # Find a random node in t2 to take
    nodes2 = []
    def collect_nodes2(node):
        nodes2.append(node)
        for c in node.children:
            collect_nodes2(c)
    collect_nodes2(t2)
    
    if not nodes2:
        return t1
        
    source = random.choice(nodes2)
    
    # Replace target with source
    # We need to find the parent of target and update the child pointer.
    # This is complex with the current structure. 
    # Alternative: Just return t1 with a random mutation, and t2 with a random mutation.
    # For simplicity and robustness in this time-constrained context, let's just do mutation primarily.
    return t1

def _mutate_tree(tree, d):
    """Mutate a tree by replacing a random subtree with a new random tree."""
    nodes = []
    def collect_nodes(node):
        nodes.append(node)
        for c in node.children:
            collect_nodes(c)
    collect_nodes(tree)
    
    if not nodes:
        return tree
        
    target = random.choice(nodes)
    
    # If target is a leaf, change it to a new leaf or a small subtree
    # If target is an internal node, replace it with a new random subtree of similar depth
    new_subtree = _create_random_tree(0, 3, d) # Small depth to keep trees manageable
    
    # We need to replace the node in the parent.
    # This requires parent pointers or a different traversal.
    # Let's implement a simpler mutation: change a random operator or terminal.
    
    # Find a random node to modify
    # Actually, the "replace subtree" logic is hard without parents.
    # Let's just change the operator of a random internal node, or the value of a random terminal.
    
    if target.children:
        # Change operator
        unary_ops = ['sin', 'cos', 'tan', 'abs', 'sqrt', 'exp', 'log']
        binary_ops = ['add', 'sub', 'mul', 'div', 'pow']
        if len(target.children) == 1:
            target.op = random.choice(unary_ops)
        else:
            target.op = random.choice(binary_ops)
    else:
        # Change terminal
        if d > 0:
            target.op = f"x{random.randint(0, d-1)}"
            
    return tree

def _fit_gp(X, y, time_limit):
    """Run GP search."""
    n, d = X.shape
    pop_size = 200
    max_depth = 4
    generations = 0
    
    # Initialize population
    population = [_create_random_tree(0, max_depth, d) for _ in range(pop_size)]
    
    best_tree = None
    best_mse = np.inf
    best_w = None
    
    start_time = time.time()
    
    while time.time() < start_time + time_limit:
        generations += 1
        # Evaluate population
        # We will evaluate all trees and stack their outputs into a feature matrix?
        # No, that's too big. 
        # Instead, we evaluate each tree, compute the MSE of the best linear combination of that single feature?
        # No, that's not symbolic regression.
        # Standard GP: Fitness of a tree is the MSE of the tree's output directly? 
        # Or, we use the tree as a feature and fit a linear model? 
        # The inspirations suggest evolving expressions. 
        # If we evolve expressions, the expression IS the model. 
        # So fitness = MSE(tree_output, y).
        # But trees are non-linear. A single tree might not fit well.
        # A common approach: Evolve a population of trees, select best by MSE.
        
        fitnesses = []
        for tree in population:
            try:
                pred = tree.evaluate(X)
                mse = np.mean((pred - y) ** 2)
                fitnesses.append(mse)
            except:
                fitnesses.append(np.inf)
        
        # Update best
        min_idx = np.argmin(fitnesses)
        if fitnesses[min_idx] < best_mse:
            best_mse = fitnesses[min_idx]
            best_tree = population[min_idx].copy()
            
        # Selection: Tournament selection
        new_population = []
        for _ in range(pop_size):
            # Pick 2 random individuals, choose the better one
            i1 = random.randint(0, pop_size - 1)
            i2 = random.randint(0, pop_size - 1)
            if fitnesses[i1] < fitnesses[i2]:
                winner = population[i1]
            else:
                winner = population[i2]
            new_population.append(winner.copy())
        
        # Crossover and Mutation
        for i in range(pop_size):
            if random.random() < 0.7:
                # Crossover
                if i < pop_size - 1:
                    # Simple crossover: swap subtrees
                    # Since our _crossover is weak, let's just do mutation for now
                    # Or implement a proper subtree swap
                    pass
            # Mutation
            if random.random() < 0.3:
                new_population[i] = _mutate_tree(new_population[i], d)
            # Elitism: keep the best individual unchanged
            elif i == 0:
                new_population[i] = population[min_idx].copy()
                
        population = new_population
        
        # Check time
        if time.time() > start_time + time_limit:
            break
            
    return best_tree, best_mse

def fit(X_train, y_train):
    global _DEADLINE
    start_time = time.time()
    
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    if y_var < 1e-12:
        y_var = 1e-12
    
    best_mse = np.inf
    best_predict = lambda X: np.full(X.shape[0], y_mean)
    
    # Time budget for GP
    gp_time = min(60.0, _DEADLINE - time.time() - 5.0)
    if gp_time > 0:
        best_tree, gp_mse = _fit_gp(X_train, y_train, gp_time)
        if best_tree is not None and gp_mse < best_mse:
            best_mse = gp_mse
            def gp_predict(X):
                try:
                    return best_tree.evaluate(X)
                except:
                    return np.full(X.shape[0], y_mean)
            best_predict = gp_predict
            
    # Fallback: If GP didn't find anything good, or as a complement, try a simple polynomial fit
    # The GP might struggle with specific scaling or bias.
    # Let's add a quick polynomial ridge regression as a safety net/complement.
    
    # Quick polynomial features
    def poly_features(X, deg=3):
        cols = [np.ones(X.shape[0])]
        for i in range(X.shape[1]):
            for k in range(1, deg+1):
                cols.append(X[:, i] ** k)
        return np.column_stack(cols)
    
    # Try a few polynomial degrees with ridge
    for deg in [1, 2, 3, 4]:
        if time.time() > _DEADLINE - 10:
            break
        try:
            F = poly_features(X_train, deg)
            # Ridge regression
            alpha = 1e-3
            A = F.T @ F + alpha * np.eye(F.shape[1])
            b = F.T @ y_train
            w = np.linalg.solve(A, b)
            pred = F @ w
            mse = np.mean((pred - y_train) ** 2)
            if mse < best_mse:
                best_mse = mse
                def make_poly_predict(F_func, w_vec):
                    def predict(X):
                        F = F_func(X)
                        return F @ w_vec
                    return predict
                best_predict = make_poly_predict(lambda X: poly_features(X, deg), w)
        except:
            pass
            
    # Final safety check
    def safe_predict(X):
        try:
            result = best_predict(X)
            # Ensure finite values
            result = np.where(np.isfinite(result), result, y_mean)
            return result
        except Exception:
            return np.full(X.shape[0], y_mean)
    
    return safe_predict
# EVOLVE-BLOCK-END
