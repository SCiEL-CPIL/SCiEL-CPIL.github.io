"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import operator
import math


def fit(X_train, y_train):
    """Symbolic regression via genetic programming for 1D data,
    with polynomial fallback for multi-dimensional inputs."""
    t0 = time.time()
    deadline = 325.0  # safety margin before 330s kill

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    if d != 1:
        # Multi-dimensional fallback: polynomial regression
        def features(X):
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            return np.column_stack(cols)

        w, *_ = np.linalg.lstsq(features(X), y, rcond=None)

        def predict_poly(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, d)
            return features(X_test) @ w

        return predict_poly

    x = X[:, 0].copy()
    y = y.copy()
    
    # Safe evaluation functions with clipping to avoid overflow
    def safe_func(x, func_name):
        x = np.asarray(x, dtype=np.float64)
        if func_name == 'sin':
            return np.sin(x)
        elif func_name == 'cos':
            return np.cos(x)
        elif func_name == 'tan':
            return np.tan(x)
        elif func_name == 'exp':
            return np.exp(np.clip(x, -50, 50))
        elif func_name == 'log':
            return np.log(np.abs(x) + 1e-10)
        elif func_name == 'sqrt':
            return np.sqrt(np.abs(x))
        elif func_name == 'abs':
            return np.abs(x)
        elif func_name == 'x':
            return x
        elif func_name == 'x2':
            return x * x
        elif func_name == 'x3':
            return x * x * x
        elif func_name == 'x4':
            return x * x * x * x
        elif func_name == 'x5':
            return x * x * x * x * x
        elif func_name == 'x6':
            return x * x * x * x * x * x
        elif func_name == 'inv':
            return 1.0 / (x + 1e-10)
        elif func_name == 'inv2':
            return 1.0 / (x * x + 1e-10)
        elif func_name == 'sinh':
            return np.sinh(np.clip(x, -30, 30))
        elif func_name == 'cosh':
            return np.cosh(np.clip(x, -30, 30))
        else:
            return x

    def safe_op(a, b, op_name):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if op_name == '+':
            return a + b
        elif op_name == '-':
            return a - b
        elif op_name == '*':
            return a * b
        elif op_name == '/':
            return a / (b + 1e-10)
        elif op_name == '**':
            return np.power(np.clip(a, -10, 10), np.clip(b, -10, 10))
        elif op_name == 'min':
            return np.minimum(a, b)
        elif op_name == 'max':
            return np.maximum(a, b)
        else:
            return a

    # Genetic programming setup
    terminal_funcs = ['x', 'x2', 'x3', 'x4', 'x5', 'x6', 'sin', 'cos', 'tan', 
                      'exp', 'log', 'sqrt', 'abs', 'inv', 'inv2', 'sinh', 'cosh']
    binary_ops = ['+', '-', '*', '/', '**', 'min', 'max']
    
    # Constants to include
    constants = [0.0, 0.5, 1.0, 2.0, 3.0, math.pi, math.e, -1.0, -0.5, -2.0]
    
    class GPNode:
        def __init__(self, name, children=None):
            self.name = name
            self.children = children or []
        
        def evaluate(self, x):
            if self.name in terminal_funcs:
                return safe_func(x, self.name)
            elif self.name in constants or (isinstance(self.name, float)):
                return np.full_like(x, self.name, dtype=np.float64)
            elif self.name in binary_ops:
                a = self.children[0].evaluate(x)
                b = self.children[1].evaluate(x)
                return safe_op(a, b, self.name)
            else:
                return x
        
        def complexity(self):
            if not self.children:
                return 1
            return 1 + sum(c.complexity() for c in self.children)
    
    def random_tree(max_depth=3, current_depth=0):
        if current_depth >= max_depth or random.random() < 0.3:
            if random.random() < 0.3:
                return GPNode(random.choice(constants))
            else:
                return GPNode(random.choice(terminal_funcs))
        else:
            op = random.choice(binary_ops)
            left = random_tree(max_depth, current_depth + 1)
            right = random_tree(max_depth, current_depth + 1)
            return GPNode(op, [left, right])
    
    def evaluate_tree(tree, x):
        try:
            result = tree.evaluate(x)
            if not np.all(np.isfinite(result)):
                return None, 1e10
            return result, np.mean((result - y)**2)
        except:
            return None, 1e10
    
    def crossover(tree1, tree2):
        # Random subtree crossover
        def get_random_node(tree):
            nodes = [tree]
            stack = [tree]
            while stack:
                node = stack.pop()
                for child in node.children:
                    nodes.append(child)
                    stack.append(child)
            return random.choice(nodes)
        
        node1 = get_random_node(tree1)
        node2 = get_random_node(tree2)
        
        # Swap subtrees
        parent1 = None
        parent2 = None
        idx1 = None
        idx2 = None
        
        # Find parent of node1 in tree1
        def find_parent(tree, target):
            if tree.name == target.name and tree.children == target.children:
                return None, None
            for i, child in enumerate(tree.children):
                if child.name == target.name and child.children == target.children:
                    return tree, i
                result = find_parent(child, target)
                if result[0] is not None:
                    return result
            return None, None
        
        parent1, idx1 = find_parent(tree1, node1)
        parent2, idx2 = find_parent(tree2, node2)
        
        if parent1 is None or parent2 is None:
            # Simple: return copies
            return tree1, tree2
        
        # Swap
        parent1.children[idx1] = node2
        parent2.children[idx2] = node1
        
        return tree1, tree2
    
    def mutate(tree, mutation_rate=0.2):
        def mutate_node(node):
            if random.random() < mutation_rate:
                if not node.children:
                    # Terminal mutation
                    if random.random() < 0.3:
                        node.name = random.choice(constants)
                    else:
                        node.name = random.choice(terminal_funcs)
                else:
                    # Operator mutation
                    node.name = random.choice(binary_ops)
            for child in node.children:
                mutate_node(child)
        
        mutate_node(tree)
        return tree
    
    # Initialize population
    pop_size = 200
    population = [random_tree(max_depth=4) for _ in range(pop_size)]
    
    # Evaluate initial population
    best_tree = None
    best_mse = 1e10
    
    for tree in population:
        _, mse = evaluate_tree(tree, x)
        if mse < best_mse:
            best_mse = mse
            best_tree = tree
    
    # Genetic algorithm loop
    n_generations = 50
    elite_size = 10
    
    for gen in range(n_generations):
        if time.time() > deadline:
            break
        
        # Sort by fitness
        scored_pop = []
        for tree in population:
            _, mse = evaluate_tree(tree, x)
            scored_pop.append((mse, tree))
        scored_pop.sort(key=lambda x: x[0])
        
        # Update best
        if scored_pop[0][0] < best_mse:
            best_mse = scored_pop[0][0]
            best_tree = scored_pop[0][1]
        
        # Create next generation
        new_population = []
        
        # Elitism
        for i in range(elite_size):
            new_population.append(scored_pop[i][1])
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
            
            # Tournament selection
            def tournament_select():
                candidates = random.sample(scored_pop, min(3, len(scored_pop)))
                return min(candidates, key=lambda x: x[0])[1]
            
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            if random.random() < 0.7:
                child1, child2 = crossover(parent1, parent2)
                new_population.append(mutate(child1, 0.1))
                if len(new_population) < pop_size:
                    new_population.append(mutate(child2, 0.1))
            else:
                new_population.append(mutate(parent1, 0.3))
        
        population = new_population
    
    # Create prediction function
    def predict_gp(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        xt = X_test[:, 0]
        
        if best_tree is None:
            # Fallback to linear interpolation
            from scipy.interpolate import interp1d
            order = np.argsort(x)
            f_interp = interp1d(x[order], y[order], kind='linear', fill_value='extrapolate')
            return f_interp(xt)
        
        try:
            preds = best_tree.evaluate(xt)
            if not np.all(np.isfinite(preds)):
                preds = np.where(np.isfinite(preds), preds, 0.0)
            return preds
        except:
            from scipy.interpolate import interp1d
            order = np.argsort(x)
            f_interp = interp1d(x[order], y[order], kind='linear', fill_value='extrapolate')
            return f_interp(xt)
    
    return predict_gp

# EVOLVE-BLOCK-END
