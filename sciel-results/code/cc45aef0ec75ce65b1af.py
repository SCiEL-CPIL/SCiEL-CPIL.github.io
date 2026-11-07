"""Genetic Programming Symbolic Regression for 1-D Noiseless Equations."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math
from collections import deque

# Global deadline guard: 330s wall-clock limit, use 310s for safety
_DEADLINE = time.time() + 310.0

# Safe math functions to handle domain errors gracefully
def _safe_div(x, y):
    return np.where(np.abs(y) < 1e-12, 0.0, x / np.where(np.abs(y) < 1e-12, 1.0, y))

def _safe_log(x):
    return np.log(np.maximum(x, 1e-12))

def _safe_sqrt(x):
    return np.sqrt(np.maximum(x, 0.0))

def _safe_exp(x):
    # Clip to avoid overflow
    return np.exp(np.clip(x, -50, 50))

def _safe_sin(x):
    return np.sin(x)

def _safe_cos(x):
    return np.cos(x)

def _safe_tan(x):
    return np.tan(x)

def _safe_asin(x):
    return np.arcsin(np.clip(x, -1, 1))

def _safe_acos(x):
    return np.arccos(np.clip(x, -1, 1))

def _safe_atan(x):
    return np.arctan(x)

def _safe_abs(x):
    return np.abs(x)

def _safe_pow(x, y):
    # Simplified power for stability, assuming y is often constant or simple
    # For GP, we usually restrict pow to integer powers or avoid it if unstable
    # Let's just use x^y with clipping
    return np.power(np.maximum(x, 1e-12), np.clip(y, -10, 10))

# Define the set of functions and terminals
# Terminals: x, constants (small integers, pi, e)
# Functions: +, -, *, /, sin, cos, tan, log, sqrt, exp, abs, pow
TERMINALS = ['x', '1', '2', '3', 'pi', 'e']
FUNCTIONS = ['+', '-', '*', '/', 'sin', 'cos', 'tan', 'log', 'sqrt', 'exp', 'abs', 'pow']

# Mapping to actual numpy functions
FUNC_MAP = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': _safe_div,
    'sin': _safe_sin,
    'cos': _safe_cos,
    'tan': _safe_tan,
    'log': _safe_log,
    'sqrt': _safe_sqrt,
    'exp': _safe_exp,
    'abs': _safe_abs,
    'pow': _safe_pow
}

class GPNode:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children if children else []

    def evaluate(self, x):
        if self.name in TERMINALS:
            if self.name == 'x':
                return x
            elif self.name == 'pi':
                return np.full_like(x, math.pi)
            elif self.name == 'e':
                return np.full_like(x, math.e)
            else:
                return np.full_like(x, float(self.name))
        
        func = FUNC_MAP[self.name]
        if len(self.children) == 1:
            return func(self.children[0].evaluate(x))
        elif len(self.children) == 2:
            return func(self.children[0].evaluate(x), self.children[1].evaluate(x))
        else:
            raise ValueError(f"Invalid number of children for {self.name}")

    def get_subtree(self):
        return self

    def depth(self):
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)

    def size(self):
        if not self.children:
            return 1
        return 1 + sum(child.size() for child in self.children)

    def copy(self):
        return _copy_node(self)

def _copy_node(node):
    new_node = GPNode(node.name)
    for child in node.children:
        new_node.children.append(_copy_node(child))
    return new_node

def _create_random_tree(max_depth, current_depth=0):
    if current_depth >= max_depth or (current_depth >= max_depth - 1 and random.random() < 0.5):
        return GPNode(random.choice(TERMINALS))
    
    func = random.choice(FUNCTIONS)
    num_args = 1 if func in ['sin', 'cos', 'tan', 'log', 'sqrt', 'exp', 'abs'] else 2
    
    children = []
    for _ in range(num_args):
        children.append(_create_random_tree(max_depth, current_depth + 1))
    
    return GPNode(func, children)

def _get_subtree(tree, pos):
    # Simple traversal to get node at position
    # This is a bit inefficient but trees are small
    stack = [tree]
    while stack:
        node = stack.pop()
        if node is tree:
            return node
        # This logic is flawed for general traversal. Let's use a list of nodes.
    pass

def _list_nodes(tree):
    nodes = []
    stack = [tree]
    while stack:
        node = stack.pop()
        nodes.append(node)
        for child in reversed(node.children):
            stack.append(child)
    return nodes

def _crossover(tree1, tree2):
    """Subtree crossover"""
    nodes1 = _list_nodes(tree1)
    nodes2 = _list_nodes(tree2)
    
    node1 = random.choice(nodes1)
    node2 = random.choice(nodes2)
    
    # Replace node1 in tree1 with copy of node2
    # We need to find the parent of node1. 
    # To simplify, we'll just reconstruct the tree if we find the parent.
    # Or, we can just swap the subtrees.
    
    # Let's do a simpler mutation/crossover by swapping subtrees in the tree structure
    # This requires parent pointers or recursive replacement.
    
    # Recursive replacement
    def replace(node, target, replacement):
        if node is target:
            return replacement
        for i, child in enumerate(node.children):
            node.children[i] = replace(child, target, replacement)
        return node
    
    new_tree1 = replace(tree1, node1, node2.copy())
    new_tree2 = replace(tree2, node2, node1.copy())
    
    return new_tree1, new_tree2

def _mutate(tree, rate=0.1):
    """Point mutation or subtree mutation"""
    nodes = _list_nodes(tree)
    for node in nodes:
        if random.random() < rate:
            # Replace node with a random new subtree of similar depth
            max_depth = min(3, node.depth())
            new_subtree = _create_random_tree(max_depth)
            
            # Replace in tree
            def replace(node, target, replacement):
                if node is target:
                    return replacement
                for i, child in enumerate(node.children):
                    node.children[i] = replace(child, target, replacement)
                return node
            
            return replace(tree, node, new_subtree)
    return tree

def _fit_gp(x_train, y_train, time_budget):
    """
    Runs GP to find the best expression.
    Returns the best tree and its fitness.
    """
    population = []
    pop_size = 100
    max_depth = 6
    
    # Initialize population
    for _ in range(pop_size):
        tree = _create_random_tree(max_depth)
        population.append(tree)
    
    best_tree = None
    best_fitness = np.inf
    best_pred = None
    
    start_time = time.time()
    generation = 0
    
    while time.time() < start_time + time_budget:
        generation += 1
        # Evaluate population
        for tree in population:
            try:
                pred = tree.evaluate(x_train)
                # Check for finite values
                if not np.all(np.isfinite(pred)):
                    fitness = 1e10
                else:
                    fitness = np.mean((pred - y_train) ** 2)
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_tree = tree.copy()
                    best_pred = pred
            except Exception:
                fitness = 1e10
        
        # Early stopping if fitness is very low (noiseless data)
        if best_fitness < 1e-10:
            break
        
        # Selection, Crossover, Mutation
        new_population = []
        
        # Elitism: keep best
        if best_tree:
            new_population.append(best_tree.copy())
        
        while len(new_population) < pop_size:
            # Tournament selection
            t1 = random.choice(population)
            t2 = random.choice(population)
            
            # Evaluate fitness for tournament if not cached (simplification: re-evaluate or cache)
            # For efficiency, we should cache fitness. Let's assume we do.
            # To keep code simple, we'll just pick random parents from top half?
            # No, let's do proper tournament with cached fitness if possible.
            # Since we don't have a global cache map here easily, let's just use random selection 
            # or re-evaluate. Re-evaluating is expensive.
            
            # Let's just pick two random parents from the current population.
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            
            # Crossover
            if random.random() < 0.8:
                child1, child2 = _crossover(parent1, parent2)
                new_population.append(child1)
                if len(new_population) < pop_size:
                    new_population.append(child2)
            else:
                # Mutation
                child1 = _mutate(parent1, rate=0.2)
                new_population.append(child1)
                if len(new_population) < pop_size:
                    child2 = _mutate(parent2, rate=0.2)
                    new_population.append(child2)
        
        population = new_population

    return best_tree, best_fitness, best_pred

def fit(X_train, y_train):
    global _DEADLINE
    
    # Ensure 1D input handling
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    x_train = X_train[:, 0]
    y_train = y_train
    
    # Sort training data (not strictly necessary for GP evaluation, but good for consistency)
    # GP evaluates pointwise, so order doesn't matter for MSE.
    
    # Time budget for GP
    # We have 4 equations. Total 300s.
    # Let's give each equation ~70s.
    time_budget = 60.0
    
    best_tree, best_fitness, best_pred = _fit_gp(x_train, y_train, time_budget)
    
    # Fallback if GP fails or finds nothing good
    if best_tree is None:
        # Fallback to simple linear regression or mean
        try:
            # Linear fit
            coeffs = np.polyfit(x_train, y_train, 1)
            def predict(X):
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                x_test = X[:, 0]
                return np.polyval(coeffs, x_test)
        except Exception:
            y_mean = np.mean(y_train)
            def predict(X):
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                return np.full(X.shape[0], y_mean)
        return predict

    # Use the best tree found
    def predict(X):
        try:
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            x_test = X[:, 0]
            preds = best_tree.evaluate(x_test)
            # Ensure finite
            preds = np.where(np.isfinite(preds), preds, np.mean(y_train))
            return preds
        except Exception:
            return np.full(X.shape[0], np.mean(y_train))
    
    return predict

# EVOLVE-BLOCK-END
