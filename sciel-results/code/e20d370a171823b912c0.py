import numpy as np
import time
from scipy.optimize import minimize

_cache = {}
_START_TIME = time.time()

def _simulate_objective(theta, params, n_paths=200, seed=42):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    h0, h1, h2, h3 = theta
    
    # Initialize
    inventory = np.zeros(n_paths, dtype=int)
    mid = np.full(n_paths, 100.0)
    objective = 0.0
    
    rng = np.random.default_rng(seed)
    
    for n in range(steps):
        t = n * dt
        tau = T - t
        
        # Calculate depths
        half_spread = h0 + h2 * tau
        skew_coeff = h1 + h3 * tau
        skew = skew_coeff * inventory
        
        bid_depths = half_spread + skew
        ask_depths = half_spread - skew
        
        # Clip depths
        bid_depths = np.clip(bid_depths, 0.0, np.inf)
        ask_depths = np.clip(ask_depths, 0.0, np.inf)
        
        # Calculate fill probabilities
        exp_term_bid = np.exp(-k * bid_depths)
        exp_term_ask = np.exp(-k * ask_depths)
        
        p_bid = 1.0 - np.exp(-A * exp_term_bid * dt)
        p_ask = 1.0 - np.exp(-A * exp_term_ask * dt)
        
        # No fill at boundaries
        p_bid[inventory == Q] = 0.0
        p_ask[inventory == -Q] = 0.0
        
        # Simulate fills
        rand_bid = rng.random(n_paths)
        rand_ask = rng.random(n_paths)
        
        bid_fill = rand_bid < p_bid
        ask_fill = rand_ask < p_ask
        
        # Update inventory
        # Bid fill -> Sell -> Inventory -1
        # Ask fill -> Buy -> Inventory +1
        inventory = inventory - bid_fill.astype(int) + ask_fill.astype(int)
        
        # Objective
        earned = bid_depths * bid_fill + ask_depths * ask_fill
        holding_cost = 0.5 * gamma * sigma**2 * inventory**2 * dt
        
        objective += np.sum(earned - holding_cost)
        
        # Update mid
        mid += sigma * np.sqrt(dt) * rng.normal(0, 1, n_paths)
        
    return objective / n_paths

def _solve_regime(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key in _cache:
        return _cache[key]
        
    if time.time() - _START_TIME > 280:
        sigma = params['sigma']
        k = params['k']
        gamma = params['gamma']
        g_s2 = gamma * sigma * sigma
        if g_s2 < 1e-9:
            h0 = 1.0 / k
        else:
            h0 = (1.0 / k) * np.log1p(k / g_s2)
        h1 = g_s2
        res = np.array([h0, h1, 0.0, 0.0])
        _cache[key] = res
        return res

    sigma = params['sigma']
    k = params['k']
    gamma = params['gamma']
    g_s2 = gamma * sigma * sigma
    if g_s2 < 1e-9:
        h0_init = 1.0 / k
    else:
        h0_init = (1.0 / k) * np.log1p(k / g_s2)
    h1_init = g_s2
    x0 = np.array([h0_init, h1_init, 0.0, 0.0])
    
    bounds = [(0.0, 5.0), (-5.0, 5.0), (-0.5, 0.5), (-0.5, 0.5)]
    
    res = minimize(_simulate_objective, x0, args=(params,), method='Nelder-Mead', 
                   options={'xatol': 1e-3, 'fatol': 1e-3, 'maxiter': 30})
    
    theta_opt = res.x
    _cache[key] = theta_opt
    return theta_opt

def quote(t, inventory, mid, params):
    theta = _solve_regime(params)
    h0, h1, h2, h3 = theta
    
    tau = params['T'] - t
    
    half_spread = h0 + h2 * tau
    skew_coeff = h1 + h3 * tau
    skew = skew_coeff * inventory
    
    bid_depths = half_spread + skew
    ask_depths = half_spread - skew
    
    bid_depths = np.maximum(bid_depths, 0.0)
    ask_depths = np.maximum(ask_depths, 0.0)
    
    return bid_depths, ask_depths
