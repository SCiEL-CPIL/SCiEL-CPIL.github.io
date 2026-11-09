"""Parallel Multi-Start Adaptive (1+1)-ES Portfolio for Schaffers F7."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        best_x_global = None
        best_f_global = float('inf')
        calls = 0
        
        # Number of parallel agents. 
        # 5 agents is a good balance for 5D with 500 budget (100 calls each if perfectly split, 
        # but we'll do round-robin so it's dynamic).
        n_agents = 5
        
        # Initialize agents with diverse random starts
        # We ensure the first call is made immediately to satisfy "exception before first call" risk
        # and to establish the first AOCC point.
        
        agents = []
        
        # First call: Start agent 0 at a random point
        x0 = np.random.uniform(-5, 5, self.dim)
        fx0 = func(x0)
        calls += 1
        
        best_x_global = x0.copy()
        best_f_global = fx0
        
        # Agent 0 state
        agents.append({
            'x': x0.copy(),
            'fx': fx0,
            'sigma': 1.0,
            'window': [],
            'successes': 0
        })
        
        # Initialize other agents. 
        # We can either spend budget now to init them, or let them start when they get their first turn.
        # To save budget for search, we will initialize their 'x' randomly but not evaluate them yet.
        # However, (1+1)-ES needs a starting fitness. 
        # Strategy: When an agent is first activated, we evaluate its start point.
        # So agents 1-4 have x defined but fx = inf (not evaluated).
        
        for i in range(1, n_agents):
            xi = np.random.uniform(-5, 5, self.dim)
            agents.append({
                'x': xi,
                'fx': float('inf'),
                'sigma': 1.0,
                'window': [],
                'successes': 0,
                'initialized': False
            })
        
        # Agent 0 is initialized
        agents[0]['initialized'] = True
        
        # Main Loop: Round-robin among agents
        # We track which agent's turn it is
        agent_idx = 0
        
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            agent = agents[agent_idx]
            
            # If this agent hasn't been evaluated yet, evaluate its start point
            if not agent.get('initialized', True):
                fx_start = func(agent['x'])
                calls += 1
                agent['fx'] = fx_start
                agent['initialized'] = True
                
                if fx_start < best_f_global:
                    best_f_global = fx_start
                    best_x_global = agent['x'].copy()
                
                # Move to next agent
                agent_idx = (agent_idx + 1) % n_agents
                continue
            
            # Normal (1+1)-ES step
            # Generate offspring
            noise = np.random.standard_normal(self.dim)
            y = agent['x'] + agent['sigma'] * noise
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls += 1
            
            # Update global best
            if fy < best_f_global:
                best_f_global = fy
                best_x_global = y.copy()
            
            # Update local agent state
            if fy < agent['fx']:
                agent['x'] = y
                agent['fx'] = fy
                agent['successes'] += 1
            else:
                agent['successes'] += 0 # Or just don't increment
            
            # Adapt sigma using 1/5 success rule over a window of 5
            # We'll use a simple counter that resets every 5 steps
            # To keep it simple and robust, we use a fixed window of 5 evaluations per agent
            # But since we are round-robining, we need to track steps per agent.
            # Let's use a simpler adaptation: 
            # Keep a local counter 'steps_since_adapt'
            if 'steps_since_adapt' not in agent:
                agent['steps_since_adapt'] = 0
                agent['local_successes'] = 0
            
            agent['steps_since_adapt'] += 1
            
            if agent['steps_since_adapt'] >= 5:
                success_rate = agent['local_successes'] / 5.0
                if success_rate > 0.5:
                    agent['sigma'] *= 1.3
                elif success_rate < 0.5:
                    agent['sigma'] *= 0.7
                
                agent['sigma'] = np.clip(agent['sigma'], 1e-8, 5.0)
                agent['steps_since_adapt'] = 0
                agent['local_successes'] = 0
            
            # Move to next agent
            agent_idx = (agent_idx + 1) % n_agents
        
        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
