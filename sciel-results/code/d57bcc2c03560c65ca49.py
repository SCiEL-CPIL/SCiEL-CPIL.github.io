"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start local search with diverse initializations and local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Broad exploration with diverse starts
        # We allocate a portion of the budget to find good starting points
        # and then refine them.
        
        # Number of parallel searches (agents)
        # With budget 500 and dim 5, we can afford several agents.
        # Let's say 5 agents, each getting ~100 calls.
        n_agents = 5
        budget_per_agent = self.budget // n_agents
        
        best_x = None
        best_f = np.inf
        
        # Initial global scan: sample some points to find rough minima
        # This helps set up the agents in promising regions
        n_scan = 20
        scan_xs = []
        scan_fxs = []
        
        # Use Latin Hypercube or random sampling for the scan
        # Random is simpler and often sufficient for 5D
        for _ in range(n_scan):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            scan_xs.append(x)
            scan_fxs.append(fx)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        # Sort scan results to pick the best starting points for agents
        if scan_xs:
            sorted_indices = np.argsort(scan_fxs)
            # Pick the top n_agents distinct regions
            # Simple strategy: just take the best n_agents from the scan
            # To ensure diversity, we might want to pick non-overlapping points,
            # but for now, let's just use the best ones as seeds for local search.
            
            agent_starts = []
            for idx in sorted_indices[:n_agents]:
                agent_starts.append(scan_xs[idx])
            
            # If we have fewer than n_agents good starts, fill with random
            while len(agent_starts) < n_agents:
                agent_starts.append(rng.uniform(-5, 5, self.dim))
        else:
            agent_starts = [rng.uniform(-5, 5, self.dim) for _ in range(n_agents)]
            
        # Phase 2: Local Refinement for each agent
        # Use a simple (1+1)-ES for each agent, but with a fixed, well-tuned sigma
        # or a simple gradient-free local search.
        # Given the oscillatory nature of Katsuura, a small step size local search is good.
        
        for i in range(n_agents):
            if time.time() > deadline:
                break
                
            x = agent_starts[i].copy()
            fx = best_f if i == 0 and agent_starts[i] is best_x else func(x)
            
            # If we didn't evaluate the start point yet (unless it was the global best from scan)
            # Actually, scan_xs were evaluated. But agent_starts might be new randoms.
            # Let's ensure we have fx for x.
            if i >= n_scan or agent_starts[i] is not best_x:
                 # Re-evaluate if necessary, or just assume we have it from scan if it's from there.
                 # To be safe and simple, let's just track best_f globally.
                 # For the first agent, we might have already evaluated it in scan.
                 # For simplicity, let's just run the local search loop.
                 pass
            
            # Local Search: (1+1)-ES with adaptive sigma
            sigma = 0.5  # Start with a moderate step size
            local_budget = budget_per_agent
            calls_made = 0
            
            for _ in range(local_budget):
                if time.time() > deadline:
                    break
                    
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    # Increase step size slightly if progress
                    sigma = min(sigma * 1.2, 2.0)
                    
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Decrease step size if no progress
                    sigma *= 0.8
                    
                    # If sigma becomes too small, we might be stuck in a local min
                    # We could restart this agent or just stop.
                    # For now, just continue with smaller steps.
                    if sigma < 0.01:
                        # Try a few random jumps to escape local min
                        for _ in range(3):
                            if time.time() > deadline:
                                break
                            y_jump = np.clip(x + rng.standard_normal(self.dim), -5, 5)
                            fy_jump = func(y_jump)
                            calls_made += 1
                            if fy_jump < fx:
                                x = y_jump
                                fx = fy_jump
                                sigma = 0.5
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                                break
                        else:
                            # No escape found, break out of local search for this agent
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
