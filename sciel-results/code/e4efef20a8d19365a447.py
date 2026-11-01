"""
Strategy: Genetic Algorithm for Signed Step Functions
The task asks for a signed step function with 400 intervals to improve an autoconvolution upper bound C3.
The seed program returns np.ones(N), which is a constant function. 
In many autocorrelation contexts (like Barker sequences or Zadoff-Chu), the goal is to minimize sidelobes.
For a signed step function (values likely restricted to +/- 1 or similar discrete set), 
a Genetic Algorithm is a robust method to explore the discrete search space without assuming convexity.

We define the score as the maximum sidelobe of the autocorrelation function.
Lower is better.
The evaluator score is seed/value. Since seed is 1.0 (implied by "1.0 equals the seed program"),
if our value (max sidelobe) is less than the seed's value, we improve.
Wait, the prompt says: "score is the upper bound on C3". 
And "evaluator measures the raw value (lower is better)".
"search score is seed/value, so 1.0 equals the seed program".
This implies:
Seed Value = $V_{seed}$.
Our Value = $V_{us}$.
Score = $V_{seed} / V_{us}$.
If Score = 1.0, then $V_{us} = V_{seed}$.
If Score > 1.0, then $V_{us} < V_{seed}$.

The seed program returns `np.ones(N)`.
Let's calculate the autocorrelation of `np.ones(N)`.
$R(k) = \sum_{i} f[i] f[i+k]$.
For $f = [1, 1, ..., 1]$ of length N.
$R(0) = N$.
$R(k) = N - |k|$ for $|k| < N$.
The "upper bound on C3" likely refers to a specific metric derived from these autocorrelation values.
Often, in these problems, $C_3$ might be related to the maximum sidelobe or a normalized peak.
However, the prompt mentions "Paper and published result notebook place the absolute value differently; execution must explicitly select a metric."
This suggests ambiguity in the exact definition of the "value".
Common metrics:
1. Max sidelobe magnitude: $\max_{k \neq 0} |R(k)|$.
2. Normalized max sidelobe: $\max_{k \neq 0} |R(k)| / R(0)$.
3. Integrated sidelobe energy.

Given "Third autocorrelation inequality", it might refer to a specific bound like $|R(k)| \le C_3$.
If we use `np.ones`, $R(k)$ is triangular. The max sidelobe is $N-1$.
If we use a random signed sequence ($\pm 1$), $R(0)=N$, and $R(k)$ for $k \neq 0$ is roughly $\sqrt{N}$ (random walk).
So a random signed sequence will have a much lower max sidelobe ($\sim \sqrt{N}$) than the constant sequence ($N-1$).
Thus, a random signed sequence should yield a much better score (value $\sim \sqrt{N}$ vs $N$).

Let's implement a GA to find a sequence of $\pm 1$ that minimizes the max sidelobe of the autocorrelation.
We will use the metric: $Value = \max_{k \neq 0} |R(k)|$.
If the seed's value is $N-1$ (for `np.ones`), and we find a value of $\sim \sqrt{N}$, the score will be $\approx \sqrt{N} \gg 1$.

Deadline: 150s.
N = 400.
Autocorrelation calculation via FFT is $O(N \log N)$.
GA steps:
1. Initialize population of random $\pm 1$ sequences.
2. Evaluate fitness (max sidelobe).
3. Selection, Crossover, Mutation.
4. Repeat until time limit.

We must be careful with the definition of "value".
If the evaluator uses a different metric, we might fail.
However, "max sidelobe" is the standard objective for "improving an autoconvolution upper bound".
The "Third autocorrelation inequality" might be a reference to a specific paper (e.g., about Barker sequences or similar).
For N=400, Barker sequences don't exist (max known is 13). So we are looking for low-autocorrelation sequences (CAZAC or similar properties, or just min-max sidelobe).

Let's refine the metric selection.
"execution must explicitly select a metric".
I will select `max_sidelobe = max(abs(autocorr[1:]))`.
This is the most standard "upper bound" improvement target.

Implementation Details:
- Use NumPy for FFT-based autocorrelation.
- GA with bit-flip mutation and uniform crossover.
- Elitism to keep the best found.
- Time management: Check `time.time()` in the loop.
"""

import numpy as np
import time

N = 400
POP_SIZE = 100
MUTATION_RATE = 0.05  # 5% of bits flipped per individual per generation
CROSSOVER_RATE = 0.8
ELITE_SIZE = 5

def _autocorrelation_max_sidelobe(seq):
    """
    Calculate the maximum absolute value of the autocorrelation sidelobes.
    seq: array of +/- 1.
    """
    # Autocorrelation via FFT
    # Pad to next power of 2 for efficiency, though N=400 is small enough that direct or standard FFT works.
    # Standard FFT size 2*N is safe for linear autocorrelation.
    n = len(seq)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    
    # Compute FFT of sequence
    f = np.fft.rfft(seq, n=fft_size)
    # Power spectrum
    power = np.abs(f) ** 2
    # Inverse FFT to get autocorrelation
    acf = np.fft.irfft(power, n=fft_size)
    
    # The autocorrelation is centered. 
    # For real signals, irfft returns [R(0), R(1), ..., R(N-1), R(-1), ..., R(-(N-1))] 
    # Wait, irfft of power spectrum gives circular autocorrelation.
    # Since we padded to 2*N, the linear autocorrelation is embedded.
    # Specifically, acf[0] is R(0). acf[1:N] are R(1)..R(N-1). acf[N+1:] are negative lags.
    # Due to symmetry, R(k) = R(-k).
    # We want max |R(k)| for k != 0.
    # So we look at acf[1:N] and acf[N+1:2*N] (which are same as 1:N).
    # Just taking acf[1:N] is sufficient due to symmetry.
    
    sidelobes = acf[1:n]
    return np.max(np.abs(sidelobes))

def _evaluate_population(pop):
    """
    Evaluate a population of sequences.
    pop: list of arrays.
    Returns: list of fitness values (lower is better).
    """
    return [_autocorrelation_max_sidelobe(ind) for ind in pop]

def _crossover(parent1, parent2):
    """Uniform crossover"""
    mask = np.random.rand(len(parent1)) < 0.5
    child = np.where(mask, parent1, parent2)
    return child

def _mutate(ind, rate):
    """Bit flip mutation"""
    mask = np.random.rand(len(ind)) < rate
    ind[mask] *= -1  # Flip sign
    return ind

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initialize population
    # Start with some random sequences and maybe one structured one
    pop = [np.random.choice([-1, 1], size=N) for _ in range(POP_SIZE)]
    
    # Add a few heuristic starting points if time permits? 
    # For now, random is a good baseline for GA.
    
    # Evaluate initial population
    fitnesses = _evaluate_population(pop)
    
    # Find best
    best_idx = np.argmin(fitnesses)
    best_seq = pop[best_idx].copy()
    best_val = fitnesses[best_idx]
    
    # Elitism: Keep the best individuals
    # Sort population by fitness
    sorted_indices = np.argsort(fitnesses)
    elite_indices = sorted_indices[:ELITE_SIZE]
    
    gen = 0
    while time.time() < deadline:
        gen += 1
        
        # Selection: Tournament selection
        new_pop = []
        
        # Add elites to new population
        for i in elite_indices:
            new_pop.append(pop[i].copy())
        
        # Generate rest of population
        while len(new_pop) < POP_SIZE:
            # Select parents
            # Tournament size 3
            idx1 = np.random.randint(0, POP_SIZE)
            idx2 = np.random.randint(0, POP_SIZE)
            idx3 = np.random.randint(0, POP_SIZE)
            
            # Pick best of 3
            best_parent_idx = min(idx1, idx2, idx3, key=lambda i: fitnesses[i])
            
            # Crossover
            if np.random.rand() < CROSSOVER_RATE:
                # Select second parent
                idx4 = np.random.randint(0, POP_SIZE)
                idx5 = np.random.randint(0, POP_SIZE)
                idx6 = np.random.randint(0, POP_SIZE)
                second_parent_idx = min(idx4, idx5, idx6, key=lambda i: fitnesses[i])
                
                p1 = pop[best_parent_idx]
                p2 = pop[second_parent_idx]
                child = _crossover(p1, p2)
            else:
                child = pop[best_parent_idx].copy()
            
            # Mutation
            child = _mutate(child, MUTATION_RATE)
            
            new_pop.append(child)
        
        pop = new_pop
        fitnesses = _evaluate_population(pop)
        
        # Update best
        current_best_idx = np.argmin(fitnesses)
        if fitnesses[current_best_idx] < best_val:
            best_val = fitnesses[current_best_idx]
            best_seq = pop[current_best_idx].copy()
            
        # Update elites for next generation
        sorted_indices = np.argsort(fitnesses)
        elite_indices = sorted_indices[:ELITE_SIZE]
        
        # Check time
        if time.time() > deadline:
            break
            
    return best_seq

# Test locally to ensure it runs and produces a valid array
if __name__ == "__main__":
    res = construct()
    print("Shape:", res.shape)
    print("Values:", np.unique(res))
    print("Max Sidelobe:", _autocorrelation_max_sidelobe(res))
