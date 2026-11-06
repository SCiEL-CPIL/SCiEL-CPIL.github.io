"""
Constructs a sparse pancyclic graph on N=40 vertices.
Minimizes edge count.
"""
N = 40

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a sparse pancyclic graph on N=40 vertices.
    
    Design:
    - Base: Hamiltonian cycle on vertices 0..39 (40 edges).
    - Chords: Search for minimal chord set from vertex 0 that covers all cycle lengths 3..39.
    - Total Edges: 59 (proven minimum for this family).
    - Value: 59 - 40 = 19.
    
    The cycles are explicitly constructed and returned.
    """
    start_time = time.time()
    deadline = 60.0  # Safety margin
    
    # For the Cycle+Chords family, chord (0,k) covers lengths {k+1, N-k+1}
    # We need to cover all lengths 3..39.
    # Length L is covered by chord k where k+1=L or N-k+1=L, i.e., k=L-1 or k=N-L+1.
    # For L=3: k=2 or k=38. But k must be in [2, N-2] for a valid chord.
    # Actually, for a chord (0,k) with 2<=k<=N-2, it creates two cycles:
    #   Cycle 1: 0-1-2-...-k-0 (length k+1)
    #   Cycle 2: 0-k-(k+1)-...-(N-1)-0 (length N-k+1)
    # So chord k covers lengths {k+1, N-k+1}.
    
    # To cover length L, we need a chord k such that k+1=L or N-k+1=L.
    # k+1=L => k=L-1
    # N-k+1=L => k=N-L+1
    
    # For L in 3..39:
    # L=3: k=2 or k=38
    # L=4: k=3 or k=37
    # ...
    # L=20: k=19 or k=21
    # L=21: k=20 or k=20
    # L=22: k=21 or k=19
    # ...
    # L=39: k=38 or k=2
    
    # Note that k and N-k give the same pair of lengths (just swapped).
    # So we can restrict k to 2..20 (since k=20 gives length 21,21).
    # k=2 covers {3,39}
    # k=3 covers {4,38}
    # ...
    # k=19 covers {20,22}
    # k=20 covers {21,21}
    
    # We need to select a subset of {2,3,...,20} that covers all lengths 3..39.
    # Each k covers two lengths (or one if k=20).
    # This is a set cover problem.
    
    # Let's verify that 19 chords are necessary.
    # Lengths 3..39 = 37 lengths to cover.
    # Each chord covers at most 2 lengths.
    # So we need at least ceil(37/2) = 19 chords.
    # The standard construction achieves exactly 19, so it's optimal.
    
    # Build the edge list
    edges = []
    
    # 1. Hamiltonian Cycle
    for i in range(N):
        edges.append((i, (i + 1) % N))
        
    # 2. Chords from vertex 0 to cover all cycle lengths
    # Use the optimal set of 19 chords: k = 2, 3, ..., 20
    for k in range(2, N // 2 + 1):
        edges.append((0, k))
        
    # Convert to list of lists for output format
    edge_list = [list(e) for e in edges]
    
    # Generate cycles
    cycles = []
    for length in range(3, N + 1):
        if length == N:
            # Hamiltonian cycle
            cycles.append(list(range(N)))
        else:
            if length <= N // 2 + 1:
                # Chord (0, length-1) creates cycle 0-1-...-(length-1)-0
                # This covers lengths 3 to N//2 + 1
                k = length - 1
                cycles.append(list(range(k + 1)))
            else:
                # Chord (0, N-length+1) creates the other cycle
                # Cycle goes 0 -> N-1 -> ... -> (N-length+1) -> 0
                # Length is N - (N-length+1) + 1 = length
                k = N - length + 1
                # Vertices: 0, N-1, N-2, ..., k
                cycles.append([0] + list(range(N - 1, k - 1, -1)))
                
    # Check deadline
    if time.time() - start_time > deadline:
         pass

    return {'edges': edge_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
