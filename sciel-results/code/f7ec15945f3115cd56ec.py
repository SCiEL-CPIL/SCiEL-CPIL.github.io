"""Few edges in 5-uniform hypergraphs without Property B (m(5))"""
import time
import random
import struct
import ctypes
import ctypes.util
import mmap
import os
import tempfile
import subprocess
import sys
from itertools import combinations

_DEADLINE = 140.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _check_time():
    return _elapsed() > _DEADLINE

# --- C++ SAT Solver Compilation ---
_sat_binary = None

def _compile_sat_solver():
    global _sat_binary
    if _sat_binary is not None:
        return _sat_binary
    
    # Simple DPLL for 2-coloring of 5-uniform hypergraph
    # Variables: x_v (0 or 1)
    # Constraint: For each edge e, not all 0 and not all 1
    # This is equivalent to: For each edge, at least one 0 and at least one 1.
    # We can model this as a SAT problem with clauses.
    # Clause 1: (x_v1 OR x_v2 OR x_v3 OR x_v4 OR x_v5) -> Not all 0
    # Clause 2: (!x_v1 OR !x_v2 OR !x_v3 OR !x_v4 OR !x_v5) -> Not all 1
    # This is a CNF with 2 clauses per edge, each of width 5.
    # We can write a simple DPLL in C++ for this specific structure.
    
    cpp_code = r"""
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// Read: n, m
// Then m lines, each with 5 ints (vertices 0..n-1)
// Output: 1 if 2-colorable, 0 if not.
// Also output the coloring if 2-colorable (n lines of 0/1)

int n, m;
std::vector<std::vector<int>> edges;
int* color; // -1 unassigned, 0, 1

// Count colors in edge
void count_edge(int e_idx, int& c0, int& c1, int& unassigned) {
    c0 = c1 = unassigned = 0;
    for (int i = 0; i < 5; i++) {
        int v = edges[e_idx][i];
        if (color[v] == 0) c0++;
        else if (color[v] == 1) c1++;
        else unassigned++;
    }
}

bool propagate() {
    bool changed = true;
    while (changed) {
        changed = false;
        for (int i = 0; i < m; i++) {
            int c0, c1, unassigned;
            count_edge(i, c0, c1, unassigned);
            if (c0 == 5 || c1 == 5) return false;
            if (c0 == 4 && unassigned == 1) {
                for (int j = 0; j < 5; j++) {
                    int v = edges[i][j];
                    if (color[v] == -1) {
                        color[v] = 1;
                        changed = true;
                        break;
                    }
                }
            } else if (c1 == 4 && unassigned == 1) {
                for (int j = 0; j < 5; j++) {
                    int v = edges[i][j];
                    if (color[v] == -1) {
                        color[v] = 0;
                        changed = true;
                        break;
                    }
                }
            }
        }
    }
    return true;
}

// MRV: pick unassigned vertex with highest degree
int pick_vertex() {
    int best = -1;
    int best_deg = -1;
    // Precomputed degrees
    for (int v = 0; v < n; v++) {
        if (color[v] == -1) {
            // Simple heuristic: just pick first? No, MRV is better.
            // But computing degree dynamically is slow. Let's use static degree.
            // We'll pass static degrees.
        }
    }
    // For simplicity and speed in this small case, let's just pick the first unassigned.
    // Or better: pick the one in the most "critical" edge.
    // Actually, for 5-uniform, just picking any unassigned is often fine with propagation.
    for (int v = 0; v < n; v++) {
        if (color[v] == -1) return v;
    }
    return -1;
}

bool backtrack() {
    if (!propagate()) return false;
    
    int v = pick_vertex();
    if (v == -1) {
        // All assigned, check if valid
        for (int i = 0; i < m; i++) {
            int c0, c1, unassigned;
            count_edge(i, c0, c1, unassigned);
            if (c0 == 5 || c1 == 5) return false;
        }
        return true;
    }
    
    // Try 0
    color[v] = 0;
    if (backtrack()) return true;
    
    // Try 1
    color[v] = 1;
    if (backtrack()) return true;
    
    color[v] = -1;
    return false;
}

int main(int argc, char* argv[]) {
    // Read from stdin
    std::cin >> n >> m;
    edges.resize(m);
    for (int i = 0; i < m; i++) {
        edges[i].resize(5);
        for (int j = 0; j < 5; j++) {
            std::cin >> edges[i][j];
        }
    }
    
    color = new int[n];
    memset(color, -1, n * sizeof(int));
    
    bool result = backtrack();
    
    std::cout << (result ? 1 : 0) << std::endl;
    if (result) {
        for (int i = 0; i < n; i++) {
            std::cout << color[i] << std::endl;
        }
    }
    
    delete[] color;
    return 0;
}
"""
    
    # Write to temp file
    tmpdir = tempfile.mkdtemp()
    cpp_file = os.path.join(tmpdir, "solver.cpp")
    bin_file = os.path.join(tmpdir, "solver")
    
    with open(cpp_file, 'w') as f:
        f.write(cpp_code)
    
    # Compile
    try:
        subprocess.run(['g++', '-O3', '-o', bin_file, cpp_file], 
                       check=True, capture_output=True, timeout=30)
        _sat_binary = bin_file
        return bin_file
    except Exception as e:
        # Fallback: use Python checker
        _sat_binary = None
        return None

def _is_2_colorable(n, edges):
    """
    Check if a 5-uniform hypergraph has a proper 2-coloring.
    Returns True if 2-colorable, False otherwise.
    Uses C++ SAT solver if available, else Python fallback.
    """
    if not edges:
        return True
    
    # Try C++ solver
    if _sat_binary and os.path.exists(_sat_binary):
        try:
            input_data = f"{n} {len(edges)}\n"
            for e in edges:
                input_data += " ".join(map(str, e)) + "\n"
            
            result = subprocess.run([_sat_binary], 
                                   input=input_data.encode(),
                                   capture_output=True, 
                                   timeout=5)
            
            lines = result.stdout.decode().strip().split('\n')
            if lines and lines[0] == '1':
                return True
            else:
                return False
        except Exception:
            pass
    
    # Python fallback (slower)
    return _is_2_colorable_python(n, edges)

def _is_2_colorable_python(n, edges):
    """Python fallback for 2-colorability check."""
    if not edges:
        return True
    
    colors = [-1] * n
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for i, e in enumerate(edges):
                c0 = sum(1 for v in e if colors[v] == 0)
                c1 = sum(1 for v in e if colors[v] == 1)
                unassigned = [v for v in e if colors[v] == -1]
                
                if c0 == 5 or c1 == 5:
                    return False
                if c0 == 4 and len(unassigned) == 1:
                    v = unassigned[0]
                    if colors[v] == -1:
                        colors[v] = 1
                        changed = True
                elif c1 == 4 and len(unassigned) == 1:
                    v = unassigned[0]
                    if colors[v] == -1:
                        colors[v] = 0
                        changed = True
        return True
    
    def backtrack():
        if not propagate():
            return False
        
        # Find unassigned vertex
        v = -1
        for i in range(n):
            if colors[i] == -1:
                v = i
                break
        
        if v == -1:
            # All assigned, verify
            for e in edges:
                c0 = sum(1 for x in e if colors[x] == 0)
                c1 = sum(1 for x in e if colors[x] == 1)
                if c0 == 5 or c1 == 5:
                    return False
            return True
        
        # Try 0
        colors[v] = 0
        if backtrack():
            return True
        
        # Try 1
        colors[v] = 1
        if backtrack():
            return True
        
        colors[v] = -1
        return False
    
    return backtrack()

def construct():
    """
    Construct a 5-uniform hypergraph with no proper 2-coloring, minimizing edges.
    Strategy: Randomized greedy set-cover with SAT guidance.
    1. Start with a small random set of edges.
    2. Find a coloring that satisfies all current edges (if exists).
    3. Add an edge that is violated by this coloring (i.e., monochromatic in it).
    4. Repeat until no coloring exists.
    5. Then prune: try to remove edges while maintaining non-2-colorability.
    """
    # Compile SAT solver
    _compile_sat_solver()
    
    best_edges = None
    best_count = float('inf')
    
    # Multiple restarts
    for restart in range(50):
        if _check_time():
            break
        
        # Start with a small random seed
        n = random.randint(9, 15)
        all_possible_edges = list(combinations(range(n), 5))
        random.shuffle(all_possible_edges)
        
        # Start with a few edges
        num_seed = random.randint(5, 15)
        current_edges = [list(e) for e in all_possible_edges[:num_seed]]
        
        # Greedy addition phase
        for step in range(200):
            if _check_time():
                break
            
            # Check if current edges are 2-colorable
            if not _is_2_colorable(n, current_edges):
                break
            
            # Find a coloring
            # We need the actual coloring to find a violated edge.
            # Let's modify the solver to output the coloring.
            # For now, use a simple heuristic: try to find a monochromatic edge
            # by random sampling? No, that's unreliable.
            
            # Let's use the Python fallback to get the coloring
            colors = [-1] * n
            vertex_edges = [[] for _ in range(n)]
            for i, e in enumerate(current_edges):
                for v in e:
                    vertex_edges[v].append(i)
            
            def propagate_colors():
                changed = True
                while changed:
                    changed = False
                    for i, e in enumerate(current_edges):
                        c0 = sum(1 for v in e if colors[v] == 0)
                        c1 = sum(1 for v in e if colors[v] == 1)
                        unassigned = [v for v in e if colors[v] == -1]
                        
                        if c0 == 4 and len(unassigned) == 1:
                            v = unassigned[0]
                            if colors[v] == -1:
                                colors[v] = 1
                                changed = True
                        elif c1 == 4 and len(unassigned) == 1:
                            v = unassigned[0]
                            if colors[v] == -1:
                                colors[v] = 0
                                changed = True
                return True
            
            def backtrack_colors():
                if not propagate_colors():
                    return False
                
                v = -1
                for i in range(n):
                    if colors[i] == -1:
                        v = i
                        break
                
                if v == -1:
                    for e in current_edges:
                        c0 = sum(1 for x in e if colors[x] == 0)
                        c1 = sum(1 for x in e if colors[x] == 1)
                        if c0 == 5 or c1 == 5:
                            return False
                    return True
                
                colors[v] = 0
                if backtrack_colors():
                    return True
                colors[v] = 1
                if backtrack_colors():
                    return True
                colors[v] = -1
                return False
            
            if not backtrack_colors():
                # Should not happen if _is_2_colorable returned True
                break
            
            # Find an edge that is monochromatic in this coloring
            violated_edge = None
            for e in all_possible_edges:
                if e in current_edges:
                    continue
                c0 = sum(1 for v in e if colors[v] == 0)
                c1 = sum(1 for v in e if colors[v] == 1)
                if c0 == 5 or c1 == 5:
                    violated_edge = list(e)
                    break
            
            if violated_edge is None:
                # No violated edge found? This shouldn't happen if 2-colorable.
                # Maybe the coloring doesn't make any edge monochromatic?
                # That means the hypergraph is 2-colorable, but we need to add an edge.
                # Pick a random edge.
                violated_edge = list(random.choice(all_possible_edges))
            
            current_edges.append(violated_edge)
        
        # Pruning phase
        if not _check_time() and not _is_2_colorable(n, current_edges):
            improved = True
            while improved and not _check_time():
                improved = False
                indices = list(range(len(current_edges)))
                random.shuffle(indices)
                
                for i in indices:
                    if _check_time():
                        break
                    candidate = current_edges[:i] + current_edges[i+1:]
                    if not _is_2_colorable(n, candidate):
                        current_edges = candidate
                        improved = True
                        break
        
        # Update best
        if not _is_2_colorable(n, current_edges):
            if len(current_edges) < best_count:
                best_count = len(current_edges)
                best_edges = [list(e) for e in current_edges]
    
    # If no valid hypergraph found, return a known one (K_9^5)
    if best_edges is None:
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
    
    return {'n': n, 'edges': best_edges}

# Alias for compatibility
def _is_2_colorable_sat(n, edges):
    return _is_2_colorable(n, edges)
