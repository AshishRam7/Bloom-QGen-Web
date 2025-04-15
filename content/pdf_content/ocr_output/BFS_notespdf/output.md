# Lecture 13: Graphs I: Breadth First Search 

## Lecture Overview

- Applications of Graph Search
- Graph Representations
- Breadth-First Search


## Recall:

Graph $G=(V, E)$

- $V=$ set of vertices (arbitrary labels)
- $E=$ set of edges i.e. vertex pairs $(v, w)$
- ordered pair $\Longrightarrow$ directed edge of graph
- unordered pair $\Longrightarrow$ undirected


**Moondream Image Description:**  The figure illustrates the concept of directed graphs, specifically undirected graphs and directed graphs. Directed graphs are characterized by the presence of edges between nodes, represented by arrows, and directed edges between nodes, represented by circles. Undirected graphs, on the other hand, lack these directional arrows and have nodes without any arrows. The figure also provides mathematical definitions for directed and undirected graphs, including the notation V = (a,b,c) and E = (a,b,c,d), which represent directed and undirected graphs respectively.

Figure 1: Example to illustrate graph terminology

## Graph Search

"Explore a graph", e.g.:

- find a path from start vertex $s$ to a desired vertex
- visit all vertices or edges of graph, or only those reachable from $s$

# Applications: 

There are many.

- web crawling (how Google finds pages)
- social networking (Facebook friend finder)
- network broadcast routing
- garbage collection
- model checking (finite state machine)
- checking mathematical conjectures
- solving puzzles and games


## Pocket Cube:

Consider a $2 \times 2 \times 2$ Rubik's cube


**Moondream Image Description:**  The cube exhibits a 3D structure with a repeating pattern of colors and numbers. The colors are arranged in a grid-like fashion, with each face exhibiting a distinct color scheme. The numbers on each face are also distinct and evenly spaced. This pattern suggests a systematic approach to problem-solving or a methodical way of visualizing data.

Configuration Graph:

- vertex for each possible state
- edge for each basic move (e.g., 90 degree turn) from one state to another
- undirected: moves are reversible

Diameter ("God's Number")
11 for $2 \times 2 \times 2,20$ for $3 \times 3 \times 3, \Theta\left(n^{2} / \lg n\right)$ for $n \times n \times n$ [Demaine, Demaine, Eisenstat Lubiw Winslow 2011]


**Moondream Image Description:**  The visualization demonstrates a hierarchical approach to problem-solving, starting with a broad search for 'hardest configurations' and then narrowing down to 'breadth-first tree' solutions.  It highlights the importance of reachability in two steps, and notes that not all solutions are reachable in two steps.

\# vertices $=8!\cdot 3^{8}=264,539,520$ where 8 ! comes from having 8 cubelets in arbitrary positions and $3^{8}$ comes as each cubelet has 3 possible twists.


**Moondream Image Description:**  The visualization demonstrates a 3D cube with a green cube at its center, surrounded by a red cube and a purple cube. The red cube is partially overlapped by the green cube, and the purple cube is partially overlapped by the green cube. The green cube appears to be the largest of the three, occupying roughly one-third of the cube's volume. The red cube is slightly smaller than the green cube, and the purple cube is slightly smaller than the red cube. The red cube is located at the bottom left corner of the cube, while the purple cube is located at the bottom right corner. The green cube is located at the top center of the cube.

This can be divided by 24 if we remove cube symmetries and further divided by 3 to account for actually reachable configurations (there are 3 connected components).

# Graph Representations: (data structures) 

## Adjacency lists:

Array $A d j$ of $|V|$ linked lists

- for each vertex $u \in V, A d j[u]$ stores $u$ 's neighbors, i.e., $\{v \in V \mid(u, v) \in E\} . \quad(u, v)$ are just outgoing edges if directed. (See Fig. 2 for an example.)


**Moondream Image Description:**  The diagram illustrates a sequence of events, starting with a single 'a' node connected to two 'b' nodes, then to a 'c' node, and finally to a 'd' node. The sequence is represented by a directed graph, where each node is a vertex and each edge represents a transition. The number of transitions from 'a' to 'b' is 2, from 'b' to 'c' is 1, and from 'c' to 'd' is 1. The diagram also shows the adjacency matrix of the graph, which is a square matrix representing the connections between nodes.

Figure 2: Adjacency List Representation: Space $\Theta(V+E)$

- in Python: $A d j=$ dictionary of list/set values; vertex $=$ any hashable object (e.g., int, tuple)
- advantage: multiple graphs on same vertices


## Implicit Graphs:

$\operatorname{Adj}(u)$ is a function - compute local structure on the fly (e.g., Rubik's Cube). This requires "Zero" Space.

# Object-oriented Variations: 

- object for each vertex $u$
- $u$.neighbors $=$ list of neighbors i.e. $A d j[u]$

In other words, this is method for implicit graphs

## Incidence Lists:

- can also make edges objects


**Moondream Image Description:**  The diagram illustrates a sequence of events or a process, starting with a single entity (e.a.) and branching out into two entities (e.b and e.c). The branching pattern suggests a hierarchical structure, possibly representing a hierarchical relationship or a network. The numerical values (e.a., e.b, e.c.) likely represent different levels or stages within the process.
- $u$.edges $=$ list of (outgoing) edges from $u$.
- advantage: store edge data without hashing


## Breadth-First Search

Explore graph level by level from $s$

- level $0=\{s\}$
- level $i=$ vertices reachable by path of $i$ edges but not fewer


**Moondream Image Description:**  The visualization demonstrates a hierarchical neural network structure.  It shows two levels of neurons, labeled 'level1' and 'level2', connected by blue lines.  Each level has a single input neuron and multiple output neurons.  The number of input neurons and output neurons for each level is clearly indicated.  The diagram also illustrates the concept of a sigmoid function, represented by the dotted line connecting the input and output neurons.

Figure 3: Illustrating Breadth-First Search

- build level $i>0$ from level $i-1$ by trying all outgoing edges, but ignoring vertices from previous levels


# Breadth-First-Search Algorithm 

$$
\begin{aligned}
& \text { BFS (V,Adj,s): } \\
& \text { level }=\{\text { s: } 0\} \\
& \text { parent }=\{s: \text { None }\} \\
& i=1 \\
& \text { frontier }=[s] \\
& \text { while frontier: } \\
& \text { next }=[] \\
& \text { for } u \text { in frontier: } \\
& \text { for } v \text { in } \operatorname{Adj}[u] \text { : } \\
& \text { if } v \text { not in level: } \\
& \text { level }[v]=i \\
& \text { parent }[v]=u \\
& \text { next.append }(v) \\
& \text { frontier }=\text { next } \\
& i+=1
\end{aligned}
$$

See CLRS for queue-based implementation

## Example



**Moondream Image Description:**  The diagram illustrates a hierarchical structure, likely representing a biological process or system.  Level 1 and Level 2 are connected by a red line, while Level 3 is isolated.  Each level is numbered sequentially.  The nodes are labeled with letters and numbers, and the connections between them are represented by blue lines.  The number of connections between each node varies, with some nodes having more than one connection.  The overall structure suggests a hierarchical model, possibly involving multiple levels of interaction or processing.
frontier $_{n}=\{\mathrm{s}\}$
frontier $_{1}=\{\mathrm{a}, \mathrm{x}\}$
frontier $_{2}=\{\mathrm{z}, \mathrm{d}, \mathrm{c}\}$
frontier $_{3}=\{\mathrm{f}, \mathrm{v}\}$
(not $\mathrm{x}, \mathrm{c}, \mathrm{d})$

Figure 4: Breadth-First Search Frontier

## Analysis:

- vertex $V$ enters next (\& then frontier) only once (because level $[v]$ then set)
base case: $v=s$

- $\Longrightarrow \operatorname{Adj}[v]$ looped through only once

$$
\text { time }=\sum_{v \in V}|A d j[V]|=\left\{\begin{array}{l}
|E| \text { for directed graphs } \\
2|E| \text { for undirected graphs }
\end{array}\right.
$$

- $\Longrightarrow O(E)$ time
- $O(V+E)$ ("LINEAR TIME") to also list vertices unreachable from $v$ (those still not assigned level)


# Shortest Paths: 

cf. L15-18

- for every vertex $v$, fewest edges to get from $s$ to $v$ is

$$
\left\{\begin{array}{l}
\text { level }[v] \text { if } v \text { assigned level } \\
\infty \quad \text { else (no path) }
\end{array}\right.
$$

- parent pointers form shortest-path tree $=$ union of such a shortest path for each $v$ $\Longrightarrow$ to find shortest path, take $v$, parent $[v]$, parent[parent $[v]]$, etc., until $s$ (or None)

MIT OpenCourseWare
http://ocw.mit.edu

# 6.006 Introduction to Algorithms 

Fall 2011

For information about citing these materials or our Terms of Use, visit: http://ocw.mit.edu/terms.