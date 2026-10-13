# KG-02 — Graph Mathematics, Adjacency Matrices, Reachability, BFS and DFS

> **Knowledge Graph and Large Graph Model curriculum, Item 2.**
>
> **Teaching state:** detailed source-grounded lesson *prepared* on 2026-10-09. Subsection 2.1 (**directed degree and global degree-sum identity**) passed its conceptual check on 2026-10-09 after a corrected diagram, learner's independent graph-level explanation and successful edge-reversal reasoning. Subsection 2.2 (**walks, trails, simple paths**) is in study. **KG-02 as a whole is not mastered and no full math competency is READY.** Do not treat preparation as mastery. KG-01 remains introduced/unassessed in authoritative trackers. Continue KG-02 one subsection at a time; do not open KG-03 without the learner's demonstration.
>
> **Academic spine:** Stanford CS224W (Fall 2026) Lecture 01, *Introduction to Machine Learning for Graphs* (slide deck 01-intro.pdf; particularly graph modeling and structural node features); MIT 6.042J (Fall 2010), Chapters 5–6 *Graph Theory* and *Directed Graphs*; MIT 6.006 (Spring 2020), Lectures 9 and 10 *Breadth-First Search* and *Depth-First Search*. Sources and links appear in **Section 15**.
>
> **Existing connections:** `notebooks/Knowledge-Graphs-01-What-Is-a-Graph.md`, `curriculum/knowledge-graphs-mathematics.md` (KG-M01–M04), `feature-engineering/chapters/13-relational-feature-engineering-pk-fk-context-and-fan-out.md`, `feature-engineering/retention/linear-algebra-for-feature-engineering.md`, and `feature-engineering/retention/CS229_LEARNING_INSTRUCTIONS.md`.
>
> **Safety:** this document uses invented letters and publicly known mathematical models, not internal SAP datasets, unreleased architecture, or company-proprietary materials.

## 1. What exactly will we learn? (intuition before notation)

At the previous step a graph was a picture of entities and arrows. Now we ask:

1. **How connected** is each entity? (degree / neighborhoods)
2. **Can we get** from A to D while respecting arrow direction? (reachability)
3. **How many steps** would it take? (path length and shortest paths)
4. **How can a computer store** those arrows? (sets, adjacency lists, adjacency matrices, sparse representation)
5. **How can it search** without following the same cycle forever? (BFS, DFS, visited sets)
6. **How expensive** is the computation? (complexity, skew, multi-hop explosion)
7. **How does this become ML input** rather than a trained model? (structural features and neighbor aggregations)

The story remains: **model relationships faithfully → measure them → query/traverse them → construct features → only later learn with graph AI.**

We use one synthetic five-node directed graph throughout, and then a **new graph** for the learner's independent transfer exercise.

## 2. Preflight mathematical language (do not presume prior exposure)

**Our graph is a finite, simple, directed graph:**

\[
G=(V,E)
\]

- \(G\): the whole graph.
- \(V\): set of unique vertices/nodes; read “vee.”
- \(E\): set of directed edges; read “ee.”
- \((u,v)\): an **ordered pair**, directed from \(u\) to \(v\). Its order matters.
- \(\{u,v\}\): a **two-element set** for an undirected edge, where \(\{u,v\}=\{v,u\}\).
- \(u\in V\): “node u belongs to vertex set V.”
- \(|V|\): the number of nodes, i.e., *cardinality*; do **not** confuse these bars with a vector norm.
- \(\sum\): add the specified terms; not a probability and not a matrix itself.
- **Simple graph** here: no multiple identical directed edges and no self-loop. We will later allow relation types, time, parallel edges, weights and hyperedges when a problem needs them.

For this lesson:

\[
V=\{A,B,C,D,E\}, \qquad |V|=5
\]

\[
E=\{(A,B),(A,C),(B,E),(B,D),(C,D),(E,D),(D,B)\},\quad |E|=7.
\]

~~~mermaid
flowchart LR
  A((A)) --> B((B))
  A --> C((C))
  B --> E((E))
  B --> D((D))
  C --> D
  E --> D
  D --> B
~~~

**Why two arrows between B and D are legal:** \((B,D)\) and \((D,B)\) are *different ordered pairs*, not duplicate edges. Their directions can represent different business predicates or even the same predicate in two directions; this bare mathematical model does not assign business meanings.

## 3. Node degree, neighbor sets, and the first proof

**Intuition:** A degree answers “how many arrows touch this node?” For *directed* graphs, the answer must distinguish outgoing and incoming arrows.

Let:

- \(N^{+}(v)=\{u\in V:(v,u)\in E\}\) — **outgoing neighbors** of node \(v\).
- \(N^{-}(v)=\{u\in V:(u,v)\in E\}\) — **incoming neighbors**.
- \(d_{\mathrm{out}}(v)=|N^{+}(v)|\) — out-degree in a simple graph.
- \(d_{\mathrm{in}}(v)=|N^{-}(v)|\) — in-degree in a simple graph.

**Caveat:** In multigraphs, degree counts *incident edges*, which can exceed the number of unique neighboring vertices. The neighbor-set definition above relies on our simple-graph assumption.

In this graph:

| Vertex | Incoming edges | Outgoing edges | In-degree | Out-degree |
|---|---|---|---:|---:|
| A | none | A→B, A→C | 0 | 2 |
| B | A→B, D→B | B→E, B→D | 2 | 2 |
| C | A→C | C→D | 1 | 1 |
| D | B→D, C→D, E→D | D→B | 3 | 1 |
| E | B→E | E→D | 1 | 1 |
| Total | 7 counted | 7 counted | **7** | **7** |

The **directed degree-counting identity** is

\[
\boxed{\sum_{v\in V} d_{\mathrm{out}}(v)=
       \sum_{v\in V} d_{\mathrm{in}}(v)=|E|}.
\]

**Pronounce every symbol:** “Summing the outgoing degree \(d_{\mathrm{out}}\) of each vertex \(v\) in the vertex set \(V\) equals summing the incoming degree of each vertex, and both equal the number of edges in \(E\).”

**Derivation (a counting proof):**
1. A directed edge \((u,v)\) contributes exactly **one** to the out-degree of \(u\).
2. No other node claims this edge as an outgoing edge.
3. Summing all outgoing degrees therefore counts **each edge exactly once** → \(|E|\).
4. The same edge contributes exactly **one** to the in-degree of \(v\).
5. Summing all in-degrees counts each edge once again → \(|E|\).
6. Thus both sums are equal. No calculus or probabilistic assumptions are involved.

**Manual numerical check:**

\[
\sum_v d_{\mathrm{out}}(v)=2+2+1+1+1=7
\]

\[
\sum_v d_{\mathrm{in}}(v)=0+2+1+3+1=7.
\]

The **mean** out-degree is \(\bar d_{\mathrm{out}}=|E|/|V|=7/5=1.4\). The mean in-degree is also \(1.4\). The mean total directed degree (in + out) is \(2.8\). These are **descriptive statistics**, not model predictions.

**Connection to MIT's undirected handshaking lemma:** If the graph were **undirected**, each edge is incident to two endpoints, so

\[
\boxed{\sum_{v\in V}\deg(v)=2|E|}.
\]

Why the factor of two? Each undirected edge is counted once at each endpoint. For simple undirected graphs, \(\deg(v)\) is simply number of neighbors. In our *directed* example, summing both in- and out-degrees similarly gives \(2|E|=14\).

**Common failure:** reporting “degree of B is 2” without specifying direction/total; confusing two distinct arrows with one bidirectional semantic relationship.

**Learner pause point 2.1:** independently explain direction and counting on a different tiny graph before continuing.

### 3.1 Additional first-principles checkpoint: sources, sinks, and matrix degree counts (2026-10-09 study session)

Before walks and reachability, consider an even smaller directed graph with
\(V=\{X,Y,Z\}\) and \(E=\{(X,Y),(X,Z),(Y,Z)\}\). There are \(3\) vertices and \(3\) directed edges.

- The outgoing degree counts arrows **starting** at a vertex: \(d_{\mathrm{out}}(X)=2\), \(d_{\mathrm{out}}(Y)=1\), \(d_{\mathrm{out}}(Z)=0\).
- The incoming degree counts arrows **ending** at a vertex: \(d_{\mathrm{in}}(X)=0\), \(d_{\mathrm{in}}(Y)=1\), \(d_{\mathrm{in}}(Z)=2\).
- A **source** has in-degree zero (here \(X\)); a **sink** has out-degree zero (here \(Z\)). This is graph structure, not proof of business authority or causality.
- Check manually: \(2+1+0=0+1+2=3=|E|\). Each ordered edge has one source and one destination.

For a gentle connection to **the later adjacency-matrix subsection**, write \(A_{ij}=1\) if the directed edge from row/source \(i\) to column/destination \(j\) exists; otherwise \(A_{ij}=0\). In node order \([X,Y,Z]\):

\[
A=\begin{bmatrix}
0&1&1\\
0&0&1\\
0&0&0
\end{bmatrix}.
\]

Row sums are \((2,1,0)\) (out-degree); column sums are \((0,1,2)\) (in-degree). Why?
\[
d_{\mathrm{out}}(i)=\sum_{j=1}^{|V|}A_{ij},
\qquad
d_{\mathrm{in}}(j)=\sum_{i=1}^{|V|}A_{ij}.
\]
Here \(i,j\) are **row and column indices**, \(A_{ij}\) is the entry at their intersection, \(\sum\) means addition, and \(|V|\) is the number of vertices. This is **not** probability or a learned weight. Because the same ones are added by rows or columns:
\[
\sum_{i=1}^{|V|}\sum_{j=1}^{|V|} A_{ij}=3=|E|.
\]
This is another derivation of the directed degree-sum identity for an unweighted simple graph.

**Transfer and sanity tests (not answered yet):** If only \(X\to Z\) is reversed, which degrees change? Why does the total edge count stay the same? Can a vertex be both source and sink? State your assumptions before answering. For a multigraph, edge-counting degree and the size of a distinct-neighbor set may differ; do not silently deduplicate repeated edges when their multiplicity matters.

**ML and enterprise connection:** Degree can be a deterministic structural feature (for example, outgoing relation counts by predicate), but it does not certify the truth of edges. A degree extracted from relationships added **after a prediction cutoff** can leak future information, just as an invalid point-in-time SQL join can. This is a model-quality and data-governance requirement, not an inference from graph shape.

**Source verification:** The direction/degree conventions and source/sink terminology are stated in MIT 6.042J Chapter 6, §6.1–6.1.1 (https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-fall-2010/resources/mit6_042jf10_chap06/); the undirected handshaking lemma appears in Chapter 5 §5.2.1 (https://ocw.mit.edu/courses/6-042j/mathematics-for-computer-science-fall-2010/resources/mit6_042jf10_chap05/). The written derivation and synthetic examples are our own.

### 3.2 Learner checkpoint and corrected chat diagram (2026-10-09)

The learner independently explained V=vertices, E=directed connections, |V|=6 and |E|=7 and correctly computed in-degree(S)=2 on the six-node transfer exercise. They reported out-degree(S)=3, correctly observing an S->Q arrow in the instructor's **inconsistent SVG** that was absent from the specified edge set. The instructor's chat SVG omitted T and substituted incorrect arrows, including S->Q and R->S, for the actual R->T and T->S. **The edge set below is authoritative; the exercise file's Mermaid diagram is consistent with it.**

\[
V=\{P,Q,R,S,T,U\},\quad E=\{(P,Q),(P,R),(Q,S),(R,T),(T,S),(S,R),(S,U)\}.
\]

~~~mermaid
flowchart LR
 P((P)) --> Q((Q))
 P --> R((R))
 Q --> S((S))
 R --> T((T))
 T --> S
 S --> R
 S --> U((U))
~~~

Incoming to S: Q->S and T->S, so in-degree(S)=2. Outgoing from S: S->R and S->U, so out-degree(S)=2. If a *new* S->Q edge is explicitly added, it becomes an eight-edge graph; the source S gains one outgoing and Q gains one incoming connection.

**Derivation to reinforce:** each edge u->v contributes one to out-degree(u) and one to in-degree(v), so sums over all vertices are equal to the number of edges. A single vertex need **not** have equal incoming and outgoing counts. For X->Y and X->Z, out-degrees (2,0,0) and in-degrees (0,1,1) each sum to 2. This is deterministic double-counting, not probability or balancing of traffic.

**Learning evidence:** graph notation/cardinality correct, genuine graph-diagram debugging observed, local S in-degree correct. Global degree-sum reasoning requested but not yet independently explained; no mastery score or READY status. Next checkpoint: learner explains the global identity in their own words and applies the S->Q hypothetical. Do not reveal full exercise reference solutions or advance to Part 2.2 yet.

### 3.3 Directed degree conceptual gate — passed (2026-10-09, second learner response)

The learner was asked two fresh conceptual transfer questions after reviewing the degree-sum theorem:

1. If a node \(X\) has out-degree 10 and in-degree 0, does that violate the theorem? **Learner's reasoning:** no; individual vertices do not have to have matching in/out degrees; the statement is at the level of all vertices in \(G\). **Correct.** The other endpoints of those ten directed edges contribute ten units to the sum of in-degrees, even though \(X\)'s in-degree is zero.
2. If \(S\to U\) is reversed to \(U\to S\), does the number of edges change, and does the theorem still hold? **Learner's reasoning:** no edge is added or removed; direction changes but does not affect the graph-wide theorem. **Correct.**

**Exact local accounting** on the authorized six-node graph:
- Before reversal: \(S\) has (in,out) \(=(2,2)\); \(U\) has \((1,0)\).
- After reversal: \(S\) has \((3,1)\); \(U\) has \((0,1)\).
- \(|E|=7\) before and after, with \(\sum_v d_{\mathrm{in}}(v)=\sum_v d_{\mathrm{out}}(v)=7\).
- Further important extension: even *adding or deleting* an edge preserves the theorem as an identity, though both degree sums and \(|E|\) then change by one.

**Assessment:** Subsection 2.1 conceptual check PASSED: learner independently explained the critical graph-level vs node-level distinction and transferred it to reversed direction; the earlier erroneous diagram was also independently caught. Do not confuse this bounded conceptual pass with completed KG-02 math/implementation or with an overall competency score. Continue with only one next foundational concept at a time, starting 2.2: directed walk versus simple path; revisit degree algebra later during adjacency matrix and feature-engineering exercises.

## 4. Walk, trail, path, cycle, reachability, distance (graph theory, not jargon)

A **walk** from \(v_0\) to \(v_k\) is a sequence \(v_0,v_1,\ldots,v_k\) where every consecutive edge \(v_{i-1}\to v_i\) exists. Vertices and edges may repeat.

A **trail** is a walk that does not repeat edges, although it may revisit nodes.

A **simple path** is a walk whose vertices are distinct. Many references, including MIT 6.042J Ch. 6, use *path* to mean this simple path; in other fields “path” is sometimes used more broadly. State the convention when reading papers.

A **directed cycle** starts and ends on the same node with no other vertex repeated. Our graph has cycles B→D→B (length 2) and B→E→D→B (length 3). A cycle is not the same as a multi-node path that happens to revisit a node.

**Edge-count path length:** \(k\) arrows traversed means length \(k\) (hops). This is distinct from the sum of edge *weights* in a weighted graph.

**Reachability:** \(t\) is reachable from \(s\) if a directed walk from \(s\) to \(t\) exists. It is directional: \(D\) is reachable from \(A\); \(A\) is not reachable from \(D\) here. We regard a node as reachable from itself via a zero-length walk.

**Unweighted shortest-path distance:**

\[
d_G(s,t)=\min\{k:\text{there exists a directed }s\to t\text{ walk of }k\text{ edges}\}.
\]

Define \(d_G(s,t)=\infty\) if no such walk exists; \(d_G(s,s)=0\). Here \(d_G(A,D)=2\) because A→B→D or A→C→D uses two edges; no direct A→D edge exists, so no 1-edge route exists.

**Definitions by hand:** explain each symbol—\(d_G(s,t)\) is a distance **in this graph**; \(s,t\) are starting and destination nodes; \(\min\) chooses the smallest valid hop count; \(\infty\) means no directed connection.

**Strongly connected:** every ordered pair of vertices is mutually reachable. **Weakly connected:** the graph becomes connected when direction is forgotten. Our graph is weakly connected but not strongly connected because no node can travel back to A.

**Caution:** “D reached” is only structural reachability. If arrows mean different business relations, a traversal across mixed predicates is not necessarily a valid business inference.

### 4.1 Current study micro-lesson — distinguish a walk, trail, and simple path

Use the original **five-node worked graph** from Section 2, reserving the separate six-node graph as independent transfer practice. Consider these **valid directed** sequences:

- \(A\to B\to E\to D\): a simple path of length 3 (all four vertices distinct); also a trail and a walk.
- \(A\to B\to D\to B\): a trail of length 3 (each of the three directed edges is distinct) but **not** a simple path (vertex \(B\) is repeated); it is still a walk.
- \(A\to B\to D\to B\to D\): a walk of length 4 but **not** a trail (edge \(B\to D\) is traversed twice), and not a simple path (repeated vertices).
- \(B\to D\to B\): a directed cycle of length 2, permitted because both \(B\to D\) and \(D\to B\) are included in the edge set.

A walk of length \(k\) is a vertex sequence \(v_0,v_1,\ldots,v_k\) satisfying \((v_{i-1},v_i)\in E\) for **every** \(i\in\{1,\dots,k\}\). Read \(\forall i\) as “for each edge position \(i\)”; the membership symbol \(\in\) means that ordered edge is present in the edge set \(E\). A sequence with \(k+1\) vertex **positions** traverses \(k\) directed edges; count positions, not unique vertices, to determine walk length.

**ML/enterprise link:** a graph walk describes a route of relationships, not by itself a justified implication such as ownership or causation. Walk multiplicity (later computed via adjacency powers) differs from the number of distinct reachable nodes and from node degree.

**Guided transfer example, not independently assessed (2026-10-09):** The learner chose to continue without submitting the earlier classification. The tutor explicitly showed that \(P\to R\to T\to S\to R\) traverses four valid directed edges and repeats vertex \(R\) but no ordered edge: it is a **walk and trail, not a simple path**. This illustration is a teaching example; learner-owned skill 2.2 remains unassessed. The repeated vertex exposes the directed cycle \(R\to T\to S\to R\). Next introduce cycle definitions and assess independently before reachability.

### 4.2 Directed cycles: formal definition, manual reasoning, and practice (2026-10-09)

**Intuition.** A directed cycle is a closed directed route that gets back to its starting vertex *without visiting any other vertex twice*. It differs from any arbitrary closed walk, which may repeat edges and internal vertices.

A **closed directed walk** \(W=(v_0,\ldots,v_k)\) satisfies:
1. Every directed edge \((v_{i-1},v_i)\in E\) for \(i=1,\ldots,k\), and
2. \(v_0=v_k\) (same starting and finishing vertex).

A **simple directed cycle** additionally requires the \(k\) vertices \(v_0,\ldots,v_{k-1}\) to be pairwise distinct. In this lesson's graph no self-loops occur, so a directed cycle has at least **two** edges; if self-loops were allowed, a cycle of length one could exist depending on convention.

**Manual example using the six-node transfer graph (authoritative E in Section 3.2):**

\[
C=(R,T,S,R)
\]

- \(v_0=R\), \(v_1=T\), \(v_2=S\), \(v_3=R\).
- \(k=3\) edges: \(R\to T\), \(T\to S\), \(S\to R\), all explicitly in \(E\).
- Start \(R\) = finish \(R\).
- \(R,T,S\) are distinct before the return, so it is a **simple directed cycle** of length three.

For contrast, \(R\to T\to S\to R\to T\to S\to R\) is a **closed directed walk of length six**, but not a simple directed cycle: internal vertices and edges are revisited. \(P\to R\to T\to S\to R\) is an **open trail containing a cycle** (it begins at \(P\), ends at \(R\)); the entire sequence is not a cycle.

**What is this math good for?** Cycles are ordinary and often valid in knowledge graphs: reciprocity and chains of business relationships need not form a hierarchy. Conversely, a dependency graph representing prerequisite ordering may be required to be acyclic (a directed acyclic graph, DAG). An actual loop in a DAG-intended workflow signals a violated design constraint. Traversal code should account for cycles via visited-state or other bounded exploration to avoid revisiting the same states indefinitely. Whether a particular business relationship is transitive or inferable cannot be concluded merely from structural cyclicity.

**Independent learner gate (answers deliberately withheld):** On the six-node graph, classify \(P\to Q\to S\to R\to T\to S\) as a directed walk, trail and/or simple path; count its length; decide whether the **entire sequence** is a directed cycle and, separately, whether it **contains** a directed cycle. Explain the distinction using vertex and edge repetitions. Do not advance to reachability until the reasoning is checked.

**Source anchors:** MIT 6.042J Chapter 6, Directed Graphs, graph-walk/path/cycle concepts: https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-fall-2010/resources/mit6_042jf10_chap06/ ; Stanford CS224W public course https://web.stanford.edu/class/cs224w/ . Explicitly label the chosen simple-path and simple-cycle conventions when interpreting other authors.

## 5. Representations: drawing, adjacency list, adjacency matrix

### 5.1 Adjacency list (for practical traversal)

Store **outgoing neighbors** for each node in a dictionary or list:

~~~python
adjacency = {
    "A": ["B", "C"],
    "B": ["E", "D"],  # this explicit order is used in the BFS/DFS traces below
    "C": ["D"],
    "D": ["B"],
    "E": ["D"],
}
~~~

It is a representation of connectivity, not a machine learning model. For an adjacency-list implementation, memory is \(O(|V|+|E|)\) if each vertex and directed edge is stored once (ignoring language/container overhead). Looking up all neighbors of a node is efficient. A list-only membership test “does u→v exist?” may require scanning that node's degree, while a hash set may reduce expected membership lookup time under conventional hash assumptions.

### 5.2 Adjacency matrix (the linear algebra bridge)

Choose and **fix** a node order:

\[
[A,B,C,D,E].
\]

The \(5\times5\) matrix \(A\) (matrix named \(A\), distinct from **node A**) stores a 1 at row \(i\), column \(j\) when node \(i\to j\) exists, else 0:

\[
A_{ij}=\begin{cases}
1 & \text{if }(v_i,v_j)\in E,\\
0 & \text{otherwise}.
\end{cases}
\]

**Symbol dictionary:**
- \(A\): the full adjacency matrix.
- \(A_{ij}\): scalar entry in **source row** \(i\), **target column** \(j\).
- \(v_i,v_j\): nodes at matrix positions \(i,j\).
- \(\in E\): “belongs to edge set E.”
- 0 and 1: absent/present *under our unweighted, single-relation projection*, **not** probabilities.
- \(\begin{cases}\cdot\end{cases}\): “use the first value when the condition is true, otherwise use the second.”

Manually fill each row by inspecting the outgoing arrows:

~~~text
             TARGET
SOURCE        A  B  C  D  E
      A       0  1  1  0  0
      B       0  0  0  1  1
      C       0  0  0  1  0
      D       0  1  0  0  0
      E       0  0  0  1  0
~~~

In LaTeX:

\[
A=\begin{bmatrix}
0&1&1&0&0\\
0&0&0&1&1\\
0&0&0&1&0\\
0&1&0&0&0\\
0&0&0&1&0
\end{bmatrix}.
\]

For a directed graph, \(A\) need not be symmetric: \(A_{BD}=1\) and \(A_{DB}=1\) happen to both be present, but \(A_{AB}=1\) whereas \(A_{BA}=0\).

**Row/column sums:**

\[
d_{\mathrm{out}}(v_i)=\sum_{j=1}^{|V|} A_{ij},
\qquad
d_{\mathrm{in}}(v_j)=\sum_{i=1}^{|V|} A_{ij}.
\]

**Read aloud:** “The out-degree for node \(v_i\) is the sum across row \(i\), over every possible destination column \(j\). The in-degree for node \(v_j\) is the sum down column \(j\), over every possible source row \(i\).”

For B, row B has \(0+0+0+1+1=2\). Column B has \(1+0+0+1+0=2\). For D, column D has \(0+1+1+0+1=3\), matching the edge-list count.

### 5.3 Why does a matrix square count two-step walks?

Matrix multiplication is defined by:

\[
(A^2)_{ij}=\sum_{k=1}^{|V|}A_{ik}A_{kj}.
\]

\(A^2=A\cdot A\), not “square each entry of \(A\).”

**Read the equation:** “The entry at source \(i\), target \(j\) in the matrix \(A\) squared equals the sum, over all possible intermediate nodes \(k\), of the indicator for \(i\to k\) multiplied by the indicator for \(k\to j\).”

**Why does it work?**
- For a chosen intermediate \(k\), the product \(A_{ik}A_{kj}\) is **one** exactly when **both** edges \(i\to k\) and \(k\to j\) exist.
- If either edge does not exist, the product is zero.
- Summing across all intermediate nodes **counts valid two-edge walks**.

**Manual example, from A to D:**

\[
\begin{aligned}
(A^2)_{A,D} &=
A_{A,A}A_{A,D}
+A_{A,B}A_{B,D}
+A_{A,C}A_{C,D}\\
&\quad + A_{A,D}A_{D,D}
+A_{A,E}A_{E,D}\\
&=(0)(0)+(1)(1)+(1)(1)+(0)(0)+(0)(1)\\
&=0+1+1+0+0=\boxed{2}.
\end{aligned}
\]

These are **A→B→D** and **A→C→D**. Both are walks of exactly two arrows, and here they are also simple paths.

**Important distinction:** \((A^k)_{ij}\) counts **k-edge walks**, not necessarily distinct simple paths and not necessarily the shortest path. In cyclic graphs, longer walks can revisit nodes. This interpretation assumes ordinary arithmetic matrix multiplication and an unweighted 0/1 adjacency; weighted adjacency powers have a different meaning.

### 5.4 Matrix reachability in finite graphs (advanced bridge, not required to master immediately)

There is a walk from \(i\) to \(j\) iff some \((A^k)_{ij}>0\). If \(i\neq j\) and a route exists, we can remove cycles to get a simple path of at most \(|V|-1\) edges.

Thus, with \(A^0=I\) (the identity matrix, encoding zero-step self-reachability), we may write:

\[
R_{ij}=\mathbf{1}\!\left[\sum_{k=0}^{|V|-1}(A^k)_{ij}>0\right].
\]

\(R_{ij}\) is 1 if reachable, else 0; \(\mathbf{1}[\cdot]\) is an indicator (returns 1 when the enclosed proposition is true); \(I\) has ones on the diagonal and zeros elsewhere. This is a mathematical characterization, **not a recommendation to compute massive matrix powers in production**.

## 6. Breadth-First Search (BFS): expanding concentric layers

**Intuition:** if searching rooms in a building, visit all rooms one doorway away before visiting those two doorways away. **Breadth first** means explore by hop count.

BFS uses a **first-in, first-out queue**, a set of **discovered/visited** nodes (mark on enqueue to avoid duplicates), a **distance** per node and optionally a **parent pointer** recording how a node was first reached.

~~~text
Start: A
Distance 0:    A
Distance 1:  B   C
Distance 2:  E   D
~~~

**Trace with outgoing neighbor order A:[B,C], B:[E,D], C:[D], D:[B], E:[D]:**

| Iteration | Dequeue | Examine outgoing neighbors | Queue after processing | New distances |
|---|---|---|---|---|
| start | – | initial discovery A | [A] | d(A)=0 |
| 1 | A | B, C | [B,C] | d(B)=d(C)=1 |
| 2 | B | E, D | [C,E,D] | d(E)=d(D)=2 |
| 3 | C | D already discovered | [E,D] | no change |
| 4 | E | D already discovered | [D] | no change |
| 5 | D | B already discovered | [] | no change |

**Dequeue order:** A, B, C, E, D. **Distances:** A:0, B:1, C:1, E:2, D:2.

**Shortest path to D:** A→B→D (or A→C→D) uses 2 arrows. Under our neighbor ordering, BFS first discovers D from B and records parent[D]=B; reversing order may choose a different valid shortest path. The shortest **distance** remains 2.

### Why does BFS find minimum-hop paths? (reasoning proof)

1. Initially A is at level/distance 0.
2. When BFS processes any node at level \(k\), newly discovered neighbors are assigned level \(k+1\).
3. FIFO queue ordering ensures level \(k\) nodes are processed before new level \(k+1\) nodes.
4. If a node could be reached via fewer than \(k+1\) edges, a predecessor on that shorter path would have been explored at an earlier level and discovered it already.
5. Hence first discovery provides the minimum number of edges from the start for **unweighted** traversal.

The proof assumes consistent direction, a FIFO queue and correct discovery marking. BFS does **not** minimize summed monetary amounts, transaction probabilities, travel durations or edge weights; weighted shortest path is another topic (e.g., Dijkstra under its own conditions).

**BFS graph-distance recurrence during discovery:**

\[
\text{dist}[u]=\text{dist}[v]+1
\]

when \(u\) is an **undiscovered neighbor** of dequeued vertex \(v\). This is an update rule, not an arbitrary equation: following exactly one more arrow adds one hop. Distances are in unit **edges**. Do not update already discovered nodes under ordinary unweighted BFS.

## 7. Depth-First Search (DFS): follow one branch deep before backing up

**Intuition:** walk along one available corridor as far as possible, backtrack when you cannot proceed, and then try the next branch.

DFS can use recursion or an **explicit LIFO stack**. It still needs a discovered/visited set: our cycles B→D→B and B→E→D→B would otherwise repeat forever.

Choose neighbor order: A:[B,C], B:[E,D], C:[D], E:[D], D:[B].

**Recursive discovery order:** A → B → E → D → C.

The tree path to D in this DFS is **A→B→E→D**, which uses **3 edges**. But we already proved a 2-edge path A→B→D exists.

This is our counterexample: **DFS can find reachability, but does not promise a minimum-hop path.** Its order is sensitive to neighbor ordering, and iterative DFS may yield a different visitation sequence depending on push order.

**Applications:** DFS is useful for cycle detection, reachability, connected-component traversal, topological ordering on DAGs (with additional logic), and understanding graph structure. Both BFS and DFS can cover a graph from a chosen start only as far as **reachable outgoing nodes**; to process disconnected parts, run additional searches.

## 8. Algorithmic complexity — why representation matters

Let \(n=|V|\) be the number of nodes, \(m=|E|\) be the number of directed edges. \(O(\cdot)\) expresses how an upper bound grows with input size under stated model assumptions, **not seconds or bytes exactly**.

| Operation | Adjacency list | Dense adjacency matrix |
|---|---|---|
| Store graph | \(O(n+m)\) storage | \(O(n^2)\) storage |
| Check one directed pair's adjacency | \(O(d_{\rm out}(u))\) if list scan; expected \(O(1)\) with hash sets | \(O(1)\) index access |
| Enumerate outgoing neighbors of one node | \(O(d_{\rm out}(u))\) | \(O(n)\) scan of the entire row |
| BFS/DFS from a start | \(O(n+m)\) upper bound, assuming visited bookkeeping and adjacency lists | \(O(n^2)\) conventional implementation |

**Derive BFS/DFS \(O(n+m)\):**
1. Each reached node is marked/discovered at most once → at most \(n\) queue/stack insertions.
2. Each reached node's outgoing adjacency list is scanned once → at most \(m\) edge examinations across the whole search.
3. Bookkeeping and initialization are \(O(n)\); work is bounded by a constant times \(n+m\).
4. Hence \(O(n+m)\). For a search restricted to one reachable component, the tighter bound uses the reached nodes and scanned outgoing edges.

**Why can it still be costly?**
- High-degree hubs make a local neighborhood enormous.
- Expanding many multi-hop **paths** can grow far faster than maintaining a single visited set. BFS of *unique nodes* and enumerating *all distinct walks* are different workload types.
- Large node dictionaries, network partitions, cache misses, authorization checks, real-time updates and temporal filtering affect constant factors and practical latency.
- A dense \(n\times n\) matrix has \(n^2\) cells, regardless of how few edges exist. A graph with one million nodes would require \(10^{12}\) cells; even one byte/cell would be about 1 TB (decimal units). This is only an illustrative lower-level dense storage estimate, not a measurement of a real graph system.
- Sparse compressed row (CSR)/column (CSC) representations enable efficient numerical operations while keeping storage proportional to nonzeros plus offsets, with format and dtype dependent overhead.

## 9. Bridges to Linear Algebra, probability, ML and Feature Engineering

### 9.1 The graph becomes an ML feature *without* training a GNN

Degree, inbound edges, number of reachable merchants, shortest distance and neighbor attributes can be engineered as numerical features for a conventional predictive model.

Given the ordered node feature vector \(x=[10,20,30,40,50]^\top\), matching \([A,B,C,D,E]\), consider

\[
y=Ax,\qquad y_i=\sum_j A_{ij}x_j.
\]

**Read aloud:** “For each source node \(i\), multiply its adjacency indicators to destination nodes \(j\) by the feature values \(x_j\), then add them.”

**Manual step (node A):**

\[
y_A=0(10)+1(20)+1(30)+0(40)+0(50)=50.
\]

The graph operation **sums the feature values of outgoing neighbors**. Likewise:

\[
Ax=[50,90,40,20,40]^\top.
\]

A simple **mean of outgoing neighbor features** uses \(\bar x_i=y_i/d_{\rm out}(v_i)\) when \(d_{\rm out}(v_i)>0\). For A, \(50/2=25\). When out-degree = 0, mean is undefined until a deliberate empty-neighborhood policy is chosen.

**IMPORTANT orientation:** Under our **row = source, column = target** convention, \(Ax\) collects information from *outgoing destination neighbors*. A GNN receiving information along incoming arrows would use a transpose or a different adjacency convention. Never copy a message-passing formula without checking this orientation and graph direction.

### 9.2 Probability bridge, for later (not a learned model yet)

For an unweighted graph with nonzero outgoing degrees, define

\[
P_{ij}=\frac{A_{ij}}{d_{\rm out}(v_i)}.
\]

\(P_{ij}\) then represents the probability of choosing outgoing neighbor \(j\) **under the explicit model assumption** of uniformly selecting one outgoing edge from node \(i\). For node A, two outgoing edges imply \(P_{A,B}=P_{A,C}=1/2\). This is a **constructed random-walk transition model**, not a probability automatically discovered by the graph or a learned fraud probability. Nodes with no outgoing edges require a specified “dangling node” policy. Random walks, PageRank, and embedding learning are taught later.

### 9.3 SQL and relational feature engineering

A relation path \(A\to B\to D\) resembles a PK/FK join. In our example there are **two** 2-hop walks from A to D, via B and via C.

For a query aggregating relationship paths:
- **COUNT(all paths)** could count **2**.
- **COUNT(DISTINCT terminal entity)** is **1** (only D).
- The correct answer is determined by your *prediction grain* and *business meaning*, not by whether you used SQL or a graph database.

This is exactly the aggregation/fan-out issue from `feature-engineering/chapters/13-relational-feature-engineering-pk-fk-context-and-fan-out.md`. Even a graph database does not automatically solve identity resolution, changing edges, late-arriving records or feature leakage. For a fraud/anomaly model, a graph feature must be constructed using the **relationships that were actually available at the prediction timestamp**, not the finished data warehouse as it looks today.

### 9.4 A glance at GNN mathematics—do not skip ahead

Later a GNN will aggregate representations of neighboring nodes, apply learned matrices, and optimize a loss with gradients. Today \(Ax\) only performs a **fixed, nonlearned, linear transformation**. Understanding its indexing, dimensions and missing neighbors is the prerequisite for computing an actual GNN layer and its gradient later.

## 10. Practical implementation — learner-written lab

No graph database or third-party graph ML library is required yet. Use a plain dictionary and then NumPy if desired. First compute results by hand, then write your implementation.

~~~python
# Save as your local scratch or continue in an exercise notebook.
# You write the BFS/DFS/degree/matrix implementations yourself;
# the teacher reviews before showing solutions.

vertices = ["A", "B", "C", "D", "E"]
adj = {
    "A": ["B", "C"],
    "B": ["E", "D"],
    "C": ["D"],
    "D": ["B"],
    "E": ["D"],
}

def out_degrees(vertices, adj):
    # TODO: create one degree for each vertex.
    raise NotImplementedError

def in_degrees(vertices, adj):
    # TODO: count incoming edges (do not silently ignore unknown targets).
    raise NotImplementedError

def adjacency_matrix(vertices, adj):
    # TODO: row=source, column=destination.
    # Verify size len(vertices) x len(vertices).
    raise NotImplementedError

def bfs_distances(adj, start):
    # TODO: FIFO queue, discover-on-enqueue, parent/distance dictionary.
    raise NotImplementedError

def dfs_discovery_order(adj, start):
    # TODO: recursive or explicit stack with a visited set.
    # State which neighbor order governs traversal.
    raise NotImplementedError
~~~

After implementing:
1. Verify sum(in-degree) = sum(out-degree) = number of edges.
2. Calculate \(A^2\) with NumPy **after** manually deriving one entry.
3. Confirm BFS shortest hop count and compare with a deliberately depth-biased DFS traversal.
4. Add a disconnected node and a cycle; explain whether the traversal terminates and why.
5. Explain the difference between **finding reachable nodes once** and **counting all paths**.

The separate transfer assignment is at `exercises/knowledge-graphs-02-math-and-traversal.md`. **Answers are not committed until after the learner attempts it.**

## 11. Limitations and failure-mode checks

- **Orientation:** transposing adjacency changes what “neighbor information” you aggregate.
- **Semantics:** a path over typed predicates is not necessarily an authorized or logically valid business assertion.
- **Duplicate identity:** count distinct nodes only when identity policy justifies it; one record ≠ one unique real-world person by default.
- **Multi-edge / typed edges:** a single binary adjacency can erase predicate types and edge multiplicity. Consider per-relation adjacency or edge lists in later KG modeling.
- **Cycle handling:** walks can repeat nodes indefinitely, but visited-based reachability traversal terminates on a finite graph.
- **Weights:** BFS minimizes hop counts, not arbitrary weighted travel time or financial cost.
- **Temporal/data leakage:** graph edges change; training must not see relationships known only after the prediction cutoff.
- **BFS/DFS order:** traversal visitation order and chosen BFS shortest-path *parent* can vary with neighbor ordering, even when reachability/distance answers are invariant.
- **Performance:** \(O(n+m)\) assumes materialized adjacency and local operations, not a distributed graph service's network, permission, storage and caching costs.
- **Mathematical interpretation:** \(A^k\) counts walks under standard arithmetic, not node identities, path probabilities, simple paths, or inherently meaningful relations.

## 12. Suggested spaced-review and anti-recognition transfer checks

- Describe *out-degree* without using the word “outgoing” and calculate it on a reversed-arrow graph.
- Find a graph where a node's in-degree and out-degree differ sharply.
- Prove the directed degree-sum identity for an arbitrary finite edge set, not just our diagram.
- Identify a cycle and explain why it creates infinitely many possible walks as k grows, even though there are finitely many vertices.
- For an unseen 4-node graph, compute one cell of \(A^2\) by hand and explain exactly what was counted.
- Diagnose a broken BFS that marks nodes only **after dequeuing** and consequently enqueues the same node repeatedly; discuss whether wrong results or avoidable work follows.
- Give a counterexample to “DFS always finds shortest paths.”
- Compare graph-derived transaction features against equivalent SQL features, defending grain, identity and point-in-time conditions.

## 13. A few distinctions to preserve for later graduate-level papers

- **Simple graph versus typed multi-relational KG:** one adjacency matrix is generally insufficient to preserve semantic relation identity.
- **Observed links versus learned links:** adjacency records an observed/constructed relation; link prediction trains a model to score candidate relations.
- **Matrix operation versus model:** \(Ax\) is aggregation; \(W Ax\) with trainable \(W\) plus nonlinearity/loss becomes part of a learnable architecture (depending on orientation, normalization and actual model).
- **Paths versus walks:** adjacency powers have an exact walk-count interpretation, not an all-simple-path enumeration shortcut.
- **Traversal versus inference:** discovering a path does not prove a logical entailment about business entities.
- **Probability on a graph versus a knowledge graph's uncertain facts:** a random walk transition model is not the confidence score on a KG fact.

## 14. Evidence/milestone status

- [x] University sources identified; material synthesized.
- [x] Lesson notebook drafted with formula dictionary, counting proofs, adjacency example, manual matrix square, BFS/DFS traces, ML/FE and production connections.
- [ ] Learner independently distinguishes directed degree, walk/path/cycle and reachability.
- [ ] Learner correctly calculates directed degree sums and a new adjacency matrix.
- [ ] Learner computes \(A^2\) for an unseen tiny graph with intermediate steps.
- [ ] Learner traces BFS/DFS and explains the shortest-path guarantee and complexity.
- [ ] Learner writes/reviews/test code on a new graph and diagnoses an adversarial case.
- [ ] KG-01 prerequisite assessment independently reconciled.
- [ ] KG-M01–M04 prerequisite readiness earned from actual learner evidence.

**Current next action:** Pause after `3` and elicit learner responses; continue `4–`5` one small concept at a time. Do not mark KG-02 complete merely because this note is comprehensive.

## 15. Credible academic anchors and provenance

1. **Stanford CS224W, Fall 2026, Lecture 01: Introduction to Machine Learning for Graphs**, Jure Leskovec, released slide deck: https://web.stanford.edu/class/cs224w/slides/01-intro.pdf. Relevant: graph data models; directed and heterogeneous graphs; node degree and shortest-path structural features. Stanford's course page: https://web.stanford.edu/class/cs224w/
2. **MIT 6.042J Mathematics for Computer Science, Fall 2010, Ch. 5: Graph Theory**, Prof. Tom Leighton / Dr. Marten van Dijk, ``5.1–5.2; degree, adjacency matrix and handshaking lemma: https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-fall-2010/resources/mit6_042jf10_chap05/
3. **MIT 6.042J Ch. 6: Directed Graphs**, ``6.1.1–6.1.4; directed degree, ordered edge, walk/path/cycle, strong/weak connectivity: https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-fall-2010/resources/mit6_042jf10_chap06/
4. **MIT 6.006 Introduction to Algorithms, Spring 2020, Lecture 9: BFS**, Justin Solomon. Graph representation, neighbors, shortest-path trees: https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/resources/lecture-9-breadth-first-search/
5. **MIT 6.006, Lecture 10: DFS**, Justin Solomon. DFS, traversal, topological sorting and cycles: https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/resources/lecture-10-depth-first-search/
6. **William L. Hamilton, Graph Representation Learning**, optional graph-ML continuity reading (not a substitute for above graph-theory definitions): https://www.cs.mcgill.ca/~wlh/grl_book/

All derivations and synthetic numbers in this notebook are independently authored educational examples, **not copied lecture passages**. Stanford slide text was readable from its public PDF, MIT Ch. 5–6 from the original OCW PDFs and MIT 6.006 from original lecture materials. The existing ChatGPT Library CS224W file is a **link index**, not a local PDF archive. Existing uploaded CS229 PDFs can supply prerequisite reinforcement on request; we do not claim to have read their specific pages in this KG-02 lesson.

---
