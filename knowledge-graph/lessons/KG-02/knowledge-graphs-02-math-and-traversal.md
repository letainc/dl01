# KG-02 — Directed Graph Mathematics & Search (Learner-owned transfer assignment)

> **State:** prepared 2026-10-09, NOT ATTEMPTED, not graded. The instructor should reveal each stage gradually and review learner work before giving reference code or full answer. This is a **different** graph from the worked example in `notebooks/Knowledge-Graphs-02-Graph-Math-and-Traversal.md`.
>
> **Source anchor:** Stanford CS224W Lecture 01 and MIT 6.042J Chapters 5/6 + MIT 6.006 Lectures 9/10.

## The new six-node graph

\[
V=\{P,Q,R,S,T,U\}
\]

\[
E=\{(P,Q),(P,R),(Q,S),(R,T),(T,S),(S,R),(S,U)\}.
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

All edges are directed. Node labels are anonymous to avoid assuming a semantic relation type. Use the ordering **[P,Q,R,S,T,U]** for all matrices; rows mean **source** and columns mean **target**.

## Phase A — Graph notation, direction and counting (stop here first)

1. Define the graph \(G=(V,E)\) in spoken language, explaining each set, parentheses and cardinality operator.
2. Count \(|V|\) and \(|E|\); explicitly state what an ordered edge pair means.
3. Find \(N^+(S)\), \(N^-(S)\), \(d_{\mathrm{out}}(S)\) and \(d_{\mathrm{in}}(S)\). Read each symbol aloud.
4. Make a two-column table of all six in-degrees and out-degrees and check the directed degree-sum identity.
5. Explain why **S→R** does not mean **R→S** unless the reverse edge is also explicitly stored.
6. Briefly contrast directed degree counts with the handshaking lemma for simple undirected graphs. **Why is its factor 2 present?**

**Assessment before advancing:** explain rather than memorize; instructor should ask for one independently calculated example with reversed direction.

## Phase B — Paths, cycles and adjacency matrices

7. Trace one directed walk that visits an already-visited vertex; identify the repeated node and a cycle, and explain why a walk is not necessarily a simple path.
8. Can S reach P by following arrow directions? Can P reach U? Explain both without guessing from drawing layout.
9. Write the \(6\times6\) adjacency matrix in the specified node order.
10. Show all intermediate terms when you compute \((A^2)_{P,S}\) **manually**. Explain what it counts and whether you are counting *walks*, *unique destinations*, or *shortest routes*.
11. Compute \((A^2)_{R,U}\); interpret why the entry is zero even though R can reach U along a longer directed walk.

**Assessment:** instructor checks shape, direction and arithmetic before sharing Python results.

## Phase C — BFS/DFS and complexity

12. Run BFS from P, using outgoing-neighbor order: P:[Q,R], Q:[S], R:[T], T:[S], S:[R,U], U:[]. Record each FIFO queue state, discovered set, distances and parent pointers.
13. Recover a minimum-hop P→U path from BFS parents. Why is it minimum-hop? What does “unweighted” mean?
14. Trace recursive DFS with P's neighbors considered in **reverse order [R,Q]**, and other nodes' outgoing neighbor order as listed above. Record discovery sequence and the first DFS tree path to U; compare with the shortest-hop answer.
15. What fails if you omit the visited set on S→R→T→S?
16. Explain a \(\mathcal O(|V|+|E|)\) BFS bound with adjacency lists and why a dense adjacency matrix can incur \(\mathcal O(|V|^2)\) traversal work.

## Phase D — Engineering transfer and deliberate failure

17. Implement `out_degrees`, `in_degrees`, `adjacency_matrix`, `bfs_distances` and `dfs_discovery_order` from a small adjacency dictionary **without graph libraries**; only after manual work should NumPy be used to check matrix products. Keep functions pure and testable.
18. Add a new disconnected vertex to test reachability and correct handling of infinity/unreachable; add a self-loop only after stating how it changes the simple-graph assumption.
19. Write a broken solution that constructs a graph from a many-to-many join without deduplication; reason whether two-hop path count can be inflated while distinct terminal-entity count stays the same.
20. If the graph represents events/transactions, describe two ways future edges or a corrected historical identity map could leak information into a prediction made yesterday.
21. An architect says: “We have a graph, so the model already knows which business entities are related.” Explain which parts are deterministic structure, semantically justified facts and learned predictions; identify how you would test their accuracy separately.

## Instructor agreement

- Reveal only the current phase; don't drown the learner in all twenty-one tasks in one chat.
- Tutor uses hints, not full solved code, until student makes a meaningful attempt.
- Every equation must receive: exact question, symbol glossary, motivation/derivation, manual steps, dimensions/units, ML/FE connection, implementation check, failure case.
- Preserve **unassessed** status until learner attempts and transfers a result.
- Review understanding before progressing to KG-03.
