# KG-01 — What Is a Graph? (First Principles)

> Study state: **introduced, not yet assessed**, 2026-10-08. Companion: `curriculum/syllabus/module-knowledge-graphs-graph-ai.md`.
>
> Rule: intuition → diagram → manual example → mathematics → relational/ML connection → code → pitfalls → transfer. Stop after Exercise 1 for learner response.

## 1. Why a graph exists: relationship is first-class

Imagine four independent tables containing customer, bank account, transaction and merchant rows. A tabular view emphasizes each object's attributes. A graph picture emphasizes **how objects relate** and lets us describe paths through the system.

A graph is not automatically smarter than a relational database; tables with primary and foreign keys can represent and query these relationships too. Graph modeling earns its value when variable-length connections, diverse relation types, contextual relationships or navigation are central to the task.

A **graph** has a set of nodes (vertices), a set of edges connecting them, and optionally labels, types, directions and properties. A **knowledge graph** usually adds semantically meaningful identities and predicates, with governed modeling rules, sometimes logical inference. Not every property graph is a knowledge graph, and there is no single universal KG definition.

### Illustration

~~~mermaid
flowchart LR
  C["Customer: Ada"] -->|OWNS| A["Account: A1"]
  A -->|RECORDED| T["Transaction: T42"]
  T -->|PAID_TO| M["Merchant: Cafe"]
  M -->|CATEGORY| K["Merchant category: Coffee shops"]
~~~

**Questions to ask before adding an edge:** Does OWNS mean the customer is legal owner, primary user, or account signatory? Is the edge current or historic? When was it known? Where did this claim come from? These are semantic, temporal, and provenance contracts, not decorations.

## 2. Vocabulary one step at a time

- **Node (vertex):** a specific entity or conceptual object, e.g., Account A1. Node *identity* differs from display name, class, or storage-row ID.
- **Edge (relationship):** a connection, e.g., Transaction T42 PAID_TO Merchant M1. Its direction and type can matter.
- **Label/type:** a category, such as `Account` or `PAID_TO`; the same spelling does not imply the same semantics everywhere.
- **Property (attribute):** e.g., transaction amount and currency. A number is often a property rather than a separate entity; choose based on relationships/query needs.
- **Path:** sequence of adjacent nodes joined by edges (subject to orientation for directed traversals).
- **Neighbor:** a directly connected node; distinguish incoming and outgoing neighbors in directed graphs.
- **Triple:** statement `(subject, predicate, object)`, e.g. `(Ada, OWNS, A1)`. In RDF, subject/predicate/object have precise term constraints; literal values can occur as objects. The diagram is a conceptual example, not a complete RDF serialization.
- **Ontology/schema:** classes, property meanings, constraints, rules; enforcement and inference differ among RDF/OWL/property graph systems.
- **Knowledge provenance:** which source supports a specific claim, with timestamps/versions when needed.

## 3. Small example — precisely four nodes

Let V = {C, A, T, M} mean customer, account, transaction, merchant.
Let E = {(C,A), (A,T), (T,M)} be directed edges; each has a distinct type.

~~~mermaid
flowchart LR
 C["C · Customer"] -->|OWNS| A["A · Account"]
 A -->|HAS_TRANSACTION| T["T · Transaction"]
 T -->|PAID_TO| M["M · Merchant"]
~~~

We can reach M from C in three directed edges. This is **not** the same as C directly paying M. A three-hop graph path is a structural fact; its *business meaning* requires explicit semantics, possible restrictions, and trustworthy edge evidence.

### Your first manual task
1. Name the **four nodes**. Are all four objects or facts?
2. Count **edges** and state each **edge direction**.
3. Is there a **direct edge** from Customer to Merchant? Can we still find a path?
4. Why might `(Transaction, PAID_TO, Merchant)` be better modeled as an edge instead of a text field on a transaction?
5. What information would you need before asserting that customer C *currently owns* account A?

Write your own responses before checking against the instructor. No learner assessment has yet occurred.

## 4. Mathematics — only after intuition

Graph notation: \(G = (V,E)\).

Read aloud: “Graph G consists of a vertex set V and an edge set E.”

For this example, |V|=4 and |E|=3. “Vertical bars” here mean **size of a set**, not a vector norm. Contrast that with \(\lVert x\rVert_2\) from Linear Algebra, where similar-looking bars measure vector length.

Choose order [C,A,T,M] and define an adjacency matrix A:
\(A_{ij} = 1\) when a directed edge exists from node i to node j, and 0 otherwise.

~~~text
          C  A  T  M   (destination columns)
C         0  1  0  0
A         0  0  1  0
T         0  0  0  1
M         0  0  0  0
(source rows)
~~~

The matrix stores **reachability in one step**, not the labels OWNS or PAID_TO. Typed multigraphs may require a matrix per relation type or a different representation. In the matrix \(A^2\), element (C,T)=1 because there is one two-edge walk C→A→T; the (C,M) entry of \(A^3\)=1, because there is one three-edge walk C→A→T→M.

Do not interpret a nonzero adjacency value as a probability. We will study random walks and probabilistic prediction separately.

## 5. Link to existing ML and Feature Engineering

- **Relational join path:** Customer → Account → Transaction resembles SQL PK/FK joins and the fan-out problems in `feature-engineering/chapters/13-relational-feature-engineering-pk-fk-context-and-fan-out.md`. Modeling as a graph does **not** fix duplicate identity or wrong cardinality.
- **Feature engineering:** “number of merchants paid by this customer over the past 30 days” is a graph-derived feature, provided the joins/edges, prediction cutoff, historical availability and entity deduplication are valid. It is not yet a GNN feature.
- **Linear Algebra:** adjacency matrix = a representation of connectivity; matrix powers enumerate walks (for simple 0/1 adjacency with arithmetic multiplication).
- **Probability:** “How likely is this unseen edge?” is a statistical modeling question (link prediction), not an automatic property of having a graph.
- **Embeddings:** future modules replace some discrete graph information with learned coordinates; they compress evidence and can discard detail.
- **Neural network:** future GNN modules use adjacency/neighbor information to perform learned aggregations. No neural network is involved in the current example.

## 6. Engineering bridge: small, inspectable Python (read-only until learner attempts)

~~~python
nodes = ["C", "A", "T", "M"]
typed_edges = [
    ("C", "OWNS", "A"),
    ("A", "HAS_TRANSACTION", "T"),
    ("T", "PAID_TO", "M"),
]
# Learning exercise: construct an integer adjacency matrix yourself,
# verify 4 nodes / 3 edges, and trace C -> M without a graph library.
~~~

Later compare:
- SQL joins / recursive CTEs;
- an adjacency-list dictionary in Python;
- Cypher with a graph DB;
- SPARQL in an RDF store;
- matrices/tensors for graph machine learning.

Do not install a database just to learn what a graph is.

## 7. Limitations and failure examples

1. A graph can encode a **false statement**. Graph shape alone does not validate truth.
2. A reversed edge can change query semantics (A OWNS C is not C OWNS A).
3. A relationship that once existed may no longer hold: temporal validity and availability matter.
4. Two identifiers may refer to one entity; or one identifier may be recycled. Both cases break graph reasoning.
5. Dense/high-degree nodes and repeated multi-hop paths can cause expansion and cost explosions.
6. A graph visualization is only a presentation; production may use relational tables, triple stores, columnar datasets or native graph engines.
7. A KG, an embedding model, and a GNN are different artifacts and are evaluated differently.

## 8. What comes next after the exercise

KG-02: adjacency representations, degree, BFS vs DFS, reachability, paths/walks, sparse storage and runtime complexity. Only introduce these after learner can explain nodes, edges and typed direction from this exercise.

## Academic anchors

- Stanford CS520: https://web.stanford.edu/class/cs520/
- MIT 6.042J, Graph Theory: https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-fall-2010/resources/mit6_042jf10_chap05/
- Hogan et al. `Knowledge Graphs`: https://kgbook.org/
- W3C RDF 1.1 Concepts: https://www.w3.org/TR/rdf11-concepts/
