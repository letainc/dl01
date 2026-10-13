# KG-01 — First Graph Reasoning Exercise

**Status:** assigned, unattempted; do not grade yet.

Graph:

~~~mermaid
flowchart LR
 C["Customer C"] -->|OWNS| A["Account A"]
 A -->|HAS_TRANSACTION| T["Transaction T"]
 T -->|PAID_TO| M["Merchant M"]
~~~

Without code: (a) list node identities/types; (b) list all directed typed edges; (c) count nodes, edges, and C-to-M path length; (d) distinguish C→M direct edge from C→A→T→M path; (e) describe what would make these edges trustworthy and point-in-time correct; (f) explain graph vs relational table without claiming SQL cannot do multi-hop queries.

**Stretch, optional:** construct 4×4 adjacency matrix ordered C,A,T,M. What is row C? What does element (C,T) in A² signify?

**Tutor gate:** wait for original learner answers, diagnose misconceptions, then use a new 5-node case to test transfer. No solutions copied here until learner has attempted.
