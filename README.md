## Cardinality from Online Sources: CardiO

Count questions are an important type of information need, and in
principle available on the Web, though often in noisy, contradictory,
or semantically not fully aligned form. In this work, we propose
CardiO, a lightweight framework for searching entity counts on the
Web. CardiO extracts all counts from a set of relevant Web snippets,
and infers the most central count based on semantic and numeric
distances from other candidates. In the absence of supporting
evidence, the system relies on closely-related sets of similar size,
to provide an estimate.  Experiments show that CardiO can produce
accurate counts better than small models based purely on LLM with
better traceable answers. Although larger models have higher
precision, when used to enhance CardiO components, they do not
contribute to the final precision or recall.

This repository contains the benchmarks and the code used in the submission to the WebConf short papers track 2024.

### Benchmarks  
1. <strong>Cardinality Questions (CQ) benchmark</strong>
1. <strong>Natural Questions (NQ) benchmark</strong>
1. <strong> Count Question Answering Dataset (CoQuAD)</strong>

### Citation

```
@inproceedings{ghosh2024cardio,
  title={CardiO: Predicting Cardinality from Online Sources},
  author={Ghosh, Shrestha and Razniewski, Simon and Graux, Damien and Weikum, Gerhard},
  booktitle={Companion Proceedings of the ACM Web Conference 2024},
  year={2024}
}
```
