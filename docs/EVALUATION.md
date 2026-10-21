# Evaluation checklist
- Use a frozen held-out test set.
- Report per-class precision, recall, mAP@0.5 and mAP@0.5:0.95.
- Measure CPU latency with warm-up and batch size 1; document hardware.
- Review false negatives, false positives, lighting variation and board generalization.
- Compare against manual inspection on the same sampled units.
- Record dataset/model versions and confidence thresholds.
Do not present demo heuristic output as validated production performance.
