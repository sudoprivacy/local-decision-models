# Local decision models

Read README.md and docs/acceptance.md before changing scope. This repository
implements the local training/execution layer described by the three sources
in docs/sources.json. The physics and private-home adapters are the first pilots.

- Keep real records, row-level labels, identity mappings, raw errors, model
  weights and checkpoints outside Git and outside research API responses.
  Private repository visibility is not permission to commit private datasets.
- Synthetic CPU smoke, real-data readiness, target-model training, domain quality
  and device acceptance are separate evidence levels. Never promote one to another.
- Physics predicts a cohort's mean score fraction, not solver confidence or a
  student-score histogram. Missing labels stay missing. Split whole years/papers
  and keep related subquestions together. The reviewed 2026 paper is retrospective.
- Model probabilities never grant home-control permission. Authenticate the
  actor, apply server-owned action/space rules, require approval, and retain the
  independent manual path. The shipped gateway is a simulator.
- Reuse PyTorch, Transformers and PEFT. Preserve existing SudoCode/Nexus agent,
  session and A2A authorities; local experiment jobs are a domain execution
  ledger, not a replacement agent platform. Product UI integration follows
  Mosswork direction after accepted contracts.
- Run owned training with one CPU thread, bounded steps/time/memory and reduced
  priority. Keep actual test processes Normal. Never weaken gates to get a pass.
- Do not change the upstream physics repository, ShareOne pages or other owners'
  data/settings as a side effect of work here.
