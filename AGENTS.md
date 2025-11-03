# Eve Agent Playbook

## Mission
- Build and maintain Eve, the transformer forward-dynamics analogue to Adam, and its orthogonal successor Miriam.
- Translate research notes from `obsidian/*.md` into high-quality code, experiments, and documentation.

## System Context
- Eve reframes residual updates as adaptive, momentum-driven integration through depth; each layer update behaves like an Adam step on hidden states.
- Miriam extends Eve with orthogonalized residuals and spectral control, mirroring Muon’s improvements over Adam.
- The Obsidian vault under `obsidian/` is the source of truth for product and research requirements.

## Knowledge & Preparation
- Before contributing, read every relevant note in `obsidian/` (`index.md`).
- Keep the README expectations in mind: agents follow this playbook and otherwise operate like human contributors.
- Maintain your own scratch context, but treat Obsidian as canonical; update it whenever work produces new insights or decisions.

## Operating Procedure
- Clarify scope: restate the task, identify stakeholders, and confirm success criteria.
- Plan before you code: outline approach, touch points, and validation strategy.
- Implement incrementally; keep commits/changesets focused and reviewable.
- Document decisions inline (comments) only when non-obvious; prefer updating markdown specs or notebooks.
- Surface open questions or risks early; do not guess when requirements are ambiguous.

## Code & Research Standards
- Match existing style guides; mirror formatting, naming, and module layout already in the repo.
- Write Eve/Miriam features so they can be toggled cleanly (feature flags, config hooks, modular modules).
- When extending dynamics, include invariants or assertions that capture assumptions (e.g., tensor shapes, numerical ranges).
- Keep experimental code reproducible: log seeds, configs, and dependencies inside scripts.

## Testing & Validation
- Run the smallest meaningful test or script that exercises your change; record the exact command and outcome.
- For research prototypes, provide evaluation notebooks or summaries with metrics that map back to the PRD goals.
- Raise blockers if sandboxed environments prevent running necessary checks; never claim unverified results as verified.

## Documentation Workflow
- When behavior or intent changes, update both code docstrings and the relevant Obsidian pages.
- Cross-link new material using Obsidian wiki links (e.g., `[[Miriam_PRD.md]]`) so the published docs stay coherent.
- Record experiment outputs, plots, or decisions in the vault with timestamps and authorship.

## Collaboration Etiquette
- Communicate in concise, factual updates; prefer checklists with statuses over prose.
- If a change might impact other contributors, flag it in advance and offer migration guidance.
- Respect existing unfinished work in the tree; never overwrite or revert files you did not author without coordination.

## Deliverables Checklist
- ✅ Code or research artifact aligned with scoped task.
- ✅ Tests or evaluation notes showing expected behavior.
- ✅ Obsidian/markdown updates capturing new knowledge.
- ✅ Summary of follow-up work, if any.

> Eve is to inference what Adam is to training; Miriam keeps depth dynamics orthogonal and stable. Build accordingly.
