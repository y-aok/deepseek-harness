# Agent Note: GLM tool-call argument values arrive JSON-stringified

Status: implemented

## Problem

`web_search` calls dispatched to a GLM-5.3-Flash route (provider `ds4`, model `glm-5.2`/`glm-5.3-flash`) failed argument validation with `invalid arguments: "queries" must be an array` even though the harness declared the parameter correctly and the model emitted the intended array. The session log preserves the ground truth: the streamed tool-call arguments were `{"queries":"[\"OpenAI latest news\"]"}` — the model serializes **every non-string argument value as a JSON string** (value-level double encoding), not only array-typed ones: `read` with numeric `offset`/`limit` failed identically with `"offset" must be a number`. String-typed parameters are unaffected, which is why `bash`, `grep`, and `web_fetch` kept working through the same pipeline.

The strict validator (`packages/core/tools/src/json-schema.ts`) is intentional — `case 'array'` demands `Array.isArray`, `case 'number'` demands `typeof value === 'number'` — and no layer between pi-ai's parsed tool-call and that validator unwrapped the stringified values. The quirk persisted across retries with different payload shapes, so no model-side or prompt-side change fixes it; the boundary that owns the declared parameter schema must tolerate it.

## Decision

`packages/core/tools/src/schema.ts` owns a type-directed, additive coercion pass at the first-party tool input boundary:

- `coerceModelArgs(args, schema)` walks the compiled parameter schema and unwraps a received **string** only when its exact JSON parse *is* the declared type (`array` → array, `object` → plain record, `number`/`integer` → finite JSON number, `boolean` → boolean). Recursion into array items and object properties is depth-bounded (`COERCE_DEPTH_CAP = 24`).
- `parseExactJsonValue` uses `JSON.parse` plus the lossless JSON boundary (`isJsonValue`), so trailing garbage, non-finite results (`1e999` → `Infinity`), and negative zero stay uncoerced and keep failing with the same diagnostics as before.
- `string`-typed parameters are identity: a `bash` command or `write` content that merely *looks* like JSON is never corrupted.
- `defineTool` applies the coercion at all four argument-validation sites (`execute`, `presentCall`, `presentResult`, `isConcurrencySafe`) and hands the **coerced** arguments — not the raw strings — to the tool body; `validateArgs` (tests and external callers) coerces through the same compiled schema.

Companion seams that reuse the helper are identified but deliberately deferred: the MCP client (which already coerces `null`/primitive args to `{}`) and the subagent structured driver, both of which throw `ToolArgsError` on their own paths. They adopt `coerceModelArgs` when GLM reaches those seams.

## Alternatives considered

- **Coerce inside the shared validator (`checkValue`).** Rejected: that validator is shared with structured-output and PTC result contracts; coercing there weakens both directions and masks genuine output violations.
- **Patch pi-ai** (`node_modules/.pnpm/@earendil-works/pi-ai`, Z.ai route) to unwrap double-encoded arguments in `openai-completions.js`. Rejected for this change: the dependency is pinned upstream (`@earendil-works/pi-ai@0.84.2`), the patch would need the tool schema plumbed into the provider parse to be type-directed safely, and the harness already owns the declared schema at its own boundary. Revisit via `patches/` only if pi-ai-level handling becomes necessary.
- **Prompt/tool-usage guidance** telling the model to emit raw JSON values. Rejected: the quirk is systematic model behavior that persisted across retries; guidance is a mitigation, not a fix, and was kept out of shipped tool sections.
- **Blind leniency** (unwrap any JSON-looking string regardless of declared type). Rejected: it would corrupt legitimately string-typed values whose content is JSON-looking (the regression guard tested via `{ path: '["x"]' }`).

## Consequences

Coercion cost a small widened boundary: a string whose JSON encoding is exactly the declared type now passes where it previously failed — including historical sessions replayed through soft presenters, which now render a card instead of falling back to the generic UI. What it bought: every GLM-5.3-Flash tool call with array, object, number, integer, or boolean parameters now executes, and the input boundary stays honest because coercion is type-directed, depth-bounded, and followed by the unchanged strict validator. Tests pin the unwrapped case, the double-encoded case, trailing garbage, quoted numerics, string-identity, nested items/properties recursion, and the coerced arguments reaching the tool body at execute time.
