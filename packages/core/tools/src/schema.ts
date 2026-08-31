/** Unified JSON-value schema DSL, inference, compilation, and typed tool helper. @module dsh-tools/schema */

import { HarnessError } from '@deepseek-ai/dsh-llm'
import type { ContentBlock } from '@deepseek-ai/dsh-llm'
import { isJsonValue, type JsonValue } from '@deepseek-ai/dsh-session'
import type { ToolDefinition, ToolExecution, ToolExecutionResult, ToolRunContext, ToolResult } from './index.ts'
import { assertSupportedJsonSchema, isJsonSchemaRecord, isPlainJsonArray, isPlainJsonRecord, JsonSchemaError, validateJsonSchemaValue } from './json-schema.ts'
import type { JsonSchemaNode, JsonSchemaScalar, ObjectJsonSchema } from './json-schema.ts'
import type { ToolCallView, ToolResultView } from './presentation.ts'

/** Annotation keywords shared by every author-facing schema node. */
export interface ValueSchemaAnnotations {
  /** Human-readable description projected into JSON Schema and generated types. */
  description?: string
  /** Human-readable title projected into JSON Schema. */
  title?: string
  /** Non-validating default annotation; it must be lossless JSON data. */
  default?: JsonValue
  /** Non-validating examples annotation; it must be lossless JSON data. */
  examples?: JsonValue
}

/** String value schema with type-correct literal constraints. */
export interface StringValueSchemaSpec extends ValueSchemaAnnotations {
  type: 'string'
  enum?: readonly string[]
  const?: string
}

/** Finite JSON-number schema with type-correct literal constraints. */
export interface NumberValueSchemaSpec extends ValueSchemaAnnotations {
  type: 'number'
  enum?: readonly number[]
  const?: number
}

/** Integer schema with type-correct literal constraints. */
export interface IntegerValueSchemaSpec extends ValueSchemaAnnotations {
  type: 'integer'
  enum?: readonly number[]
  const?: number
}

/** Boolean value schema with type-correct literal constraints. */
export interface BooleanValueSchemaSpec extends ValueSchemaAnnotations {
  type: 'boolean'
  enum?: readonly boolean[]
  const?: boolean
}

/** Null value schema with type-correct literal constraints. */
export interface NullValueSchemaSpec extends ValueSchemaAnnotations {
  type: 'null'
  enum?: readonly null[]
  const?: null
}

/** Array value schema; omitted `items` accepts any lossless JSON item. */
export interface ArrayValueSchemaSpec extends ValueSchemaAnnotations {
  type: 'array'
  items?: ValueSchemaSpec
}

/**
 * Explicit object value schema. Openness is mandatory so a nested or output
 * object never acquires an accidental JSON Schema default.
 */
export interface ObjectValueSchemaSpec extends ValueSchemaAnnotations {
  type: 'object'
  properties?: ParameterSchemaSpec
  additionalProperties: boolean
}

/** Author-only unconstrained lossless JSON node. */
export interface JsonValueSchemaSpec extends ValueSchemaAnnotations {
  type: 'json'
}

/** Exact-one union schema; at least two branches are required. */
export interface OneOfValueSchemaSpec extends ValueSchemaAnnotations {
  oneOf: readonly [ValueSchemaSpec, ValueSchemaSpec, ...ValueSchemaSpec[]]
}

/** One author-facing schema for any lossless JSON value root. */
export type ValueSchemaSpec =
  | StringValueSchemaSpec
  | NumberValueSchemaSpec
  | IntegerValueSchemaSpec
  | BooleanValueSchemaSpec
  | NullValueSchemaSpec
  | ArrayValueSchemaSpec
  | ObjectValueSchemaSpec
  | JsonValueSchemaSpec
  | OneOfValueSchemaSpec

/** One implicit parameter-root property, optionally required. */
export type ParameterPropertySpec = ValueSchemaSpec & { required?: true }

/**
 * Tool parameter schema. The map itself is an implicit open object root;
 * requiredness remains a per-property `required: true` annotation.
 */
export type ParameterSchemaSpec = {
  [key: string]: ParameterPropertySpec
  [key: symbol]: never
}

/** Raw JSON Schema projection of the implicit parameter object. */
export interface ParameterJsonSchema extends ObjectJsonSchema {
  properties: Record<string, JsonSchemaNode>
}

/** Flatten an intersection into one object type for readable hovers. */
type Simplify<T> = { [K in keyof T]: T[K] } & {}

/** String keys of one property map; runtime compilation rejects symbol keys. */
type StringKeyOf<S> = Extract<keyof S, string>

/** Keys of a property map marked `required: true`. */
type RequiredKeys<S> = {
  [K in StringKeyOf<S>]: S[K] extends { required: true } ? K : never
}[StringKeyOf<S>]

/** Infer the declared value of one parameter property without key optionality. */
type InferProperty<P, Depth extends unknown[]> = InferValueAt<P, Depth>

/** Infer an implicit property map into required and optional object keys. */
type InferProperties<S, Depth extends unknown[]> = Simplify<
  & { [K in RequiredKeys<S>]: InferProperty<S[K], Depth> }
  & { [K in Exclude<StringKeyOf<S>, RequiredKeys<S>>]?: InferProperty<S[K], Depth> }
>

/** Infer an explicit object node, including its declared openness. */
type InferObject<S extends { additionalProperties: boolean }, Depth extends unknown[]> =
  S extends { properties: infer P }
    ? S['additionalProperties'] extends true
      ? InferProperties<P, Depth> & Record<string, JsonValue>
      : InferProperties<P, Depth>
    : S['additionalProperties'] extends true
      ? Record<string, JsonValue>
      : Record<string, never>

/** Infer a scalar node's literal constraint before its broad primitive type. */
type InferScalar<S, Fallback> =
  S extends { const: infer C } ? C :
    S extends { enum: readonly (infer E)[] } ? E :
      Fallback

/** Add one schema-container level to bounded compile-time inference. */
type NextInferenceDepth<Depth extends unknown[]> = [unknown, ...Depth]

/** Infer one node without recursively checking it against the full author union. */
type InferValueAt<S, Depth extends unknown[]> =
  Depth['length'] extends 16 ? JsonValue :
    S extends { type: 'string' } ? InferScalar<S, string> :
      S extends { type: 'number' | 'integer' } ? InferScalar<S, number> :
        S extends { type: 'boolean' } ? InferScalar<S, boolean> :
          S extends { type: 'null' } ? null :
            S extends { type: 'array' }
              ? S extends { items: infer I } ? InferValueAt<I, NextInferenceDepth<Depth>>[] : JsonValue[]
              : S extends { type: 'object'; additionalProperties: boolean }
                ? InferObject<S, NextInferenceDepth<Depth>>
                : S extends { type: 'json' } ? JsonValue :
                  S extends { oneOf: readonly unknown[] }
                    ? InferValueAt<S['oneOf'][number], NextInferenceDepth<Depth>>
                    : never

/**
 * Infer the TypeScript value accepted by an author-facing value schema. Exact
 * inference is bounded to 16 container levels, then falls back to `JsonValue`.
 */
export type InferValue<S> = InferValueAt<S, []>

/** Infer the TypeScript argument object for an implicit parameter schema. */
export type InferArgs<S> = InferProperties<S, []>

const ANNOTATION_KEYS = ['description', 'title', 'default', 'examples'] as const

/** Throw one author-schema violation through the shared schema error type. */
function authorError(message: string): never {
  throw new JsonSchemaError([message])
}

/** Copy own annotation fields for validation by the raw-schema boundary. */
function copyAnnotations(source: Record<string, unknown>, target: JsonSchemaNode): void {
  if (Object.hasOwn(source, 'description')) target.description = source.description as string
  if (Object.hasOwn(source, 'title')) target.title = source.title as string
  if (Object.hasOwn(source, 'default')) target.default = source.default as JsonValue
  if (Object.hasOwn(source, 'examples')) target.examples = source.examples as JsonValue
}

/** Reject author-only keys outside one node's declared vocabulary. */
function assertAuthorKeys(source: Record<string, unknown>, path: string, allowed: readonly string[]): void {
  for (const key of Object.keys(source)) {
    if (!allowed.includes(key)) authorError(`${path}.${key} is not supported by the value schema DSL`)
  }
}

/** Compiled form of one implicit property map. */
interface CompiledPropertyMap {
  properties: Record<string, JsonSchemaNode>
  required?: string[]
}

/** Mutable holder used only while an iterative compilation root is unresolved. */
interface CompileRoot<T> {
  value?: T
}

/** Where one compiled value node is installed. */
type NodeDestination =
  | { kind: 'root'; holder: CompileRoot<JsonSchemaNode> }
  | { kind: 'property'; target: Record<string, JsonSchemaNode>; key: string }
  | { kind: 'item'; target: JsonSchemaNode }
  | { kind: 'one-of'; target: JsonSchemaNode[]; index: number }

/** Where one compiled property map is installed. */
type PropertyMapDestination =
  | { kind: 'root'; holder: CompileRoot<CompiledPropertyMap> }
  | { kind: 'object'; target: JsonSchemaNode }

/** Deferred work for stack-safe author-schema compilation. */
type CompileTask =
  | { kind: 'value'; input: unknown; path: string; allowRequired: boolean; destination: NodeDestination }
  | { kind: 'property-map'; input: unknown; path: string; destination: PropertyMapDestination }
  | {
    kind: 'property'
    property: unknown
    path: string
    key: string
    properties: Record<string, JsonSchemaNode>
    required: string[]
  }
  | {
    kind: 'property-map-tail'
    compiled: CompiledPropertyMap
    required: string[]
    destination: PropertyMapDestination
  }
  | { kind: 'leave'; input: object }

/** Install a compiled node without giving `__proto__` assignment semantics. */
function assignCompiledNode(destination: NodeDestination, node: JsonSchemaNode): void {
  switch (destination.kind) {
    case 'root':
      destination.holder.value = node
      break
    case 'property':
      Object.defineProperty(destination.target, destination.key, {
        value: node,
        enumerable: true,
        configurable: true,
        writable: true,
      })
      break
    case 'item':
      destination.target.items = node
      break
    case 'one-of':
      destination.target[destination.index] = node
      break
  }
}

/** Install a compiled property map at its root or containing object node. */
function assignCompiledPropertyMap(destination: PropertyMapDestination, compiled: CompiledPropertyMap): void {
  if (destination.kind === 'root') {
    destination.holder.value = compiled
  } else {
    destination.target.properties = compiled.properties
  }
}

/** Execute an author-schema compilation task graph without recursive descent. */
function runSchemaCompiler(initial: CompileTask): void {
  const seen = new Set<object>()
  const tasks: CompileTask[] = [initial]
  for (let task = tasks.pop(); task !== undefined; task = tasks.pop()) {
    if (task.kind === 'leave') {
      seen.delete(task.input)
      continue
    }
    if (task.kind === 'property-map-tail') {
      if (task.required.length > 0) {
        task.compiled.required = task.required
        if (task.destination.kind === 'object') task.destination.target.required = task.required
      }
      continue
    }
    if (task.kind === 'property') {
      if (!isJsonSchemaRecord(task.property)) authorError(`${task.path} must be a value schema object`)
      if (Object.hasOwn(task.property, 'required') && task.property.required !== true) {
        authorError(`${task.path}.required must be true when present`)
      }
      if (Object.hasOwn(task.property, 'required') && task.property.required === true) task.required.push(task.key)
      tasks.push({
        kind: 'value',
        input: task.property,
        path: task.path,
        allowRequired: true,
        destination: { kind: 'property', target: task.properties, key: task.key },
      })
      continue
    }
    if (task.kind === 'property-map') {
      if (!isJsonSchemaRecord(task.input)) authorError(`${task.path} must be an object of value schemas`)
      if (seen.has(task.input)) authorError(`${task.path} is circular`)
      seen.add(task.input)
      const compiled: CompiledPropertyMap = { properties: {} }
      const required: string[] = []
      assignCompiledPropertyMap(task.destination, compiled)
      tasks.push({ kind: 'leave', input: task.input })
      tasks.push({ kind: 'property-map-tail', compiled, required, destination: task.destination })
      const entries = Object.entries(task.input)
      for (let index = entries.length - 1; index >= 0; index--) {
        const entry = entries[index]
        /* v8 ignore next -- the loop is bounded by the captured entry count. */
        if (entry === undefined) continue
        tasks.push({
          kind: 'property',
          property: entry[1],
          path: `${task.path}.${entry[0]}`,
          key: entry[0],
          properties: compiled.properties,
          required,
        })
      }
      continue
    }

    const { input, path } = task
    if (!isJsonSchemaRecord(input)) authorError(`${path} must be a value schema object`)
    if (seen.has(input)) authorError(`${path} is circular`)
    seen.add(input)
    const authorKeys = [...ANNOTATION_KEYS, ...(task.allowRequired ? ['required'] : [])]
    const node: JsonSchemaNode = {}
    assignCompiledNode(task.destination, node)
    tasks.push({ kind: 'leave', input })

    if (Object.hasOwn(input, 'oneOf')) {
      assertAuthorKeys(input, path, [...authorKeys, 'oneOf', 'type'])
      if (Object.hasOwn(input, 'type')) authorError(`${path} cannot declare both type and oneOf`)
      if (!isPlainJsonArray(input.oneOf)) authorError(`${path}.oneOf must be an array of at least two value schemas`)
      const branches: JsonSchemaNode[] = []
      node.oneOf = branches
      copyAnnotations(input, node)
      for (let index = input.oneOf.length - 1; index >= 0; index--) {
        tasks.push({
          kind: 'value',
          input: input.oneOf[index],
          path: `${path}.oneOf[${index}]`,
          allowRequired: false,
          destination: { kind: 'one-of', target: branches, index },
        })
      }
      continue
    }

    const inputType = Object.hasOwn(input, 'type') ? input.type : undefined
    switch (inputType) {
      case 'json':
        assertAuthorKeys(input, path, [...authorKeys, 'type'])
        copyAnnotations(input, node)
        break
      case 'object':
        assertAuthorKeys(input, path, [...authorKeys, 'type', 'properties', 'additionalProperties'])
        if (!Object.hasOwn(input, 'additionalProperties') || typeof input.additionalProperties !== 'boolean') {
          authorError(`${path}.additionalProperties must be explicitly true or false`)
        }
        node.type = 'object'
        copyAnnotations(input, node)
        node.additionalProperties = input.additionalProperties
        if (Object.hasOwn(input, 'properties')) {
          tasks.push({
            kind: 'property-map',
            input: input.properties,
            path: `${path}.properties`,
            destination: { kind: 'object', target: node },
          })
        }
        break
      case 'array':
        assertAuthorKeys(input, path, [...authorKeys, 'type', 'items'])
        node.type = 'array'
        copyAnnotations(input, node)
        if (Object.hasOwn(input, 'items')) {
          tasks.push({
            kind: 'value',
            input: input.items,
            path: `${path}.items`,
            allowRequired: false,
            destination: { kind: 'item', target: node },
          })
        }
        break
      case 'string':
      case 'number':
      case 'integer':
      case 'boolean':
      case 'null':
        assertAuthorKeys(input, path, [...authorKeys, 'type', 'enum', 'const'])
        node.type = inputType
        copyAnnotations(input, node)
        if (Object.hasOwn(input, 'enum')) {
          if (!isPlainJsonArray(input.enum)) authorError(`${path}.enum must be a non-empty array of scalar values`)
          node.enum = Array.from(input.enum, entry => entry as JsonSchemaScalar)
        }
        if (Object.hasOwn(input, 'const')) node.const = input.const as JsonSchemaScalar
        break
      default:
        authorError(`${path}.type must be string/number/integer/boolean/null/array/object/json, or use oneOf`)
    }
  }
}

/** Compile one implicit property map, collecting per-property requiredness. */
function compilePropertyMap(input: unknown, path: string): CompiledPropertyMap {
  const holder: CompileRoot<CompiledPropertyMap> = {}
  runSchemaCompiler({ kind: 'property-map', input, path, destination: { kind: 'root', holder } })
  /* v8 ignore next -- the root task assigns before scheduling any descendants. */
  return holder.value ?? authorError(`${path} did not compile`)
}

/** Compile one author node without applying any consumer root restriction. */
function compileValueSchema(input: unknown, path: string): JsonSchemaNode {
  const holder: CompileRoot<JsonSchemaNode> = {}
  runSchemaCompiler({ kind: 'value', input, path, allowRequired: false, destination: { kind: 'root', holder } })
  /* v8 ignore next -- the root task assigns before scheduling any descendants. */
  return holder.value ?? authorError(`${path} did not compile`)
}

/**
 * Compile one author-facing value schema to the enforced raw JSON Schema
 * subset. The author-only `json` node becomes an annotation-only schema.
 * @param spec - schema for any JSON-value root.
 * @returns The asserted raw schema projection.
 */
export function valueSchemaSpecToJsonSchema(spec: ValueSchemaSpec): JsonSchemaNode {
  const schema = compileValueSchema(spec, 'schema')
  assertSupportedJsonSchema(schema)
  return schema
}

/**
 * Compile the implicit open parameter object into raw JSON Schema.
 * @param spec - per-property parameter definitions.
 * @returns An object-rooted raw schema with no implicit-root openness override.
 */
export function parameterSchemaSpecToJsonSchema(spec: ParameterSchemaSpec): ParameterJsonSchema {
  const compiled = compilePropertyMap(spec, 'parameters')
  const schema: ParameterJsonSchema = {
    type: 'object',
    properties: compiled.properties,
    ...(compiled.required === undefined ? {} : { required: compiled.required }),
  }
  assertSupportedJsonSchema(schema)
  return schema
}

/** Invalid model-generated arguments for a typed tool. */
export class ToolArgsError extends HarnessError {
  /** Individual violations in schema-walk order. */
  readonly violations: string[]

  constructor(violations: string[]) {
    super(`invalid arguments: ${violations.join('; ')}`, 'INVALID_ARGS')
    this.name = 'ToolArgsError'
    this.violations = violations
  }
}

/** Maximum recursion depth for unwrapping model-stringified argument values. */
const COERCE_DEPTH_CAP = 24

/**
 * Strictly parse one JSON text that encodes exactly one lossless JSON value.
 *
 * `JSON.parse` refuses trailing garbage itself; non-finite and negative-zero
 * results (which it can still produce from exponent or `-0` spellings) are
 * refused by the lossless JSON boundary, so the strict validator rejects them
 * exactly as it did before coercion existed.
 *
 * @param text - candidate JSON text.
 * @returns The parsed lossless JSON value, or `undefined` when the text does
 *   not encode exactly one lossless JSON value.
 */
function parseExactJsonValue(text: string): JsonValue | undefined {
  try {
    const parsed: unknown = JSON.parse(text)
    return isJsonValue(parsed) ? parsed as JsonValue : undefined
  } catch {
    // `JSON.parse` throws on trailing garbage (for example `{"a":1} x`); treat
    // such text as not-a-JSON-value so the caller keeps the raw string.
    return undefined
  }
}

/**
 * Coerce model-generated arguments before strict validation.
 *
 * GLM-5.3-Flash (and other Z.ai/GLM routes) serialize tool-call argument
 * values as JSON strings even when the declared parameter type is array,
 * object, number, integer, or boolean — streaming `{"queries":
 * "[\"OpenAI latest news\"]"}` instead of a real array, so a strict validator
 * fails every such call with `invalid arguments: "queries" must be an array`.
 *
 * Coercion here is type-directed and additive, never a general leniency:
 * `string`-typed parameters stay verbatim (so a `bash` command or `write`
 * content that merely *looks* like JSON is never corrupted), only a string
 * whose exact JSON parse is the declared type is unwrapped, recursion into
 * array items and object properties is depth-bounded, and the strict
 * validator runs on the coerced value afterwards, so genuinely malformed
 * input fails with the same diagnostics as before.
 *
 * @param args - candidate arguments, however malformed.
 * @param schema - the compiled parameter object schema.
 * @returns The coerced arguments tree.
 */
export function coerceModelArgs(args: unknown, schema: ParameterJsonSchema): unknown {
  if (!isJsonSchemaRecord(schema)) return args
  return coerceValueAt(schema, args, 0)
}

/** Recursive scalar/array/object unwrap over one compiled schema node. */
function coerceValueAt(node: Record<string, unknown>, value: unknown, depth: number): unknown {
  if (depth > COERCE_DEPTH_CAP) return value
  const type = Object.hasOwn(node, 'type') ? node.type : undefined
  if (typeof type !== 'string') return value
  switch (type) {
    case 'array': {
      const unwrapped = typeof value === 'string' && value.length > 0 ? parseExactJsonValue(value) : undefined
      const candidate = isPlainJsonArray(unwrapped) ? unwrapped : value
      if (!isPlainJsonArray(candidate)) return value
      const items = Object.hasOwn(node, 'items') ? node.items : undefined
      if (!isJsonSchemaRecord(items)) return candidate
      return candidate.map(entry => coerceValueAt(items, entry, depth + 1))
    }
    case 'object': {
      const properties = Object.hasOwn(node, 'properties') ? node.properties : undefined
      if (isPlainJsonRecord(value)) {
        const coerced: Record<string, unknown> = { ...value }
        if (isJsonSchemaRecord(properties)) {
          for (const [key, child] of Object.entries(properties)) {
            if (!Object.hasOwn(value, key)) continue
            if (!isJsonSchemaRecord(child)) continue
            coerced[key] = coerceValueAt(child, value[key], depth + 1)
          }
        }
        return coerced
      }
      const unwrapped = typeof value === 'string' && value.length > 0 ? parseExactJsonValue(value) : undefined
      const candidate = isPlainJsonRecord(unwrapped) ? unwrapped : value
      if (!isPlainJsonRecord(candidate)) return value
      const coerced: Record<string, unknown> = { ...candidate }
      if (isJsonSchemaRecord(properties)) {
        for (const [key, child] of Object.entries(properties)) {
          if (!Object.hasOwn(candidate, key)) continue
          if (!isJsonSchemaRecord(child)) continue
          coerced[key] = coerceValueAt(child, candidate[key], depth + 1)
        }
      }
      return coerced
    }
    case 'number': {
      const parsed = typeof value === 'string' && value.length > 0 ? parseExactJsonValue(value) : undefined
      const candidate = typeof parsed === 'number' ? parsed : value
      if (typeof candidate !== 'number' || !Number.isFinite(candidate)) return value
      return candidate === 0 ? 0 : candidate
    }
    case 'integer': {
      const parsed = typeof value === 'string' && value.length > 0 ? parseExactJsonValue(value) : undefined
      const candidate = typeof parsed === 'number' && Number.isInteger(parsed) ? parsed : value
      if (typeof candidate !== 'number' || !Number.isInteger(candidate)) return value
      return candidate === 0 ? 0 : candidate
    }
    case 'boolean': {
      const parsed = typeof value === 'string' && value.length > 0 ? parseExactJsonValue(value) : undefined
      return typeof parsed === 'boolean' ? parsed : value
    }
    default:
      return value
  }
}

/**
 * Validate model-generated arguments against an implicit parameter schema.
 * @param spec - declared parameter schema.
 * @param args - candidate arguments, however malformed.
 * @returns Path-qualified violations; empty means valid.
 */
export function validateArgs(spec: ParameterSchemaSpec, args: unknown): string[] {
  const schema = parameterSchemaSpecToJsonSchema(spec)
  return validateJsonSchemaValue(schema, coerceModelArgs(args, schema), '')
}

/** Options for {@link defineTool}. */
export interface DefineToolOptions<S extends ParameterSchemaSpec, O extends ValueSchemaSpec> {
  /** Tool name (must be unique). */
  readonly name: string
  /** Human-readable description sent to the model. */
  readonly description: string
  /** Per-property parameter schema compiled to an implicit open object root. */
  readonly parameters: S
  /** Canonical output schema plus pure Native and presentation projections. */
  readonly output: {
    /** Schema enforced against every successful body or policy-replaced value. */
    readonly schema: O
    /** Pure Native/model rendering of one validated canonical value. */
    render(args: InferArgs<S>, value: InferValue<NoInfer<O>>): ContentBlock[]
    /** Pure replayable presentation metadata for direct top-level calls. */
    presentationMeta?(args: InferArgs<S>, value: InferValue<NoInfer<O>>): JsonValue
  }
  /** Optional positive cooperative timeout budget in milliseconds. */
  readonly timeoutMs?: number
  /**
   * Pure classifier for sibling overlap.
   * @param args - typed validated arguments.
   * @returns Whether the call may join a parallel group.
   */
  isConcurrencySafe?(args: InferArgs<S>): boolean
  /**
   * Execute the tool after argument validation.
   * @param args - typed validated arguments.
   * @param exec - execution identity, caller, cancellation, and nesting data.
   * @returns The canonical value declared by `output.schema`.
   */
  execute(args: InferArgs<S>, exec: ToolRunContext): Promise<InferValue<NoInfer<O>>>
  /**
   * Optional last-mile content transform for every normalized outcome. Unlike
   * `execute`, arguments remain `unknown` because invalid-input failures also
   * reach this callback. See {@link ToolDefinition.finalizeContent}.
   * @param exec - immutable execution identity and arguments.
   * @param result - complete normalized outcome before materialization.
   * @returns replacement content, or `undefined` to preserve it.
   */
  finalizeContent?(exec: Readonly<ToolExecution>, result: Readonly<ToolExecutionResult>): ContentBlock[] | undefined
  /**
   * Pure pending-state presenter.
   * @param args - typed validated arguments.
   * @returns Tool-owned render intent, or `undefined` for the generic card.
   */
  presentCall?(args: InferArgs<S>): ToolCallView | undefined
  /**
   * Pure completed-state presenter.
   * @param args - typed validated arguments.
   * @param result - final model-facing tool result.
   * @returns Tool-owned render intent, or `undefined` for the generic card.
   */
  presentResult?(args: InferArgs<S>, result: ToolResult): ToolResultView | undefined
}

/**
 * Define a first-party tool with inferred arguments and strict execution
 * validation. Replay-only presenters validate softly and fall back to generic
 * rendering for obsolete logged arguments. Model-side JSON-stringified
 * argument values are unwrapped once at this boundary (see
 * {@link coerceModelArgs}); strict validation still runs on the coerced
 * arguments.
 * @param options - typed definition and optional finalizer and presenters.
 * @returns A registry-ready definition.
 */
export function defineTool<const S extends ParameterSchemaSpec, const O extends ValueSchemaSpec>(
  options: DefineToolOptions<S, O>,
): ToolDefinition {
  // Object-literal methods do not use `this`; retaining references is safe.
  // oxlint-disable-next-line typescript/unbound-method
  const userExecute = options.execute
  // oxlint-disable-next-line typescript/unbound-method
  const userFinalizeContent = options.finalizeContent
  // oxlint-disable-next-line typescript/unbound-method
  const userRender = options.output.render
  // oxlint-disable-next-line typescript/unbound-method
  const userPresentationMeta = options.output.presentationMeta
  // oxlint-disable-next-line typescript/unbound-method
  const userPresentCall = options.presentCall
  // oxlint-disable-next-line typescript/unbound-method
  const userPresentResult = options.presentResult
  // oxlint-disable-next-line typescript/unbound-method
  const userIsConcurrencySafe = options.isConcurrencySafe
  if (options.timeoutMs !== undefined && (!Number.isFinite(options.timeoutMs) || options.timeoutMs <= 0)) {
    throw new Error(`defineTool(${options.name}): timeoutMs must be a positive finite number`)
  }
  const parameters = parameterSchemaSpecToJsonSchema(options.parameters)
  const outputSchema = valueSchemaSpecToJsonSchema(options.output.schema)
  const normalize = (args: unknown): unknown => coerceModelArgs(args, parameters)
  const validate = (args: unknown): string[] => validateJsonSchemaValue(parameters, normalize(args), '')
  const tool: ToolDefinition = {
    name: options.name,
    description: options.description,
    parameters: parameters as unknown as Record<string, unknown>,
    output: {
      schema: outputSchema,
      render(args: unknown, value: JsonValue): ContentBlock[] {
        return userRender(args as InferArgs<S>, value as unknown as InferValue<NoInfer<O>>)
      },
      ...userPresentationMeta !== undefined ? {
        presentationMeta(args: unknown, value: JsonValue): JsonValue {
          return userPresentationMeta(args as InferArgs<S>, value as unknown as InferValue<NoInfer<O>>)
        },
      } : {},
    },
    ...(options.timeoutMs !== undefined ? { timeoutMs: options.timeoutMs } : {}),
    async execute(args: unknown, exec: ToolRunContext): Promise<JsonValue> {
      const coerced = normalize(args)
      const violations = validateJsonSchemaValue(parameters, coerced, '')
      if (violations.length > 0) throw new ToolArgsError(violations)
      return userExecute(coerced as InferArgs<S>, exec) as Promise<JsonValue>
    },
  }
  if (userFinalizeContent) {
    tool.finalizeContent = (exec, result) => userFinalizeContent(exec, result)
  }
  // Presentation is display-only and may run on REPLAY of arbitrary logged args
  // (possibly from an older schema), so it must never throw: validate softly and
  // fall back to `undefined` (a generic UI presentation) on any mismatch, rather
  // than the hard `ToolArgsError` the execute path raises.
  if (userPresentCall) {
    tool.presentCall = (args: unknown): ToolCallView | undefined => {
      const coerced = normalize(args)
      if (validate(coerced).length > 0) return undefined
      return userPresentCall(coerced as InferArgs<S>)
    }
  }
  if (userPresentResult) {
    tool.presentResult = (args: unknown, result: ToolResult): ToolResultView | undefined => {
      const coerced = normalize(args)
      if (validate(coerced).length > 0) return undefined
      return userPresentResult(coerced as InferArgs<S>, result)
    }
  }
  if (userIsConcurrencySafe) {
    tool.isConcurrencySafe = (args: unknown): boolean => {
      const coerced = normalize(args)
      if (validate(coerced).length > 0) return false
      return userIsConcurrencySafe(coerced as InferArgs<S>)
    }
  }
  return tool
}
