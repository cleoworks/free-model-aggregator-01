// free-model-aggregator-01: DUSKLINE routing aggregator.
//
// One server, two client wires (OpenAI Chat Completions, Anthropic Messages), ten free-tier
// providers behind three wires (OpenAI, Anthropic, Gemini). The score is mostly quality:
// for each of the trial's twenty requests, answer with the highest-rated free model that
// meets the request's requirements — and when that provider refuses (budget spent, failure
// schedule, outage), fall over to the next-best candidate instead of failing the request.
//
// Routing, in one paragraph: candidates are every free (provider, model) pair whose
// capabilities meet the request, ranked by the quality table's rating; ties break toward
// the provider that speaks the client's own wire (a passthrough cannot mistranslate), then
// toward the provider that has refused less so far. One extra rule the ratings cannot see:
// a JSON-mode request on the OpenAI wire never routes to an Anthropic-wire provider,
// because that wire cannot carry response_format and its echo would report json_mode false.
// Calls are strictly one at a time (the client sends one request at a time), so no provider
// ever sees a concurrent call from here. Every choice derives from the brief and from the
// answers seen so far — never from the wall clock — so a replay walks the same route.
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicStreamFrames,
  answerFromJson,
  answerToJson,
  echoFieldsOf,
  openAIError,
  openAIModelList,
  openAIStreamFrames,
  providerDeltas,
  providerRequest,
  type Answer,
  type ClientWire,
  type ProviderWire,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 6000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** The manifest is in arrival order and the client sends one request at a time. */
let manifestIndex = 0;
/** Per-provider trial history: calls made and refusals seen, for tie-breaks and logging. */
const providerState = new Map<string, { calls: number; refusals: number }>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}
function stateOf(providerId: string): { calls: number; refusals: number } {
  let s = providerState.get(providerId);
  if (s === undefined) { s = { calls: 0, refusals: 0 }; providerState.set(providerId, s); }
  return s;
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

// ---- the model directory -------------------------------------------------------------

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

// ---- routing ---------------------------------------------------------------------------

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
  wire: ProviderWire;
}

/**
 * Every legal pick for one request, best first. Legal means: the provider answered its
 * catalog probe, the model is free, its capabilities meet the request, and — for a
 * JSON-mode request from an OpenAI client — the provider's wire can carry JSON mode.
 */
function candidatesFor(clientWire: ClientWire, body: Json, requires: Requirement): Candidate[] {
  const fields = echoFieldsOf(clientWire, body);
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const wire = probe.provider.wire as ProviderWire;
    if (fields.jsonMode && wire === 'anthropic') continue; // its echo would lose json_mode
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0, wire });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    const aSame = a.wire === clientWire ? 0 : 1;
    const bSame = b.wire === clientWire ? 0 : 1;
    if (aSame !== bSame) return aSame - bSame;
    const aRef = stateOf(a.probe.provider.id).refusals;
    const bRef = stateOf(b.probe.provider.id).refusals;
    if (aRef !== bRef) return aRef - bRef;
    return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
  });
  return out;
}

function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // No manifest row for this arrival (a brief without a workload): read the body itself.
  const fields = echoFieldsOf('openai', body);
  const vision = fields.messages.some((m) => Array.isArray(m.content) &&
    (m.content as Json[]).some((b) => isRecord(b) && (b.type === 'image' || b.type === 'image_url')));
  return { tools: fields.tools.length > 0, vision, json: fields.jsonMode };
}

// ---- upstream calls ----------------------------------------------------------------------

interface UpstreamOk { ok: true; answer: Answer; model: string }
interface UpstreamFail { ok: false; status: number; detail: string }

function upstreamUrl(candidate: Candidate, stream: boolean): string {
  const base = joinBase(ENV_URL, candidate.probe.provider.base);
  if (candidate.wire === 'gemini') {
    const verb = stream ? 'streamGenerateContent' : 'generateContent';
    return `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${verb}`;
  }
  return candidate.wire === 'anthropic' ? `${base}/v1/messages` : `${base}/v1/chat/completions`;
}

function upstreamHeaders(candidate: Candidate, clientHeaders: IncomingMessage['headers'], stream: boolean): Record<string, string> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(candidate.probe.provider.auth, candidate.probe.provider.key),
  };
  if (candidate.wire === 'anthropic') {
    const version = clientHeaders['anthropic-version'];
    headers['anthropic-version'] = typeof version === 'string' && version !== '' ? version : '2023-06-01';
  }
  if (stream) headers.accept = 'text/event-stream';
  return headers;
}

/** One provider call. Any refusal (402/429/5xx), error body or network failure is a failover signal. */
async function callCandidate(
  candidate: Candidate,
  clientWire: ClientWire,
  body: Json,
  stream: boolean,
  clientHeaders: IncomingMessage['headers'],
): Promise<UpstreamOk | UpstreamFail> {
  const state = stateOf(candidate.probe.provider.id);
  state.calls += 1;
  const upstreamBody = providerRequest(clientWire, candidate.wire, body, candidate.model.id);
  if (stream && candidate.wire !== 'gemini') upstreamBody.stream = true;
  let res: Response;
  try {
    res = await fetch(upstreamUrl(candidate, stream), {
      method: 'POST',
      headers: upstreamHeaders(candidate, clientHeaders, stream),
      body: JSON.stringify(upstreamBody),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (err) {
    state.refusals += 1;
    return { ok: false, status: 0, detail: (err as Error).message };
  }
  if (!res.ok) {
    state.refusals += 1;
    const text = await res.text().catch(() => '');
    return { ok: false, status: res.status, detail: text.slice(0, 200) };
  }
  try {
    if (stream) {
      if (res.body === null) return { ok: false, status: 0, detail: 'the provider sent no stream' };
      const gen = providerDeltas(candidate.wire, res.body);
      let result = await gen.next();
      while (!result.done) result = await gen.next();
      const answer: Answer = {
        text: result.value.text,
        inputTokens: result.value.inputTokens,
        outputTokens: result.value.outputTokens,
        stopRaw: result.value.stopRaw,
      };
      return { ok: true, answer, model: candidate.model.id };
    }
    const json: unknown = await res.json();
    if (!isRecord(json)) return { ok: false, status: 0, detail: 'the provider answer was not one JSON object' };
    return { ok: true, answer: answerFromJson(candidate.wire, json), model: candidate.model.id };
  } catch (err) {
    state.refusals += 1;
    return { ok: false, status: 0, detail: (err as Error).message };
  }
}

// ---- client routes -------------------------------------------------------------------------

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: ClientWire, status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** Serve one client request: walk the ranked candidates until one answers. */
async function serve(clientWire: ClientWire, req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, clientWire, 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const stream = body.stream === true;
  const candidates = candidatesFor(clientWire, body, requirementsFor(body));
  if (candidates.length === 0) {
    return sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model',
      'no free model on the directory meets this request');
  }
  let lastFail: UpstreamFail | null = null;
  for (const candidate of candidates) {
    const result = await callCandidate(candidate, clientWire, body, stream, req.headers);
    if (!result.ok) {
      lastFail = result;
      log(`refused: ${candidate.probe.provider.id}/${candidate.model.id} (HTTP ${result.status}) — failing over`);
      continue;
    }
    if (stream) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      const frames = clientWire === 'openai'
        ? openAIStreamFrames(result.answer, result.model)
        : anthropicStreamFrames(result.answer, result.model);
      for (const frame of frames) res.write(frame);
      res.end();
    } else {
      sendJson(res, 200, answerToJson(clientWire, result.answer, result.model));
    }
    return;
  }
  return sendError(res, clientWire, 502, clientWire === 'anthropic' ? 'api_error' : 'upstream_error',
    `every candidate refused (last: HTTP ${lastFail?.status ?? 0} ${lastFail?.detail ?? ''})`);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') return models(req, res);
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') return serve('openai', req, res);
  if (req.method === 'POST' && url.pathname === '/v1/messages') return serve('anthropic', req, res);
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

// ---- trial lifecycle -------------------------------------------------------------------------

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
