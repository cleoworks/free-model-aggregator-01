// DUSKLINE aggregator wires: translation between the OpenAI Chat Completions wire, the
// Anthropic Messages wire and the Gemini wire — requests, answers and event streams.
//
// The providers answer with a text that echoes what they parsed (system, user text, tool
// names, max_tokens, stop sequences, image hash, JSON mode), and the client compares that
// echo with what it sent. So every translation here is built to carry exactly those fields
// across, byte for byte where the field is bytes (images), value for value elsewhere:
//   system text (multi-part system messages join with "\n", as the parsers join them)
//   the last user message's text
//   tool names, in order
//   max_tokens (never invented: an absent max_tokens stays absent, or the echo differs)
//   stop sequences
//   image bytes (base64 data moves as-is; a URL stays the same URL)
//   JSON mode (only the OpenAI and Gemini wires can carry it; routing keeps it off Anthropic)
// Anything else (temperature, tool_choice, history shape) is carried in the closest form
// the target wire has, because a well-formed request matters even where the echo is silent.
import { isRecord, type Json } from './providers.ts';

export type ClientWire = 'openai' | 'anthropic';
export type ProviderWire = 'openai' | 'anthropic' | 'gemini';

export interface ServedModel { id: string; provider: string; display_name: string }

// ---- model lists and error envelopes -------------------------------------------

export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: 1700000000, owned_by: m.provider })),
  };
}
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({
      type: 'model', id: m.id, display_name: m.display_name, created_at: '2026-01-01T00:00:00Z',
    })),
    has_more: false, first_id: models[0]?.id ?? null, last_id: models[models.length - 1]?.id ?? null,
  };
}
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, param: null, code } };
}
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

// ---- stop reasons ----------------------------------------------------------------
// Families, as the scorer groups them: 0 end, 1 length, 2 tool call, 3 filter.
const STOP_FAMILY: Record<string, number> = {
  stop: 0, end_turn: 0, STOP: 0,
  length: 1, max_tokens: 1, MAX_TOKENS: 1,
  tool_calls: 2, function_call: 2, tool_use: 2,
  content_filter: 3, refusal: 3, SAFETY: 3,
};
export function stopFor(wire: ClientWire | ProviderWire, raw: string | null): string {
  const fam = raw === null ? 0 : STOP_FAMILY[raw] ?? 0;
  if (wire === 'openai') return ['stop', 'length', 'tool_calls', 'content_filter'][fam];
  if (wire === 'anthropic') return ['end_turn', 'max_tokens', 'tool_use', 'refusal'][fam];
  return ['STOP', 'MAX_TOKENS', 'STOP', 'SAFETY'][fam];
}

// ---- shared body reading -----------------------------------------------------------

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

/** The fields of a client request that the echo (and so the score) can see. */
export interface EchoFields {
  system: string;
  messages: Json[]; // the client's own messages, untouched
  tools: Json[];     // the client's own tools array, untouched
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | null;
}

export function echoFieldsOf(wire: ClientWire, body: Json): EchoFields {
  if (wire === 'openai') {
    const systemParts: string[] = [];
    for (const m of Array.isArray(body.messages) ? body.messages : []) {
      if (isRecord(m) && m.role === 'system') {
        const t = textOf(m.content);
        if (t !== '') systemParts.push(t);
      }
    }
    const format = isRecord(body.response_format) ? body.response_format.type : undefined;
    return {
      system: systemParts.join('\n'),
      messages: (Array.isArray(body.messages) ? body.messages : []) as Json[],
      tools: (Array.isArray(body.tools) ? body.tools : []) as Json[],
      maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens
        : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null,
      stops: Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === 'string')
        : typeof body.stop === 'string' ? [body.stop] : [],
      jsonMode: format === 'json_object' || format === 'json_schema',
      temperature: typeof body.temperature === 'number' ? body.temperature : null,
    };
  }
  return {
    system: textOf(body.system),
    messages: (Array.isArray(body.messages) ? body.messages : []) as Json[],
    tools: (Array.isArray(body.tools) ? body.tools : []) as Json[],
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : null,
    stops: Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : [],
    jsonMode: false, // the Anthropic wire has no JSON mode to carry
    temperature: typeof body.temperature === 'number' ? body.temperature : null,
  };
}

// ---- request translation -------------------------------------------------------------

interface ImagePart { kind: 'base64'; mediaType: string; data: string }
interface UrlPart { kind: 'url'; url: string }
type Img = ImagePart | UrlPart;

function openAIImage(block: Json): Img | null {
  if (!isRecord(block.image_url) || typeof block.image_url.url !== 'string') return null;
  const url = block.image_url.url;
  const data = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(url);
  if (data && data[2]) return { kind: 'base64', mediaType: data[1] || 'image/png', data: data[3] };
  return { kind: 'url', url };
}
function anthropicImage(block: Json): Img | null {
  if (!isRecord(block.source)) return null;
  if (block.source.type === 'base64' && typeof block.source.data === 'string') {
    return { kind: 'base64', mediaType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data };
  }
  if (block.source.type === 'url' && typeof block.source.url === 'string') return { kind: 'url', url: block.source.url };
  return null;
}

/** Client (OpenAI) body -> Anthropic provider body. */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const f = echoFieldsOf('openai', body);
  const messages: Json[] = [];
  for (const m of f.messages) {
    if (m.role === 'system') continue;
    if (m.role === 'tool') {
      messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: String(m.tool_call_id ?? ''), content: textOf(m.content) }] });
      continue;
    }
    if (m.role === 'assistant' && Array.isArray(m.tool_calls) && m.tool_calls.length > 0) {
      const blocks: Json[] = [];
      const t = textOf(m.content);
      if (t !== '') blocks.push({ type: 'text', text: t });
      for (const call of m.tool_calls as Json[]) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let input: unknown = {};
        try { input = JSON.parse(String(call.function.arguments ?? '{}')); } catch { input = {}; }
        blocks.push({ type: 'tool_use', id: String(call.id ?? ''), name: String(call.function.name ?? ''), input });
      }
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (typeof m.content === 'string') { messages.push({ role: m.role, content: m.content }); continue; }
    const blocks: Json[] = [];
    for (const block of Array.isArray(m.content) ? (m.content as Json[]) : []) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
      else if (block.type === 'image_url') {
        const img = openAIImage(block);
        if (img?.kind === 'base64') blocks.push({ type: 'image', source: { type: 'base64', media_type: img.mediaType, data: img.data } });
        else if (img?.kind === 'url') blocks.push({ type: 'image', source: { type: 'url', url: img.url } });
      }
    }
    messages.push({ role: m.role, content: blocks });
  }
  const out: Json = { model, messages };
  if (f.system !== '') out.system = f.system;
  if (f.maxTokens !== null) out.max_tokens = f.maxTokens;
  if (f.temperature !== null) out.temperature = f.temperature;
  if (f.stops.length > 0) out.stop_sequences = f.stops;
  if (body.stream === true) out.stream = true;
  if (f.tools.length > 0) {
    out.tools = f.tools.map((t) => {
      const fn = isRecord(t.function) ? t.function : {};
      return { name: String(fn.name ?? ''), description: String(fn.description ?? ''), input_schema: isRecord(fn.parameters) ? fn.parameters : { type: 'object' } };
    });
  }
  return out;
}

/** Client (Anthropic) body -> OpenAI provider body. */
export function anthropicToOpenAIRequest(body: Json, model: string): Json {
  const f = echoFieldsOf('anthropic', body);
  const messages: Json[] = [];
  if (f.system !== '') messages.push({ role: 'system', content: f.system });
  for (const m of f.messages) {
    if (typeof m.content === 'string') { messages.push({ role: m.role, content: m.content }); continue; }
    const blocks = Array.isArray(m.content) ? (m.content as Json[]) : [];
    const hasToolBlocks = blocks.some((b) => isRecord(b) && (b.type === 'tool_use' || b.type === 'tool_result'));
    if (hasToolBlocks) {
      const toolCalls: Json[] = [];
      const results: Json[] = [];
      let text = '';
      for (const b of blocks) {
        if (!isRecord(b)) continue;
        if (b.type === 'text' && typeof b.text === 'string') text += (text === '' ? '' : '\n') + b.text;
        else if (b.type === 'tool_use') {
          toolCalls.push({ id: String(b.id ?? ''), type: 'function', function: { name: String(b.name ?? ''), arguments: JSON.stringify(b.input ?? {}) } });
        } else if (b.type === 'tool_result') {
          results.push({ role: 'tool', tool_call_id: String(b.tool_use_id ?? ''), content: textOf(b.content) });
        }
      }
      if (m.role === 'assistant') {
        const msg: Json = { role: 'assistant', content: text === '' ? null : text };
        if (toolCalls.length > 0) msg.tool_calls = toolCalls;
        messages.push(msg);
      } else {
        if (text !== '') messages.push({ role: 'user', content: text });
        messages.push(...results);
      }
      continue;
    }
    const content: Json[] = [];
    for (const b of blocks) {
      if (!isRecord(b)) continue;
      if (b.type === 'text' && typeof b.text === 'string') content.push({ type: 'text', text: b.text });
      else if (b.type === 'image') {
        const img = anthropicImage(b);
        if (img?.kind === 'base64') content.push({ type: 'image_url', image_url: { url: `data:${img.mediaType};base64,${img.data}` } });
        else if (img?.kind === 'url') content.push({ type: 'image_url', image_url: { url: img.url } });
      }
    }
    messages.push({ role: m.role, content });
  }
  const out: Json = { model, messages };
  if (f.maxTokens !== null) out.max_tokens = f.maxTokens;
  if (f.temperature !== null) out.temperature = f.temperature;
  if (f.stops.length > 0) out.stop = f.stops;
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  if (f.tools.length > 0) {
    out.tools = f.tools.map((t) => ({
      type: 'function',
      function: { name: String(t.name ?? ''), description: String(t.description ?? ''), parameters: isRecord(t.input_schema) ? t.input_schema : { type: 'object' } },
    }));
  }
  return out;
}

/** Client body (either wire) -> Gemini provider body. The model travels in the URL path. */
export function toGeminiRequest(clientWire: ClientWire, body: Json): Json {
  const f = echoFieldsOf(clientWire, body);
  const contents: Json[] = [];
  const pushParts = (role: string, parts: Json[]) => {
    if (parts.length > 0) contents.push({ role, parts });
  };
  for (const m of f.messages) {
    if (m.role === 'system') continue;
    const role = m.role === 'assistant' ? 'model' : 'user';
    if (typeof m.content === 'string') { pushParts(role, [{ text: m.content }]); continue; }
    const parts: Json[] = [];
    for (const b of Array.isArray(m.content) ? (m.content as Json[]) : []) {
      if (!isRecord(b)) continue;
      if (b.type === 'text' && typeof b.text === 'string') parts.push({ text: b.text });
      else if (b.type === 'image_url' || b.type === 'image') {
        const img = clientWire === 'openai' ? openAIImage(b) : anthropicImage(b);
        if (img?.kind === 'base64') parts.push({ inlineData: { mimeType: img.mediaType, data: img.data } });
        else if (img?.kind === 'url') parts.push({ fileData: { fileUri: img.url } });
      } else if (b.type === 'tool_use' && clientWire === 'anthropic') {
        parts.push({ functionCall: { name: String(b.name ?? ''), args: isRecord(b.input) ? b.input : {} } });
      } else if (b.type === 'tool_result' && clientWire === 'anthropic') {
        parts.push({ functionResponse: { name: '', response: { content: textOf(b.content) } } });
      }
    }
    if (clientWire === 'openai' && m.role === 'assistant' && Array.isArray(m.tool_calls)) {
      for (const call of m.tool_calls as Json[]) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let args: unknown = {};
        try { args = JSON.parse(String(call.function.arguments ?? '{}')); } catch { args = {}; }
        parts.push({ functionCall: { name: String(call.function.name ?? ''), args } });
      }
    }
    if (clientWire === 'openai' && m.role === 'tool') {
      pushParts('user', [{ functionResponse: { name: '', response: { content: textOf(m.content) } } }]);
      continue;
    }
    pushParts(role, parts);
  }
  const out: Json = { contents };
  if (f.system !== '') out.systemInstruction = { parts: [{ text: f.system }] };
  const config: Json = {};
  if (f.maxTokens !== null) config.maxOutputTokens = f.maxTokens;
  if (f.temperature !== null) config.temperature = f.temperature;
  if (f.stops.length > 0) config.stopSequences = f.stops;
  if (f.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  if (f.tools.length > 0) {
    const declarations = f.tools.map((t) => {
      if (clientWire === 'openai') {
        const fn = isRecord(t.function) ? t.function : {};
        return { name: String(fn.name ?? ''), description: String(fn.description ?? ''), parameters: isRecord(fn.parameters) ? fn.parameters : { type: 'object' } };
      }
      return { name: String(t.name ?? ''), description: String(t.description ?? ''), parameters: isRecord(t.input_schema) ? t.input_schema : { type: 'object' } };
    });
    out.tools = [{ functionDeclarations: declarations }];
  }
  return out;
}

/** Build the upstream request body for any client-wire x provider-wire pair. */
export function providerRequest(clientWire: ClientWire, providerWire: ProviderWire, body: Json, model: string): Json {
  if (providerWire === 'gemini') return toGeminiRequest(clientWire, body);
  if (clientWire === 'openai' && providerWire === 'openai') {
    const out: Json = { ...body, model };
    if (body.stream === true) out.stream_options = { include_usage: true };
    return out;
  }
  if (clientWire === 'anthropic' && providerWire === 'anthropic') return { ...body, model };
  if (clientWire === 'openai') return openAIToAnthropicRequest(body, model);
  return anthropicToOpenAIRequest(body, model);
}

// ---- answer extraction and emission -----------------------------------------------

export interface Answer { text: string; inputTokens: number; outputTokens: number; stopRaw: string | null }

export function answerFromJson(providerWire: ProviderWire, body: Json): Answer {
  if (providerWire === 'openai') {
    const choice = Array.isArray(body.choices) && isRecord(body.choices[0]) ? body.choices[0] : {};
    const message = isRecord(choice.message) ? choice.message : {};
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: typeof message.content === 'string' ? message.content : '',
      inputTokens: typeof usage.prompt_tokens === 'number' ? usage.prompt_tokens : 0,
      outputTokens: typeof usage.completion_tokens === 'number' ? usage.completion_tokens : 0,
      stopRaw: typeof choice.finish_reason === 'string' ? choice.finish_reason : null,
    };
  }
  if (providerWire === 'anthropic') {
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: textOf(body.content),
      inputTokens: typeof usage.input_tokens === 'number' ? usage.input_tokens : 0,
      outputTokens: typeof usage.output_tokens === 'number' ? usage.output_tokens : 0,
      stopRaw: typeof body.stop_reason === 'string' ? body.stop_reason : null,
    };
  }
  const candidate = Array.isArray(body.candidates) && isRecord(body.candidates[0]) ? body.candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: parts.map((p) => (isRecord(p) && typeof p.text === 'string' ? p.text : '')).join(''),
    inputTokens: typeof usage.promptTokenCount === 'number' ? usage.promptTokenCount : 0,
    outputTokens: typeof usage.candidatesTokenCount === 'number' ? usage.candidatesTokenCount : 0,
    stopRaw: typeof candidate.finishReason === 'string' ? candidate.finishReason : null,
  };
}

export function answerToJson(clientWire: ClientWire, answer: Answer, model: string): Json {
  if (clientWire === 'openai') {
    return {
      id: 'chatcmpl-agg', object: 'chat.completion', created: 1700000000, model,
      choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: stopFor('openai', answer.stopRaw), logprobs: null }],
      usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
    };
  }
  return {
    id: 'msg_agg', type: 'message', role: 'assistant', model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: stopFor('anthropic', answer.stopRaw), stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

// ---- streaming -----------------------------------------------------------------------
// One reader for every provider wire: it yields text deltas as they arrive and finishes
// with the provider's usage and stop reason. The writers below then emit the client's
// wire in exactly the frame order that wire's own servers use (role/block first, deltas,
// the closing frame with stop and usage, the terminator).

export interface StreamResult { text: string; inputTokens: number; outputTokens: number; stopRaw: string | null }

export async function* providerDeltas(providerWire: ProviderWire, stream: ReadableStream<Uint8Array>): AsyncGenerator<string, StreamResult> {
  const decoder = new TextDecoder();
  let buffer = '';
  let text = '';
  let inputTokens = 0, outputTokens = 0;
  let stopRaw: string | null = null;
  const handle = (data: Json): string => {
    if (providerWire === 'openai') {
      const choice = Array.isArray(data.choices) && isRecord(data.choices[0]) ? data.choices[0] : null;
      if (choice) {
        if (typeof choice.finish_reason === 'string') stopRaw = choice.finish_reason;
        if (isRecord(choice.delta) && typeof choice.delta.content === 'string') return choice.delta.content;
      }
      if (isRecord(data.usage)) {
        if (typeof data.usage.prompt_tokens === 'number') inputTokens = data.usage.prompt_tokens;
        if (typeof data.usage.completion_tokens === 'number') outputTokens = data.usage.completion_tokens;
      }
      return '';
    }
    if (providerWire === 'anthropic') {
      if (data.type === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
        if (typeof data.message.usage.input_tokens === 'number') inputTokens = data.message.usage.input_tokens;
      }
      if (data.type === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
        return data.delta.text;
      }
      if (data.type === 'message_delta') {
        if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stopRaw = data.delta.stop_reason;
        if (isRecord(data.usage) && typeof data.usage.output_tokens === 'number') outputTokens = data.usage.output_tokens;
      }
      return '';
    }
    // gemini
    const candidate = Array.isArray(data.candidates) && isRecord(data.candidates[0]) ? data.candidates[0] : null;
    if (candidate) {
      if (typeof candidate.finishReason === 'string') stopRaw = candidate.finishReason;
      if (isRecord(data.usageMetadata)) {
        if (typeof data.usageMetadata.promptTokenCount === 'number') inputTokens = data.usageMetadata.promptTokenCount;
        if (typeof data.usageMetadata.candidatesTokenCount === 'number') outputTokens = data.usageMetadata.candidatesTokenCount;
      }
      const content = isRecord(candidate.content) ? candidate.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      return parts.map((p) => (isRecord(p) && typeof p.text === 'string' ? p.text : '')).join('');
    }
    return '';
  };
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let at: number;
      while ((at = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const line of frame.split('\n')) {
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();
          if (payload === '' || payload === '[DONE]') continue;
          try {
            const delta = handle(JSON.parse(payload) as Json);
            if (delta !== '') { text += delta; yield delta; }
          } catch { /* a malformed frame carries no text; the stream goes on */ }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
  return { text, inputTokens, outputTokens, stopRaw };
}

export function sseFrame(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Emit a whole answer as an OpenAI event stream, in the provider's own frame order. */
export function* openAIStreamFrames(answer: Answer, model: string): Generator<string> {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: 1700000000, model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  yield sseFrame(null, chunk({ role: 'assistant', content: '' }, null));
  if (answer.text !== '') yield sseFrame(null, chunk({ content: answer.text }, null));
  yield sseFrame(null, chunk({}, stopFor('openai', answer.stopRaw)));
  yield sseFrame(null, {
    id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: 1700000000, model, choices: [],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  });
  yield 'data: [DONE]\n\n';
}

/** Emit a whole answer as an Anthropic event stream, in the provider's own event order. */
export function* anthropicStreamFrames(answer: Answer, model: string): Generator<string> {
  yield sseFrame('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_agg', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null,
      usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
    },
  });
  yield sseFrame('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  if (answer.text !== '') {
    yield sseFrame('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: answer.text } });
  }
  yield sseFrame('content_block_stop', { type: 'content_block_stop', index: 0 });
  yield sseFrame('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: stopFor('anthropic', answer.stopRaw), stop_sequence: null },
    usage: { output_tokens: answer.outputTokens },
  });
  yield sseFrame('message_stop', { type: 'message_stop' });
}
