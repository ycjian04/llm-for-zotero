import { CodexAppServerProcess } from "../utils/codexAppServerProcess";
import { pathExists } from "../utils/geckoFs";
import { getRuntimePlatformInfo } from "../utils/runtimePlatform";
import { appLogger } from "../core/logging";
import {
  describeAgyFailure,
  encodeAgyUserEvent,
  isAgyAuthenticationFailure,
  parseAgyStreamLine,
  type AgyEvent,
} from "./agyStream";

/**
 * Runs one turn of the Antigravity CLI in headless stream-json mode.
 *
 * Every turn is its own `agy` process: the conversation lives on agy's side
 * and is resumed with `--conversation <id>`, so nothing has to stay running
 * between turns. The prompt goes in as one stdin `user` event; stdin stays
 * open until the turn's `result` arrives, then closing it ends the session.
 *
 * Authentication is agy's own: it uses the Google sign-in saved by an
 * interactive `agy` run. The plugin never reads or forwards those credentials.
 */

export type AgyTurnOptions = {
  binaryPath?: string;
  prompt: string;
  model?: string;
  effort?: "low" | "medium" | "high";
  mode?: "plan";
  skipPermissions?: boolean;
  conversationId?: string;
  cwd?: string;
  signal?: AbortSignal;
  onEvent: (event: AgyEvent) => void;
};

export type AgyTurnResult = Extract<AgyEvent, { kind: "result" }>;

const AGY_ENV_KEYS = [
  "AGY_PATH",
  "HOME",
  "USERPROFILE",
  "LOCALAPPDATA",
  "PATH",
];

function getRuntimeEnvValue(key: string): string | undefined {
  const processValue = (globalThis as any).process?.env?.[key];
  if (typeof processValue === "string" && processValue.trim()) {
    return processValue.trim();
  }
  try {
    const servicesValue = (globalThis as any).Services?.env?.get?.(key);
    if (typeof servicesValue === "string" && servicesValue.trim()) {
      return servicesValue.trim();
    }
  } catch {
    /* ignore */
  }
  return undefined;
}

function getAgyRuntimeEnv(): Record<string, string | undefined> {
  const env: Record<string, string | undefined> = {};
  for (const key of AGY_ENV_KEYS) env[key] = getRuntimeEnvValue(key);
  return env;
}

function joinPath(separator: string, ...parts: string[]): string {
  return parts
    .map((part, index) =>
      index === 0
        ? part.replace(/[\\/]+$/, "")
        : part.replace(/^[\\/]+|[\\/]+$/g, ""),
    )
    .filter(Boolean)
    .join(separator);
}

function normalizeExplicitPath(value?: string | null): string {
  const trimmed = typeof value === "string" ? value.trim() : "";
  const quoted = trimmed.match(/^(['"])(.*)\1$/);
  return (quoted?.[2] ?? trimmed).trim();
}

export function listAgyBinaryCandidates(params: {
  platform: "windows" | "macos" | "linux";
  home?: string;
  localAppData?: string;
}): string[] {
  const { platform, home } = params;
  if (platform === "windows") {
    return [
      home ? joinPath("\\", home, ".local", "bin", "agy.exe") : "",
      home ? joinPath("\\", home, ".antigravity", "bin", "agy.exe") : "",
      params.localAppData
        ? joinPath("\\", params.localAppData, "Programs", "agy", "agy.exe")
        : "",
      params.localAppData
        ? joinPath("\\", params.localAppData, "agy", "agy.exe")
        : "",
    ].filter(Boolean);
  }
  return [
    home ? joinPath("/", home, ".local", "bin", "agy") : "",
    home ? joinPath("/", home, ".antigravity", "bin", "agy") : "",
    ...(platform === "macos" ? ["/opt/homebrew/bin/agy"] : []),
    "/usr/local/bin/agy",
    "/usr/bin/agy",
  ].filter(Boolean);
}

async function lookUpAgyOnShellPath(Subprocess: any): Promise<string> {
  const info = getRuntimePlatformInfo();
  try {
    const proc = await Subprocess.call({
      command: info.shellPath,
      arguments:
        info.platform === "windows"
          ? [info.shellFlag, "where agy"]
          : // A login shell reads the profile that usually puts ~/.local/bin
            // on PATH; Zotero launched from the dock does not inherit it.
            ["-lc", "command -v agy"],
    });
    let out = "";
    while (true) {
      const chunk = await proc.stdout?.readString?.();
      if (!chunk) break;
      out += chunk;
    }
    const exit = await proc.wait?.();
    if (typeof exit?.exitCode === "number" && exit.exitCode !== 0) return "";
    const first = out
      .split(/\r?\n/)
      .map((line) => line.trim())
      .find(Boolean);
    return first || "";
  } catch {
    return "";
  }
}

export function createAgyNotFoundError(): Error {
  return new Error(
    "Antigravity CLI (agy) was not found. Install it from https://antigravity.google/docs/cli/install, run `agy` once in a terminal to sign in, " +
      "or set the Antigravity CLI path in the plugin's Agent settings.",
  );
}

export async function resolveAgyBinary(explicitPath?: string): Promise<string> {
  const explicit = normalizeExplicitPath(explicitPath);
  if (explicit) return explicit;
  const env = getAgyRuntimeEnv();
  const fromEnv = normalizeExplicitPath(env.AGY_PATH);
  if (fromEnv) return fromEnv;
  const Subprocess = await CodexAppServerProcess.loadSubprocessModule();
  const found = await lookUpAgyOnShellPath(Subprocess);
  if (found) return found;
  const info = getRuntimePlatformInfo();
  for (const candidate of listAgyBinaryCandidates({
    platform: info.platform,
    home: env.HOME || env.USERPROFILE,
    localAppData: env.LOCALAPPDATA,
  })) {
    if (await pathExists(candidate)) return candidate;
  }
  throw createAgyNotFoundError();
}

export function buildAgyArguments(
  options: Pick<
    AgyTurnOptions,
    "model" | "effort" | "mode" | "skipPermissions" | "conversationId"
  >,
): string[] {
  // A valueless --print reads its prompts from stdin.
  const args = [
    "--print",
    "--input-format",
    "stream-json",
    "--output-format",
    "stream-json",
  ];
  if (options.model?.trim()) args.push("--model", options.model.trim());
  if (options.effort) args.push("--effort", options.effort);
  if (options.mode) args.push("--mode", options.mode);
  if (options.skipPermissions) args.push("--dangerously-skip-permissions");
  if (options.conversationId?.trim()) {
    args.push("--conversation", options.conversationId.trim());
  }
  return args;
}

function buildLaunchPath(binary: string): string {
  const info = getRuntimePlatformInfo();
  const separator = info.platform === "windows" ? ";" : ":";
  const env = getAgyRuntimeEnv();
  const home = env.HOME || env.USERPROFILE || "";
  const binaryDir = binary.replace(/[\\/][^\\/]*$/, "");
  const entries =
    info.platform === "windows"
      ? [binaryDir]
      : [
          binaryDir,
          home ? `${home}/.local/bin` : "",
          ...(info.platform === "macos" ? ["/opt/homebrew/bin"] : []),
          "/usr/local/bin",
          "/usr/bin",
          "/bin",
        ];
  const seen = new Set<string>();
  return [...entries, ...(env.PATH || "").split(separator)]
    .map((entry) => entry.trim())
    .filter((entry) => entry && !seen.has(entry) && seen.add(entry))
    .join(separator);
}

class AgyTurnError extends Error {
  constructor(
    message: string,
    readonly authRequired = false,
  ) {
    super(message);
    this.name = "AgyTurnError";
  }
}

export function isAgyAuthRequiredError(error: unknown): boolean {
  return error instanceof AgyTurnError && error.authRequired;
}

function createAbortError(): Error {
  const error = new Error("Aborted");
  error.name = "AbortError";
  return error;
}

export async function runAgyTurn(
  options: AgyTurnOptions,
): Promise<AgyTurnResult> {
  if (options.signal?.aborted) throw createAbortError();
  const Subprocess = await CodexAppServerProcess.loadSubprocessModule();
  const binary = await resolveAgyBinary(options.binaryPath);
  const args = buildAgyArguments(options);
  const isWindowsShim =
    getRuntimePlatformInfo().platform === "windows" &&
    !/\.(exe|com)$/i.test(binary);
  const info = getRuntimePlatformInfo();
  const command = isWindowsShim ? info.shellPath : binary;
  const commandArgs = isWindowsShim
    ? ["/d", "/s", "/c", [binary, ...args].map(quoteWindowsArg).join(" ")]
    : args;

  let proc: any;
  try {
    proc = await Subprocess.call({
      command,
      arguments: commandArgs,
      stderr: "pipe",
      environment: { PATH: buildLaunchPath(binary) },
      environmentAppend: true,
      ...(options.cwd ? { workdir: options.cwd } : {}),
    });
  } catch (error) {
    throw new AgyTurnError(
      `Failed to start Antigravity CLI (${binary}): ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }

  let stderr = "";
  let stdinClosed = false;
  const closeStdin = async () => {
    if (stdinClosed) return;
    stdinClosed = true;
    try {
      await proc.stdin?.close?.();
    } catch {
      /* already closed */
    }
  };
  const onAbort = () => {
    try {
      proc.kill?.();
    } catch {
      /* already exited */
    }
  };
  options.signal?.addEventListener("abort", onAbort, { once: true });

  const stderrLoop = (async () => {
    while (true) {
      let chunk = "";
      try {
        chunk = await proc.stderr?.readString?.();
      } catch {
        break;
      }
      if (!chunk) break;
      stderr = (stderr + chunk).slice(-8000);
    }
  })();

  try {
    await proc.stdin.write(encodeAgyUserEvent(options.prompt));
    let buffer = "";
    let result: AgyTurnResult | null = null;
    while (!result) {
      let chunk = "";
      try {
        chunk = await proc.stdout.readString();
      } catch {
        break;
      }
      if (!chunk) break;
      buffer += chunk;
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        const event = parseAgyStreamLine(line);
        if (!event) continue;
        options.onEvent(event);
        if (event.kind === "result") {
          result = event;
          break;
        }
      }
    }
    await closeStdin();
    const exit = await proc.wait?.().catch(() => undefined);
    await stderrLoop;
    if (options.signal?.aborted) throw createAbortError();
    if (result) {
      if (/^(SUCCESS|OK|COMPLETED?)$/i.test(result.status) || result.response) {
        return result;
      }
      const message =
        result.error || describeAgyFailure(stderr) || result.status;
      throw new AgyTurnError(
        `Antigravity CLI failed: ${message}`,
        isAgyAuthenticationFailure(message),
      );
    }
    const message =
      describeAgyFailure(stderr) ||
      `agy exited with code ${String(exit?.exitCode ?? "unknown")}`;
    appLogger.warn("Antigravity CLI turn ended without a result", {
      exitCode: exit?.exitCode,
      stderr: stderr.slice(-2000),
    });
    throw new AgyTurnError(
      isAgyAuthenticationFailure(message)
        ? "Antigravity CLI is not signed in. Run `agy` once in a terminal and sign in with your Google account, then try again."
        : `Antigravity CLI failed: ${message}`,
      isAgyAuthenticationFailure(message),
    );
  } finally {
    options.signal?.removeEventListener("abort", onAbort);
    await closeStdin();
  }
}

function quoteWindowsArg(value: string): string {
  return /[\s"]/.test(value) ? `"${value.replace(/"/g, '\\"')}"` : value;
}

/** `agy models --output-format json`: the model slugs this account can use. */
export async function listAgyModels(binaryPath?: string): Promise<string[]> {
  const Subprocess = await CodexAppServerProcess.loadSubprocessModule();
  const binary = await resolveAgyBinary(binaryPath);
  const proc = await Subprocess.call({
    command: binary,
    arguments: ["models", "--output-format", "json"],
    stderr: "pipe",
    environment: { PATH: buildLaunchPath(binary) },
    environmentAppend: true,
  });
  let out = "";
  while (true) {
    const chunk = await proc.stdout?.readString?.();
    if (!chunk) break;
    out += chunk;
  }
  await proc.wait?.();
  return parseAgyModelList(out);
}

export function parseAgyModelList(raw: string): string[] {
  const slugs: string[] = [];
  const add = (value: unknown) => {
    if (typeof value === "string" && value.trim()) slugs.push(value.trim());
    else if (value && typeof value === "object") {
      const record = value as Record<string, unknown>;
      add(record.slug ?? record.id ?? record.name ?? record.model);
    }
  };
  const visit = (parsed: unknown) => {
    if (Array.isArray(parsed)) parsed.forEach(add);
    else if (parsed && typeof parsed === "object") {
      const record = parsed as Record<string, unknown>;
      if (Array.isArray(record.models)) record.models.forEach(add);
      else add(record);
    }
  };
  const trimmed = raw.trim();
  try {
    visit(JSON.parse(trimmed));
  } catch {
    // stream-json style: one object per line.
    for (const line of trimmed.split(/\r?\n/)) {
      try {
        visit(JSON.parse(line));
      } catch {
        /* not JSON */
      }
    }
  }
  return [...new Set(slugs)];
}
