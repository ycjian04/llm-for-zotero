import type { ChatMessage, ModelTurnOutcome, UsageStats } from "../shared/llm";
import { getUserSkillsRuntimeRootDir } from "../agent/skills/userSkills";
import { withConversationWriteLock } from "../shared/conversationWriteFence";
import {
  getAntigravityConversationSummary,
  upsertAntigravityConversationSummary,
} from "./store";
import {
  AgyTextAccumulator,
  buildAgyPrompt,
  fingerprintSystemContext,
  toUsageStats,
} from "./agyStream";
import { runAgyTurn, type AgyTurnOptions } from "./agyProcess";
import {
  getAntigravityBinaryPathPref,
  getAntigravityEffortPref,
  getAntigravityPermissionModePref,
  getAntigravityRuntimeModelPref,
} from "./prefs";

/**
 * The system context last sent into each agy conversation, by conversation
 * key. In memory only: after a restart the context is sent once more, which
 * costs tokens but never loses it.
 */
const sentSystemContextByConversation = new Map<number, string>();

export function forgetAntigravitySentContext(conversationKey: number): void {
  sentSystemContextByConversation.delete(conversationKey);
}

export function resolveAntigravityRuntimeCwd(): string | undefined {
  try {
    // The same root Codex uses: agy also discovers `.agents/skills` there.
    return getUserSkillsRuntimeRootDir();
  } catch {
    return undefined;
  }
}

export type AntigravityPanelTurn = {
  conversationKey: number;
  messages: ChatMessage[];
  model?: string;
  signal?: AbortSignal;
  /**
   * Start a new agy conversation instead of resuming the stored one. A retry
   * or an edit rewrites history agy already holds, so it starts over with
   * the transcript the panel shows.
   */
  freshConversation?: boolean;
};

export type AntigravityPanelCallbacks = {
  isLive: () => boolean;
  onDelta: (delta: string) => void;
  onStatus: (text: string) => void;
  onUsage?: (usage: UsageStats) => void;
};

async function rememberConversationId(
  conversationKey: number,
  conversationId: string,
): Promise<void> {
  if (!conversationId) return;
  await withConversationWriteLock(conversationKey, async () => {
    const summary = await getAntigravityConversationSummary(conversationKey);
    if (!summary || summary.providerSessionId === conversationId) return;
    await upsertAntigravityConversationSummary({
      conversationKey: summary.conversationKey,
      instanceID: summary.instanceID,
      libraryID: summary.libraryID,
      kind: summary.kind,
      paperItemID: summary.paperItemID,
      createdAt: summary.createdAt,
      updatedAt: Date.now(),
      title: summary.title,
      providerSessionId: conversationId,
      scopedConversationKey: summary.scopedConversationKey,
      scopeType: summary.scopeType,
      scopeId: summary.scopeId,
      scopeLabel: summary.scopeLabel,
      cwd: summary.cwd,
      model: summary.model,
      effort: summary.effort,
    });
  });
}

function formatToolStatus(toolName: string, summary?: string): string {
  const label = toolName.replace(/_/g, " ").trim() || "a tool";
  return summary ? `Antigravity: ${summary}` : `Antigravity: using ${label}`;
}

/** One Antigravity turn from the panel, shared by the send and retry flows. */
export async function runAntigravityPanelTurn(
  turn: AntigravityPanelTurn,
  panel: AntigravityPanelCallbacks,
  runTurn: (
    options: AgyTurnOptions,
  ) => ReturnType<typeof runAgyTurn> = runAgyTurn,
): Promise<ModelTurnOutcome> {
  const summary = await getAntigravityConversationSummary(turn.conversationKey);
  const storedConversationId = turn.freshConversation
    ? ""
    : summary?.providerSessionId?.trim() || "";
  const resumed = Boolean(storedConversationId);
  const contextFingerprint = fingerprintSystemContext(turn.messages);
  const includeSystem =
    !resumed ||
    sentSystemContextByConversation.get(turn.conversationKey) !==
      contextFingerprint;
  const prompt = buildAgyPrompt({
    messages: turn.messages,
    resumed,
    includeSystem,
  });

  const permissionMode = getAntigravityPermissionModePref();
  const effort = getAntigravityEffortPref();
  const accumulator = new AgyTextAccumulator();
  panel.onStatus("Antigravity is thinking");
  const result = await runTurn({
    binaryPath: getAntigravityBinaryPathPref(),
    prompt,
    model: (turn.model ?? getAntigravityRuntimeModelPref()).trim() || undefined,
    effort: effort === "auto" ? undefined : effort,
    mode: permissionMode === "plan" ? "plan" : undefined,
    skipPermissions: permissionMode === "full",
    conversationId: storedConversationId || undefined,
    cwd: resolveAntigravityRuntimeCwd(),
    signal: turn.signal,
    onEvent: (event) => {
      if (!panel.isLive()) return;
      if (event.kind === "text") {
        const delta = accumulator.push(
          event.stepIndex,
          event.delta,
          event.state,
        );
        if (delta) panel.onDelta(delta);
      } else if (event.kind === "tool" && event.state !== "DONE") {
        panel.onStatus(formatToolStatus(event.toolName, event.summary));
      }
    },
  });

  sentSystemContextByConversation.set(turn.conversationKey, contextFingerprint);
  const conversationId = result.conversationId || storedConversationId;
  if (conversationId) {
    await rememberConversationId(turn.conversationKey, conversationId);
  }
  const usage = toUsageStats(result.usage);
  if (usage && panel.isLive()) panel.onUsage?.(usage);

  let text = result.response.trim() ? result.response : accumulator.text();
  if (result.deniedActions.length) {
    text +=
      "\n\n> Antigravity was not allowed to: " +
      result.deniedActions.join("; ") +
      ". Change the Antigravity permission mode in settings to allow it.";
  }
  return { text, completion: { status: "complete" } };
}
