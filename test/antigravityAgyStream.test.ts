import { assert } from "chai";
import {
  AgyTextAccumulator,
  buildAgyPrompt,
  describeAgyFailure,
  encodeAgyUserEvent,
  fingerprintSystemContext,
  isAgyAuthenticationFailure,
  parseAgyStreamLine,
  toUsageStats,
} from "../src/antigravity/agyStream";
import {
  buildAgyArguments,
  listAgyBinaryCandidates,
  parseAgyModelList,
} from "../src/antigravity/agyProcess";
import {
  ANTIGRAVITY_DEFAULT_MODEL_ID,
  buildAntigravityRuntimeModelEntries,
  loadAntigravityModelCatalog,
  getAntigravityModelCatalogSnapshot,
  resetAntigravityModelCatalogForTests,
  selectAntigravityRuntimeEntry,
  toAntigravityModelPref,
} from "../src/antigravity/modelCatalog";
import type { ChatMessage } from "../src/shared/llm";

describe("Antigravity agy headless protocol", function () {
  describe("parseAgyStreamLine", function () {
    it("reads init, answer deltas, tool steps and the result", function () {
      assert.deepEqual(
        parseAgyStreamLine(
          '{"event":"init","conversation_id":"c-1","model":"gemini"}',
        ),
        { kind: "init", conversationId: "c-1", model: "gemini" },
      );
      assert.deepEqual(
        parseAgyStreamLine(
          '{"event":"step_update","step_type":"agent_response","step_index":2,"state":"ACTIVE","text_delta":"Hel"}',
        ),
        { kind: "text", stepIndex: 2, state: "ACTIVE", delta: "Hel" },
      );
      assert.deepEqual(
        parseAgyStreamLine(
          '{"event":"step_update","step_update":{"step_type":"tool","step_index":3,"state":"ACTIVE","tool_name":"view_file","tool_info":{"summary":"Reading paper.md"}}}',
        ),
        {
          kind: "tool",
          stepIndex: 3,
          toolName: "view_file",
          state: "ACTIVE",
          summary: "Reading paper.md",
        },
      );
      const result = parseAgyStreamLine(
        '{"event":"result","result":{"conversation_id":"c-1","status":"SUCCESS","response":"Done","usage":{"input_tokens":10,"output_tokens":4},"denied_actions":[{"summary":"run ls"},"write file"]}}',
      );
      assert.deepEqual(result, {
        kind: "result",
        conversationId: "c-1",
        status: "SUCCESS",
        response: "Done",
        error: undefined,
        usage: {
          input_tokens: 10,
          output_tokens: 4,
          thinking_tokens: undefined,
          cache_read_tokens: undefined,
          total_tokens: undefined,
        },
        deniedActions: ["run ls", "write file"],
      });
    });

    it("ignores lines that are not protocol events", function () {
      assert.isNull(parseAgyStreamLine(""));
      assert.isNull(parseAgyStreamLine("Loading model…"));
      assert.isNull(parseAgyStreamLine("{not json"));
      assert.isNull(parseAgyStreamLine('{"event":"heartbeat"}'));
      assert.isNull(
        parseAgyStreamLine(
          '{"event":"step_update","step_type":"user_input","text_delta":"x"}',
        ),
      );
    });
  });

  describe("AgyTextAccumulator", function () {
    it("streams fragments, trims cumulative snapshots and skips DONE", function () {
      const acc = new AgyTextAccumulator();
      assert.equal(acc.push(1, "Hello"), "Hello");
      assert.equal(acc.push(1, "Hello, wor"), ", wor");
      assert.equal(acc.push(1, "ld"), "ld");
      assert.equal(acc.push(1, "Hello, world", "DONE"), "");
      assert.equal(acc.push(4, "After the tool."), "\n\nAfter the tool.");
      assert.equal(acc.text(), "Hello, world\n\nAfter the tool.");
    });
  });

  describe("buildAgyPrompt", function () {
    const messages: ChatMessage[] = [
      { role: "system", content: "You read papers." },
      { role: "system", content: "Paper text: ..." },
      { role: "user", content: "What is it about?" },
      { role: "assistant", content: "Diffusion." },
      {
        role: "user",
        content: [
          { type: "text", text: "And this figure?" },
          { type: "image_url", image_url: { url: "data:image/png;base64,x" } },
        ],
      } as ChatMessage,
    ];

    it("sends context, the transcript and the question to a fresh conversation", function () {
      const prompt = buildAgyPrompt({
        messages,
        resumed: false,
        includeSystem: true,
      });
      assert.include(
        prompt,
        "<instructions_and_context>\nYou read papers.\n\nPaper text: ...\n</instructions_and_context>",
      );
      assert.include(
        prompt,
        "<earlier_conversation>\nUser: What is it about?\n\nAssistant: Diffusion.\n</earlier_conversation>",
      );
      assert.include(
        prompt,
        "<question>\nAnd this figure?\n[An image was attached here.]\n</question>",
      );
    });

    it("sends only the question to a resumed conversation whose context is unchanged", function () {
      assert.equal(
        buildAgyPrompt({ messages, resumed: true, includeSystem: false }),
        "And this figure?\n[An image was attached here.]",
      );
      const updated = buildAgyPrompt({
        messages,
        resumed: true,
        includeSystem: true,
      });
      assert.match(updated, /^<updated_context>/);
      assert.notInclude(updated, "<earlier_conversation>");
    });

    it("fingerprints only the system context", function () {
      const a = fingerprintSystemContext(messages);
      const b = fingerprintSystemContext([
        ...messages,
        { role: "user", content: "more" },
      ]);
      const c = fingerprintSystemContext([
        { role: "system", content: "Other paper" },
      ]);
      assert.equal(a, b);
      assert.notEqual(a, c);
    });
  });

  it("encodes one stdin user event per turn", function () {
    const line = encodeAgyUserEvent('Say "hi"\nplease');
    assert.isTrue(line.endsWith("\n"));
    assert.deepEqual(JSON.parse(line), {
      event: "user",
      message: { content: 'Say "hi"\nplease' },
    });
  });

  it("reads AGY_ERROR lines and recognises a missing sign-in", function () {
    assert.equal(
      describeAgyFailure(
        'starting\nAGY_ERROR: {"status":"MODEL_ERROR","message":"quota exceeded"}\n',
      ),
      "quota exceeded",
    );
    assert.equal(describeAgyFailure("one\nlast line\n"), "last line");
    assert.isTrue(isAgyAuthenticationFailure("Please sign in to continue"));
    assert.isFalse(isAgyAuthenticationFailure("quota exceeded"));
  });

  it("maps agy usage to plugin usage stats", function () {
    assert.deepEqual(
      toUsageStats({
        input_tokens: 100,
        cache_read_tokens: 20,
        output_tokens: 30,
        thinking_tokens: 5,
      }),
      {
        promptTokens: 120,
        completionTokens: 35,
        totalTokens: 155,
        cacheReadTokens: 20,
      },
    );
    assert.isNull(toUsageStats(undefined));
    assert.isNull(toUsageStats({}));
  });
});

describe("Antigravity agy process helpers", function () {
  it("builds the headless arguments from the turn options", function () {
    assert.deepEqual(buildAgyArguments({}), [
      "--print",
      "--input-format",
      "stream-json",
      "--output-format",
      "stream-json",
    ]);
    assert.deepEqual(
      buildAgyArguments({
        model: " gemini-x ",
        effort: "high",
        mode: "plan",
        skipPermissions: true,
        conversationId: "c-9",
      }).slice(5),
      [
        "--model",
        "gemini-x",
        "--effort",
        "high",
        "--mode",
        "plan",
        "--dangerously-skip-permissions",
        "--conversation",
        "c-9",
      ],
    );
  });

  it("lists install locations per platform", function () {
    const linux = listAgyBinaryCandidates({
      platform: "linux",
      home: "/home/max",
    });
    assert.equal(linux[0], "/home/max/.local/bin/agy");
    assert.include(linux, "/usr/local/bin/agy");
    assert.include(
      listAgyBinaryCandidates({ platform: "macos", home: "/Users/max" }),
      "/opt/homebrew/bin/agy",
    );
    assert.isTrue(
      listAgyBinaryCandidates({
        platform: "windows",
        home: "C:\\Users\\max",
      }).every((path) => path.endsWith("agy.exe")),
    );
  });

  it("parses agy models output in its JSON shapes", function () {
    assert.deepEqual(parseAgyModelList('["a","b","a"]'), ["a", "b"]);
    assert.deepEqual(
      parseAgyModelList('{"models":[{"slug":"x"},{"id":"y"},{"name":"z"}]}'),
      ["x", "y", "z"],
    );
    assert.deepEqual(parseAgyModelList('{"slug":"p"}\n{"slug":"q"}\n'), [
      "p",
      "q",
    ]);
    assert.deepEqual(parseAgyModelList("not json"), []);
  });
});

describe("Antigravity model menu", function () {
  afterEach(function () {
    resetAntigravityModelCatalogForTests();
  });

  it("offers the CLI default first and keeps a saved model the catalog dropped", function () {
    const entries = buildAntigravityRuntimeModelEntries({
      models: ["gemini-a", "gemini-b"],
      selectedModel: "gemini-old",
    });
    assert.deepEqual(
      entries.map((entry) => entry.model),
      [ANTIGRAVITY_DEFAULT_MODEL_ID, "gemini-old", "gemini-a", "gemini-b"],
    );
    assert.isTrue(
      entries.every((entry) => entry.providerLabel === "Antigravity"),
    );
    assert.isTrue(entries.every((entry) => entry.apiBase.length > 0));
    assert.equal(
      selectAntigravityRuntimeEntry(entries, "").model,
      ANTIGRAVITY_DEFAULT_MODEL_ID,
    );
    assert.equal(
      selectAntigravityRuntimeEntry(entries, "gemini-b").model,
      "gemini-b",
    );
    assert.equal(toAntigravityModelPref(ANTIGRAVITY_DEFAULT_MODEL_ID), "");
    assert.equal(toAntigravityModelPref("gemini-a"), "gemini-a");
  });

  it("caches the model list and keeps the last one on failure", async function () {
    let calls = 0;
    const ok = async () => {
      calls += 1;
      return ["m1"];
    };
    assert.deepEqual(await loadAntigravityModelCatalog("", { list: ok }), [
      "m1",
    ]);
    assert.deepEqual(await loadAntigravityModelCatalog("", { list: ok }), [
      "m1",
    ]);
    assert.equal(calls, 1);
    const failing = async (): Promise<string[]> => {
      throw new Error("agy not found");
    };
    assert.deepEqual(
      await loadAntigravityModelCatalog("", { force: true, list: failing }),
      ["m1"],
    );
    const snapshot = getAntigravityModelCatalogSnapshot();
    assert.equal(snapshot.status, "error");
    assert.equal(snapshot.error, "agy not found");
    assert.deepEqual(snapshot.models, ["m1"]);
  });
});
