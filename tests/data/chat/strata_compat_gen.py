"""Record Strata's own answers to the custom-tool cases, for Radiance's compatibility test.

Run inside a Strata checkout with its venv (no GPU, no model: Strata's MockEngine and real HTTP server):
    .venv/bin/python -I strata_compat_gen.py OUT.json
For each case it records, raw: request_tools() (the tools offered and the name map), input_messages() (the chat
conversation), and the HTTP /v1/responses answer to the case's scripted model reply, not streamed and streamed (every
SSE event). Radiance's tests/custom_tools_test.cpp runs the same cases and compares with one normaliser.

A script may name a namespaced FUNCTION as @NSFN:<namespace>:<name>@: Strata offers it as <namespace>.<name>,
Radiance as <namespace>__<name> (its existing naming, a named deviation). Everything else is the same text for both.
"""
import contextlib, hashlib, http.client, io, json, subprocess, sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from serve import responses  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.responses import input_messages, request_tools  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

TEMPLATE = ChatTemplate(ROOT / "serve/chat_template.jinja")
PATCH = {"type": "custom", "name": "apply_patch", "description": "Edit files."}
LARK = {"type": "custom", "name": "apply_patch", "description": "Patch.",
        "format": {"type": "grammar", "syntax": "lark", "definition": "start: \"x\""}}
TOOLS = [{"type": "function", "name": "exec_command", "description": "Runs a command.", "strict": False,
          "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}]
AGENTS = {"type": "namespace", "name": "multi_agent_v1", "description": "Agents.", "tools": [
    {"type": "function", "name": "spawn_agent", "strict": False,
     "parameters": {"type": "object", "properties": {"message": {"type": "string"}}}}]}
NS_CUSTOM = {"type": "namespace", "name": "crm", "description": "CRM.", "tools": [
    {"type": "custom", "name": "raw", "description": "Raw.",
     "format": {"type": "grammar", "syntax": "regex", "definition": "^q+$"}}]}


def call(fn, param, value):
    return f"<tool_call>\n<function={fn}>\n<parameter={param}>\n{value}\n</parameter>\n</function>\n</tool_call>\n"


LONG = "*** Begin Patch\n*** Add File: big.txt\n" + "".join(f"+line {i}\tend\n" for i in range(1500)) + "*** End Patch"
INPUTS = {
    "multiline": "*** Begin Patch\n*** Update File: a.py\n@@ def f():\n-    return 1\n+    return 2\n*** End Patch",
    "tabs": "a\tb\t\tc", "quotes": "say \"hi\" and 'bye'", "backslashes": "C:\\path\\to\\file and \\n literal",
    "unicode": "caf\u00e9 \u2192 \U0001F680 \u6f22\u5b57", "json_object": "{\"input\": \"nested\", \"n\": 1}",
    "json_array": "[1, 2, 3]", "empty": "", "long": LONG,
}

CASES = []
# Strata's Parsing.test_tools_namespaces_custom_and_hosted (offer only)
CASES.append({"name": "strata_tools_namespaces_custom_and_hosted", "row": "A2 A5 E2", "reply": None, "request": {
    "input": "x", "tools": [*TOOLS, {"type": "namespace", "name": "mcp_fs", "description": "Files.",
                                     "tools": [{"type": "function", "name": "read", "parameters": {"type": "object"}}]},
                            LARK, {"type": "web_search"}]}})
CASES.append({"name": "tool_choice_none", "row": "A6", "reply": None,
              "request": {"input": "x", "tools": [*TOOLS, PATCH], "tool_choice": "none"}})
CASES.append({"name": "custom_without_description_with_grammar", "row": "A1 A2", "reply": None, "request": {
    "input": "x", "tools": [{"type": "custom", "name": "sql",
                             "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"}}]}})
CASES.append({"name": "namespaced_custom_offer", "row": "A5", "reply": None,
              "request": {"input": "x", "tools": [NS_CUSTOM, PATCH]}})
# Strata's ToolRoundTrip.test_namespace_and_custom_calls_come_back_under_their_own_names
CASES.append({"name": "strata_namespace_and_custom_calls_round_trip", "row": "B1 B4 D2", "request": {
    "model": "m", "input": "go", "max_output_tokens": 40000, "tools": [AGENTS, PATCH]},
    "reply": "</think>\n\n" + call("@NSFN:multi_agent_v1:spawn_agent@", "message", "hi") +
             call("apply_patch", "input", "*** Begin Patch"),
    "history_from_reply": True})
for key, value in INPUTS.items():
    CASES.append({"name": f"input_bytes_{key}", "row": "B2 C2", "reply": "</think>\n\n" + call("apply_patch", "input", value),
                  "request": {"model": "m", "input": "edit", "max_output_tokens": 40000, "tools": [PATCH]}})
CASES.append({"name": "text_then_custom_call", "row": "B5", "reply": "</think>\n\nEditing now.\n" + call("apply_patch", "input", "p"),
              "request": {"model": "m", "input": "go", "max_output_tokens": 40000, "tools": [PATCH]}})
CASES.append({"name": "reasoning_function_and_custom", "row": "C4",
              "reply": "Plan the edit.\n</think>\n\n" + call("exec_command", "cmd", "ls") + call("apply_patch", "input", "p"),
              "request": {"model": "m", "input": "go", "max_output_tokens": 40000, "tools": [*TOOLS, PATCH]}})
CASES.append({"name": "custom_and_custom", "row": "B4",
              "reply": "</think>\n\n" + call("apply_patch", "input", "one") + call("apply_patch", "input", "two"),
              "request": {"model": "m", "input": "go", "max_output_tokens": 40000, "tools": [PATCH]}})
CASES.append({"name": "custom_call_with_another_parameter", "row": "B2",
              "reply": "</think>\n\n" + call("apply_patch", "patch", "x"),
              "request": {"model": "m", "input": "go", "max_output_tokens": 40000, "tools": [PATCH]}})
CASES.append({"name": "namespaced_custom_call", "row": "B1",
              "reply": "</think>\n\n" + call("crm.raw", "input", "qq"),
              "request": {"model": "m", "input": "go", "max_output_tokens": 40000, "tools": [NS_CUSTOM]}})
CASES.append({"name": "tool_choice_custom_form", "row": "A6", "reply": "</think>\n\n" + call("apply_patch", "input", "p"),
              "request": {"model": "m", "input": "go", "max_output_tokens": 40000, "tools": [PATCH],
                          "tool_choice": {"type": "custom", "name": "apply_patch"}}})
CASES.append({"name": "custom_call_cut_short", "row": "C (cut)",
              "reply": "</think>\n\n<tool_call>\n<function=apply_patch>\n<parameter=input>\n*** Begin Patch\n*** Add",
              "request": {"model": "m", "input": "go", "max_output_tokens": 40000, "tools": [PATCH]}})
# history only (D rows Strata covers)
CASES.append({"name": "history_custom_pairs", "row": "D1 D2 D3", "reply": None, "request": {"input": [
    {"role": "user", "content": "go"},
    {"type": "custom_tool_call", "call_id": "c1", "name": "apply_patch", "input": "*** Begin Patch\n*** End Patch"},
    {"type": "custom_tool_call_output", "call_id": "c1", "output": "done"},
    {"type": "function_call", "call_id": "c2", "name": "exec_command", "arguments": "{\"cmd\": \"ls\"}"},
    {"type": "function_call_output", "call_id": "c2", "output": "a b"},
    {"type": "custom_tool_call", "call_id": "c3", "namespace": "crm", "name": "raw", "input": "q"},
    {"type": "custom_tool_call_output", "call_id": "c3", "output": [{"type": "input_text", "text": "ok"}]},
    {"type": "custom_tool_call", "call_id": "c4", "name": "apply_patch", "input": {"not": "a string"}},
    {"type": "custom_tool_call_output", "call_id": "c4", "output": "x"},
    {"role": "user", "content": "and now?"}], "tools": [*TOOLS, PATCH, NS_CUSTOM]}})


def strata_script(text):
    import re
    return re.sub(r"@NSFN:([^:@]+):([^@]+)@", r"\1.\2", text)


def post(port, body):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    with contextlib.redirect_stdout(io.StringIO()):
        c.request("POST", "/v1/responses", body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        raw = r.read().decode()
    c.close()
    return r.status, raw


def jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: repr(o)))


tok = ByteTokenizer()
engine = MockEngine(tok, "x", max_context=1 << 20)
httpd = serve(Service(engine, tok, TEMPLATE), port=0)
port = httpd.server_address[1]
out = []
for case in CASES:
    rec = {k: v for k, v in case.items() if k != "history_from_reply"}
    req = case["request"]
    try:
        tools, names, skipped = request_tools(req)
        rec["strata_tools"] = jsonable({"tools": tools, "names": {k: list(v) for k, v in names.items()}, "skipped": skipped})
    except Exception as e:  # noqa: BLE001
        rec["strata_tools"] = {"error": str(e)}
    try:
        rec["strata_messages"] = jsonable(input_messages(req))
    except Exception as e:  # noqa: BLE001
        rec["strata_messages"] = {"error": str(e)}
    if case["reply"] is not None:
        script = tok.encode(strata_script(case["reply"]) + "<|im_end|>", parse_special=True)
        for stream in (False, True):
            engine.scripts = [script]
            engine.script = script
            status, raw = post(port, {**req, "stream": stream})
            rec["strata_stream" if stream else "strata_body"] = {"status": status, "raw": raw}
        if case.get("history_from_reply"):
            body = json.loads(rec["strata_body"]["raw"])
            ns_item, custom_item = body["output"]
            hist = {"input": [{"role": "user", "content": "go"}, ns_item, custom_item,
                              {"type": "function_call_output", "call_id": ns_item["call_id"], "output": "1"},
                              {"type": "custom_tool_call_output", "call_id": custom_item["call_id"], "output": "2"}]}
            rec["history_request"] = hist
            rec["strata_history_messages"] = jsonable(input_messages(hist))
    out.append(rec)
httpd.shutdown()
httpd.server_close()

src = (ROOT / "serve/responses.py").read_bytes()
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
doc = {"source": "Strata serve/responses.py (MIT, Copyright (c) 2026 Niko1221 and the Strata contributors)",
       "strata_commit": head, "responses_py_sha256": hashlib.sha256(src).hexdigest(),
       "generator": "tests/data/chat/strata_compat_gen.py (Strata's MockEngine + HTTP server, no GPU)", "cases": out}
Path(sys.argv[1]).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
print(f"{len(out)} cases, strata {head[:10]}, responses.py {doc['responses_py_sha256'][:12]}")
