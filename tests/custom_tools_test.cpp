/* tests/custom_tools_test.cpp -- the Responses API's freeform custom tools (oai_custom_tools.h),
 * driven through responses_to_chat and the HTTP layer as server_test.cpp does.
 *
 * Ported from Strata's serve/test_responses.py custom cases (MIT, Copyright (c) 2026 Niko1221 and
 * the Strata contributors) and widened to a tool-call test matrix (rows A-F):
 *   A offer     A1 a_custom_tool_is_offered_as_a_function_with_one_string_input
 *               A2 a_grammar_format_is_shown_to_the_model_as_text
 *               A3 codexs_own_tools_are_offered_with_apply_patch_as_a_custom_tool (+ the real template)
 *               A4 several_custom_tools_are_offered_together
 *               A5 custom_plain_and_namespaced_tools_in_one_request
 *                  a_namespaced_custom_tool_is_served_as_strata_serves_it (offer, reply, stream,
 *                  history)
 *   Strata      the_same_requests_answer_as_strata_does (every Strata custom case and the
 *               follow-ups: result order, tool_choice, function offers, additional_tools,
 *               Codex's local compaction; the named deviations normalised)
 *               A6 tool_choice_auto_required_none_and_the_custom_form
 *               A7 parallel_tool_calls_passes_through_both_ways
 *   B reply     B1 a_custom_call_comes_back_as_a_custom_tool_call_item
 *               B2 a_custom_calls_input_is_kept_byte_for_byte (fake parser and real Qwen parser)
 *               B3 a_call_is_answered_as_the_kind_of_tool_it_names (routing; the model's choice is live)
 *               B4 several_calls_in_one_turn_keep_their_kinds_and_order
 *               B5 text_and_a_custom_call_in_one_reply
 *   C stream    C1-C3 a_streamed_custom_call_sends_its_events_in_order
 *               C2    the_streamed_input_is_the_same_bytes_everywhere
 *               C4    a_stream_with_reasoning_a_function_call_and_a_custom_call
 *               C5    function_calls_stream_as_in_1_17_0 (golden)
 *               (cut) a_custom_call_cut_short_is_incomplete_with_no_input
 *   D history   D1 custom_call_history_maps_onto_the_chat_conversation
 *               D2 custom_and_function_history_interleave
 *               D3 several_round_trips_of_history
 *               D4 unmatched_and_repeated_call_ids_are_passed_through
 *                  each_turns_results_are_put_in_its_calls_order (Strata's result order)
 *                  codexs_local_compaction_turn_is_served (codex 0.167, a custom provider)
 *               D5 previous_response_id_is_still_refused
 *   E errors    E1 a_custom_tool_needs_a_name
 *               E2 other_tool_types_are_still_left_out
 *               E3 tool_choice_naming_no_offered_tool (acts as "auto", as in Strata)
 *               E4 a_malformed_custom_tool_call_item_is_refused
 *   F regress   F1/F2 function_and_chat_tool_replies_are_as_in_1_17_0 (golden from 19d4e8fe0e18)
 */
#include "server_fakes.h"

#include "server/oai_custom_tools.h"
#include "text/chat.h"

#include <fstream>
#include <regex>
#include <sstream>

namespace {

struct Conv {
    int rc = 0;
    ApiError err;
    json chat, echo;
    std::map<std::string, std::pair<std::string, std::string>> ns;
    std::set<std::string> custom;
};

Conv convert(const std::string& body) {
    Conv c;
    OaiLimits lim;
    std::string chat, echo;
    c.rc = responses_to_chat(body, lim, &chat, &echo, &c.err, &c.ns, &c.custom);
    if (c.rc == RAD_OK) {
        c.chat = json::parse(chat);
        c.echo = json::parse(echo);
    }
    return c;
}

std::string read_fixture(const std::string& name) {
    std::ifstream f(std::string(RAD_CHAT_FIXTURES) + "/" + name);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

/* The chat tool named `name` in a translated request, or null. */
json chat_tool(const json& chat, const std::string& name) {
    if (chat.contains("tools"))
        for (const json& t : chat["tools"])
            if (t["function"].value("name", "") == name) return t;
    return json();
}

/* One Responses stream event. */
struct Ev {
    std::string type;
    json data;
};
std::vector<Ev> events(const std::string& body) {
    std::vector<Ev> out;
    size_t at = 0;
    while ((at = body.find("event: ", at)) != std::string::npos) {
        const size_t eol = body.find('\n', at);
        const size_t d = body.find("data: ", eol);
        const size_t end = body.find("\n\n", d);
        out.push_back({ body.substr(at + 7, eol - at - 7), json::parse(body.substr(d + 6, end - d - 6)) });
        at = end;
    }
    return out;
}
std::vector<std::string> types_of(const std::vector<Ev>& ev) {
    std::vector<std::string> t;
    for (const Ev& e : ev) t.push_back(e.type);
    return t;
}

/* A Qwen-format call, as the model writes it: the value between one newline at each end. */
std::string qwen_call(const std::string& fn, const std::string& param, const std::string& value) {
    return "<tool_call>\n<function=" + fn + ">\n<parameter=" + param + ">\n" + value +
           "\n</parameter>\n</function>\n</tool_call>\n";
}

/* A fixture whose replies the real Qwen parser reads, for the tools the Responses body offers. */
void use_qwen(Fixture& f, const std::string& body) {
    const Conv c = convert(body);
    REQUIRE(c.rc == RAD_OK);
    f.tmpl.real_parser = qwen_reply_parser(c.chat.contains("tools") ? c.chat["tools"] : json::array());
    REQUIRE(f.tmpl.real_parser != nullptr);
}

const char* kPatchTool = R"({"type":"custom","name":"apply_patch","description":"Edit files."})";

/* The same normaliser the goldens were recorded with (ids and clocks only). */
std::string normalize(std::string s) {
    /* id VALUES only: a key such as "call_id" is kept, so a renamed or dropped field shows */
    static const std::regex id(R"re("(resp|fc|call|msg|rs|ctc|chatcmpl|cmpl)_[A-Za-z0-9]+"(?!:))re");
    s = std::regex_replace(s, id, "\"$1_ID\"");
    static const std::regex cid(R"re(\bchatcmpl-[A-Za-z0-9]+)re");
    s = std::regex_replace(s, cid, "chatcmpl-ID");
    static const std::regex clock(R"re("(created_at|created)":[0-9]+)re");
    s = std::regex_replace(s, clock, "\"$1\":0");
    static const std::regex timings(R"re("timings":\{[^}]*\})re");
    s = std::regex_replace(s, timings, "\"timings\":{}");
    return s;
}
/* A /v1/chat/completions stream sends a call's arguments in the chunks the engine's timing makes
 * ("{" then the rest, or all at once: the fake engine's pace varies under load), so the goldens
 * compare a stream with each run of argument chunks of one call joined into one chunk: the bytes
 * and their order still count, the chunk boundaries do not. Other events are compared as sent. */
std::string join_argument_chunks(const std::string& s) {
    std::string out;
    json held;   /* the argument chunk being joined, or null */
    auto args_of = [](const json& c) -> const json* {
        if (!c.is_object() || !c.contains("choices") || c["choices"].size() != 1) return nullptr;
        const json& d = c["choices"][0]["delta"];
        if (!d.is_object() || d.size() != 1 || !d.contains("tool_calls") || d["tool_calls"].size() != 1) return nullptr;
        const json& t = d["tool_calls"][0];
        if (t.size() != 2 || !t.contains("index") || !t.contains("function") || t["function"].size() != 1 ||
            !t["function"].contains("arguments"))
            return nullptr;
        return &t;
    };
    auto flush = [&]() {
        if (!held.is_null()) out += "data: " + held.dump() + "\n\n";
        held = json();
    };
    size_t at = 0;
    while (at < s.size()) {
        size_t end = s.find("\n\n", at);
        const std::string block = s.substr(at, end == std::string::npos ? std::string::npos : end - at);
        at = end == std::string::npos ? s.size() : end + 2;
        const json c = block.rfind("data: {", 0) == 0 ? json::parse(block.substr(6), nullptr, false) : json();
        const json* t = args_of(c);
        if (t && !held.is_null() && (*args_of(held))["index"] == (*t)["index"]) {
            json& a = held["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"];
            a = a.get<std::string>() + (*t)["function"]["arguments"].get<std::string>();
            continue;
        }
        flush();
        if (t) { held = c; continue; }
        out += block + (end == std::string::npos ? "" : "\n\n");
    }
    flush();
    return out;
}

std::map<std::string, std::string> goldens() {
    std::map<std::string, std::string> g;
    const std::string t = read_fixture("responses-function-golden-1.17.0.txt");
    static const std::regex block(R"re(@@GOLDEN (\S+)@@\n([\s\S]*?)\n@@END@@)re");
    for (auto it = std::sregex_iterator(t.begin(), t.end(), block); it != std::sregex_iterator(); ++it)
        g[(*it)[1]] = (*it)[2];
    return g;
}

}  // namespace

/* ================================================================== A. offer */

TEST(a_custom_tool_is_offered_as_a_function_with_one_string_input) {
    const Conv c = convert(R"({"input":"x","tools":[{"type":"custom","name":"note","description":"Write a note."}]})");
    CHECK_OK(c.rc);
    const json t = chat_tool(c.chat, "note");
    REQUIRE(t.is_object());
    CHECK_EQ(t.value("type", ""), std::string("function"));
    CHECK_EQ(t["function"].value("description", ""), std::string("Write a note."));
    CHECK_EQ(t["function"]["parameters"].dump(),
             json::parse(R"({"type":"object","properties":{"input":{"type":"string","description":"the raw input"}},
                            "required":["input"]})").dump());
    CHECK_EQ(c.custom.size(), (size_t)1);
    CHECK(c.custom.count("note") == 1);
    CHECK(c.ns.empty());
    /* echoed as sent, as the function tools are */
    CHECK_EQ(c.echo["tools"].size(), (size_t)1);
    CHECK_EQ(c.echo["tools"][0].value("type", ""), std::string("custom"));
    /* no description: none, and still the one string input */
    const Conv bare = convert(R"({"input":"x","tools":[{"type":"custom","name":"note"}]})");
    CHECK_OK(bare.rc);
    CHECK_EQ(chat_tool(bare.chat, "note")["function"].value("description", "?"), std::string(""));
}

TEST(a_grammar_format_is_shown_to_the_model_as_text) {
    /* Strata's wording: the description, a blank line, then the grammar under its syntax's name. */
    const Conv lark = convert(R"({"input":"x","tools":[{"type":"custom","name":"apply_patch","description":"Patch.",
        "format":{"type":"grammar","syntax":"lark","definition":"start: \"x\""}}]})");
    CHECK_OK(lark.rc);
    CHECK_EQ(chat_tool(lark.chat, "apply_patch")["function"].value("description", ""),
             std::string("Patch.\n\nThe input must follow this lark grammar:\nstart: \"x\""));
    const Conv regex = convert(R"({"input":"x","tools":[{"type":"custom","name":"id","description":"An id.",
        "format":{"type":"grammar","syntax":"regex","definition":"^[a-z]{3}\\d+$"}}]})");
    CHECK_OK(regex.rc);
    CHECK_EQ(chat_tool(regex.chat, "id")["function"].value("description", ""),
             std::string("An id.\n\nThe input must follow this regex grammar:\n^[a-z]{3}\\d+$"));
    /* a text format, or a grammar with no definition, adds nothing */
    for (const char* fmt : { R"({"type":"text"})", R"({"type":"grammar","syntax":"lark","definition":""})",
                             R"({"type":"grammar","syntax":"lark"})" }) {
        const Conv c = convert(std::string(R"({"input":"x","tools":[{"type":"custom","name":"t","description":"D.","format":)") +
                               fmt + "}]}");
        CHECK_OK(c.rc);
        CHECK_EQ(chat_tool(c.chat, "t")["function"].value("description", ""), std::string("D."));
    }
}

/* Codex 0.167.0's own tool list (tests/data/chat), captured from a throwaway Codex session against
 * a recording server: nine function tools, a namespace, web search, and apply_patch as a custom
 * tool with a lark grammar. */
TEST(codexs_own_tools_are_offered_with_apply_patch_as_a_custom_tool) {
    const json tools = json::parse(read_fixture("codex-0.167.0-tools.json"));
    REQUIRE(tools.is_array() && tools.size() == 10);
    json patch;
    for (const json& t : tools)
        if (t.value("type", "") == "custom") patch = t;
    REQUIRE(patch.value("name", "") == "apply_patch");
    const std::string definition = patch["format"]["definition"].get<std::string>();
    CHECK(has(definition, "*** Begin Patch"));

    const Conv c = convert(json{ { "input", "x" }, { "tools", tools } }.dump());
    CHECK_OK(c.rc);
    std::vector<std::string> names;
    for (const json& t : c.chat["tools"]) names.push_back(t["function"]["name"].get<std::string>());
    CHECK(std::find(names.begin(), names.end(), "apply_patch") != names.end());
    CHECK(std::find(names.begin(), names.end(), "exec_command") != names.end());
    CHECK(std::find(names.begin(), names.end(), "web_search") == names.end());   /* hosted: left out */
    for (const std::string& n : names)
        if (n.rfind("multi_agent_v1__", 0) == 0) CHECK(c.ns.count(n) == 1);
    CHECK_EQ(c.custom.size(), (size_t)1);
    CHECK(c.custom.count("apply_patch") == 1);
    CHECK_EQ(chat_tool(c.chat, "apply_patch")["function"].value("description", ""),
             patch["description"].get<std::string>() + "\n\nThe input must follow this lark grammar:\n" + definition);
    /* the function tools are offered exactly as before: their schemas untouched */
    for (const json& t : tools)
        if (t.value("type", "") == "function")
            CHECK_EQ(chat_tool(c.chat, t["name"].get<std::string>())["function"]["parameters"].dump(),
                     t["parameters"].dump());

    /* The production model's own template (Qwen3.8) renders the request: apply_patch and its
     * grammar reach the prompt, under its one string parameter. */
    ChatTemplate tmpl;
    const std::string src = read_fixture("qwen3.8/chat_template.jinja");
    if (src.empty()) { ::radtest::artifact_missing(__FILE__, __LINE__, "qwen3.8/chat_template.jinja"); return; }
    REQUIRE(tmpl.load(src, "", "") == RAD_OK);
    ChatOptions opt;
    opt.add_generation_prompt = true;
    ChatPrompt p;
    CHECK_OK(tmpl.apply(nlohmann::ordered_json::parse(c.chat["messages"].dump()),
                        nlohmann::ordered_json::parse(c.chat["tools"].dump()), opt, &p));
    CHECK(has(p.prompt, "apply_patch"));
    CHECK(has(p.prompt, "*** Begin Patch"));
    CHECK(has(p.prompt, "The input must follow this lark grammar"));
    CHECK(has(p.prompt, "\"input\""));
}

TEST(several_custom_tools_are_offered_together) {
    const Conv c = convert(R"({"input":"x","tools":[{"type":"custom","name":"apply_patch"},
        {"type":"custom","name":"sql","format":{"type":"grammar","syntax":"lark","definition":"start: /.+/"}},
        {"type":"custom","name":"note"}]})");
    CHECK_OK(c.rc);
    CHECK_EQ(c.chat["tools"].size(), (size_t)3);
    CHECK_EQ(c.custom.size(), (size_t)3);
    for (const char* n : { "apply_patch", "sql", "note" }) CHECK(c.custom.count(n) == 1);
    /* two custom tools may not share a name, as function tools may not */
    CHECK(convert(R"({"input":"x","tools":[{"type":"custom","name":"a"},{"type":"custom","name":"a"}]})").rc < 0);
    CHECK(convert(R"({"input":"x","tools":[{"type":"custom","name":"a"},{"type":"function","name":"a"}]})").rc < 0);
}

TEST(custom_plain_and_namespaced_tools_in_one_request) {
    const Conv c = convert(R"({"input":"x","tools":[
        {"type":"function","name":"plain","parameters":{"type":"object"}},
        {"type":"namespace","name":"crm","description":"d","tools":[
            {"type":"function","name":"find","parameters":{"type":"object"}},
            {"type":"custom","name":"raw","description":"Raw."}]},
        {"type":"custom","name":"apply_patch"}]})");
    CHECK_OK(c.rc);
    std::vector<std::string> names;
    for (const json& t : c.chat["tools"]) names.push_back(t["function"]["name"].get<std::string>());
    CHECK_EQ(json(names).dump(), json::parse(R"(["plain","crm__find","crm.raw","apply_patch"])").dump());
    CHECK_EQ(c.custom.size(), (size_t)2);
    CHECK(c.custom.count("crm.raw") == 1 && c.custom.count("apply_patch") == 1);
    CHECK(c.custom.count("plain") == 0 && c.custom.count("crm__find") == 0);
    CHECK_EQ(c.ns.size(), (size_t)2);
    CHECK_EQ(c.ns.at("crm.raw").second, std::string("raw"));
    CHECK_EQ(chat_tool(c.chat, "plain")["function"]["parameters"].dump(), std::string("{\"type\":\"object\"}"));
}

/* A custom tool inside a namespace, as Strata serves it: offered as <namespace>.<name> with the
 * namespace's description leading its own, its reply item (streamed and not) carrying namespace and
 * name apart, and its history item joined back as <namespace>.<name>. A namespaced FUNCTION beside
 * it keeps Radiance's <namespace>__<name> (deviation NS-FN). */
TEST(a_namespaced_custom_tool_is_served_as_strata_serves_it) {
    const std::string tools = R"("tools":[{"type":"namespace","name":"crm","description":"CRM.","tools":[
        {"type":"function","name":"find","parameters":{"type":"object"}},
        {"type":"custom","name":"raw","description":"Raw."}]}],)";
    /* offer */
    const Conv c = convert("{" + tools + R"("input":"x"})");
    CHECK_OK(c.rc);
    CHECK(chat_tool(c.chat, "crm__find").is_object());
    const json raw = chat_tool(c.chat, "crm.raw");
    REQUIRE(raw.is_object());
    CHECK_EQ(raw["function"].value("description", ""), std::string("CRM.\n\nRaw."));
    CHECK_EQ(c.ns.at("crm.raw").first, std::string("crm"));
    CHECK_EQ(c.ns.at("crm.raw").second, std::string("raw"));
    CHECK_EQ(c.ns.at("crm__find").second, std::string("find"));
    CHECK(c.custom.count("crm.raw") == 1 && c.custom.count("crm__find") == 0);
    /* a namespaced custom tool with no description of its own: the namespace's alone */
    const Conv bare = convert(R"({"input":"x","tools":[{"type":"namespace","name":"crm","description":"CRM.",
        "tools":[{"type":"custom","name":"raw"}]}]})");
    CHECK_EQ(chat_tool(bare.chat, "crm.raw")["function"].value("description", ""), std::string("CRM."));
    /* history: joined back as offered */
    const Conv h = convert("{" + tools + R"("input":[
        {"type":"function_call","call_id":"c1","namespace":"crm","name":"find","arguments":"{}"},
        {"type":"custom_tool_call","call_id":"c2","namespace":"crm","name":"raw","input":"q"}]})");
    CHECK_OK(h.rc);
    CHECK_EQ(h.chat["messages"][0]["tool_calls"][0]["function"].value("name", ""), std::string("crm__find"));
    CHECK_EQ(h.chat["messages"][0]["tool_calls"][1]["function"].value("name", ""), std::string("crm.raw"));
    /* reply, not streamed and streamed: namespace and name apart, the flat name nowhere */
    for (const auto& [reply, type] : { std::pair<std::string, std::string>{ R"(TOOL:crm__find:{"q":"P"})", "function_call" },
                                       std::pair<std::string, std::string>{ R"(TOOL:crm.raw:{"input":"q"})", "custom_tool_call" } }) {
        Fixture f;
        f.sched.reply = reply;
        const json o = json::parse(f.post("/v1/responses", "{" + tools + R"("max_output_tokens":200,"input":"x"})").body);
        CHECK_EQ(o["output"][0].value("type", ""), type);
        CHECK_EQ(o["output"][0].value("namespace", ""), std::string("crm"));
        CHECK_EQ(o["output"][0].value("name", ""), type == "function_call" ? std::string("find") : std::string("raw"));
        f.sched.join_all();
        HttpResponse s = f.post("/v1/responses", "{" + tools + R"("max_output_tokens":200,"stream":true,"input":"x"})");
        const std::string body = drain(s);
        CHECK(!has(body, "crm__") && !has(body, "crm.raw"));
        int with_ns = 0;
        for (const Ev& e : events(body))
            if ((e.type == "response.output_item.added" || e.type == "response.output_item.done") &&
                e.data["item"].value("type", "") == type && e.data["item"].value("namespace", "") == "crm")
                ++with_ns;
        CHECK_EQ(with_ns, 2);
        f.sched.join_all();
    }
}

TEST(tool_choice_auto_required_none_and_the_custom_form) {
    const std::string tools = R"("tools":[{"type":"custom","name":"apply_patch"},
        {"type":"namespace","name":"crm","tools":[{"type":"custom","name":"raw"}]}],)";
    for (const char* mode : { "auto", "required", "none" }) {
        const Conv c = convert("{" + tools + R"("input":"x","tool_choice":")" + mode + "\"}");
        CHECK_OK(c.rc);
        CHECK_EQ(c.chat["tool_choice"], json(mode));
        CHECK_EQ(c.echo["tool_choice"], json(mode));
    }
    /* The {type:custom} form is served as Strata serves it: Strata's tool_choice reads only the
     * function form, so a custom one is "auto" (the model may still call the tool, and does here). */
    const Conv one = convert("{" + tools + R"("input":"x","tool_choice":{"type":"custom","name":"apply_patch"}})");
    CHECK_OK(one.rc);
    CHECK_EQ(one.chat["tool_choice"], json("auto"));
    CHECK_EQ(one.echo["tool_choice"]["type"], json("custom"));   /* echoed as sent */
    const Conv ns = convert("{" + tools + R"("input":"x","tool_choice":{"type":"custom","namespace":"crm","name":"raw"}})");
    CHECK_OK(ns.rc);
    CHECK_EQ(ns.chat["tool_choice"], json("auto"));
    /* the function form is still forced, as in 1.17.0 */
    const Conv fn = convert(R"({"tools":[{"type":"function","name":"f","parameters":{"type":"object"}}],"input":"x",
        "tool_choice":{"type":"function","name":"f"}})");
    CHECK_OK(fn.rc);
    CHECK_EQ(fn.chat["tool_choice"]["function"]["name"], json("f"));
    /* accepted end to end, and the call comes back as a custom_tool_call */
    Fixture f;
    f.sched.reply = R"(TOOL:apply_patch:{"input":"p"})";
    HttpResponse r = f.post("/v1/responses", "{" + tools + R"("input":"x","tool_choice":{"type":"custom","name":"apply_patch"}})");
    CHECK_EQ(r.status, 200);
    CHECK_EQ(json::parse(r.body)["output"][0].value("type", ""), std::string("custom_tool_call"));
    f.sched.join_all();
}

TEST(parallel_tool_calls_passes_through_both_ways) {
    for (const bool p : { true, false }) {
        const Conv c = convert(std::string(R"({"input":"x","tools":[{"type":"custom","name":"apply_patch"}],"parallel_tool_calls":)") +
                               (p ? "true" : "false") + "}");
        CHECK_OK(c.rc);
        CHECK_EQ(c.chat["parallel_tool_calls"], json(p));
        CHECK_EQ(c.echo["parallel_tool_calls"], json(p));
    }
}

/* ================================================================== B. reply items */

TEST(a_custom_call_comes_back_as_a_custom_tool_call_item) {
    Fixture f;
    f.sched.reply = R"(TOOL:apply_patch:{"input":"*** Begin Patch\n*** End Patch"})";
    const std::string body = std::string(R"({"input":"edit","max_output_tokens":2000,"tools":[)") + kPatchTool + "]}";
    HttpResponse r = f.post("/v1/responses", body);
    CHECK_EQ(r.status, 200);
    const json o = json::parse(r.body);
    REQUIRE(o["output"].size() == 1);
    const json& it = o["output"][0];
    CHECK_EQ(it.value("type", ""), std::string("custom_tool_call"));
    CHECK_EQ(it.value("name", ""), std::string("apply_patch"));
    CHECK_EQ(it.value("status", ""), std::string("completed"));
    CHECK_EQ(it.value("input", ""), std::string("*** Begin Patch\n*** End Patch"));
    CHECK(!it.contains("arguments"));
    CHECK(!it.contains("namespace"));
    const std::string id = it.value("id", ""), call = it.value("call_id", "");
    CHECK(id.rfind("ctc_", 0) == 0);
    CHECK(call.rfind("call_", 0) == 0);
    CHECK_EQ(id.substr(4), call.substr(5));   /* the item id follows the call id, as fc_ does */
    CHECK_EQ(o.value("status", ""), std::string("completed"));
    f.sched.join_all();
    /* a second request's call has its own ids */
    const json o2 = json::parse(f.post("/v1/responses", body).body);
    CHECK(o2["output"][0].value("id", "") != id);
    CHECK(o2["output"][0].value("call_id", "") != call);
    f.sched.join_all();
    /* a namespaced custom tool reports its namespace and name apart */
    Fixture g;
    g.sched.reply = R"(TOOL:crm.raw:{"input":"q"})";
    const json n = json::parse(g.post("/v1/responses", R"({"input":"x","tools":[{"type":"namespace","name":"crm",
        "tools":[{"type":"custom","name":"raw"}]}]})").body);
    CHECK_EQ(n["output"][0].value("type", ""), std::string("custom_tool_call"));
    CHECK_EQ(n["output"][0].value("namespace", ""), std::string("crm"));
    CHECK_EQ(n["output"][0].value("name", ""), std::string("raw"));
    g.sched.join_all();
}

namespace {
/* B2's inputs: what a patch carries. */
std::vector<std::pair<std::string, std::string>> tricky_inputs() {
    std::string long_patch = "*** Begin Patch\n*** Add File: big.txt\n";
    while (long_patch.size() < 20000) long_patch += "+line " + std::to_string(long_patch.size()) + "\tend\n";
    long_patch += "*** End Patch";
    return {
        { "multi-line patch", "*** Begin Patch\n*** Update File: a.py\n@@ def f():\n-    return 1\n+    return 2\n*** End Patch" },
        { "tabs", "a\tb\t\tc" },
        { "quotes", "say \"hi\" and 'bye'" },
        { "backslashes", "C:\\path\\to\\file and \\n literal" },
        { "unicode", "caf\xc3\xa9 \xe2\x86\x92 \xf0\x9f\x9a\x80 \xe6\xbc\xa2\xe5\xad\x97" },
        { "looks like JSON", "{\"input\": \"nested\", \"n\": 1}" },
        { "a JSON array", "[1, 2, 3]" },
        { "empty", "" },
        { "long (~20 KB)", long_patch },
    };
}
}  // namespace

TEST(a_custom_calls_input_is_kept_byte_for_byte) {
    const std::string body = std::string(R"({"input":"edit","max_output_tokens":40000,"tools":[)") + kPatchTool + "]}";
    for (const auto& [what, input] : tricky_inputs()) {
        /* the arguments as JSON (the fake parser hands them over as written) */
        Fixture f;
        f.sched.piece_bytes = 64;   /* the ~20 KB reply as ~330 tokens: far below its Sink's capacity */
        f.sched.reply = "TOOL:apply_patch:" + json{ { "input", input } }.dump();
        const json o = json::parse(f.post("/v1/responses", body).body);
        REQUIRE(o.contains("output") && o["output"].size() == 1);   /* an error body has none: fail, not UB */
        if (o["output"][0].value("input", "?") != input) fprintf(stderr, "    fake parser: %s differs\n", what.c_str());
        CHECK_EQ(o["output"][0].value("input", "?"), input);
        f.sched.join_all();

        /* as the production model writes it (Qwen's format, the real parser): the value between
         * the one newline at each end that is the format's punctuation */
        Fixture q;
        q.sched.piece_bytes = 64;
        use_qwen(q, body);
        q.sched.reply = "</think>\n\n" + qwen_call("apply_patch", "input", input);
        const json oq = json::parse(q.post("/v1/responses", body).body);
        REQUIRE(oq.contains("output") && oq["output"].size() >= 1);
        const json& last = oq["output"][oq["output"].size() - 1];
        if (last.value("input", "?") != input) fprintf(stderr, "    qwen parser: %s differs\n", what.c_str());
        CHECK_EQ(last.value("type", ""), std::string("custom_tool_call"));
        CHECK_EQ(last.value("input", "?"), input);
        q.sched.join_all();
    }
    /* arguments that are not {"input": string} (a parameter of another name): written out as
     * Strata writes them, Python's json.dumps */
    Fixture f;
    f.sched.reply = R"(TOOL:apply_patch:{"patch":"x","n":[1,2]})";
    CHECK_EQ(json::parse(f.post("/v1/responses", body).body)["output"][0].value("input", ""),
             std::string(R"({"patch": "x", "n": [1, 2]})"));
    f.sched.join_all();
}

/* Which kind of item a call comes back as follows the tool it names; which tool the model picks is
 * the model's (a live probe, matrix B3). */
TEST(a_call_is_answered_as_the_kind_of_tool_it_names) {
    const std::string body = std::string(R"({"input":"x","tools":[)") + kPatchTool +
                             R"(,{"type":"function","name":"exec_command","parameters":{"type":"object"}}]})";
    Fixture f;
    f.sched.reply = R"(TOOL:exec_command:{"cmd":"ls"})";
    const json fn = json::parse(f.post("/v1/responses", body).body)["output"][0];
    CHECK_EQ(fn.value("type", ""), std::string("function_call"));
    CHECK_EQ(fn.value("arguments", ""), std::string(R"({"cmd":"ls"})"));
    f.sched.join_all();
    f.sched.reply = R"(TOOL:apply_patch:{"input":"p"})";
    CHECK_EQ(json::parse(f.post("/v1/responses", body).body)["output"][0].value("type", ""),
             std::string("custom_tool_call"));
    f.sched.join_all();
    /* Strata's kept-tools case: a call to a name no custom tool of this request has is a
     * function_call, whatever an earlier request offered */
    f.sched.reply = R"(TOOL:apply_patch:{"input":"p"})";
    const json no = json::parse(f.post("/v1/responses",
        R"({"input":"x","tools":[{"type":"function","name":"apply_patch","parameters":{"type":"object"}}]})").body);
    CHECK_EQ(no["output"][0].value("type", ""), std::string("function_call"));
    f.sched.join_all();
}

/* Strata's test_namespace_and_custom_calls_come_back_under_their_own_names, and custom + custom. */
TEST(several_calls_in_one_turn_keep_their_kinds_and_order) {
    const std::string body = R"({"input":"go","max_output_tokens":2000,"tools":[
        {"type":"namespace","name":"multi_agent_v1","description":"Agents.","tools":[
            {"type":"function","name":"spawn_agent","strict":false,
             "parameters":{"type":"object","properties":{"message":{"type":"string"}}}}]},
        {"type":"custom","name":"apply_patch","description":"Edit files."},
        {"type":"custom","name":"note"}]})";
    Fixture f;
    use_qwen(f, body);
    f.sched.reply = "</think>\n\n" + qwen_call("multi_agent_v1__spawn_agent", "message", "hi") +
                    qwen_call("apply_patch", "input", "*** Begin Patch") + qwen_call("note", "input", "n1") +
                    qwen_call("apply_patch", "input", "*** Begin Patch 2");
    HttpResponse r = f.post("/v1/responses", body);
    CHECK_EQ(r.status, 200);
    const json out = json::parse(r.body)["output"];
    if (out.size() != 4) fprintf(stderr, "    output: %s\n", out.dump().c_str());
    REQUIRE(out.size() == 4);
    CHECK_EQ(out[0].value("type", ""), std::string("function_call"));
    CHECK_EQ(out[0].value("namespace", ""), std::string("multi_agent_v1"));
    CHECK_EQ(out[0].value("name", ""), std::string("spawn_agent"));
    CHECK_EQ(json::parse(out[0].value("arguments", "{}")).dump(), std::string(R"({"message":"hi"})"));
    const char* names[] = { "apply_patch", "note", "apply_patch" };
    const char* inputs[] = { "*** Begin Patch", "n1", "*** Begin Patch 2" };
    std::set<std::string> ids, calls;
    for (size_t i = 1; i < 4; ++i) {
        CHECK_EQ(out[i].value("type", ""), std::string("custom_tool_call"));
        CHECK_EQ(out[i].value("name", ""), std::string(names[i - 1]));
        CHECK_EQ(out[i].value("input", ""), std::string(inputs[i - 1]));
        ids.insert(out[i].value("id", ""));
        calls.insert(out[i].value("call_id", ""));
    }
    CHECK_EQ(ids.size(), (size_t)3);     /* unique per call */
    CHECK_EQ(calls.size(), (size_t)3);
    f.sched.join_all();
}

TEST(text_and_a_custom_call_in_one_reply) {
    const std::string body = std::string(R"({"input":"go","max_output_tokens":2000,"tools":[)") + kPatchTool + "]}";
    Fixture f;
    use_qwen(f, body);
    f.sched.reply = "</think>\n\nEditing now.\n" + qwen_call("apply_patch", "input", "*** Begin Patch");
    const json out = json::parse(f.post("/v1/responses", body).body)["output"];
    if (out.size() != 2) fprintf(stderr, "    output: %s\n", out.dump().c_str());
    REQUIRE(out.size() == 2);
    CHECK_EQ(out[0].value("type", ""), std::string("message"));
    CHECK(has(out[0]["content"][0].value("text", ""), "Editing now."));
    CHECK_EQ(out[1].value("type", ""), std::string("custom_tool_call"));
    f.sched.join_all();
    /* streamed: the message at output_index 0, the call at 1 */
    HttpResponse s = f.post("/v1/responses", body.substr(0, body.size() - 1) + R"(,"stream":true})");
    const std::vector<Ev> ev = events(drain(s));
    int64_t msg_at = -1, call_at = -1;
    for (const Ev& e : ev)
        if (e.type == "response.output_item.added") {
            if (e.data["item"].value("type", "") == "message") msg_at = e.data.value("output_index", -2);
            if (e.data["item"].value("type", "") == "custom_tool_call") call_at = e.data.value("output_index", -2);
        }
    CHECK_EQ(msg_at, (int64_t)0);
    CHECK_EQ(call_at, (int64_t)1);
    f.sched.join_all();
}

/* ================================================================== C. streaming */

TEST(a_streamed_custom_call_sends_its_events_in_order) {
    Fixture f;
    f.sched.reply = R"(TOOL:apply_patch:{"input":"*** Begin Patch\n*** End Patch"})";
    HttpResponse s = f.post("/v1/responses", std::string(R"({"input":"edit","stream":true,"max_output_tokens":2000,"tools":[)") + kPatchTool + "]}");
    CHECK_EQ(s.status, 200);
    const std::vector<Ev> ev = events(drain(s));
    /* C1: the order, and no function-call events for it */
    CHECK_EQ(json(types_of(ev)).dump(),
             json::parse(R"(["response.created","response.in_progress","response.output_item.added",
                             "response.custom_tool_call_input.delta","response.custom_tool_call_input.done",
                             "response.output_item.done","response.completed"])").dump());
    REQUIRE(ev.size() == 7);
    const json& added = ev[2].data;
    CHECK_EQ(added["item"].value("type", ""), std::string("custom_tool_call"));
    CHECK_EQ(added["item"].value("status", ""), std::string("in_progress"));
    CHECK_EQ(added["item"].value("input", "?"), std::string(""));
    /* C3: sequence numbers count up by one from 0; one item id and output index throughout */
    for (size_t i = 0; i < ev.size(); ++i) CHECK_EQ(ev[i].data.value("sequence_number", -1), (int64_t)i);
    const std::string id = added["item"].value("id", "");
    CHECK(id.rfind("ctc_", 0) == 0);
    for (size_t i = 3; i <= 4; ++i) {
        CHECK_EQ(ev[i].data.value("item_id", ""), id);
        CHECK_EQ(ev[i].data.value("output_index", -1), (int64_t)0);
    }
    CHECK_EQ(ev[5].data["item"].value("id", ""), id);
    CHECK_EQ(ev[5].data.value("output_index", -1), (int64_t)0);
    CHECK_EQ(ev[5].data["item"].value("status", ""), std::string("completed"));
    const json& done = ev[6].data["response"];
    CHECK_EQ(done.value("status", ""), std::string("completed"));
    REQUIRE(done["output"].size() == 1);
    CHECK_EQ(done["output"][0].value("id", ""), id);
    f.sched.join_all();
}

TEST(the_streamed_input_is_the_same_bytes_everywhere) {
    const std::string body = std::string(R"({"input":"edit","stream":true,"max_output_tokens":40000,"tools":[)") + kPatchTool + "]}";
    for (const auto& [what, input] : tricky_inputs()) {
        for (const bool real : { false, true }) {
            Fixture f;
            f.sched.piece_bytes = 64;   /* the ~20 KB reply streamed as ~330 tokens: far below its Sink's capacity */
            if (real) {
                use_qwen(f, body);
                f.sched.reply = "</think>\n\n" + qwen_call("apply_patch", "input", input);
            } else {
                f.sched.reply = "TOOL:apply_patch:" + json{ { "input", input } }.dump();
            }
            HttpResponse s = f.post("/v1/responses", body);
            std::string deltas, done_input, item_input, completed_input;
            const auto ev = events(drain(s));
            REQUIRE(!ev.empty());
            CHECK_EQ(ev.back().type, std::string("response.completed"));   /* not response.failed */
            for (const Ev& e : ev) {
                if (e.type == "response.custom_tool_call_input.delta") deltas += e.data.value("delta", "");
                if (e.type == "response.custom_tool_call_input.done") done_input = e.data.value("input", "?");
                if (e.type == "response.output_item.done" && e.data["item"].value("type", "") == "custom_tool_call")
                    item_input = e.data["item"].value("input", "?");
                if (e.type == "response.completed")
                    for (const json& o : e.data["response"]["output"])
                        if (o.value("type", "") == "custom_tool_call") completed_input = o.value("input", "?");
            }
            if (deltas != input) fprintf(stderr, "    %s (%s parser) differs\n", what.c_str(), real ? "qwen" : "fake");
            CHECK_EQ(deltas, input);
            CHECK_EQ(done_input, input);
            CHECK_EQ(item_input, input);
            CHECK_EQ(completed_input, input);
            f.sched.join_all();
        }
    }
}

TEST(a_stream_with_reasoning_a_function_call_and_a_custom_call) {
    const std::string body = R"({"input":"go","stream":true,"max_output_tokens":2000,"tools":[
        {"type":"function","name":"exec_command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},
         "required":["cmd"]}},
        {"type":"custom","name":"apply_patch"}]})";
    Fixture f;
    use_qwen(f, body);
    f.sched.reply = "Plan the edit.\n</think>\n\n" + qwen_call("exec_command", "cmd", "ls") +
                    qwen_call("apply_patch", "input", "*** Begin Patch");
    HttpResponse s = f.post("/v1/responses", body);
    const std::vector<Ev> ev = events(drain(s));
    std::vector<std::string> added;
    std::string fn_args, custom_input;
    int64_t last = -1;
    for (const Ev& e : ev) {
        CHECK_EQ(e.data.value("sequence_number", -1), last + 1);
        last = e.data.value("sequence_number", -1);
        if (e.type == "response.output_item.added") added.push_back(e.data["item"].value("type", ""));
        if (e.type == "response.function_call_arguments.delta") fn_args += e.data.value("delta", "");
        if (e.type == "response.custom_tool_call_input.delta") custom_input += e.data.value("delta", "");
    }
    CHECK_EQ(json(added).dump(), json::parse(R"(["reasoning","function_call","custom_tool_call"])").dump());
    CHECK_EQ(json::parse(fn_args).dump(), std::string(R"({"cmd":"ls"})"));
    CHECK_EQ(custom_input, std::string("*** Begin Patch"));
    const json out = ev.back().data["response"]["output"];
    REQUIRE(out.size() == 3);
    CHECK_EQ(out[0].value("type", ""), std::string("reasoning"));
    CHECK_EQ(out[1].value("type", ""), std::string("function_call"));
    CHECK_EQ(out[2].value("type", ""), std::string("custom_tool_call"));
    f.sched.join_all();
}

/* A reply that ends inside a custom call: the item is "incomplete" with no input, so a client
 * acting on it does not apply half a patch (Strata does the same). */
TEST(a_custom_call_cut_short_is_incomplete_with_no_input) {
    const std::string body = std::string(R"({"input":"go","stream":true,"max_output_tokens":2000,"tools":[)") + kPatchTool + "]}";
    Fixture f;
    use_qwen(f, body);
    f.sched.reply = "</think>\n\n<tool_call>\n<function=apply_patch>\n<parameter=input>\n*** Begin Patch\n*** Add";
    HttpResponse s = f.post("/v1/responses", body);
    const std::vector<Ev> ev = events(drain(s));
    bool delta = false;
    std::string status, done_input = "?";
    for (const Ev& e : ev) {
        if (e.type == "response.custom_tool_call_input.delta") delta = true;
        if (e.type == "response.custom_tool_call_input.done") done_input = e.data.value("input", "?");
        if (e.type == "response.output_item.done" && e.data["item"].value("type", "") == "custom_tool_call")
            status = e.data["item"].value("status", "");
    }
    CHECK(!delta);
    CHECK_EQ(done_input, std::string(""));
    CHECK_EQ(status, std::string("incomplete"));
    f.sched.join_all();
}

/* ================================================================== D. input history */

TEST(custom_call_history_maps_onto_the_chat_conversation) {
    for (const char* output : { R"("2 files changed")", R"([{"type":"input_text","text":"2 files changed"}])" }) {
        const Conv c = convert(std::string(R"({"input":[{"role":"user","content":"edit"},
            {"type":"custom_tool_call","call_id":"call_a","name":"apply_patch","input":"*** Begin Patch\n*** End Patch"},
            {"type":"custom_tool_call_output","call_id":"call_a","output":)") + output + "}]," +
            R"("tools":[{"type":"custom","name":"apply_patch"}]})");
        CHECK_OK(c.rc);
        const json& m = c.chat["messages"];
        REQUIRE(m.size() == 3);
        CHECK_EQ(m[1].value("role", ""), std::string("assistant"));
        REQUIRE(m[1]["tool_calls"].size() == 1);
        const json& call = m[1]["tool_calls"][0];
        CHECK_EQ(call.value("id", ""), std::string("call_a"));
        CHECK_EQ(call.value("type", ""), std::string("function"));
        CHECK_EQ(call["function"].value("name", ""), std::string("apply_patch"));
        CHECK_EQ(json::parse(call["function"].value("arguments", "")).value("input", ""),
                 std::string("*** Begin Patch\n*** End Patch"));
        CHECK_EQ(m[2].value("role", ""), std::string("tool"));
        CHECK_EQ(m[2].value("tool_call_id", ""), std::string("call_a"));
        if (m[2]["content"].is_string()) CHECK_EQ(m[2]["content"], json("2 files changed"));
        else CHECK_EQ(m[2]["content"][0].value("text", ""), std::string("2 files changed"));
    }
    /* a namespaced custom call comes back joined; a missing input is an empty one */
    const Conv ns = convert(R"({"input":[{"type":"custom_tool_call","call_id":"c","namespace":"crm","name":"raw"}]})");
    CHECK_OK(ns.rc);
    CHECK_EQ(ns.chat["messages"][0]["tool_calls"][0]["function"].value("name", ""), std::string("crm.raw"));
    CHECK_EQ(ns.chat["messages"][0]["tool_calls"][0]["function"].value("arguments", ""), std::string(R"({"input":""})"));
    /* and the server serves such a history */
    Fixture f;
    HttpResponse r = f.post("/v1/responses", R"({"tools":[{"type":"custom","name":"apply_patch"}],"input":[
        {"role":"user","content":"edit"},
        {"type":"custom_tool_call","call_id":"call_a","name":"apply_patch","input":"x"},
        {"type":"custom_tool_call_output","call_id":"call_a","output":"ok"}]})");
    CHECK_EQ(r.status, 200);
    f.sched.join_all();
}

/* Strata's round trip: the reply's items, sent back as history, are the calls they were. */
TEST(custom_and_function_history_interleave) {
    const Conv c = convert(R"({"input":[{"role":"user","content":"go"},
        {"type":"function_call","call_id":"c1","namespace":"multi_agent_v1","name":"spawn_agent","arguments":"{\"message\":\"hi\"}"},
        {"type":"custom_tool_call","call_id":"c2","name":"apply_patch","input":"*** Begin Patch"},
        {"type":"function_call_output","call_id":"c1","output":"1"},
        {"type":"custom_tool_call_output","call_id":"c2","output":"2"}]})");
    CHECK_OK(c.rc);
    const json& m = c.chat["messages"];
    REQUIRE(m.size() == 4);
    std::vector<std::string> names;
    for (const json& t : m[1]["tool_calls"]) names.push_back(t["function"]["name"].get<std::string>());
    CHECK_EQ(json(names).dump(), json::parse(R"(["multi_agent_v1__spawn_agent","apply_patch"])").dump());
    CHECK_EQ(m[2].value("tool_call_id", ""), std::string("c1"));
    CHECK_EQ(m[3].value("tool_call_id", ""), std::string("c2"));
}

TEST(several_round_trips_of_history) {
    json input = json::array({ json{ { "role", "user" }, { "content", "edit three files" } } });
    for (int i = 0; i < 4; ++i) {
        const std::string id = "call_" + std::to_string(i);
        input.push_back(json{ { "type", "reasoning" }, { "summary", json::array() },
                              { "content", json::array({ json{ { "type", "reasoning_text" }, { "text", "step" } } }) } });
        if (i % 2 == 0)
            input.push_back(json{ { "type", "custom_tool_call" }, { "call_id", id }, { "name", "apply_patch" },
                                  { "input", "*** Begin Patch\n*** Add File: f" + std::to_string(i) + "\n+x\n*** End Patch" } });
        else
            input.push_back(json{ { "type", "function_call" }, { "call_id", id }, { "name", "exec_command" },
                                  { "arguments", R"({"cmd":"ls"})" } });
        input.push_back(json{ { "type", i % 2 == 0 ? "custom_tool_call_output" : "function_call_output" },
                              { "call_id", id }, { "output", "done " + std::to_string(i) } });
    }
    input.push_back(json{ { "role", "user" }, { "content", "and now?" } });
    const json req = { { "input", input }, { "max_output_tokens", 2000 },
                       { "tools", json::parse(R"([{"type":"custom","name":"apply_patch"},
        {"type":"function","name":"exec_command","parameters":{"type":"object"}}])") } };
    const Conv c = convert(req.dump());
    CHECK_OK(c.rc);
    int calls = 0, results = 0;
    for (const json& m : c.chat["messages"]) {
        if (m.contains("tool_calls")) calls += (int)m["tool_calls"].size();
        if (m.value("role", "") == "tool") ++results;
    }
    CHECK_EQ(calls, 4);
    CHECK_EQ(results, 4);
    CHECK_EQ(c.chat["messages"].back().value("role", ""), std::string("user"));
    Fixture f;
    f.sched.reply = R"(TOOL:apply_patch:{"input":"*** Begin Patch\n*** End Patch"})";
    HttpResponse r = f.post("/v1/responses", req.dump());
    CHECK_EQ(r.status, 200);
    CHECK_EQ(json::parse(r.body)["output"][0].value("type", ""), std::string("custom_tool_call"));
    f.sched.join_all();
}

/* No call id is refused (Strata does not refuse them either): an output with no call, a call
 * with no call_id and a repeated call_id all pass through to the chat request, the same as for
 * function calls. (Ids do order a turn's results: each_turns_results_are_put_in_its_calls_order.) */
TEST(unmatched_and_repeated_call_ids_are_passed_through) {
    const Conv orphan = convert(R"({"input":[{"role":"user","content":"x"},
        {"type":"custom_tool_call_output","call_id":"nobody","output":"?"}]})");
    CHECK_OK(orphan.rc);
    CHECK_EQ(orphan.chat["messages"][1].value("tool_call_id", ""), std::string("nobody"));
    const Conv noid = convert(R"({"input":[{"type":"custom_tool_call","name":"apply_patch","input":"x"}]})");
    CHECK_OK(noid.rc);
    CHECK_EQ(noid.chat["messages"][0]["tool_calls"][0].value("id", "?"), std::string(""));
    const Conv twice = convert(R"({"input":[{"type":"custom_tool_call","call_id":"c","name":"apply_patch","input":"1"},
        {"type":"custom_tool_call","call_id":"c","name":"apply_patch","input":"2"}]})");
    CHECK_OK(twice.rc);
    CHECK_EQ(twice.chat["messages"][0]["tool_calls"].size(), (size_t)2);
    /* the same as a function call's: an orphan output is not refused by the translation */
    const Conv fn = convert(R"({"input":[{"role":"user","content":"x"},
        {"type":"function_call_output","call_id":"nobody","output":"?"}]})");
    CHECK_EQ(fn.rc, orphan.rc);
}

/* As Strata (_order_tool_results): the template places results by position, so each run of results
 * is put in the order of the calls of the turn before it, by call id; a result no call of that
 * turn has keeps its place after them. Function and custom results alike. */
TEST(each_turns_results_are_put_in_its_calls_order) {
    const Conv c = convert(R"({"input":[{"role":"user","content":"go"},
        {"type":"function_call","call_id":"a","name":"exec_command","arguments":"{}"},
        {"type":"custom_tool_call","call_id":"b","name":"apply_patch","input":"p"},
        {"type":"function_call","call_id":"c","name":"exec_command","arguments":"{}"},
        {"type":"function_call_output","call_id":"stray","output":"?"},
        {"type":"function_call_output","call_id":"c","output":"3"},
        {"type":"custom_tool_call_output","call_id":"b","output":"2"},
        {"type":"function_call_output","call_id":"a","output":"1"},
        {"role":"user","content":"next"},
        {"type":"function_call_output","call_id":"z","output":"9"},
        {"type":"function_call_output","call_id":"y","output":"8"}]})");
    CHECK_OK(c.rc);
    std::vector<std::string> order;
    for (const json& m : c.chat["messages"])
        if (m.value("role", "") == "tool") order.push_back(m.value("tool_call_id", ""));
    CHECK_EQ(json(order).dump(), std::string(R"(["a","b","c","stray","z","y"])"));
    /* a turn already in order is unchanged, and so is the conversation around it */
    const Conv same = convert(R"({"input":[{"role":"user","content":"go"},
        {"type":"function_call","call_id":"a","name":"f","arguments":"{}"},
        {"type":"function_call_output","call_id":"a","output":"1"}]})");
    CHECK_EQ(same.chat["messages"].size(), (size_t)3);
    CHECK_EQ(same.chat["messages"][2].value("tool_call_id", ""), std::string("a"));
}

/* Codex's own compaction, with a provider that is not OpenAI (codex 0.167, recorded against a fake
 * server): no compaction item and no /responses/compact call, only an ordinary request
 * with the conversation, `tools: []` and the compaction prompt last; then the next turn carries the
 * summary as a user message. Both are served, streamed as Codex asks. (The shape below is the
 * recorded one, its texts shortened.) */
TEST(codexs_local_compaction_turn_is_served) {
    const std::string meta = R"("client_metadata":{"turn_id":"t1","x-codex-turn-metadata":"{\"request_kind\":\"compaction\"}"},)";
    const std::string compaction = R"({"model":"m","stream":true,"tools":[],"tool_choice":"auto","parallel_tool_calls":false,
        "reasoning":{"effort":"medium"},"store":false,"include":["reasoning.encrypted_content"],
        "prompt_cache_key":"k","text":{"verbosity":"low"},)" + meta + R"("max_output_tokens":200,"input":[
        {"type":"message","role":"developer","content":[{"type":"input_text","text":"<permissions instructions>"}]},
        {"type":"message","role":"developer","content":[{"type":"input_text","text":"<collaboration_mode>"}]},
        {"type":"message","role":"user","content":[{"type":"input_text","text":"<environment_context>"}]},
        {"type":"message","role":"user","content":[{"type":"input_text","text":"Run true, then say ok."}]},
        {"type":"function_call","status":"completed","id":"fc_0","name":"exec_command","arguments":"{\"cmd\": \"true\"}","call_id":"call_0"},
        {"type":"function_call_output","id":"fco_1","call_id":"call_0","output":"Process exited with code 0"},
        {"type":"message","role":"user","content":[{"type":"input_text","text":"You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task."}]}]})";
    const Conv c = convert(compaction);
    CHECK_OK(c.rc);
    CHECK(!c.chat.contains("tools"));
    CHECK_EQ(c.chat["tool_choice"], json("auto"));
    Fixture f;
    f.sched.reply = "Summary: true ran; say ok.";
    HttpResponse s = f.post("/v1/responses", compaction);
    CHECK_EQ(s.status, 200);
    const auto ev = events(drain(s));
    REQUIRE(!ev.empty());
    CHECK_EQ(ev.back().type, std::string("response.completed"));
    CHECK_EQ(ev.back().data["response"]["output"].back().value("type", ""), std::string("message"));
    f.sched.join_all();
    /* the turn after it: the summary as a user message, the tools again */
    const std::string next = std::string(R"({"model":"m","stream":true,"tool_choice":"auto","max_output_tokens":200,"tools":[)") + kPatchTool +
        R"(],"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Run true, then say ok."}]},
        {"type":"message","role":"user","content":[{"type":"input_text","text":"Another language model started to solve this problem and produced a summary of its thinking process. Summary: true ran."}]}]})";
    HttpResponse n = f.post("/v1/responses", next);
    CHECK_EQ(n.status, 200);
    const auto nev = events(drain(n));
    REQUIRE(!nev.empty());
    CHECK_EQ(nev.back().type, std::string("response.completed"));
    f.sched.join_all();
}

TEST(previous_response_id_is_still_refused) {
    const Conv c = convert(R"({"input":"x","previous_response_id":"resp_1","tools":[{"type":"custom","name":"apply_patch"}]})");
    CHECK(c.rc < 0);
    CHECK_EQ(c.err.param, std::string("previous_response_id"));
}

/* ================================================================== E. negatives */

TEST(a_custom_tool_needs_a_name) {
    for (const char* t : { R"({"type":"custom"})", R"({"type":"custom","name":""})", R"({"type":"custom","name":7})" }) {
        const Conv c = convert(std::string(R"({"input":"x","tools":[)") + t + "]}");
        CHECK(c.rc < 0);
        CHECK_EQ(c.err.status, 400);
        CHECK_EQ(c.err.param, std::string("tools[].name"));
    }
    const Conv ns = convert(R"({"input":"x","tools":[{"type":"namespace","name":"crm","tools":[{"type":"custom"}]}]})");
    CHECK(ns.rc < 0);
    CHECK_EQ(ns.err.param, std::string("tools[].tools[].name"));
    Fixture f;
    HttpResponse r = f.post("/v1/responses", R"({"input":"x","tools":[{"type":"custom","name":""}]})");
    CHECK_EQ(r.status, 400);
    CHECK(has(r.body, "a custom tool needs a name"));
}

TEST(other_tool_types_are_still_left_out) {
    const Conv c = convert(R"({"input":"x","tools":[{"type":"web_search"},{"type":"file_search"},
        {"type":"computer_use_preview"},{"type":"mcp","server_label":"s"},{"type":"local_shell"},{"name":"untyped"},
        {"type":"custom","name":"apply_patch"}]})");
    CHECK_OK(c.rc);
    CHECK_EQ(c.chat["tools"].size(), (size_t)1);
    CHECK_EQ(c.custom.size(), (size_t)1);
    /* only custom: no other tool type became custom */
    const Conv none = convert(R"({"input":"x","tools":[{"type":"web_search"},{"type":"file_search"}]})");
    CHECK_OK(none.rc);
    CHECK(!none.chat.contains("tools"));
    CHECK(none.custom.empty());
}

/* A tool_choice the request cannot honour acts as "auto", as in Strata (frontend.py forced_call),
 * not a 400; "required" with one tool forces it; a choice it can honour passes on. */
TEST(tool_choice_naming_no_offered_tool) {
    const std::string tools = R"("tools":[{"type":"custom","name":"apply_patch"},
        {"type":"function","name":"f","parameters":{"type":"object"}}],)";
    struct Case { const char* choice; json chat; };
    const json named_f = json::parse(R"({"type":"function","function":{"name":"f"}})");
    for (const Case& k : { Case{ R"({"type":"custom","name":"nope"})", json("auto") },
                           Case{ R"({"type":"function","name":"nope"})", json("auto") },
                           Case{ R"({"type":"function","function":{"name":"nope"}})", json("auto") },
                           Case{ R"({"type":"function","name":7})", json("auto") },
                           Case{ R"({"type":"allowed_tools","mode":"auto","tools":[]})", json("auto") },
                           Case{ R"("sometimes")", json("auto") },
                           Case{ "5", json("auto") },
                           Case{ R"({"type":"function","name":"f"})", named_f },
                           Case{ R"({"type":"function","function":{"name":"f"}})", named_f },
                           Case{ R"({"type":"tool","name":"f"})", named_f },
                           Case{ R"("required")", json("required") },
                           Case{ R"({"type":"any"})", json("required") },
                           Case{ R"({"type":"auto"})", json("auto") },
                           Case{ R"("none")", json("none") } }) {
        const Conv c = convert("{" + tools + R"("input":"x","tool_choice":)" + k.choice + "}");
        CHECK_OK(c.rc);
        if (c.chat["tool_choice"] != k.chat) fprintf(stderr, "    %s -> %s\n", k.choice, c.chat["tool_choice"].dump().c_str());
        CHECK_EQ(c.chat["tool_choice"], k.chat);
    }
    /* "required" with one tool names it; with none, it is "auto" */
    const Conv one = convert(R"({"input":"x","tools":[{"type":"custom","name":"apply_patch"}],"tool_choice":"required"})");
    CHECK_EQ(one.chat["tool_choice"], json::parse(R"({"type":"function","function":{"name":"apply_patch"}})"));
    const Conv none = convert(R"({"input":"x","tools":[{"type":"web_search"}],"tool_choice":"required"})");
    CHECK_EQ(none.chat["tool_choice"], json("auto"));
    /* a namespace joins the name, as the tools are named */
    const Conv ns = convert(R"({"input":"x","tools":[{"type":"namespace","name":"crm","tools":[{"type":"function","name":"find"}]}],
        "tool_choice":{"type":"function","namespace":"crm","name":"find"}})");
    CHECK_EQ(ns.chat["tool_choice"], json::parse(R"({"type":"function","function":{"name":"crm__find"}})"));
    /* served end to end, where a 400 was */
    Fixture f;
    for (const char* choice : { R"({"type":"function","name":"nope"})", R"({"type":"function","name":7})" }) {
        HttpResponse r = f.post("/v1/responses", "{" + tools + R"("input":"x","tool_choice":)" + choice + "}");
        CHECK_EQ(r.status, 200);
        f.sched.join_all();
    }
}

/* As Strata: an item with no usable name is refused; an input that is not a string reads as "", a
 * call_id is passed through as sent, and a namespace joins the name when it is set at all. (An
 * output item's call_id is checked as a function_call_output's is: the same, 1.17.0, path.) */
TEST(a_malformed_custom_tool_call_item_is_refused) {
    struct Case { const char* item; const char* param; };
    for (const Case& k : { Case{ R"({"type":"custom_tool_call","call_id":"c","input":"x"})", "input[].name" },
                           Case{ R"({"type":"custom_tool_call","call_id":"c","name":"","input":"x"})", "input[].name" },
                           Case{ R"({"type":"custom_tool_call","call_id":"c","name":5,"input":"x"})", "input[].name" },
                           Case{ R"({"type":"custom_tool_call_output","call_id":5,"output":"x"})", "input[].call_id" } }) {
        const Conv c = convert(std::string(R"({"input":[)") + k.item + "]}");
        CHECK(c.rc < 0);
        CHECK_EQ(c.err.status, 400);
        if (c.err.param != k.param) fprintf(stderr, "    %s -> %s\n", k.item, c.err.param.c_str());
        CHECK_EQ(c.err.param, std::string(k.param));
    }
    for (const char* in : { R"({"a":1})", "7", "null", "[]" }) {
        const Conv c = convert(std::string(R"({"input":[{"type":"custom_tool_call","call_id":"c","name":"apply_patch","input":)") + in + "}]}");
        CHECK_OK(c.rc);
        CHECK_EQ(c.chat["messages"][0]["tool_calls"][0]["function"].value("arguments", ""), std::string(R"({"input":""})"));
    }
    struct Pass { const char* fields; json id; const char* name; };
    for (const Pass& k : { Pass{ R"("call_id":9,"name":"p")", json(9), "p" },
                           Pass{ R"("call_id":null,"name":"p")", json(""), "p" },
                           Pass{ R"("name":"p","namespace":[])", json(""), "p" },
                           Pass{ R"("name":"p","namespace":"")", json(""), "p" },
                           Pass{ R"("name":"p","namespace":0)", json(""), "p" },
                           Pass{ R"("name":"p","namespace":5)", json(""), "5.p" },
                           Pass{ R"("name":"p","namespace":true)", json(""), "True.p" } }) {
        const Conv c = convert(std::string(R"({"input":[{"type":"custom_tool_call",)") + k.fields + R"(,"input":"x"}]})");
        CHECK_OK(c.rc);
        if (c.rc != RAD_OK) { fprintf(stderr, "    %s refused\n", k.fields); continue; }
        const json& call = c.chat["messages"][0]["tool_calls"][0];
        CHECK_EQ(call["id"], k.id);
        CHECK_EQ(call["function"].value("name", ""), std::string(k.name));
    }
    /* and the server serves a history whose call_id is not a string */
    Fixture f;
    HttpResponse r = f.post("/v1/responses", R"({"tools":[{"type":"custom","name":"apply_patch"}],"input":[
        {"role":"user","content":"edit"},
        {"type":"custom_tool_call","call_id":9,"name":"apply_patch","input":"x"},
        {"type":"custom_tool_call_output","call_id":"9","output":"ok"}]})");
    CHECK_EQ(r.status, 200);
    f.sched.join_all();
}

/* ================================================================== F. regression */

/* The function-call paths answer byte for byte as 1.17.0 (19d4e8fe0e18) did, normalised for ids
 * and clocks: plain and namespaced function tools, streamed and not, the real Qwen parser's
 * reasoning + text + two calls, and /v1/chat/completions. The goldens were recorded by these same
 * requests on the unmodified sources (tests/data/chat/responses-function-golden-1.17.0.txt). */
TEST(function_and_chat_tool_replies_are_as_in_1_17_0) {
    const auto g = goldens();
    REQUIRE(g.size() == 8);
    /* the normaliser keeps field names: a function_call's call_id, which Codex joins outputs on */
    CHECK(has(g.at("responses_fn"), R"("call_id":"call_ID")"));
    CHECK(has(g.at("responses_ns_stream"), R"("call_id":"call_ID")"));
    auto same = [&](const char* name, const std::string& got) {
        const std::string n = join_argument_chunks(normalize(got)), want = join_argument_chunks(g.at(name));
        if (n != want) fprintf(stderr, "    %s differs from 1.17.0\n%s\n", name, getenv("CT_FULL") ? n.c_str() : "");
        CHECK_EQ(n, want);
    };
    const std::string fn = R"("tools":[{"type":"function","name":"get","description":"Weather.","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}],)";
    const std::string ns = R"("tools":[{"type":"namespace","name":"crm","description":"d","tools":[{"type":"function","name":"find","parameters":{"type":"object"}}]}],)";
    {
        Fixture f;
        f.sched.reply = R"(TOOL:get:{"city":"P"})";
        same("responses_fn", f.post("/v1/responses", "{" + fn + R"("input":"weather in P"})").body);
        f.sched.join_all();
        HttpResponse s = f.post("/v1/responses", "{" + fn + R"("stream":true,"input":"weather in P"})");
        same("responses_fn_stream", drain(s));
        f.sched.join_all();
    }
    {
        Fixture f;
        f.sched.reply = R"(TOOL:crm__find:{"q":"P"})";
        same("responses_ns", f.post("/v1/responses", "{" + ns + R"("input":"find P"})").body);
        f.sched.join_all();
        HttpResponse s = f.post("/v1/responses", "{" + ns + R"("stream":true,"input":"find P"})");
        same("responses_ns_stream", drain(s));
        f.sched.join_all();
    }
    {
        Fixture f;
        f.sched.reply = R"(TOOL:get:{"city":"P"})";
        const std::string body = R"({"messages":[{"role":"user","content":"weather in P"}],"tools":[{"type":"function","function":{"name":"get","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}])";
        same("chat_fn", f.post("/v1/chat/completions", body + "}").body);
        f.sched.join_all();
        HttpResponse s = f.post("/v1/chat/completions", body + R"(,"stream":true})");
        same("chat_fn_stream", drain(s));
        f.sched.join_all();
    }
    {
        const json tools = json::parse(R"([{"type":"function","function":{"name":"get","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},{"type":"function","function":{"name":"put","parameters":{"type":"object","properties":{"v":{"type":"string"}},"required":["v"]}}}])");
        Fixture f;
        f.tmpl.real_parser = qwen_reply_parser(tools);
        REQUIRE(f.tmpl.real_parser != nullptr);
        f.sched.reply = "Think.\n</think>\n\nOn it.\n<tool_call>\n<function=get>\n<parameter=city>\nP\n</parameter>\n</function>\n</tool_call>\n"
                        "<tool_call>\n<function=put>\n<parameter=v>\nx\n</parameter>\n</function>\n</tool_call>\n";
        const std::string tl = R"("tools":[{"type":"function","name":"get","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}},{"type":"function","name":"put","parameters":{"type":"object","properties":{"v":{"type":"string"}},"required":["v"]}}],)";
        same("responses_qwen_two_fn", f.post("/v1/responses", "{" + tl + R"("max_output_tokens":1000,"input":"go"})").body);
        f.sched.join_all();
        HttpResponse s = f.post("/v1/responses", "{" + tl + R"("max_output_tokens":1000,"stream":true,"input":"go"})");
        same("responses_qwen_two_fn_stream", drain(s));
        f.sched.join_all();
    }
}

/* ================================================================== Strata compatibility */

namespace {

/* One normaliser for both servers' answers: ids, sequence numbers, usage and the response
 * snapshots' echo fields are not compared; every item kind, field and value is, and so is the
 * event sequence (consecutive deltas of one item joined, as chunking differs by engine). */
json norm_item(const json& it) {
    const std::string type = it.value("type", "");
    json o = { { "type", type } };
    if (it.contains("status")) o["status"] = it["status"];
    if (type == "message" || type == "reasoning") {
        std::string text;
        if (it.contains("content"))
            for (const json& p : it["content"]) text += p.value("text", "");
        o["text"] = text;
        return o;
    }
    for (auto k = it.begin(); k != it.end(); ++k) {
        if (k.key() == "id") {   /* the prefix only */
            const std::string id = k.value().get<std::string>();
            o["id"] = id.substr(0, id.find('_') + 1);
        }
        else if (k.key() == "call_id") o["call_id"] = "CALL";
        else if (k.key() == "arguments") {   /* as JSON; text that is not JSON yet (""), as sent */
            const json a = json::parse(k.value().get<std::string>(), nullptr, false);
            o["arguments"] = a.is_discarded() ? k.value() : a;
        }
        else o[k.key()] = k.value();
    }
    return o;
}

json norm_events(const std::string& raw, bool strata) {
    json out = json::array();
    for (const Ev& e : events(raw)) {
        json d = e.data;
        /* Deviation REASONING-PARTS: Radiance 1.17.0 wraps a reasoning item's text in
         * content_part.added / .done (unchanged here, golden C4/C5); Strata sends none. */
        if (!strata && (e.type == "response.content_part.added" || e.type == "response.content_part.done") &&
            d["part"].value("type", "") == "reasoning_text")
            continue;
        /* Deviation FN-ARGS-DONE: Strata's function_call_arguments.done also names the function;
         * Radiance 1.17.0's does not (function streaming unchanged, golden C5). */
        if (strata && e.type == "response.function_call_arguments.done") d.erase("name");
        d.erase("sequence_number");
        d.erase("type");
        d.erase("logprobs");
        d.erase("obfuscation");
        if (d.contains("item_id")) d["item_id"] = "ITEM";
        if (d.contains("item")) d["item"] = norm_item(d["item"]);
        if (d.contains("part")) d["part"] = json{ { "type", d["part"].value("type", "") }, { "text", d["part"].value("text", "") } };
        if (d.contains("response")) {
            json items = json::array();
            for (const json& i : d["response"]["output"]) items.push_back(norm_item(i));
            d = json{ { "status", d["response"].value("status", "") }, { "output", items } };
        }
        const bool delta = e.type.size() > 6 && e.type.compare(e.type.size() - 6, 6, ".delta") == 0;
        if (delta && !out.empty() && out.back().value("type", "") == e.type &&
            out.back().value("output_index", -1) == d.value("output_index", -2)) {
            out.back()["delta"] = out.back().value("delta", "") + d.value("delta", "");
            continue;
        }
        d["type"] = e.type;
        out.push_back(d);
    }
    return out;
}

json norm_body(const std::string& raw) {
    const json o = json::parse(raw, nullptr, false);
    json items = json::array();
    if (o.is_object() && o.contains("output"))
        for (const json& i : o["output"]) items.push_back(norm_item(i));
    return json{ { "status", o.is_object() ? o.value("status", "") : "?" }, { "output", items } };
}

/* The conversation as both build it: roles, the calls each assistant turn makes (name and
 * arguments) and the tool results' text. */
json norm_messages(const json& msgs, bool strata) {
    json out = json::array();
    for (const json& m : msgs) {
        json o = { { "role", m.value("role", "") } };
        if (m.contains("tool_calls")) {
            json calls = json::array();
            for (const json& c : m["tool_calls"]) {
                const json a = c["function"]["arguments"];
                calls.push_back(json{ { "name", c["function"].value("name", "") },
                                      { "arguments", a.is_string() ? json::parse(a.get<std::string>(), nullptr, false) : a } });
            }
            o["tool_calls"] = calls;
        }
        if (o["role"] == "tool") {
            const json& c = m["content"];
            std::string text = c.is_string() ? c.get<std::string>() : "";
            if (c.is_array())
                for (const json& p : c) text += p.value("text", "");
            o["content"] = text;
        }
        (void) strata;
        out.push_back(o);
    }
    return out;
}

/* Deviation NS-FN: a namespaced FUNCTION is offered as <namespace>__<name> on Radiance (1.17.0's
 * naming, kept) and <namespace>.<name> on Strata; the scripts name it @NSFN:<ns>:<name>@. */
std::string radiance_script(std::string s) {
    static const std::regex nsfn(R"re(@NSFN:([^:@]+):([^@]+)@)re");
    return std::regex_replace(s, nsfn, "$1__$2");
}
std::string ns_fn_as_strata(std::string name, const std::map<std::string, std::pair<std::string, std::string>>& ns) {
    auto it = ns.find(name);
    return it != ns.end() && name.find("__") != std::string::npos ? it->second.first + "." + it->second.second : name;
}

}  // namespace

/* For each of Strata's custom-tool cases, and each matrix row Strata covers, the same request gives
 * the same offered tools, conversation, items and stream events as Strata's own server (recorded
 * from Strata's serve/responses.py by its MockEngine and HTTP server:
 * tests/data/chat/strata-compat-custom-tools.json). The named deviations are normalised here,
 * each with its reason (see the GUIDE and oai_custom_tools.h). */
TEST(the_same_requests_answer_as_strata_does) {
    const json doc = json::parse(read_fixture("strata-compat-custom-tools.json"));
    REQUIRE(doc.contains("cases") && doc["cases"].size() >= 20);
    int compared = 0;
    for (const json& c : doc["cases"]) {
        const std::string name = c["name"].get<std::string>();
        auto differ = [&](const char* what, const json& strata, const json& radiance) {
            if (strata == radiance) { ++compared; return; }
            const size_t cut = getenv("CT_FULL") ? std::string::npos : 600;   /* CT_FULL=1: whole answers */
            fprintf(stderr, "    %s %s\n      strata:   %s\n      radiance: %s\n", name.c_str(), what,
                    strata.dump().substr(0, cut).c_str(), radiance.dump().substr(0, cut).c_str());
            ::radtest::fail(__FILE__, __LINE__, name + ": " + what + " differs from Strata");
        };
        const std::string req = c["request"].dump();
        const Conv rc = convert(req);
        /* the offer: every tool's name in order (NS-FN mapped), and each custom tool exactly */
        const json& st = c["strata_tools"];
        if (st.contains("tools") && rc.rc == RAD_OK) {
            json sn = json::array(), rn = json::array(), sc = json::array(), rcust = json::array();
            json sf = json::array(), rf = json::array();
            /* the namespaces the request (and its additional_tools items) declares: their descriptions,
             * and their untyped members */
            std::map<std::string, std::string> ns_desc;
            std::set<std::string> untyped;
            json declared = c["request"].value("tools", json::array());
            if (c["request"]["input"].is_array())
                for (const json& it : c["request"]["input"])
                    if (it.is_object() && it.value("type", "") == "additional_tools" && it.contains("tools") && it["tools"].is_array())
                        for (const json& t : it["tools"]) declared.push_back(t);
            for (const json& t : declared)
                if (t.is_object() && t.value("type", "") == "namespace") {
                    ns_desc[t.value("name", "")] = t.value("description", "");
                    for (const json& m : t.value("tools", json::array()))
                        if (!m.contains("type")) untyped.insert(t.value("name", "") + "." + m.value("name", ""));
                }
            if (st["tools"].is_array()) {
                /* DUP-ADDITIONAL: Strata offers every definition of a name an additional_tools item repeats
                 * (its name map keeps the last); Radiance offers the last one only, where Strata's last is. */
                std::map<std::string, size_t> last;
                for (size_t k = 0; k < st["tools"].size(); ++k) last[st["tools"][k]["name"].get<std::string>()] = k;
                for (size_t k = 0; k < st["tools"].size(); ++k) {
                    const json& t = st["tools"][k];
                    const std::string n = t["name"].get<std::string>();
                    if (last[n] != k) continue;
                    if (untyped.count(n)) continue;   /* UNTYPED-MEMBER: Strata offers an untyped member as a function; Radiance leaves it out */
                    sn.push_back(t["name"]);
                    const json& kind = st["names"].contains(n) ? st["names"][n][2] : json();
                    if (kind == "custom")
                        sc.push_back(json{ { "name", t["name"] }, { "description", t["description"] }, { "parameters", t["parameters"] } });
                    if (kind == "function") {
                        /* NS-DESC: Strata leads a namespace's function member's description with the namespace's;
                         * Radiance keeps the member's own (1.17.0, function tools unchanged) */
                        std::string d = t["description"].get<std::string>();
                        const json& space = st["names"][n][0];
                        if (space.is_string() && !ns_desc[space.get<std::string>()].empty()) {
                            const std::string lead = ns_desc[space.get<std::string>()];
                            if (d == lead) d = "";
                            else if (d.rfind(lead + "\n\n", 0) == 0) d = d.substr(lead.size() + 2);
                        }
                        sf.push_back(json{ { "name", t["name"] }, { "description", d }, { "parameters", t["parameters"] } });
                    }
                }
            }
            /* Deviation TOOLS-NONE is checked separately below: Strata offers no tools for "none". */
            const bool none = c["request"].value("tool_choice", json()) == "none";
            if (rc.chat.contains("tools") && !none)
                for (const json& t : rc.chat["tools"]) {
                    const std::string n = t["function"]["name"].get<std::string>();
                    rn.push_back(ns_fn_as_strata(n, rc.ns));
                    if (rc.custom.count(n)) {
                        rcust.push_back(json{ { "name", n }, { "description", t["function"].value("description", "") },
                                              { "parameters", t["function"]["parameters"] } });
                    } else {
                        /* FN-DEFAULTS: Strata gives a function without them the description "" and the parameters
                         * {"type":"object","properties":{}}; Radiance omits the one and sends {} (1.17.0) */
                        json params = t["function"]["parameters"];
                        if (params == json::object()) params = json::parse(R"({"type":"object","properties":{}})");
                        rf.push_back(json{ { "name", ns_fn_as_strata(n, rc.ns) }, { "description", t["function"].value("description", "") },
                                           { "parameters", params } });
                    }
                }
            if (!none) differ("function tools", sf, rf);
            if (!none) differ("tool names", sn, rn);
            differ("custom tools", sc, rcust);
            std::map<std::string, json> sm, rm;   /* compared as maps: the order is not the point */
            for (auto it = st["names"].begin(); it != st["names"].end(); ++it)
                if (it.value()[2] == "custom") sm[it.key()] = json::array({ it.value()[0], it.value()[1] });
            for (const std::string& n : rc.custom) {
                auto ns = rc.ns.find(n);
                rm[n] = ns != rc.ns.end() ? json::array({ ns->second.first, ns->second.second }) : json::array({ nullptr, n });
            }
            differ("custom name map", sm, rm);
            if (none) {
                /* TOOLS-NONE: Strata leaves every tool out; Radiance sends them with tool_choice
                 * "none", which its chat path honours (no call is made). Same for function tools. */
                CHECK(st["tools"].is_null());
                CHECK_EQ(rc.chat["tool_choice"], json("none"));
            }
        }
        /* the conversation */
        if (c["strata_messages"].is_array() && rc.rc == RAD_OK) {
            json sm = norm_messages(c["strata_messages"], true), rm = norm_messages(rc.chat["messages"], false);
            for (json& m : rm)
                if (m.contains("tool_calls"))
                    for (json& t : m["tool_calls"]) t["name"] = ns_fn_as_strata(t["name"].get<std::string>(), rc.ns);
            /* Strata keeps the instructions-less system message out, as Radiance does: no change */
            differ("conversation", sm, rm);
        }
        if (c.contains("history_request")) {
            const Conv h = convert(c["history_request"].dump());
            CHECK_OK(h.rc);
            json rm = norm_messages(h.chat["messages"], false);
            for (json& m : rm)
                if (m.contains("tool_calls"))
                    for (json& t : m["tool_calls"]) {
                        std::string n = t["name"].get<std::string>();
                        const size_t u = n.find("__");
                        if (u != std::string::npos) n = n.substr(0, u) + "." + n.substr(u + 2);   /* NS-FN */
                        t["name"] = n;
                    }
            differ("history round trip", norm_messages(c["strata_history_messages"], true), rm);
        }
        /* the answers */
        if (c.contains("strata_body")) {
            Fixture f;
            f.sched.piece_bytes = 64;   /* the long input as ~330 tokens: far below its Sink's capacity */
            use_qwen(f, req);
            f.sched.reply = radiance_script(c["reply"].get<std::string>());
            HttpResponse r = f.post("/v1/responses", req);
            f.sched.join_all();
            json want = norm_body(c["strata_body"]["raw"].get<std::string>()), got = norm_body(r.body);
            if (name == "custom_call_cut_short") {
                /* CUT-BODY: a reply that ends inside a call is, not streamed, the text the model
                 * wrote on Radiance (1.17.0, every tool kind: a_reply_ending_inside_a_call...);
                 * Strata answers an incomplete call item. The stream is compared below. */
                CHECK_EQ(got["output"][0].value("type", ""), std::string("message"));
                CHECK_EQ(want["output"][0].value("status", ""), std::string("incomplete"));
            } else {
                differ("answer", want, got);
            }
            const std::string sreq = req.substr(0, req.size() - 1) + R"(,"stream":true})";
            HttpResponse s = f.post("/v1/responses", sreq);
            const std::string body = drain(s);
            f.sched.join_all();
            differ("stream events", norm_events(c["strata_stream"]["raw"].get<std::string>(), true), norm_events(body, false));
        }
    }
    fprintf(stderr, "    %d comparisons with Strata (%s, responses.py %s)\n", compared,
            doc.value("strata_commit", "").substr(0, 10).c_str(), doc.value("responses_py_sha256", "").substr(0, 12).c_str());
}

RAD_TEST_MAIN()
