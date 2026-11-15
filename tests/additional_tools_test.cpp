/* tests/additional_tools_test.cpp -- the Responses API's `additional_tools` input items
 * (oai_additional_tools.h), through responses_to_chat and the HTTP layer as server_test.cpp does.
 *
 * Ported from Strata's serve/test_responses.py test_additional_tools_input_items_are_accepted and
 * JsonFormatWithTools (Strata #782; MIT, Copyright (c) 2026 Niko1221 and the Strata contributors).
 */
#include "server_fakes.h"

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

std::vector<std::string> tool_names(const json& chat) {
    std::vector<std::string> n;
    if (chat.contains("tools"))
        for (const json& t : chat["tools"]) n.push_back(t["function"]["name"].get<std::string>());
    return n;
}

const char* kTools = R"("tools":[{"type":"function","name":"exec_command","description":"Runs a command.",
    "parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}])";

}  // namespace

/* Strata's test: the item adds no message, and its tools follow the request's; an item with no
 * tools array adds nothing. */
TEST(additional_tools_input_items_are_accepted) {
    const Conv c = convert(std::string(R"({"input":[{"type":"message","role":"user","content":"hi"},
        {"type":"additional_tools","tools":[{"type":"function","name":"later_tool","parameters":{"type":"object"}}]},
        {"type":"additional_tools"}],)") + kTools + "}");
    CHECK_OK(c.rc);
    REQUIRE(c.chat["messages"].size() == 1);
    CHECK_EQ(c.chat["messages"][0].value("role", ""), std::string("user"));
    CHECK_EQ(json(tool_names(c.chat)).dump(), std::string(R"(["exec_command","later_tool"])"));
}

/* Each kind of tool an item carries is served as the same tool in `tools` would be; the response
 * echoes the request's own tools only, as Strata does. */
TEST(an_additional_tool_is_served_as_a_request_tool_is) {
    const Conv c = convert(std::string(R"({"input":[{"type":"additional_tools","tools":[
        {"type":"custom","name":"apply_patch"},
        {"type":"namespace","name":"crm","tools":[{"type":"function","name":"find","parameters":{"type":"object"}}]},
        {"type":"web_search"}]},
        {"role":"user","content":"go"},
        {"type":"additional_tools","tools":"not an array"}],)") + kTools + "}");
    CHECK_OK(c.rc);
    CHECK_EQ(json(tool_names(c.chat)).dump(), std::string(R"(["exec_command","apply_patch","crm__find"])"));
    CHECK(c.custom.count("apply_patch") == 1);
    CHECK_EQ(c.ns.at("crm__find").first, std::string("crm"));
    REQUIRE(c.echo["tools"].size() == 1);
    CHECK_EQ(c.echo["tools"][0].value("name", ""), std::string("exec_command"));
    /* a request with no tools of its own */
    const Conv only = convert(R"({"input":[{"role":"user","content":"go"},
        {"type":"additional_tools","tools":[{"type":"function","name":"later_tool"}]}]})");
    CHECK_OK(only.rc);
    CHECK_EQ(json(tool_names(only.chat)).dump(), std::string(R"(["later_tool"])"));
    CHECK(only.echo["tools"].empty());
    /* and nothing changes for a request without the item */
    const Conv none = convert(std::string(R"({"input":"hi",)") + kTools + "}");
    CHECK_OK(none.rc);
    CHECK_EQ(json(tool_names(none.chat)).dump(), std::string(R"(["exec_command"])"));
    CHECK_EQ(none.echo["tools"].size(), (size_t)1);
}

/* A tool an additional_tools item names again replaces the earlier one, as in Strata (its later
 * definition wins): Codex re-sending a tool must not fail the turn. Two tools of one name in the
 * request's own tools are still refused (Radiance's rule, deviation DUP-NAMES). */
TEST(an_additional_tool_named_again_replaces_the_earlier_one) {
    const Conv c = convert(std::string(R"({"input":[{"role":"user","content":"x"},
        {"type":"additional_tools","tools":[{"type":"function","name":"exec_command","description":"Again."},
                                            {"type":"custom","name":"later"}]},
        {"type":"additional_tools","tools":[{"type":"function","name":"later","parameters":{"type":"object"}}]}],)") + kTools + "}");
    CHECK_OK(c.rc);
    CHECK_EQ(json(tool_names(c.chat)).dump(), std::string(R"(["exec_command","later"])"));
    const json& tools = c.chat["tools"];
    REQUIRE(tools.size() == 2);
    CHECK_EQ(tools[0]["function"].value("description", ""), std::string("Again."));
    CHECK(c.custom.empty());   /* "later" was a custom tool, then a function tool: the function wins */
    REQUIRE(c.echo["tools"].size() == 1);
    CHECK_EQ(c.echo["tools"][0].value("description", ""), std::string("Runs a command."));
    /* a custom tool replacing a function tool is served as custom, and its call comes back so */
    const std::string body = std::string(R"({"max_output_tokens":200,"input":[{"role":"user","content":"x"},
        {"type":"additional_tools","tools":[{"type":"custom","name":"exec_command"}]}],)") + kTools + "}";
    const Conv k = convert(body);
    CHECK_OK(k.rc);
    CHECK(k.custom.count("exec_command") == 1);
    Fixture f;
    f.sched.reply = R"(TOOL:exec_command:{"input":"ls"})";
    const json o = json::parse(f.post("/v1/responses", body).body);
    CHECK_EQ(o["output"][0].value("type", ""), std::string("custom_tool_call"));
    f.sched.join_all();
    /* a namespace's tool named again */
    const Conv n = convert(R"({"input":[{"role":"user","content":"x"},
        {"type":"additional_tools","tools":[{"type":"namespace","name":"crm","tools":[{"type":"function","name":"find"}]}]}],
        "tools":[{"type":"namespace","name":"crm","tools":[{"type":"function","name":"find","description":"old"}]}]})");
    CHECK_OK(n.rc);
    CHECK_EQ(json(tool_names(n.chat)).dump(), std::string(R"(["crm__find"])"));
    CHECK(!n.chat["tools"][0]["function"].contains("description"));
    CHECK_EQ(n.ns.at("crm__find").first, std::string("crm"));
    /* the request's own tools: refused, as before */
    const Conv own = convert(R"({"input":"x","tools":[{"type":"function","name":"a"},{"type":"function","name":"a"}]})");
    CHECK(own.rc < 0);
    CHECK_EQ(own.err.status, 400);
}

/* Strata's JsonFormatWithTools request (Codex's review turn): the item, a developer role on it,
 * first in the input. The server answers it, and calls an additional tool as any other. */
TEST(codexs_review_request_with_an_additional_tools_item_is_served) {
    const std::string body = std::string(R"({"model":"m","instructions":"You are a reviewer.","input":[
        {"type":"additional_tools","role":"developer","tools":[{"type":"function","name":"later_tool","parameters":{"type":"object"}}]},
        {"type":"message","role":"user","content":[{"type":"input_text","text":"review this call"}]}],
        "tool_choice":"auto","parallel_tool_calls":false,"store":false,"max_output_tokens":200,)") + kTools + "}";
    const Conv c = convert(body);
    CHECK_OK(c.rc);
    CHECK_EQ(json(tool_names(c.chat)).dump(), std::string(R"(["exec_command","later_tool"])"));
    for (const json& m : c.chat["messages"]) CHECK(m.value("role", "") != "developer");
    Fixture f;
    f.sched.reply = R"(TOOL:later_tool:{})";
    HttpResponse r = f.post("/v1/responses", body);
    CHECK_EQ(r.status, 200);
    const json o = json::parse(r.body);
    REQUIRE(o["output"].size() >= 1);
    CHECK_EQ(o["output"].back().value("type", ""), std::string("function_call"));
    CHECK_EQ(o["output"].back().value("name", ""), std::string("later_tool"));
    f.sched.join_all();
}

RAD_TEST_MAIN()
