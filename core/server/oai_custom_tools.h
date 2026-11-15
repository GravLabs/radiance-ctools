/* core/server/oai_custom_tools.h -- the Responses API's freeform custom tools on the chat path.
 *
 * A custom tool (`{"type":"custom","name":...,"format":...}`, Codex's apply_patch) takes a raw
 * string, not JSON arguments. The chat path has only function tools, so a custom tool is offered
 * to the model as a function tool with one string parameter, `input`; a grammar `format` is
 * written into its description for the model to read. The reply's call to it is a
 * custom_tool_call item with that raw `input`, and the history's custom_tool_call /
 * custom_tool_call_output items map onto an assistant tool call and a tool message.
 *
 * Ported from Strata's serve/responses.py (MIT, Copyright (c) 2026 Niko1221 and the Strata
 * contributors): the same offer, the same input/reply mapping and the same stream events
 * (response.custom_tool_call_input.delta / .done, the input sent whole at the call's end).
 *
 * Kept in its own header, with only thin hooks in oai.cpp, so the change rebases onto new
 * releases as one piece. Nothing here throws (oai.h).
 */
#pragma once
#include "oai.h"

#include <nlohmann/json.hpp>

#include <string>

namespace rad {
namespace server {
namespace custom_tools {

/* `json` is the server's ordered_json (iface.h), as in oai.cpp. */

/* The function tool a custom tool is offered as, in the Responses shape the tool list's `add`
 * reads (name, description, parameters). A tool needs a non-empty name. */
inline int as_function(const json& t, json* fn, ApiError* err, const char* param) {
    auto n = t.find("name");
    if (n == t.end() || !n->is_string() || n->get<std::string>().empty()) {
        *err = err_bad(param, "a custom tool needs a name");
        return RAD_E_INVAL;
    }
    std::string description;
    auto d = t.find("description");
    if (d != t.end() && d->is_string()) description = d->get<std::string>();
    /* A grammar the input must follow (Codex: lark or regex) is shown to the model as text. */
    auto f = t.find("format");
    if (f != t.end() && f->is_object() && f->value("type", json()) == "grammar") {
        auto def = f->find("definition");
        if (def != f->end() && def->is_string() && !def->get<std::string>().empty()) {
            auto syn = f->find("syntax");
            const std::string syntax = syn != f->end() && syn->is_string() ? syn->get<std::string>() : "";
            description += "\n\nThe input must follow this " + syntax + " grammar:\n" + def->get<std::string>();
        }
    }
    *fn = json{ { "type", "function" }, { "name", *n }, { "description", description },
                { "parameters",
                  { { "type", "object" },
                    { "properties", { { "input", { { "type", "string" }, { "description", "the raw input" } } } } },
                    { "required", json::array({ "input" }) } } } };
    return RAD_OK;
}

/* The chat tool call a custom_tool_call input item stands for: its raw input as the arguments
 * {"input": ...}, under the name its tool was offered as (<namespace>__<name>). */
inline int input_call(const json& item, json* call, ApiError* err) {
    std::string id, name, input, space;
    auto str = [&](const char* k, const char* param, std::string* out) {
        auto it = item.find(k);
        if (it == item.end() || it->is_null()) return true;
        if (!it->is_string()) { *err = err_bad(param, "expected a string"); return false; }
        *out = it->get<std::string>();
        return true;
    };
    if (!str("call_id", "input[].call_id", &id) || !str("name", "input[].name", &name) ||
        !str("input", "input[].input", &input) || !str("namespace", "input[].namespace", &space))
        return RAD_E_INVAL;
    if (name.empty()) {
        *err = err_bad("input[].name", "a custom_tool_call needs a name");
        return RAD_E_INVAL;
    }
    if (!space.empty()) name = space + "__" + name;
    *call = json{ { "id", id }, { "type", "function" },
                  { "function", { { "name", name }, { "arguments", json{ { "input", input } }.dump() } } } };
    return RAD_OK;
}

/* Whether the reply's call (under its flattened name) is to a custom tool of this request. */
inline bool is_custom(const OaiRequest& r, const ToolCall& tc) {
    return r.tool_custom.count(tc.name) != 0;
}

/* The raw input of a call to a custom tool: the arguments' `input` string, or the arguments as
 * the model wrote them when they are not that (a parameter of another name, not JSON). */
inline std::string input_of(const std::string& arguments) {
    const json a = json::parse(arguments, nullptr, false);
    if (a.is_object()) {
        auto it = a.find("input");
        if (it != a.end() && it->is_string()) return it->get<std::string>();
    }
    return arguments;
}

/* The id a custom_tool_call item carries, from the call id this server minted ("call_..."). */
inline std::string item_id(const std::string& call_id) {
    return "ctc_" + (call_id.rfind("call_", 0) == 0 ? call_id.substr(5) : call_id);
}

/* The reply's custom_tool_call item; a namespaced tool reports its namespace and name apart. */
inline json item(const OaiRequest& r, const ToolCall& tc, const char* status, const std::string& input) {
    json it = { { "id", item_id(tc.id) }, { "type", "custom_tool_call" }, { "call_id", tc.id },
                { "name", tc.name }, { "input", input }, { "status", status } };
    auto ns = r.tool_namespace.find(tc.name);
    if (ns != r.tool_namespace.end()) {
        it["namespace"] = ns->second.first;
        it["name"] = ns->second.second;
    }
    return it;
}

}  /* namespace custom_tools */
}  /* namespace server */
}  /* namespace rad */
