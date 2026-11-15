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

/* A namespaced custom tool's offered name, Strata's "<namespace>.<name>". (Radiance's namespaced
 * function tools keep their own "<namespace>__<name>".) */
inline std::string flat_name(const std::string& space, const std::string& name) {
    return space.empty() ? name : space + "." + name;
}

/* The function tool a custom tool is offered as, in the Responses shape the tool list's `add`
 * reads (name, description, parameters), under its flat name. A tool needs a non-empty name. In a
 * namespace, the namespace's description leads the tool's own, as in Strata. */
inline int as_function(const json& t, json* fn, ApiError* err, const char* param,
                       const std::string& space = std::string(), const std::string& ns_description = std::string()) {
    auto n = t.find("name");
    if (n == t.end() || !n->is_string() || n->get<std::string>().empty()) {
        *err = err_bad(param, "a custom tool needs a name");
        return RAD_E_INVAL;
    }
    std::string description;
    auto d = t.find("description");
    if (d != t.end() && d->is_string()) description = d->get<std::string>();
    if (!ns_description.empty())
        description = ns_description + (description.empty() ? "" : "\n\n" + description);
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
    *fn = json{ { "type", "function" }, { "name", flat_name(space, n->get<std::string>()) }, { "description", description },
                { "parameters",
                  { { "type", "object" },
                    { "properties", { { "input", { { "type", "string" }, { "description", "the raw input" } } } } },
                    { "required", json::array({ "input" }) } } } };
    return RAD_OK;
}

/* Python's truth of a JSON value, which Strata tests a field's presence by. */
inline bool py_truthy(const json& v) {
    if (v.is_null()) return false;
    if (v.is_boolean()) return v.get<bool>();
    if (v.is_number()) return v != json(0);
    if (v.is_string() || v.is_array() || v.is_object()) return !v.empty();
    return true;
}

/* The chat tool call a custom_tool_call input item stands for: its raw input as the arguments
 * {"input": ...}, under the name its tool was offered as (flat_name). As in Strata: a name is
 * required; an input that is not a string reads as ""; the call_id is passed through as sent; and
 * a namespace joins the name when it is set at all ("crm" -> crm.<name>, 5 -> 5.<name>; Strata
 * writes a list or object with Python's str, this its JSON). */
inline int input_call(const json& item, json* call, ApiError* err) {
    auto n = item.find("name");
    if (n == item.end() || !n->is_string() || n->get<std::string>().empty()) {
        *err = err_bad("input[].name", "a custom_tool_call needs a name");
        return RAD_E_INVAL;
    }
    std::string name = n->get<std::string>(), space;
    auto ns = item.find("namespace");
    if (ns != item.end() && py_truthy(*ns)) {
        if (ns->is_string()) space = ns->get<std::string>();
        else if (ns->is_boolean()) space = "True";
        else space = ns->dump();
    }
    auto c = item.find("call_id");
    const json id = c != item.end() && !c->is_null() ? *c : json("");
    auto in = item.find("input");
    const std::string input = in != item.end() && in->is_string() ? in->get<std::string>() : "";
    name = flat_name(space, name);
    *call = json{ { "id", id }, { "type", "function" },
                  { "function", { { "name", name }, { "arguments", json{ { "input", input } }.dump() } } } };
    return RAD_OK;
}

/* Whether the reply's call (under its flattened name) is to a custom tool of this request. */
inline bool is_custom(const OaiRequest& r, const ToolCall& tc) {
    return r.tool_custom.count(tc.name) != 0;
}

/* JSON written as Python's json.dumps(v, ensure_ascii=False) writes it (", " and ": "
 * separators, members in order), which is how Strata reports arguments that are not an input. */
inline void py_dumps(const json& v, std::string* out) {
    if (v.is_object()) {
        *out += '{';
        bool first = true;
        for (auto it = v.begin(); it != v.end(); ++it) {
            if (!first) *out += ", ";
            first = false;
            *out += json(it.key()).dump();
            *out += ": ";
            py_dumps(it.value(), out);
        }
        *out += '}';
    } else if (v.is_array()) {
        *out += '[';
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) *out += ", ";
            py_dumps(v[i], out);
        }
        *out += ']';
    } else {
        *out += v.dump();
    }
}

/* The raw input of a call to a custom tool: the arguments' `input` string; otherwise, as in
 * Strata, the arguments written out (json.dumps); arguments that are not JSON at all (a reply cut
 * inside them) as the model wrote them. */
inline std::string input_of(const std::string& arguments) {
    const json a = json::parse(arguments, nullptr, false);
    if (a.is_discarded()) return arguments;
    if (a.is_object()) {
        auto it = a.find("input");
        if (it != a.end() && it->is_string()) return it->get<std::string>();
    }
    std::string s;
    py_dumps(a, &s);
    return s;
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
