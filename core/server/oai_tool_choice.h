/* core/server/oai_tool_choice.h -- a Responses request's tool_choice, as the chat path is to serve it.
 *
 * A choice the request can be served by passes on: "auto", "none", "required", or a function the
 * request offers ({"type":"function","name":...}, a namespace joined as the tools are). A choice
 * it cannot honour is logged and acts as "auto", not a 400, so a client's odd choice does not stop
 * its request: a name no offered tool has, "required" with no tools, an unknown string or shape
 * (a custom tool's {"type":"custom",...} among them). "required" with exactly one tool forces that
 * tool, since the model cannot call another.
 *
 * Ported from Strata's serve/frontend.py tool_choice_of and forced_call (MIT, Copyright (c) 2026
 * Niko1221 and the Strata contributors). Kept in its own header, with one hook in oai.cpp, so the
 * change rebases as one piece. Nothing here throws (oai.h).
 */
#pragma once
#include "oai.h"
#include "rad_internal.h"

#include <nlohmann/json.hpp>

#include <set>
#include <string>

namespace rad {
namespace server {
namespace tool_choice {

/* `json` is the server's ordered_json (iface.h), as in oai.cpp. */

inline json named(const std::string& name) {
    return json{ { "type", "function" }, { "function", { { "name", name } } } };
}

/* The chat request's tool_choice for the Responses `choice`, given the chat `tools` the request
 * offers (each {"type":"function","function":{"name":...}}). */
inline json served(const json& choice, const json& tools) {
    std::set<std::string> names;
    if (tools.is_array())
        for (const json& t : tools)
            if (t.contains("function") && t["function"].contains("name") && t["function"]["name"].is_string())
                names.insert(t["function"]["name"].get<std::string>());
    std::string kind = "unknown", name;
    if (choice.is_string()) {
        const std::string s = choice.get<std::string>();
        if (s == "auto" || s == "none" || s == "required") kind = s;
    } else if (choice.is_object()) {
        const json type = choice.value("type", json());
        if (type == "auto" || type == "none") {
            kind = type.get<std::string>();
        } else if (type == "any" || type == "required") {
            kind = "required";
        } else if (type == "function" || type == "tool") {
            auto fn = choice.find("function");
            json n = fn != choice.end() && fn->is_object() ? fn->value("name", json()) : json();
            if (!n.is_string() || n.get<std::string>().empty()) n = choice.value("name", json());
            if (n.is_string() && !n.get<std::string>().empty()) {
                kind = "named";
                name = n.get<std::string>();
                /* Radiance's own addition: a namespace joins the name as its tools were (oai.cpp) */
                const json space = choice.value("namespace", json());
                if (space.is_string() && !space.get<std::string>().empty())
                    name = space.get<std::string>() + "__" + name;
            }
        }
    }
    if (kind == "auto" || kind == "none") return json(kind);
    if (kind == "required" && names.size() == 1) return named(*names.begin());
    if (kind == "required" && !names.empty()) return json("required");
    if (kind == "named" && names.count(name)) return named(name);
    RAD_DEBUG("responses: tool_choice %s is not served here (or names no tool of the request): the model "
              "decides, as with \"auto\"", choice.dump().substr(0, 200).c_str());
    return json("auto");
}

}  /* namespace tool_choice */
}  /* namespace server */
}  /* namespace rad */
