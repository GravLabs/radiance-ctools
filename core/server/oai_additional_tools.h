/* core/server/oai_additional_tools.h -- the Responses API's `additional_tools` input items.
 *
 * Codex (v0.160 and later) can send more tool definitions as an input item,
 * `{"type":"additional_tools","tools":[...]}`, a `tools` array as the request has. Those tools join
 * the request's own, after them, and the item adds nothing to the conversation; an item without a
 * usable `tools` array adds no tool. The response echoes the request's own `tools` only.
 *
 * Ported from Strata's serve/responses.py, request_tools and input_messages (Strata #782; MIT,
 * Copyright (c) 2026 Niko1221 and the Strata contributors), with its test
 * test_additional_tools_input_items_are_accepted.
 *
 * Kept in its own header, with thin hooks in oai.cpp, so the change rebases as one piece. Nothing
 * here throws (oai.h).
 */
#pragma once
#include "oai.h"

#include <nlohmann/json.hpp>

#include <cstddef>

namespace rad {
namespace server {
namespace additional_tools {

/* `json` is the server's ordered_json (iface.h), as in oai.cpp. */

/* The tools a request offers: its own `tools` (when an array), then every additional_tools input
 * item's. *own is how many of them are the request's own. Null when the request has neither. */
inline json offered(const json& body, size_t* own) {
    json all;
    *own = 0;
    auto t = body.find("tools");
    if (t != body.end() && t->is_array()) {
        all = *t;
        *own = t->size();
    }
    auto in = body.find("input");
    if (in == body.end() || !in->is_array()) return all;
    for (const json& item : *in) {
        if (!item.is_object() || item.value("type", json()) != "additional_tools") continue;
        auto more = item.find("tools");
        if (more == item.end() || !more->is_array()) continue;
        if (all.is_null()) all = json::array();
        for (const json& tool : *more) all.push_back(tool);
    }
    return all;
}

}  /* namespace additional_tools */
}  /* namespace server */
}  /* namespace rad */
