/* core/server/oai_tool_results.h -- tool results in the order of their calls, on /v1/responses.
 *
 * A chat template places tool results by position: the first tool message after an assistant turn
 * answers its first call. A Responses client may send a turn's function_call_output /
 * custom_tool_call_output items in the order the calls finished, which would pair a result with
 * the wrong call in the prompt. So each run of tool messages is put in the order of the calls of
 * the assistant message just before it, by call id (stably; a result whose id that turn does not
 * call keeps its place after the matched ones).
 *
 * Ported from Strata's serve/responses.py _order_tool_results (MIT, Copyright (c) 2026 Niko1221
 * and the Strata contributors). Kept in its own header, with one hook in oai.cpp, so the change
 * rebases as one piece. Nothing here throws (oai.h).
 */
#pragma once
#include "oai.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace rad {
namespace server {
namespace tool_results {

/* `json` is the server's ordered_json (iface.h), as in oai.cpp. */

/* Python's truth of a call id, which Strata ranks only set ids by. */
inline bool id_set(const json& v) {
    if (v.is_null()) return false;
    if (v.is_boolean()) return v.get<bool>();
    if (v.is_number()) return v != json(0);
    if (v.is_string() || v.is_array() || v.is_object()) return !v.empty();
    return true;
}

/* Reorders each run of role "tool" messages in `msgs` (chat messages) into the call order of the
 * message before the run. Ids compare as JSON values: the call "9" is not the call 9. */
inline void order(json* msgs) {
    const size_t n = msgs->size();
    size_t i = 0;
    while (i < n) {
        if ((*msgs)[i].value("role", "") != "tool") { ++i; continue; }
        size_t j = i;
        while (j < n && (*msgs)[j].value("role", "") == "tool") ++j;
        std::map<std::string, size_t> rank;   /* a call id (as JSON text) -> its place; the last wins */
        size_t calls = 0;
        if (i > 0) {
            const json& prev = (*msgs)[i - 1];
            auto tc = prev.find("tool_calls");
            if (tc != prev.end() && tc->is_array()) {
                calls = tc->size();
                for (size_t k = 0; k < calls; ++k) {
                    auto id = (*tc)[k].find("id");
                    if (id != (*tc)[k].end() && id_set(*id)) rank[id->dump()] = k;
                }
            }
        }
        if (!rank.empty()) {
            auto place = [&](const json& m) {
                auto id = m.find("tool_call_id");
                if (id == m.end()) return calls;
                auto r = rank.find(id->dump());
                return r == rank.end() ? calls : r->second;
            };
            std::vector<json> run((*msgs).begin() + (std::ptrdiff_t)i, (*msgs).begin() + (std::ptrdiff_t)j);
            std::stable_sort(run.begin(), run.end(), [&](const json& a, const json& b) { return place(a) < place(b); });
            for (size_t k = 0; k < run.size(); ++k) (*msgs)[i + k] = std::move(run[k]);
        }
        i = j;
    }
}

}  /* namespace tool_results */
}  /* namespace server */
}  /* namespace rad */
