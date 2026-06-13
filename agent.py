import json
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, MAX_TOOL_ROUNDS
from tools import lookup_plant, get_seasonal_conditions, _plant_db

_client = Groq(api_key=GROQ_API_KEY)

# ──────────────────────────────────────────────
# Tool definitions
#
# These are the schemas that tell the LLM what tools are available and how to
# call them. The LLM reads these descriptions and decides when (and how) to use
# each tool. They're already complete — your job is to implement the tool
# functions in tools.py and the agent loop below.
# ──────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_plant",
            "description": (
                "Look up care information for a specific houseplant by name. "
                "Returns detailed watering, light, humidity, and temperature requirements. "
                "Use this whenever the user asks about a specific plant."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plant_name": {
                        "type": "string",
                        "description": "The plant name to look up. Can be a common name, scientific name, or nickname (e.g., 'pothos', 'devil's ivy', 'Monstera deliciosa').",
                    }
                },
                "required": ["plant_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_seasonal_conditions",
            "description": (
                "Get seasonal care adjustments for houseplants. "
                "Returns guidance on watering, fertilizing, light, and pests for the current or specified season. "
                "Use this when a user asks a season-specific question, or to complement plant care advice with seasonal context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "season": {
                        "type": "string",
                        "description": "The season to get care conditions for. If omitted, the current season is detected automatically.",
                        "enum": ["spring", "summer", "fall", "winter"],
                    }
                },
                "required": [],
            },
        },
    },
]

# ──────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a knowledgeable and friendly plant care advisor. "
    "Help users care for their houseplants by looking up specific plant information "
    "and current seasonal conditions using your available tools.\n\n"
    "Always use your tools to look up plant-specific information before answering — "
    "don't rely on your general knowledge alone.\n\n"
    "When lookup_plant returns found: False, follow this pattern exactly:\n"
    "  1. Acknowledge clearly that the plant isn't in your database.\n"
    "  2. Identify what category of plant it likely is (succulent, tropical, fern, etc.).\n"
    "  3. Give 2-3 practical care tips for that category.\n"
    "  4. Suggest a reliable resource for specific data (e.g. the ASPCA plant database, "
    "     a local nursery, or the American Horticultural Society).\n"
    "Never invent specific watering schedules or measurements for plants not in your database.\n\n"
    "When you have seasonal data, always state the specific season by name and "
    "give at least one concrete adjustment from it.\n\n"
    "You only answer questions about houseplants and plant care. If the user asks "
    "about something unrelated — cars, food, sports, etc. — politely explain that "
    "you're a plant care advisor and decline to answer."
)

# ──────────────────────────────────────────────
# Tool dispatch
#
# This is already complete. It routes tool calls from the LLM to the actual
# Python functions in tools.py, and returns results as JSON strings (which is
# what the Groq API expects for tool results).
# ──────────────────────────────────────────────


def dispatch_tool(tool_name: str, tool_args: dict) -> str:
    """Route a tool call to the correct function and return the result as a JSON string."""
    # Some models send arguments as JSON "null" for no-argument tools, which
    # json.loads() turns into None — normalize so .get() below is always safe.
    if not isinstance(tool_args, dict):
        tool_args = {}
    print(f"  → Tool call: {tool_name}({tool_args})")
    if tool_name == "lookup_plant":
        result = lookup_plant(tool_args["plant_name"])
    elif tool_name == "get_seasonal_conditions":
        result = get_seasonal_conditions(tool_args.get("season"))
    else:
        result = {"error": f"Unknown tool: {tool_name}"}
    print(
        f"  ← Result: {json.dumps(result)[:120]}{'...' if len(json.dumps(result)) > 120 else ''}"
    )
    return json.dumps(result)


# ──────────────────────────────────────────────
# Agent loop
# ──────────────────────────────────────────────


"""
def run_agent(user_message: str, history: list) -> str:
   
    Run the plant care agent for one user turn and return its response.

    TODO — Milestone 2:

    The agent loop follows a specific pattern that you'll implement here. Read
    specs/agent-loop-spec.md carefully before writing any code — understand the
    full loop before implementing any part of it.

    The loop works like this:
      1. Build a messages list: system prompt + conversation history + new user message
      2. Call the LLM with messages and TOOL_DEFINITIONS
      3. If the response contains tool_calls:
           a. Append the assistant message (with tool_calls) to messages
           b. For each tool call: execute via dispatch_tool(), append the result
           c. Call the LLM again with the updated messages
           d. Repeat until no more tool_calls (or MAX_TOOL_ROUNDS is reached)
      4. Return the final text response

    Key details to get right:
      - The assistant message must be appended BEFORE tool results
      - Tool result messages use role="tool" with a tool_call_id field
      - Append the assistant's message object directly (not just its content)
      - The history format from Gradio: list of {"role": ..., "content": ...} dicts

    Before writing code, complete specs/agent-loop-spec.md.


    """


def extract_plant_memory(history: list) -> str:
    """
    Scan conversation history for plant names and issues the user has mentioned.
    Returns a memory block to inject into the system prompt, or empty string if none.
    """
    mentioned_plants = []
    known_issues = []

    for human, assistant in history:
        if not human:
            continue

        human_lower = human.lower()

        # Check every plant in the database against what the user said
        for slug, plant in _plant_db.items():
            names_to_check = (
                [slug]
                + [plant["display_name"].lower()]
                + [a.lower() for a in plant.get("aliases", [])]
            )
            for name in names_to_check:
                if (
                    name in human_lower
                    and plant["display_name"] not in mentioned_plants
                ):
                    mentioned_plants.append(plant["display_name"])

        # Simple issue detection — look for problem keywords near plant mentions
        issue_keywords = [
            "yellow",
            "brown",
            "drooping",
            "wilting",
            "dying",
            "mushy",
            "crispy",
            "spots",
            "bugs",
            "gnats",
            "root rot",
            "overwater",
        ]
        for keyword in issue_keywords:
            if keyword in human_lower:
                # Pair the issue with the most recently mentioned plant if possible
                plant_ref = (
                    mentioned_plants[-1] if mentioned_plants else "unknown plant"
                )
                issue = f"{plant_ref} — {keyword}"
                if issue not in known_issues:
                    known_issues.append(issue)

    if not mentioned_plants:
        return ""

    lines = [
        "\n--- Conversation memory ---",
        f"Plants this user has mentioned: {', '.join(mentioned_plants)}",
    ]
    if known_issues:
        lines.append(f"Issues described: {', '.join(known_issues)}")
    lines.append(
        "When answering general questions, connect your advice to these specific "
        "plants where relevant. Do not re-look them up unless the user asks something new."
    )
    lines.append("--- End memory ---")

    return "\n".join(lines)


def run_agent(user_message: str, history: list) -> str:
    memory_block = extract_plant_memory(history)
    system_content = SYSTEM_PROMPT + memory_block

    messages = [{"role": "system", "content": system_content}]

    for human, assistant in history:
        if human:
            messages.append({"role": "user", "content": human})
        if assistant:
            # Only append assistant messages that are plain text
            # Skip anything that looks like it contains tool call artifacts
            if isinstance(assistant, str):
                messages.append({"role": "assistant", "content": assistant})

    messages.append({"role": "user", "content": user_message})

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            response = _client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=TOOL_DEFINITIONS,
            )
        except Exception as e:
            print(f"  ✗ API error: {e}")
            return (
                "Something went wrong while processing your request. Please try again."
            )

        assistant_message = response.choices[0].message

        if not assistant_message.tool_calls:
            return assistant_message.content

        # Append raw object — never reconstruct this as a dict
        messages.append(assistant_message)

        for tool_call in assistant_message.tool_calls:
            tool_result = dispatch_tool(
                tool_call.function.name,
                json.loads(tool_call.function.arguments),
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

    # MAX_TOOL_ROUNDS hit
    print("  ✗ MAX_TOOL_ROUNDS reached")
    tool_results_collected = [
        m["content"]
        for m in messages
        if isinstance(m, dict) and m.get("role") == "tool"
    ]

    if tool_results_collected:
        messages.append(
            {
                "role": "user",
                "content": (
                    "You've reached the maximum number of tool calls. "
                    "Give the best answer you can with the information already retrieved. "
                    "Clearly note if any part of the question couldn't be fully addressed."
                ),
            }
        )
        try:
            final = _client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
            )
            return final.choices[0].message.content
        except Exception as e:
            print(f"  ✗ Final summary error: {e}")

    # Safety valve — hit MAX_TOOL_ROUNDS without a final answer
    return "🌱 Agent not yet implemented. Complete Milestone 2 to activate the Plant Advisor."
