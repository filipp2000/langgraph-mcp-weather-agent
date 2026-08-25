import sys
from pathlib import Path

# Add project root (MCPcourse/) to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import asyncio

import streamlit as st

from agent.graph import run_agent

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------

st.set_page_config(
    page_title="MCP Weather Agent",
    page_icon="🌦️",
    layout="centered",
)


st.title("🌦️ MCP Weather Agent")

st.caption("LangGraph · Groq · Model Context Protocol · NWS")


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("Agent")

    st.markdown(
        """
        **Model:** GPT-OSS 20B  
        **LLM Provider:** Groq  
        **Orchestration:** LangGraph  
        **Tools:** MCP  
        **Weather API:** NWS
        """
    )

    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------
# Render previous conversation
# ---------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Show debugging information for assistant responses
        if message["role"] == "assistant" and message.get("tool_trace"):
            with st.expander(f"🔧 MCP tool calls ({len(message['tool_trace'])})"):
                for index, call in enumerate(
                    message["tool_trace"],
                    start=1,
                ):
                    st.markdown(f"**{index}. `{call['name']}`**")

                    st.caption("Arguments")

                    st.json(call["arguments"])

                    if call["is_error"]:
                        st.error("Tool returned an error.")

        st.markdown(message["content"])


# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------

if prompt := st.chat_input("Ask about US weather or weather alerts..."):
    # Keep history BEFORE current prompt.
    history = st.session_state.messages.copy()

    # Store/display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # -----------------------------------------------------
    # Run agent
    # -----------------------------------------------------

    with st.chat_message("assistant"):
        try:
            with st.status(
                "Agent is working...",
                expanded=False,
            ) as status:
                result = asyncio.run(
                    run_agent(
                        history=history,
                        query=prompt,
                    )
                )

                tool_trace = result["tool_trace"]

                answer = result["answer"]

                status.update(
                    label="Completed",
                    state="complete",
                )

            # ---------------------------------------------
            # Debug / MCP tool calls
            # ---------------------------------------------

            if tool_trace:
                with st.expander(
                    f"🔧 MCP tool calls ({len(tool_trace)})",
                    expanded=True,
                ):
                    for index, call in enumerate(
                        tool_trace,
                        start=1,
                    ):
                        st.markdown(f"**{index}. `{call['name']}`**")

                        st.caption("Arguments")

                        st.json(call["arguments"])

                        if call["is_error"]:
                            st.error("Tool returned an error.")

            # ---------------------------------------------
            # Final formatted answer
            # ---------------------------------------------

            st.markdown(answer)

            # Save assistant response
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "tool_trace": tool_trace,
                }
            )

        except Exception as e:
            st.error(f"Agent error: {e}")
        
        
