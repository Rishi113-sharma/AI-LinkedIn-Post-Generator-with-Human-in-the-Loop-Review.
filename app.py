import streamlit as st
from workflow import app
from langgraph.types import Command


st.set_page_config(
    page_title="LinkedIn Post Generator",
    page_icon="💼"
)

st.title("LinkedIn Post Generator")
st.write("Generate and review a LinkedIn post using your LangGraph workflow.")


if "thread_id" not in st.session_state:
    st.session_state.thread_id = "linkedin_post_1"

if "waiting_for_review" not in st.session_state:
    st.session_state.waiting_for_review = False


topic = st.text_area(
    "Enter your topic",
    placeholder="Example: Impact of Artificial Intelligence on entry-level jobs"
)


if not st.session_state.waiting_for_review:

    if st.button("Generate Post"):

        if not topic.strip():
            st.warning("Please enter a topic.")

        else:

            initial_state = {
                "topic": topic,
                "message": [],
                "draft": "",
                "reviewer_feedback": "",
                "is_approved": False,
                "attempt": 0
            }

            config = {
                "configurable": {
                    "thread_id": st.session_state.thread_id
                }
            }

            with st.spinner("Generating LinkedIn post..."):

                result = app.invoke(
                    initial_state,
                    config=config
                )

            if "__interrupt__" in result:
                st.session_state.waiting_for_review = True
                st.rerun()


if st.session_state.waiting_for_review:

    state = app.get_state({
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    })

    values = state.values

    st.subheader("Generated LinkedIn Post")

    st.write(values["draft"])

    st.subheader("Human Review")

    st.write(
        f"Attempt: {values['attempt']}"
    )

    human_response = st.text_input(
        "Type 'approved' to accept or enter feedback for a rewrite"
    )

    if st.button("Submit Review"):

        if not human_response.strip():
            st.warning("Please enter your approval or feedback.")

        else:

            config = {
                "configurable": {
                    "thread_id": st.session_state.thread_id
                }
            }

            with st.spinner("Processing review..."):

                result = app.invoke(
                    Command(resume=human_response),
                    config=config
                )

            if "__interrupt__" in result:

                st.rerun()

            else:

                final_state = app.get_state(config)
                values = final_state.values

                if values["is_approved"]:

                    st.success("APPROVED")

                    st.subheader("Final LinkedIn Post")
                    st.write(values["draft"])

                    st.session_state.waiting_for_review = False

                else:

                    st.warning("REJECTED")

                    st.write(
                        values["reviewer_feedback"]
                    )

                    st.session_state.waiting_for_review = True

                    st.rerun()