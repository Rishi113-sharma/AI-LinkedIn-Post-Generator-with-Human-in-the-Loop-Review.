**AI LinkedIn Post Generator with Human-in-the-Loop Review**

Developed a Generative AI-powered LinkedIn content generation system using **LangGraph and Qwen2.5-7B-Instruct**. Implemented a multi-step workflow that generates LinkedIn posts, uses **Tavily Search** for fresh web context when required, and pauses for **human review and approval** using LangGraph interrupts. Rejected drafts can be regenerated using human feedback, with a maximum of four attempts. Integrated **LangGraph checkpointing and thread-based state management** to maintain workflow state across Streamlit interactions. Optimized local LLM inference using **4-bit NF4 quantization and CUDA** and built a Streamlit interface for generating, reviewing, and approving posts.

**Tech Stack:** Python • Generative AI • LangGraph • Qwen2.5-7B-Instruct • LangChain • Hugging Face Transformers • Streamlit • Tavily Search • Tool Calling • Human-in-the-Loop • LangGraph Interrupts • Checkpointing • PyTorch • CUDA • BitsAndBytes • 4-bit NF4 Quantization • Prompt Engineering

