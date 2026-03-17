from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from app.config import Config
from langchain_openai import ChatOpenAI


class LLMService:
    """
    Level 2 — Intelligence Layer.
    Owns the LLM connection, prompt, memory, and retrieval logic.
    The LLM is initialized ONCE. The vectorstore and page-1 cache
    are swapped per document upload.
    """

    _llm = None
    _chat_model = None

    def __init__(self, vectorstore, page1_text: str = ""):
        self.vectorstore = vectorstore

        # FIX (Page 1 pinning): Store raw Page 1 text directly from the
        # loader output — bypasses the metadata filter entirely.
        # This is guaranteed to always contain the title and authors.
        self.page1_text = page1_text

        if LLMService._llm is None:
            print("🤖 Level 2: Initializing LLM (first time only)...")
            LLMService._llm = ChatOpenAI(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            openai_api_base="https://router.huggingface.co/featherless-ai/v1",
            openai_api_key=Config.HUGGINGFACEHUB_API_TOKEN,
            temperature=0.1,
            max_tokens=512
        )
            print("✅ Level 2: LLM ready.")

        self.chat_model = LLMService._llm

        # Plain list-based memory — no LangChain memory class needed.
        self.chat_history: list[dict] = []
        self.max_history = 5

        self.QA_PROMPT = PromptTemplate(
            template="""You are DocuMind, a precise Research Assistant. Answer questions using ONLY the document context below. Never use outside knowledge or make up information.

STRICT RULES — follow without exception:
1. TITLE & AUTHORS: These are found in [PAGE 1 - GUARANTEED METADATA] below. Always use this for identity questions. The title is the large heading. The authors are listed below it.
2. DO NOT use titles or author names that appear only in the References, Related Work, or Citations sections of the document.
3. HONESTY: If the exact answer is not present in the context, say: "I couldn't find that information in this document."
4. NO HALLUCINATION: Do not guess, infer, or generate any information not explicitly stated in the context.
5. FOLLOW-UPS: Use the Chat History to understand pronouns like "the paper", "the authors", "their method".

--- CHAT HISTORY (last {history_count} exchanges) ---
{chat_history}

--- [PAGE 1 - GUARANTEED METADATA - Use this for title/author questions] ---
{page1_context}

--- [RETRIEVED CONTEXT for the question] ---
{retrieved_context}

--- QUESTION ---
{question}

--- ANSWER ---""",
            input_variables=["history_count", "chat_history", "page1_context",
                             "retrieved_context", "question"]
        )

    def update_vectorstore(self, new_vectorstore, new_page1_text: str = ""):
        """Swaps vectorstore and page-1 cache. Clears conversation history."""
        self.vectorstore = new_vectorstore
        self.page1_text = new_page1_text
        self.chat_history = []
        print("🔄 Level 2: Vectorstore and page-1 cache updated, history cleared.")

    def _format_history(self) -> str:
        if not self.chat_history:
            return "None"
        lines = []
        for turn in self.chat_history:
            lines.append(f"Human: {turn['human']}")
            lines.append(f"AI: {turn['ai']}")
        return "\n".join(lines)

    def get_response(self, question: str) -> str:
        """
        Two-context strategy:
          - page1_context: always the raw first page text (title/authors guaranteed)
          - retrieved_context: MMR results for the actual question
        These are kept separate in the prompt so the LLM always sees Page 1
        regardless of what MMR returns.
        """
        try:
            # MMR retrieval for the actual question
            mmr_docs = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 6, "fetch_k": 20}
            ).invoke(question)

            retrieved_context = "\n\n---\n\n".join([
                f"[Page {d.metadata.get('page', '?') + 1}]\n{d.page_content}"
                for d in mmr_docs
            ])

            history_str = self._format_history()

            full_prompt = self.QA_PROMPT.format(
                history_count=len(self.chat_history),
                chat_history=history_str,
                page1_context=self.page1_text or "Not available.",
                retrieved_context=retrieved_context,
                question=question
            )

            answer = self.chat_model.invoke(full_prompt).content

            self.chat_history.append({"human": question, "ai": answer})
            if len(self.chat_history) > self.max_history:
                self.chat_history.pop(0)

            return answer

        except Exception as e:
            print(f"❌ Level 2 LLM Error: {e}")
            return "Sorry, I encountered an error while processing your question. Please try again."