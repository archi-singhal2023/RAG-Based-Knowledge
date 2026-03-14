from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate 
from app.config import Config

class LLMService:
    def __init__(self, vectorstore):
        # 1. Initialize the Base Endpoint
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=Config.HUGGINGFACEHUB_API_TOKEN,
            temperature=0.1, 
            max_new_tokens=512
        )

        # 2. Wrap it in ChatHuggingFace
        self.chat_model = ChatHuggingFace(llm=llm)
        self.vectorstore = vectorstore

        #3. Strict instructions to ignore citations
        # This forces the AI to only use the text found in Archi's resume or the paper
        self.template = """You are a professional assistant for a Knowledge Management System. Answer based ONLY on the context.
        
        STRICT EVALUATION RULES:
        1. Primary Identity: The TITLE and AUTHORS of this paper are always on the first page.
        2. Citation Filter: Do NOT cite papers from the 'References' section as the current paper.
        3. Accuracy: If the answer is not in the context, say you don't know.
        Context: {context}
        Question: {question}
        Answer:"""
        self.prompt = PromptTemplate(template=self.template, input_variables=["context", "question"])

    def get_response(self, question):
        # Level 2 Intent Check: Identify if user is asking for metadata
        id_query = any(k in question.lower() for k in ["title", "author", "name of paper"])
        
        # k=10 for identity to capture the whole title page; k=5 for general content
        search_k = 10 if id_query else 5
        docs = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": search_k}).invoke(question)
        
        context = "\n\n".join([d.page_content for d in docs])
        return self.chat_model.invoke(self.prompt.format(context=context, question=question)).content