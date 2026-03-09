from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate 
from app.config import Config


class LLMService:
    def __init__(self, vectorstore):
        # 1. Initialize the Base Endpoint
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=Config.HUGGINGFACEHUB_API_TOKEN,
            temperature=0.1, # Lowered temperature for higher accuracy
            max_new_tokens=512
        )

        # 2. Wrap it in ChatHuggingFace
        self.chat_model = ChatHuggingFace(llm=llm)

        # 3. Setup Memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # 4. Define a Strict Prompt Template
        # This forces the AI to only use the text found in Archi's resume or the paper
        template = """You are a professional assistant for a Knowledge Management System. 
        Use the following pieces of retrieved context to answer the question. 
        If the answer is not in the context, say that you don't know. 
        Do not use outside knowledge.
        CRITICAL: Ignore the 'References' or 'Bibliography' sections if they conflict with the main text.

        Context: {context}

        Question: {question}
        
        Helpful Answer:"""
        
        QA_PROMPT = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )

        # 5. Create the Chain with the custom prompt
        self.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=self.chat_model,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), # Increased k
        memory=self.memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )

    def get_response(self, question):
        try:
            # Debugging step to see what Archi's resume/paper chunks look like
            docs = self.qa_chain.retriever.invoke(question)
            
            print(f"\n--- DEBUG: CONTEXT RETRIEVED ---")
            for i, d in enumerate(docs):
                print(f"Chunk {i+1}: {d.page_content[:200]}...")
            print(f"---------------------------------\n")
            
            response = self.qa_chain.invoke({"question": question})
            return response['answer']
            
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"