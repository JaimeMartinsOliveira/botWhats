import os
import shutil
import asyncio
import logging
import time
from typing import Optional, Dict, Any, Set
from dataclasses import dataclass

import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("whatsapp_bot")

load_dotenv()


@dataclass
class Config:
    """Configurações da aplicação"""
    openai_api_key: str
    openai_model: str
    openai_temperature: float
    contextualize_prompt: str
    system_prompt: str
    vector_store_path: str
    rag_files_dir: str
    evolution_api_url: str
    evolution_instance_name: str
    evolution_api_key: str
    redis_url: str
    debounce_seconds: float

    @classmethod
    def from_env(cls) -> 'Config':
        required_vars = [
            'OPENAI_API_KEY',
            'EVOLUTION_API_URL',
            'EVOLUTION_INSTANCE_NAME',
            'EVOLUTION_AUTHENTICATION_API_KEY'
        ]

        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Variáveis obrigatórias não encontradas: {missing}")

        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
            openai_temperature=float(os.getenv("OPENAI_MODEL_TEMPERATURE", "0.3")),
            contextualize_prompt=os.getenv("AI_CONTEXTUALIZE_PROMPT", "Reformule a pergunta se necessário"),
            system_prompt=os.getenv("AI_SYSTEM_PROMPT", "Você é o Jaime Martins"),
            vector_store_path=os.getenv("VECTOR_STORE_PATH", "./vectorstore_data"),
            rag_files_dir=os.getenv("RAG_FILES_DIR", "./rag_files"),
            evolution_api_url=os.getenv("EVOLUTION_API_URL"),
            evolution_instance_name=os.getenv("EVOLUTION_INSTANCE_NAME"),
            evolution_api_key=os.getenv("EVOLUTION_AUTHENTICATION_API_KEY"),
            redis_url=os.getenv("CACHE_REDIS_URI", "redis://redis:6379/0"),
            debounce_seconds=float(os.getenv("DEBOUNCE_SECONDS", "3.0"))
        )


class DocumentService:
    """Gerencia documentos e vector store"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def load_documents(self):
        """Carrega documentos para o RAG"""
        docs = []
        processed_dir = os.path.join(self.config.rag_files_dir, 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(self.config.rag_files_dir, exist_ok=True)
        
        logger.info(f"📂 Procurando documentos em: {os.path.abspath(self.config.rag_files_dir)}")
        
        # Lista todos os arquivos no diretório
        try:
            all_files = os.listdir(self.config.rag_files_dir)
            logger.info(f"📋 Arquivos encontrados: {all_files}")
        except FileNotFoundError:
            logger.error(f"❌ Diretório não encontrado: {self.config.rag_files_dir}")
            return docs
        
        files = [
            os.path.join(self.config.rag_files_dir, f)
            for f in all_files
            if f.endswith(('.pdf', '.txt')) and os.path.isfile(os.path.join(self.config.rag_files_dir, f))
        ]
        
        logger.info(f"📄 Arquivos válidos encontrados: {[os.path.basename(f) for f in files]}")
        
        for file in files:
            try:
                logger.info(f"🔄 Carregando: {file}")
                
                if file.endswith('.pdf'):
                    loader = PyPDFLoader(file)
                else:
                    loader = TextLoader(file, encoding='utf-8')
                
                file_docs = loader.load()
                docs.extend(file_docs)
                
                # Log do conteúdo carregado
                for i, doc in enumerate(file_docs):
                    logger.info(f"📄 Doc {i+1}: {len(doc.page_content)} caracteres")
                
                # Move para processed
                dest_path = os.path.join(processed_dir, os.path.basename(file))
                shutil.move(file, dest_path)
                logger.info(f"✅ {os.path.basename(file)} processado e movido")
                
            except Exception as e:
                logger.error(f"❌ Erro ao carregar {file}: {e}")
        
        logger.info(f"🔎 Total: {len(docs)} documentos carregados")
        return docs
    
    def build_vectorstore(self):
        """Constrói vector store"""
        try:
            embeddings = OpenAIEmbeddings(
                openai_api_key=self.config.openai_api_key,
                model="text-embedding-3-small"
            )
            
            # Sempre tenta carregar documentos primeiro
            docs = self.load_documents()
            
            if docs:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=100,
                    separators=["\n\n", "\n", ".", " "]
                )
                splits = text_splitter.split_documents(docs)
                
                logger.info(f"📊 Criando {len(splits)} chunks dos documentos")
                
                vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings,
                    persist_directory=self.config.vector_store_path,
                )
                logger.info(f"✅ Vector store criado com {len(splits)} chunks")
                return vectorstore
            else:
                # Verifica se existe vector store anterior
                if os.path.exists(self.config.vector_store_path):
                    try:
                        vectorstore = Chroma(
                            embedding_function=embeddings,
                            persist_directory=self.config.vector_store_path
                        )
                        logger.info("📦 Vector store anterior carregado")
                        return vectorstore
                    except Exception as e:
                        logger.warning(f"⚠️ Erro ao carregar vector store anterior: {e}")
                
                logger.warning("⚠️ Criando vector store vazio")
                return Chroma(
                    embedding_function=embeddings,
                    persist_directory=self.config.vector_store_path,
                )
                
        except Exception as e:
            logger.error(f"❌ Erro crítico no vector store: {e}")
            return Chroma(
                embedding_function=OpenAIEmbeddings(openai_api_key=self.config.openai_api_key),
                persist_directory=self.config.vector_store_path,
            )


class WhatsAppBot:
    """Bot principal do WhatsApp"""
    
    def __init__(self, config: Config):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Sistema anti-duplicação
        self.processed_messages: Set[str] = set()
        self.processing_lock = asyncio.Lock()
        
        # Inicializa RAG
        self._setup_rag()

    def _setup_rag(self):
        """Configura o sistema RAG"""
        # Carrega documentos
        doc_service = DocumentService(self.config)
        self.vectorstore = doc_service.build_vectorstore()

        # Configura LLM
        self.llm = ChatOpenAI(
            api_key=self.config.openai_api_key,
            model=self.config.openai_model,
            temperature=self.config.openai_temperature
        )

        # Prompts otimizados
        self.contextualize_prompt = ChatPromptTemplate.from_messages([
            ('system', self.config.contextualize_prompt),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}'),
        ])

        # Prompt melhorado para respostas concisas
        improved_system_prompt = """Você é Jaime Martins respondendo a um recrutador no WhatsApp.

REGRAS IMPORTANTES:
- Seja CONCISO (máximo 3-4 linhas)
- Responda de forma NATURAL e DIRETA
- Use as informações do contexto fornecido
- Seja profissional mas amigável
- Use emojis moderadamente

SUAS INFORMAÇÕES PRINCIPAIS:
- Formado em Análise e Desenvolvimento de Sistemas (Unisul, 2024)
- Cursando Engenharia de Software (PUC Minas, conclusão 2026)  
- Desenvolvedor Back-end Python especialista em FastAPI, Django
- Experiência com IA, APIs REST, PostgreSQL, Redis, Docker
- Freelancer na Workana desde 2023

Use o contexto abaixo para responder com precisão:"""

        self.qa_prompt = ChatPromptTemplate.from_messages([
            ('system', improved_system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}'),
            ('system', "CONTEXTO DO CURRÍCULO:\n{context}"),
        ])

        # Constrói cadeia RAG com configurações otimizadas
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",  # Mudou para similarity (mais simples)
            search_kwargs={"k": 4}  # Aumentou para 4 documentos
        )

        history_aware = create_history_aware_retriever(
            self.llm, retriever, self.contextualize_prompt
        )
        qa_chain = create_stuff_documents_chain(llm=self.llm, prompt=self.qa_prompt)
        retrieval_chain = create_retrieval_chain(history_aware, qa_chain)

        self.rag_chain = RunnableWithMessageHistory(
            runnable=retrieval_chain,
            get_session_history=self._get_session_history,
            input_messages_key='input',
            history_messages_key='chat_history',
            output_messages_key='answer',
        )

        # Testa o RAG
        self._test_rag()
        logger.info("🤖 Sistema RAG configurado com sucesso")

    def _test_rag(self):
        """Testa se o RAG está funcionando"""
        try:
            # Lista documentos no vector store
            try:
                collection = self.vectorstore._collection
                count = collection.count()
                logger.info(f"📊 Vector store contém {count} documentos")
            except:
                logger.info("📊 Não foi possível contar documentos no vector store")
            
            # Testa busca
            test_queries = ["formação graduação", "experiência profissional", "desenvolvedor python"]
            
            for query in test_queries:
                try:
                    test_docs = self.vectorstore.similarity_search(query, k=2)
                    if test_docs:
                        logger.info(f"✅ Query '{query}': {len(test_docs)} docs encontrados")
                        logger.info(f"📄 Exemplo: {test_docs[0].page_content[:100]}...")
                        break
                    else:
                        logger.info(f"⚠️ Query '{query}': nenhum doc encontrado")
                except Exception as e:
                    logger.error(f"❌ Erro na query '{query}': {e}")
            
            if not any(self.vectorstore.similarity_search(q, k=1) for q in test_queries):
                logger.warning("⚠️ RAG não encontrou documentos em nenhum teste")
            
        except Exception as e:
            logger.error(f"❌ Erro no teste RAG: {e}")

    def _get_session_history(self, session_id: str):
        """Obtém histórico da sessão"""
        return RedisChatMessageHistory(session_id=session_id, url=self.config.redis_url)

    async def start(self):
        """Inicia o bot"""
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        logger.info("🚀 Bot iniciado com sucesso!")

    async def stop(self):
        """Para o bot"""
        if self.session:
            await self.session.close()
        logger.info("🛑 Bot finalizado")

    def _create_message_id(self, chat_id: str, message: str) -> str:
        """Cria ID único para evitar duplicatas"""
        time_window = int(time.time() / 15) * 15
        return f"{chat_id}:{hash(message)}:{time_window}"

    async def process_webhook(self, payload: Dict[str, Any]) -> Dict[str, str]:
        """Processa webhook da Evolution API"""
        try:
            # Valida evento
            if payload.get("event") != "messages.upsert":
                return {"status": "ignored", "reason": "not_message_event"}

            data = payload.get("data", {})
            message_info = data.get("message", {})
            key_info = data.get("key", {})

            # Extrai dados
            chat_id = key_info.get("remoteJid")
            message_text = message_info.get("conversation")
            is_from_me = key_info.get("fromMe", False)

            # Validações
            if not chat_id or not message_text:
                return {"status": "ignored", "reason": "missing_data"}

            if is_from_me:
                return {"status": "ignored", "reason": "own_message"}

            if chat_id.endswith("@g.us"):
                return {"status": "ignored", "reason": "group_message"}

            # Anti-duplicação
            message_id = self._create_message_id(chat_id, message_text)

            async with self.processing_lock:
                if message_id in self.processed_messages:
                    logger.info(f"🔄 Mensagem duplicada ignorada: {chat_id}")
                    return {"status": "duplicate_ignored"}

                self.processed_messages.add(message_id)

            logger.info(f"📨 Nova mensagem de {chat_id}: {message_text}")

            # Processa em background
            asyncio.create_task(self._handle_message(chat_id, message_text))

            # Limpa cache
            if len(self.processed_messages) > 200:
                old_messages = list(self.processed_messages)[:50]
                for msg in old_messages:
                    self.processed_messages.discard(msg)

            return {"status": "processing"}

        except Exception as e:
            logger.error(f"❌ Erro no webhook: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    async def _send_message(self, number: str, text: str) -> bool:
        """Envia mensagem via Evolution API com delay fixo de 10s"""
        url = f"{self.config.evolution_api_url}/message/sendText/{self.config.evolution_instance_name}"
        headers = {
            "apikey": self.config.evolution_api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "number": number,
            "text": text,
            "delay": 60  # Delay fixo de 10 segundos
        }

        try:
            async with self.session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    logger.info(f"📤 Enviado para {number} com 10s delay: {text[:80]}...")
                    return True
                else:
                    response_text = await response.text()
                    logger.error(f"❌ Erro no envio - Status {response.status}: {response_text}")
                    return False

        except Exception as e:
            logger.error(f"❌ Erro ao enviar mensagem: {e}")
            return False
    async def _handle_message(self, chat_id: str, message: str):
        """Processa mensagem com RAG"""
        try:
            # Debounce
            await asyncio.sleep(self.config.debounce_seconds)
            
            logger.info(f"🤖 Processando com RAG: {message[:50]}...")
            start_time = time.time()

            # Testa retrieval primeiro (opcional para debug)
            try:
                relevant_docs = self.vectorstore.similarity_search(message, k=3)
                logger.info(f"🔍 RAG encontrou {len(relevant_docs)} documentos relevantes")
            except Exception as e:
                logger.error(f"❌ Erro no retrieval: {e}")

            # Processa com RAG
            response = await asyncio.wait_for(
                self.rag_chain.ainvoke(
                    {"input": message},
                    config={"configurable": {"session_id": chat_id}}
                ),
                timeout=30
            )
            
            processing_time = round(time.time() - start_time, 2)
            answer = response.get("answer", "").strip()

            if not answer:
                logger.warning("⚠️ RAG retornou resposta vazia, usando fallback")
                answer = self._get_fallback_response(message)

            logger.info(f"✅ Resposta gerada em {processing_time}s: {answer[:100]}...")

            # Envia resposta (Evolution API fará o delay/"digitando" automaticamente)
            number = self._extract_number(chat_id)
            await self._send_message(number, answer)

        except asyncio.TimeoutError:
            logger.error(f"⏱️ Timeout no processamento para {chat_id}")
            await self._send_timeout_response(chat_id)
        except Exception as e:
            logger.error(f"❌ Erro no processamento: {e}", exc_info=True)
            await self._send_error_response(chat_id)

    async def _send_timeout_response(self, chat_id: str):
        """Resposta de timeout"""
        message = "Aguarde um momento, estou processando sua mensagem! 🤖"
        number = self._extract_number(chat_id)
        await self._send_message(number, message)

    async def _send_error_response(self, chat_id: str):
        """Resposta de erro"""
        message = "Ops! Tive um problema técnico. Pode repetir? 🔧"
        number = self._extract_number(chat_id)
        await self._send_message(number, message)

    def _extract_number(self, chat_id: str) -> str:
        """Extrai número do chat_id"""
        return chat_id.split('@')[0]

    def _get_fallback_response(self, original_message: str) -> str:
        """Resposta de fallback inteligente baseada na mensagem"""
        message_lower = original_message.lower()

        if any(word in message_lower for word in ['formação', 'formado', 'graduação', 'faculdade', 'curso']):
            return (
                "Sim! Sou formado em Análise e Desenvolvimento de Sistemas pela Unisul (2024) "
                "e estou cursando Engenharia de Software na PUC Minas 🎓"
            )
        elif any(word in message_lower for word in ['experiência', 'trabalho', 'emprego', 'carreira']):
            return (
                "Trabalho como Desenvolvedor Back-end freelancer na Workana desde 2023, "
                "focado em Python, APIs REST e IA Generativa 💻"
            )
        elif any(word in message_lower for word in ['tecnologia', 'linguagem', 'stack', 'ferramenta']):
            return (
                "Trabalho principalmente com Python, FastAPI, Django, PostgreSQL, "
                "Redis, Docker e ferramentas de IA como LangChain 🚀"
            )
        else:
            return (
                "Oi! Eu sou o Jaime Martins 👋\n"
                "Desenvolvedor Back-end Python especialista em IA e APIs REST.\n"
                "O que gostaria de saber sobre minha experiência? 😊"
            )


# Configuração global
config = Config.from_env()
bot = WhatsAppBot(config)

# FastAPI App
app = FastAPI(
    title="WhatsApp Bot RAG - Jaime Martins",
    description="Bot com RAG para conversar com recrutadores",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    await bot.start()


@app.on_event("shutdown")
async def shutdown_event():
    await bot.stop()


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "WhatsApp Bot RAG - Jaime Martins",
        "version": "2.1.0",
        "rag_enabled": True
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "bot_active": bot.session is not None,
        "vectorstore_ready": bot.vectorstore is not None,
        "timestamp": time.time()
    }


@app.get("/test-rag")
async def test_rag():
    """Endpoint para testar o RAG"""
    try:
        test_query = "formação acadêmica"
        docs = bot.vectorstore.similarity_search(test_query, k=3)
        return {
            "query": test_query,
            "found_docs": len(docs),
            "docs": [{"content": doc.page_content[:200], "metadata": doc.metadata} for doc in docs]
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/debug-files")
async def debug_files():
    """Debug dos arquivos no diretório"""
    try:
        rag_dir = config.rag_files_dir
        files_info = {
            "rag_dir_path": os.path.abspath(rag_dir),
            "rag_dir_exists": os.path.exists(rag_dir),
            "files": []
        }
        
        if os.path.exists(rag_dir):
            all_files = os.listdir(rag_dir)
            for f in all_files:
                file_path = os.path.join(rag_dir, f)
                files_info["files"].append({
                    "name": f,
                    "is_file": os.path.isfile(file_path),
                    "size": os.path.getsize(file_path) if os.path.isfile(file_path) else 0,
                    "extension": os.path.splitext(f)[1]
                })
        
        return files_info
        
    except Exception as e:
        return {"error": str(e)}


@app.post("/webhook")
async def webhook(request: Request):
    """Endpoint principal do webhook"""
    try:
        payload = await request.json()
        result = await bot.process_webhook(payload)
        return result

    except Exception as e:
        logger.error(f"❌ Erro crítico no webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )