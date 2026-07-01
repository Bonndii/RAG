import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage
from chains import build_conversational_rag_chain
from ui import ask_mode, setup_from_uploaded_pdf, setup_from_existing_collection

@cl.on_chat_start
async def on_chat_start():
    mode = await ask_mode()

    if mode is None:
        await cl.Message(content="Режим не выбран. Перезапусти чат.").send()
        return
    
    if mode == "upload_pdf":
        vector_store = await setup_from_uploaded_pdf()

    elif mode == "existing_collection":
        vector_store = await setup_from_existing_collection()

    else:
        await cl.Message(content="Неизвестный режим.").send()
        return
    
    if vector_store is None:
        return
    
    chain = build_conversational_rag_chain(vector_store)

    cl.user_session.set("chain", chain)
    cl.user_session.set("chat_history", [])

    await cl.Message(content="База знаний подключена. Можно задавать вопросы.").send()

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")

    if chain is None:
        await cl.Message(
            content="База знаний не подключена. Перезапусти чат и выбери режим."
        ).send()
        return
    
    chat_history = cl.user_session.get("chat_history")

    cb = cl.AsyncLangchainCallbackHandler()

    result = await chain.ainvoke(
        {
            "question": message.content,
            "chat_history": chat_history
        },
        config={"callbacks": [cb]}
    )
    
    answer = result["answer"]
    source_docs = result["docs"]

    text_chunks = []
    if source_docs:
        for source_id, source_doc in enumerate(source_docs, start=1):
            page = source_doc.metadata.get("page", "?")
            source_name = f"source_{source_id}_page_{page}"

            text_chunks.append(
                cl.Text(
                    name = source_name,
                    content=source_doc.page_content
                )
            )

    if text_chunks:
        answer += "\n\nИсточники: " + ", ".join(chunk.name for chunk in text_chunks)
    
    chat_history.extend(
        [
            HumanMessage(content=message.content),
            AIMessage(content=answer),
        ]
    )
    
    cl.user_session.set("chat_history", chat_history[-8:])
    
    await cl.Message(content=answer, elements=text_chunks).send()