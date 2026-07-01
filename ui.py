import chainlit as cl

from embeddings import get_embeddings
from loaders import load_pdf_pages
from splitters import split_documents
from vectorstore import (
    get_collection_names,
    create_vectorstore_from_documents,
    load_existing_vectorstore,
)


async def ask_mode() -> str | None:
    response = await cl.AskActionMessage(
        content="Выбери режим работы:",
        actions=[
            cl.Action(
                name="upload_pdf",
                payload={"mode": "upload_pdf"},
                label="Загрузить PDF",
            ),
            cl.Action(
                name="existing_collection",
                payload={"mode": "existing_collection"},
                label="Использовать коллекцию",
            ),
        ],
        timeout=180,
    ).send()

    if not response:
        await cl.Message(content="Режим не выбран. Перезапусти чат.").send()
        return None

    return response["payload"]["mode"]


async def setup_from_uploaded_pdf():
    embeddings = get_embeddings()

    files = None

    while files is None:
        files = await cl.AskFileMessage(
            content="Загрузи PDF-файл",
            accept=["application/pdf"],
            max_size_mb=100,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Обрабатываю `{file.name}`...")
    await msg.send()

    docs = await cl.make_async(load_pdf_pages)(file.path, file.name)
    splits = split_documents(docs)

    collection_name = (f"pdf_{file.name}")

    vector_store = await cl.make_async(create_vectorstore_from_documents)(
        documents=splits,
        embeddings=embeddings,
        collection_name=collection_name,
    )

    msg.content = (
        f"Файл `{file.name}` обработан.\n"
        f"Коллекция: `{collection_name}`\n"
        f"Чанков: `{len(splits)}`"
    )
    await msg.update()

    return vector_store


async def setup_from_existing_collection():
    embeddings = get_embeddings()

    collections = await cl.make_async(get_collection_names)()
    collections = sorted(collections)

    if not collections:
        await cl.Message(
            content="В Qdrant нет коллекций. Сначала загрузи PDF."
        ).send()
        return None

    actions = [
        cl.Action(
            name=f"select_collection_{i}",
            payload={"collection_name": collection_name},
            label=collection_name,
        )
        for i, collection_name in enumerate(collections)
    ]

    response = await cl.AskActionMessage(
        content="Выбери коллекцию Qdrant:",
        actions=actions,
        timeout=180,
    ).send()

    if not response:
        await cl.Message(content="Коллекция не выбрана. Перезапусти чат.").send()
        return None

    collection_name = response["payload"]["collection_name"]

    vector_store = load_existing_vectorstore(
        embeddings=embeddings,
        collection_name=collection_name,
    )

    await cl.Message(
        content=f"Подключена коллекция: `{collection_name}`"
    ).send()

    return vector_store