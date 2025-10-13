# build_faiss_vectorstores.py

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings 

# ⚠️ Keep your API key in .env later, for now it's hardcoded
# GEMINI_API_KEY = "AIzaSyAbYSZiclAZbshAznnMK7C2TrmWzxHnnYg"  

def build_vectorstore(input_dir, output_dir):
    print(f"📁 Processing folder: {input_dir}")
    documents = []
    skipped = []

    filenames = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    if not filenames:
        print("❌ No PDF files found in the directory.")
        return

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    chunk_size = 20  # Process in batches
    vector_stores = []

    for i in range(0, len(filenames), chunk_size):
        batch = filenames[i:i + chunk_size]
        batch_docs = []

        print(f"\n🔹 Processing batch {i // chunk_size + 1} ({len(batch)} files)")

        for file in batch:
            file_path = os.path.join(input_dir, file)
            try:
                loader = PyPDFLoader(file_path)
                loaded = loader.load()
                if loaded:
                    batch_docs.extend(loaded)
                    print(f" ✅ Loaded: {file} ({len(loaded)} pages)")
                else:
                    raise Exception("Empty or unreadable PDF")
            except Exception as e:
                print(f" ❌ Skipped {file}: {str(e)}")
                skipped.append(file)

        if batch_docs:
            try:
                vs = FAISS.from_documents(batch_docs, embeddings)
                vector_stores.append(vs)
            except Exception as e:
                print(f" ❌ Could not create vector store for this batch: {str(e)}")

    if not vector_stores:
        print("❌ No usable documents found. Nothing was indexed.")
        return

    print(f"\n🧠 Merging {len(vector_stores)} partial indexes...")
    merged_store = vector_stores[0]
    for vs in vector_stores[1:]:
        merged_store.merge_from(vs)

    os.makedirs(output_dir, exist_ok=True)
    merged_store.save_local(output_dir)
    print(f"\n✅ Vector store saved at: {os.path.abspath(output_dir)}")

    if skipped:
        print("\n⚠️ Skipped Files:")
        for file in skipped:
            print(f" - {file}")


if __name__ == "__main__":
    os.makedirs("vectorstores_1.0", exist_ok=True)  # ensure parent exists
    build_vectorstore("data", "vectorstores_1.0/acts")
