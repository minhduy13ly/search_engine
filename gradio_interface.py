import gradio as gr
import requests


# Define the function to interact with FastAPI
def search_documents(query):
    url = "http://127.0.0.1:8000/api/search"  # FastAPI endpoint

    # Send POST request to FastAPI
    response = requests.post(url, json={"query": query})

    if response.status_code == 200:
        # Parse the response
        result = response.json()
        return f"Document ID: {result['doc_id']}\nSimilar Score: {round(result['score'], 3)}\nText: {result['text']}"
    else:
        return f"Error: {response.status_code}, {response.text}"


# Create the Gradio interface
interface = gr.Interface(
    fn=search_documents,
    inputs="text",
    outputs="text",
    title="Document Search System",
    description="Enter a query to search for relevant documents.",
    allow_flagging='never'  # Disable the flag button
)
# Launch the Gradio interface
interface.launch(server_name="0.0.0.0", server_port=7860)