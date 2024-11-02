# Start FastAPI in the background
echo "Starting FastAPI..."
uvicorn server:app --host 127.0.0.1 --port 8000 --reload &

# Store the PID of FastAPI so we can stop it later
FASTAPI_PID=$!

# Start Gradio in the foreground
echo "Starting Gradio..."
python gradio_interface.py

# When Gradio is closed, kill the FastAPI process
kill $FASTAPI_PID