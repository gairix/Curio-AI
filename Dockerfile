# Step 1: Use an official, lightweight Python blueprint base
FROM python:3.10-slim

# Step 2: Install system-level binaries needed for audio processing and layout tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Establish the active working folder inside our container environment
WORKDIR /app

# Step 4: Copy over just the requirements list first (optimizes Docker build caching)
COPY requirements.txt /app/

# Step 5: Install all Python dependencies smoothly inside the box
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Step 6: Copy the rest of your local source code files into the container
COPY . /app/

# Step 7: Open the standard network port that Streamlit communicates through
EXPOSE 8501

# Step 8: Define the network commands to launch your application live
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]