# Use NVIDIA NGC PyTorch image
FROM nvcr.io/nvidia/pytorch:24.11-py3

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OPENCV_LOG_LEVEL=ERROR
ENV OPENCV_VIDEOIO_DEBUG=0
ENV TZ="Asia/Tokyo"

# Copy dependency files
WORKDIR /app
COPY pyproject.toml .

# Install additional dependencies
RUN pip install -U "accelerate>=0.34.2" \
    "bitsandbytes>=0.43.3" \
    "datasets>=3.0.0" \
    "huggingface-hub[all,cli,hf-transfer]>=0.24.7" \
    "humanize>=4.10.0" \
    "ipykernel>=6.29.5" \
    "ipywidgets>=8.1.5" \
    "loguru>=0.7.2" \
    "methodtools>=0.4.7" \
    "nltk>=3.9.1" \
    "pycountry>=24.6.1" \
    "pydantic>=2.9.1" \
    "python-dotenv>=1.0.1" \
    "pyyaml>=6.0.2" \
    "seaborn>=0.13.2" \
    "setuptools>=75.3.0" \
    "tqdm>=4.66.5" \
    "transformers>=4.44.2"

# Set default command
CMD ["python3"]