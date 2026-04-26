FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PORT=7860

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends libgomp1 git \
  && rm -rf /var/lib/apt/lists/*

# Install Git LFS
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
  && apt-get install -y --no-install-recommends git-lfs \
  && git lfs install

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . /app

# Configure Git for LFS
RUN git config --global credential.helper store

# Pull LFS files with authentication
RUN git lfs pull --include="models/*.pth"

EXPOSE 7860

CMD ["gunicorn", "--workers", "1", "--threads", "1", "--bind", "0.0.0.0:7860", "app:app"]
