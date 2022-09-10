
FROM python:3.10.2

EXPOSE 8501

WORKDIR /app
ADD . /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# ARG DEBIAN_FRONTEND=noninteractive
# ARG DEBCONF_NOWARNINGS="yes"

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc

# RUN python -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

RUN python3 -m pip install -r requirements.txt



# FROM python:3.8.3-alpine

# RUN pip install --upgrade pip

# RUN adduser -D myuser
# USER myuser
# WORKDIR /home/myuser

# COPY --chown=myuser:myuser requirements.txt requirements.txt
# RUN pip install --user -r requirements.txt

# ENV PATH="/home/user/.local/bin:${PATH}"

# COPY --chown=myuser:myuser . .

# CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]